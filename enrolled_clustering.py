"""
EnrolledSpeakerClustering — extends diart's OnlineSpeakerClustering with:

1. **Centroid injection**: pre-fill centroids from enrollment embeddings so that
   known speakers are immediately recognised on the first chunk.
2. **Frozen enrolled centroids**: enrolled centroids are NEVER updated — they
   always stay at their enrollment anchor.
3. **Enrolled-priority identify()**: enrolled speakers are matched FIRST using
   ``delta_enrolled`` *before* the general distance map
   is computed for remaining speakers.  This prevents unknown centroids from
   stealing enrolled identities.
4. **Label mapping**: ``get_label(global_idx)`` returns the enrolled speaker
   name or ``"Unknown-N"`` for strangers.
5. **Exposed SpeakerMap**: ``last_speaker_map`` is stored after every
   ``__call__`` so the pipeline can map PixIT source channels to global
   speakers.

Continuity with RealtimePipeline (why this file matters for “choppiness”)
────────────────────────────────────────────────────────────────────────
- **tau_active** (config ``clustering.tau_active``): a local PixIT channel is
  “active” only if max frame score ≥ tau.  If tau is high, channels drop in and
  out → ``valid_assignments()`` shrinks → pipeline may emit no audio for a step.
  Lower tau (e.g. 0.28 vs 0.4) reduces empty-map steps; pipeline **holdover**
  (see ``pipeline.py``) covers the remaining flicker for enrolled speakers only.

- **Grace period** (``_grace_steps``, ``_last_enrolled_step``): for a short
  window after an enrolled match, only the **previously used local channel**
  gets a small extra distance margin so brief silence→speech transitions do not
  lose the enrolled ID.

- **Channel persistence** (``_last_local_for_enrolled``): when several locals
  are within threshold of the same enrolled anchor, we prefer the **same**
  PixIT channel as last step.  Otherwise we choose the **closest** candidate,
  not merely the loudest one.  Switching channels every 500 ms caused audible
  discontinuities and unstable maps; persistence aligns with the pipeline’s
  per-global crossfade state.

- **Enrolled leakage suppression**: active locals that are still within a
  **stricter duplicate threshold** of an enrolled anchor but **lost** the
  greedy enrolled pass are treated as duplicate/leakage of that voice on a
  secondary separation channel — they are blocked from general matching so one
  enrolled speaker does not split into two globals or steal unknown slots.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from diart.blocks.clustering import OnlineSpeakerClustering
from diart.mapping import SpeakerMap, SpeakerMapBuilder
from pyannote.core import SlidingWindowFeature
from scipy.spatial.distance import cdist

log = logging.getLogger(__name__)


class EnrolledSpeakerClustering(OnlineSpeakerClustering):
    """OnlineSpeakerClustering with enrolled-priority identification.

    Key difference from base class: ``identify()`` is fully overridden so that
    enrolled speakers are matched **before** the general distance map is built.
    This guarantees that no unknown centroid can out-compete a frozen enrolled
    anchor, regardless of how many embeddings accumulate in that unknown slot.

    Parameters
    ----------
    tau_active : float
        Min activation to consider a local speaker active.
    rho_update : float
        Min mean activation to update a centroid.
    delta_new : float
        Max cosine distance before a new global speaker is created.
    delta_enrolled : float
        Max cosine distance to match an active speaker to an enrolled anchor.
        Should be more lenient than ``delta_new`` (e.g. 0.80 vs 0.55) to
        account for domain gap between enrollment and live separated sources.
    metric : str
        Distance metric (default ``"cosine"``).
    max_speakers : int
        Max global speakers to track.
    """

    def __init__(
        self,
        tau_active: float = 0.5,
        rho_update: float = 0.3,
        delta_new: float = 1.0,
        metric: str = "cosine",
        max_speakers: int = 8,
        max_drift: float = 0.0,   # kept for config compat, unused
        delta_enrolled: float = 0.80,
        enrolled_grace_margin: float = 0.05,
        leakage_delta: Optional[float] = None,
        new_center_grace_margin: float = 0.05,
        enrolled_preference_margin: float = 0.03,
        unknown_reuse_delta: Optional[float] = None,
        unknown_min_voiced_sec: float = 0.18,
        unknown_min_tail_rms: float = 0.0,
        enrolled_session_alpha: float = 0.15,
    ):
        super().__init__(tau_active, rho_update, delta_new, metric, max_speakers)
        self.delta_enrolled = delta_enrolled
        self.enrolled_grace_margin = enrolled_grace_margin
        self.leakage_delta = (
            leakage_delta if leakage_delta is not None
            else min(delta_enrolled, delta_new)
        )
        self.new_center_grace_margin = new_center_grace_margin
        self.enrolled_preference_margin = enrolled_preference_margin
        self.unknown_reuse_delta = (
            unknown_reuse_delta if unknown_reuse_delta is not None
            else max(delta_new, min(delta_enrolled, delta_new + 0.10))
        )
        self.unknown_min_voiced_sec = unknown_min_voiced_sec
        self.unknown_min_tail_rms = unknown_min_tail_rms
        self.enrolled_session_alpha = enrolled_session_alpha

        # Enrollment state
        self._anchors: Dict[int, np.ndarray] = {}      # global_idx → L2-normed anchor
        self._profiles: Dict[int, np.ndarray] = {}     # global_idx → (k, dim) profile vectors
        self._session_centroids: Dict[int, np.ndarray] = {}  # global_idx → live EMA
        self._session_counts: Dict[int, int] = {}
        self._labels: Dict[int, str] = {}               # global_idx → speaker name
        self._unknown_counter: int = 0
        self._current_source_stats: Optional[list[dict]] = None

        # Populated after every __call__
        self.last_speaker_map: Optional[SpeakerMap] = None
        self._call_count: int = 0

        # Grace period: track when each enrolled speaker was last matched.
        # During a grace window after the last match, only the previously used
        # local channel gets a small extra distance margin.
        self._last_enrolled_step: Dict[int, int] = {}   # g_idx → call_count
        self._grace_steps: int = 20                      # ~10 s at 0.5 s step

        # Channel persistence: prefer the same local speaker as last step to
        # prevent source switching (which causes audio discontinuities).
        self._last_local_for_enrolled: Dict[int, int] = {}  # g_idx → local_spk

    # ─── Enrollment ──────────────────────────────────────────────

    def inject_centroids(self, enrolled: Dict[str, np.ndarray]) -> None:
        """Pre-fill centroids from enrollment embeddings.

        Must be called **before** the first ``__call__()``.
        """
        if not enrolled:
            return

        emb_dim = next(iter(enrolled.values())).shape[-1]
        self.init_centers(emb_dim)

        for name, emb in enrolled.items():
            emb = np.asarray(emb, dtype=np.float64).ravel()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            idx = self.add_center(emb)
            self._anchors[idx] = emb.copy()
            self._profiles[idx] = emb.reshape(1, -1).copy()
            self._session_centroids[idx] = emb.copy()
            self._session_counts[idx] = 0
            self._labels[idx] = name
            # Seed grace period so the first few steps can reuse the initial
            # local channel with a small extra margin.
            self._last_enrolled_step[idx] = 0

    def inject_profiles(self, profiles: Dict[str, np.ndarray]) -> None:
        """Attach additional enrollment profile vectors per enrolled speaker."""
        if not profiles:
            return
        label_to_idx = {label: idx for idx, label in self._labels.items()}
        for name, prof in profiles.items():
            idx = label_to_idx.get(name)
            if idx is None:
                continue
            prof = np.asarray(prof, dtype=np.float64)
            if prof.ndim == 1:
                prof = prof.reshape(1, -1)
            norms = np.linalg.norm(prof, axis=1, keepdims=True)
            prof = prof / np.maximum(norms, 1e-12)
            self._profiles[idx] = prof.copy()
            if idx not in self._session_centroids:
                self._session_centroids[idx] = prof.mean(axis=0)
                self._session_counts[idx] = 0

    # ─── Labels ──────────────────────────────────────────────────

    def get_label(self, global_idx: int) -> str:
        if global_idx in self._labels:
            return self._labels[global_idx]
        label = f"Unknown-{global_idx}"
        self._labels[global_idx] = label
        return label

    def get_all_labels(self) -> Dict[int, str]:
        return dict(self._labels)

    def is_enrolled(self, global_idx: int) -> bool:
        return global_idx in self._anchors

    # ─── Centroid management ─────────────────────────────────────

    def _freeze_enrolled(self):
        """Reset enrolled centroids to their anchors."""
        if self.centers is not None:
            for idx, anchor in self._anchors.items():
                self.centers[idx] = anchor.copy()

    def update(
        self, assignments: Iterable[Tuple[int, int]], embeddings: np.ndarray
    ) -> None:
        """Update centroids — enrolled speakers are FROZEN (skipped)."""
        if self.centers is None:
            return
        for l_spk, g_spk in assignments:
            if g_spk not in self.active_centers:
                continue
            if g_spk in self._anchors:
                continue   # never update enrolled centroids
            self.centers[g_spk] += embeddings[l_spk]

    def _in_grace(self, anchor_idx: int) -> bool:
        steps_since = self._call_count - self._last_enrolled_step.get(anchor_idx, -999)
        return steps_since <= self._grace_steps

    def _effective_enrolled_delta(self, anchor_idx: int, local_spk: int) -> float:
        """Relax matching only for the previously used local channel."""
        eff = self.delta_enrolled
        prev_local = self._last_local_for_enrolled.get(anchor_idx)
        if prev_local == local_spk and self._in_grace(anchor_idx):
            eff += self.enrolled_grace_margin
        return eff

    def enrolled_distance(
        self, anchor_idx: int, embedding: np.ndarray, local_spk: Optional[int] = None
    ) -> float:
        """Distance between a live embedding and the enrolled speaker profile."""
        emb = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
        candidates = [self._anchors[anchor_idx].reshape(1, -1)]
        profile = self._profiles.get(anchor_idx)
        if profile is not None and profile.size > 0:
            candidates.append(profile)
        session = self._session_centroids.get(anchor_idx)
        if session is not None:
            candidates.append(session.reshape(1, -1))
        cand = np.concatenate(candidates, axis=0)
        return float(cdist(emb, cand, metric=self.metric).min())

    def _update_enrolled_session(self, anchor_idx: int, embedding: np.ndarray) -> None:
        emb = np.asarray(embedding, dtype=np.float64).ravel()
        norm = np.linalg.norm(emb)
        if norm == 0:
            return
        emb = emb / norm
        prev = self._session_centroids.get(anchor_idx)
        if prev is None:
            self._session_centroids[anchor_idx] = emb.copy()
            self._session_counts[anchor_idx] = 1
            return
        alpha = float(np.clip(self.enrolled_session_alpha, 0.0, 1.0))
        merged = (1.0 - alpha) * prev + alpha * emb
        merged_norm = np.linalg.norm(merged)
        if merged_norm > 0:
            merged = merged / merged_norm
        self._session_centroids[anchor_idx] = merged.astype(np.float64, copy=False)
        self._session_counts[anchor_idx] = self._session_counts.get(anchor_idx, 0) + 1

    # ─── Core override: enrolled-priority identify ───────────────

    def identify(
        self, segmentation: SlidingWindowFeature, embeddings: torch.Tensor
    ) -> SpeakerMap:
        """Assign local speakers to global centroids with enrolled priority.

        1. Compute active / long / valid speakers (same as base).
        2. **Enrolled priority**: greedily match active speakers to enrolled
           anchors using ``delta_enrolled`` (lenient).  These speakers and
           centroids are locked — the general matching cannot touch them.
        3. **General matching**: remaining active speakers compete for
           remaining (non-enrolled) centroids using ``delta_new`` (strict).
        4. Centroid update: only non-enrolled centroids are updated.
        """
        embeddings = embeddings.detach().cpu().numpy()

        active_speakers = np.where(
            np.max(segmentation.data, axis=0) >= self.tau_active
        )[0]
        long_speakers = np.where(
            np.mean(segmentation.data, axis=0) >= self.rho_update
        )[0]
        no_nan = np.where(~np.isnan(embeddings).any(axis=1))[0]
        active_speakers = np.intersect1d(active_speakers, no_nan)

        num_local = segmentation.data.shape[1]

        # ── First-ever call: bootstrap centres ──
        if self.centers is None:
            self.init_centers(embeddings.shape[1])
            assignments = [
                (spk, self.add_center(embeddings[spk])) for spk in active_speakers
            ]
            return SpeakerMapBuilder.hard_map(
                shape=(num_local, self.max_speakers),
                assignments=assignments,
                maximize=False,
            )

        # ── 1. Enrolled priority pass ──
        # For each enrolled anchor, find speakers within the effective
        # threshold, then pick the closest one. Activity is only a tie-breaker.
        enrolled_assignments = {}          # local_spk → global_idx
        enrolled_assignment_dist: Dict[int, float] = {}
        taken_enrolled: set = set()
        mean_act = np.mean(segmentation.data, axis=0)  # (n_local,)
        anchor_ids = sorted(self._anchors.keys()) if self._anchors else []

        if self._anchors and len(active_speakers) > 0:
            used_local: set = set()
            for anchor_idx in anchor_ids:
                if anchor_idx in taken_enrolled:
                    continue

                # Collect all candidates within effective threshold
                candidates = []
                for li, local_spk in enumerate(active_speakers):
                    local_spk = int(local_spk)
                    if local_spk in used_local:
                        continue
                    dist = self.enrolled_distance(anchor_idx, embeddings[local_spk], local_spk)
                    eff_delta = self._effective_enrolled_delta(anchor_idx, local_spk)
                    if dist < eff_delta:
                        candidates.append((li, local_spk, dist))

                if not candidates:
                    continue

                # Prefer the previously assigned local speaker (channel
                # persistence) to avoid source switching when it is still
                # close enough. Otherwise choose the CLOSEST candidate first,
                # using activity only as a tie-breaker.
                prev_local = self._last_local_for_enrolled.get(anchor_idx)
                prev_match = [c for c in candidates if c[1] == prev_local]
                if prev_match:
                    _, best_local, best_dist = min(prev_match, key=lambda x: x[2])
                else:
                    _, best_local, best_dist = min(
                        candidates,
                        key=lambda x: (x[2], -mean_act[x[1]]),
                    )
                enrolled_assignments[best_local] = anchor_idx
                enrolled_assignment_dist[anchor_idx] = float(best_dist)
                taken_enrolled.add(anchor_idx)
                used_local.add(best_local)
                self._last_enrolled_step[anchor_idx] = self._call_count
                self._last_local_for_enrolled[anchor_idx] = best_local

        # ── 1b. Enrolled leakage suppression ──
        # Remaining active speakers extremely close to an enrolled anchor are
        # most likely the SAME voice leaking through a secondary PixIT channel.
        # Suppress entirely — no centroid, no mapping, no audio.
        enrolled_leakage: set = set()
        if self._anchors:
            for s in active_speakers:
                if s in enrolled_assignments:
                    continue
                if np.isnan(embeddings[s]).any():
                    continue
                best_anchor = min(
                    anchor_ids,
                    key=lambda a_idx: self.enrolled_distance(a_idx, embeddings[s], int(s)),
                )
                min_anchor = int(best_anchor)
                min_dist = self.enrolled_distance(min_anchor, embeddings[s], int(s))
                if (
                    min_anchor in taken_enrolled
                    and min_dist
                    <= max(
                        self.leakage_delta,
                        self.delta_enrolled,
                        enrolled_assignment_dist.get(min_anchor, 0.0) + 0.10,
                    )
                ):
                    enrolled_leakage.add(int(s))

        # ── 2. General matching for remaining speakers ──
        remaining_active = np.array(
            [s for s in active_speakers
             if s not in enrolled_assignments and s not in enrolled_leakage],
            dtype=int,
        )

        # Full distance map: all local speakers × all centres
        dist_map = SpeakerMapBuilder.dist(embeddings, self.centers, self.metric)

        # Block rows: inactive speakers + enrolled-assigned speakers
        block_local = [
            s for s in range(num_local) if s not in remaining_active
        ]
        # Block cols: inactive centres + taken enrolled centres
        block_global = list(self.inactive_centers)
        for g in taken_enrolled:
            if g not in block_global:
                block_global.append(g)

        dist_map = dist_map.unmap_speakers(block_local, block_global)

        # Threshold at delta_new
        valid_map = dist_map.unmap_threshold(self.delta_new)

        strict_local, strict_global = valid_map.valid_assignments(strict=True)
        strict_local_set = {int(s) for s in strict_local}

        # Missed: active remaining speakers still unmatched
        missed = [
            s for s in remaining_active
            if int(s) not in strict_local_set
        ]

        # Try to create new centres for missed speakers, or fall back.
        # During a grace period, suppress new-centroid creation only when the
        # SAME local channel that was just mapped to an enrolled speaker
        # flickers into a slightly worse embedding.
        new_center_speakers = []
        for spk in missed:
            src_stats = None
            if self._current_source_stats is not None and spk < len(self._current_source_stats):
                src_stats = self._current_source_stats[spk]

            min_dist = None
            min_anchor = None
            prev_local = None
            grace_delta = None
            assigned_globals = {int(g) for g in valid_map.valid_assignments(strict=True)[1]}
            if self._anchors and not np.isnan(embeddings[spk]).any():
                min_anchor = min(
                    anchor_ids,
                    key=lambda a_idx: self.enrolled_distance(a_idx, embeddings[spk], int(spk)),
                )
                min_dist = self.enrolled_distance(min_anchor, embeddings[spk], int(spk))
                prev_local = self._last_local_for_enrolled.get(min_anchor)
                grace_delta = (
                    self._effective_enrolled_delta(min_anchor, spk)
                    + self.new_center_grace_margin
                )
                anchor_available = (
                    min_anchor not in taken_enrolled and min_anchor not in assigned_globals
                )
                if (
                    anchor_available
                    and
                    prev_local == spk
                    and self._in_grace(min_anchor)
                    and min_dist < grace_delta
                ):
                    valid_map = valid_map.set_source_speaker(spk, min_anchor)
                    taken_enrolled.add(min_anchor)
                    self._last_enrolled_step[min_anchor] = self._call_count
                    self._last_local_for_enrolled[min_anchor] = int(spk)
                    continue

                # If the enrolled anchor is already claimed this step and this
                # local is still reasonably close to it, treat the local as
                # separation leakage rather than spawning/reusing Unknown-*.
                if (
                    min_anchor in taken_enrolled
                    and min_dist
                    <= max(
                        self.leakage_delta,
                        self.delta_enrolled,
                        enrolled_assignment_dist.get(min_anchor, 0.0) + 0.10,
                    )
                ):
                    continue

            if (
                src_stats is not None
                and float(src_stats.get("tail_rms", 0.0)) < self.unknown_min_tail_rms
            ):
                continue

            # Unknown speakers are much less stable than enrolled anchors.
            # Reuse the nearest existing unknown centroid with a slightly more
            # permissive threshold before spawning another Unknown-* lane.
            unknown_globals = [
                g for g in self.active_centers
                if g not in self._anchors and g not in taken_enrolled
            ]
            if unknown_globals:
                best_unknown = min(
                    unknown_globals,
                    key=lambda g: float(dist_map.mapping_matrix[spk, g]),
                )
                best_unknown_dist = float(dist_map.mapping_matrix[spk, best_unknown])
                if (
                    self._anchors
                    and min_anchor is not None
                    and min_dist is not None
                    and min_anchor not in taken_enrolled
                    and min_anchor not in assigned_globals
                    and min_dist < (self.delta_enrolled + self.enrolled_preference_margin)
                    and min_dist <= (best_unknown_dist + self.enrolled_preference_margin)
                ):
                    valid_map = valid_map.set_source_speaker(spk, min_anchor)
                    taken_enrolled.add(min_anchor)
                    self._last_enrolled_step[min_anchor] = self._call_count
                    self._last_local_for_enrolled[min_anchor] = int(spk)
                    continue
                if best_unknown_dist < self.unknown_reuse_delta:
                    valid_map = valid_map.set_source_speaker(spk, best_unknown)
                    continue

            if (
                self._anchors
                and min_anchor is not None
                and min_dist is not None
                and min_anchor not in taken_enrolled
                and min_anchor not in assigned_globals
                and min_dist < (self.delta_enrolled + self.enrolled_preference_margin)
            ):
                valid_map = valid_map.set_source_speaker(spk, min_anchor)
                taken_enrolled.add(min_anchor)
                self._last_enrolled_step[min_anchor] = self._call_count
                self._last_local_for_enrolled[min_anchor] = int(spk)
                continue

            if (
                src_stats is not None
                and float(src_stats.get("voiced_sec", 0.0)) < self.unknown_min_voiced_sec
            ):
                continue

            has_space = len(new_center_speakers) < self.num_free_centers
            if has_space and spk in long_speakers:
                new_center_speakers.append(spk)
            else:
                prefs = np.argsort(dist_map.mapping_matrix[spk, :])
                prefs = [
                    g for g in prefs
                    if g not in self.active_centers and g not in taken_enrolled
                ]
                _, g_assigned = valid_map.valid_assignments(strict=True)
                free = [g for g in prefs if g not in g_assigned]
                if free:
                    valid_map = valid_map.set_source_speaker(spk, free[0])

        # Update non-enrolled centroids (only long, non-missed speakers)
        to_update = [
            (ls, gs)
            for ls, gs in zip(*valid_map.valid_assignments(strict=True))
            if ls not in missed and ls in long_speakers
        ]
        self.update(to_update, embeddings)

        # Add new centres
        for spk in new_center_speakers:
            valid_map = valid_map.set_source_speaker(
                spk, self.add_center(embeddings[spk])
            )

        # ── 3. Merge enrolled assignments into the map ──
        for local_spk, global_idx in enrolled_assignments.items():
            valid_map = valid_map.set_source_speaker(local_spk, global_idx)

        for local_spk, global_idx in zip(*valid_map.valid_assignments(strict=True)):
            if global_idx in self._anchors:
                self._update_enrolled_session(int(global_idx), embeddings[int(local_spk)])

        return valid_map

    # ─── Override __call__ ───────────────────────────────────────

    def __call__(
        self, segmentation: SlidingWindowFeature, embeddings: torch.Tensor
    ) -> SlidingWindowFeature:
        # Always reset enrolled centroids to clean anchors
        self._freeze_enrolled()

        self._call_count += 1
        speaker_map = self.identify(segmentation, embeddings)
        self.last_speaker_map = speaker_map
        if (
            log.isEnabledFor(logging.DEBUG)
            and self.centers is not None
            and self._call_count % 10 == 1
        ):
            self._log_distances(segmentation, embeddings, speaker_map)

        return SlidingWindowFeature(
            self._apply_strict_map(speaker_map, segmentation.data),
            segmentation.sliding_window,
        )

    @staticmethod
    def _apply_strict_map(speaker_map: SpeakerMap, source_scores: np.ndarray) -> np.ndarray:
        """Project local scores to globals using only strict valid assignments."""
        num_frames = source_scores.shape[0]
        projected_scores = np.zeros((num_frames, speaker_map.num_target_speakers))
        for src_speaker, tgt_speaker in zip(*speaker_map.valid_assignments(strict=True)):
            projected_scores[:, tgt_speaker] = source_scores[:, src_speaker]
        return projected_scores

    def _log_distances(
        self,
        segmentation: SlidingWindowFeature,
        embeddings: torch.Tensor,
        speaker_map: SpeakerMap,
    ):
        emb_np = embeddings.detach().cpu().numpy()
        if emb_np.ndim == 1:
            emb_np = emb_np.reshape(1, -1)
        active_spk = np.where(
            np.max(segmentation.data, axis=0) >= self.tau_active
        )[0]
        no_nan = np.where(~np.isnan(emb_np).any(axis=1))[0]
        valid_local = np.intersect1d(active_spk, no_nan)
        if len(valid_local) == 0 or not self._anchors:
            return
        anchor_ids = sorted(self._anchors.keys())
        anchor_labels = [self._labels.get(i, f"g{i}") for i in anchor_ids]
        l_assigned, g_assigned = speaker_map.valid_assignments(strict=True)
        assign_str = ", ".join(
            f"L{l}→{self.get_label(g)}" for l, g in zip(l_assigned, g_assigned)
        )
        log.debug(
            "ENROLLED-DIST (active=%s, map=[%s]):\n  %s\n  %s",
            valid_local.tolist(),
            assign_str,
            "  ".join(f"{l:>12s}" for l in anchor_labels),
            "\n  ".join(
                f"local-{valid_local[i]}: "
                + "  ".join(
                    f"{self.enrolled_distance(anchor_ids[j], emb_np[valid_local[i]], int(valid_local[i])):12.3f}"
                    for j in range(len(anchor_ids))
                )
                for i in range(len(valid_local))
            ),
        )
