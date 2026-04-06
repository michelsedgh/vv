"""
EnrolledSpeakerClustering — extends diart's OnlineSpeakerClustering with:

1. **Centroid injection**: pre-fill centroids from enrollment embeddings so that
   known speakers are immediately recognised on the first chunk.
2. **Frozen enrolled centroids**: enrolled centroids are NEVER updated — they
   always stay at their enrollment anchor.
3. **Enrolled-priority identify()**: enrolled speakers are matched FIRST using
   a lenient ``delta_enrolled`` threshold *before* the general distance map
   is computed for remaining speakers.  This prevents unknown centroids from
   stealing enrolled identities.
4. **Label mapping**: ``get_label(global_idx)`` returns the enrolled speaker
   name or ``"Unknown-N"`` for strangers.
5. **Exposed SpeakerMap**: ``last_speaker_map`` is stored after every
   ``__call__`` so the pipeline can map PixIT source channels to global
   speakers.
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
    ):
        super().__init__(tau_active, rho_update, delta_new, metric, max_speakers)
        self.delta_enrolled = delta_enrolled

        # Enrollment state
        self._anchors: Dict[int, np.ndarray] = {}      # global_idx → L2-normed anchor
        self._labels: Dict[int, str] = {}               # global_idx → speaker name
        self._unknown_counter: int = 0

        # Populated after every __call__
        self.last_speaker_map: Optional[SpeakerMap] = None
        self._call_count: int = 0

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
            self._labels[idx] = name

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
        enrolled_assignments = {}          # local_spk → global_idx
        taken_enrolled: set = set()

        if self._anchors and len(active_speakers) > 0:
            anchor_ids = sorted(self._anchors.keys())
            anchor_mat = np.array([self._anchors[i] for i in anchor_ids])
            active_embs = embeddings[active_speakers]
            dists = cdist(active_embs, anchor_mat, metric=self.metric)

            # Greedy: pick closest (speaker, anchor) pair until none < delta
            used_local = set()
            while True:
                best_val, best_li, best_ai = np.inf, -1, -1
                for li in range(len(active_speakers)):
                    if active_speakers[li] in used_local:
                        continue
                    for ai in range(len(anchor_ids)):
                        if anchor_ids[ai] in taken_enrolled:
                            continue
                        if dists[li, ai] < best_val:
                            best_val = dists[li, ai]
                            best_li, best_ai = li, ai
                if best_li < 0 or best_val >= self.delta_enrolled:
                    break
                local_spk = int(active_speakers[best_li])
                global_idx = anchor_ids[best_ai]
                enrolled_assignments[local_spk] = global_idx
                taken_enrolled.add(global_idx)
                used_local.add(local_spk)
                log.debug(
                    "ENROLLED-MATCH: local-%d → %s (dist=%.3f)",
                    local_spk, self._labels.get(global_idx, f"g{global_idx}"),
                    best_val,
                )

        # ── 2. General matching for remaining speakers ──
        remaining_active = np.array(
            [s for s in active_speakers if s not in enrolled_assignments],
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

        # Missed: active remaining speakers still unmatched
        missed = [
            s for s in remaining_active
            if not valid_map.is_source_speaker_mapped(s)
        ]

        # Try to create new centres for missed speakers, or fall back
        new_center_speakers = []
        for spk in missed:
            has_space = len(new_center_speakers) < self.num_free_centers
            if has_space and spk in long_speakers:
                new_center_speakers.append(spk)
            else:
                # Fall back: closest free non-enrolled global slot
                prefs = np.argsort(dist_map.mapping_matrix[spk, :])
                prefs = [
                    g for g in prefs
                    if g not in self.active_centers and g not in taken_enrolled
                ]
                _, g_assigned = valid_map.valid_assignments()
                free = [g for g in prefs if g not in g_assigned]
                if free:
                    valid_map = valid_map.set_source_speaker(spk, free[0])

        # Update non-enrolled centroids (only long, non-missed speakers)
        to_update = [
            (ls, gs)
            for ls, gs in zip(*valid_map.valid_assignments())
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

        return valid_map

    # ─── Override __call__ ───────────────────────────────────────

    def __call__(
        self, segmentation: SlidingWindowFeature, embeddings: torch.Tensor
    ) -> SlidingWindowFeature:
        # Always reset enrolled centroids to clean anchors
        self._freeze_enrolled()

        speaker_map = self.identify(segmentation, embeddings)
        self.last_speaker_map = speaker_map

        # Periodic debug logging
        self._call_count += 1
        if self.centers is not None and self._call_count % 10 == 1:
            self._log_distances(segmentation, embeddings, speaker_map)

        return SlidingWindowFeature(
            speaker_map.apply(segmentation.data),
            segmentation.sliding_window,
        )

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
        anchor_mat = np.array([self._anchors[i] for i in anchor_ids])
        dists = cdist(emb_np[valid_local], anchor_mat, metric="cosine")
        anchor_labels = [self._labels.get(i, f"g{i}") for i in anchor_ids]
        l_assigned, g_assigned = speaker_map.valid_assignments()
        assign_str = ", ".join(
            f"L{l}→{self.get_label(g)}" for l, g in zip(l_assigned, g_assigned)
        )
        log.info(
            "ENROLLED-DIST (active=%s, map=[%s]):\n  %s\n  %s",
            valid_local.tolist(),
            assign_str,
            "  ".join(f"{l:>12s}" for l in anchor_labels),
            "\n  ".join(
                f"local-{valid_local[i]}: "
                + "  ".join(f"{d:12.3f}" for d in row)
                for i, row in enumerate(dists)
            ),
        )
