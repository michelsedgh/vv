"""
Minimal enrolled-first online clustering.

This keeps only the custom behavior we actually need on top of diart:

1. inject frozen enrolled centroids before streaming starts
2. match enrolled speakers before general unknown clustering
3. suppress duplicate PixIT leakage channels that are still close to a
   matched enrolled speaker

Everything else should stay close to vanilla diart OnlineSpeakerClustering.

Why this file still exists
──────────────────────────
During debugging we tried both extremes:

- lots of custom rescue / holdover / remap logic:
  this made the output choppy and hard to reason about
- totally vanilla diart:
  this fragmented the enrolled speaker into ``Unknown-*`` lanes too easily in
  our PixIT + enrollment setup

This module is the small compromise that survived:

- frozen enrolled anchors for persistent identity
- optional multi-vector enrollment profiles for robustness
- enrolled-first assignment
- duplicate-leakage suppression
- unknown-lane reuse before spawning more speakers

Anything beyond that should be treated skeptically unless logs clearly show why
it is necessary.
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
    """Online clustering with frozen enrolled centroids.

    Important distinction:
    diart's clustering solves online speaker consistency *within a session*.
    This subclass adds the minimum needed to recognize enrolled speakers across
    sessions without letting those frozen identities drift.
    """

    def __init__(
        self,
        tau_active: float = 0.5,
        rho_update: float = 0.3,
        delta_new: float = 1.0,
        metric: str = "cosine",
        max_speakers: int = 8,
        max_drift: float = 0.0,  # kept for config compatibility, unused
        delta_enrolled: float = 0.80,
        leakage_delta: Optional[float] = None,
        enrolled_preference_margin: float = 0.03,
        unknown_reuse_delta: Optional[float] = None,
    ):
        super().__init__(tau_active, rho_update, delta_new, metric, max_speakers)
        self.delta_enrolled = float(delta_enrolled)
        self.leakage_delta = float(
            leakage_delta if leakage_delta is not None else delta_enrolled
        )
        self.enrolled_preference_margin = float(enrolled_preference_margin)
        self.unknown_reuse_delta = float(
            unknown_reuse_delta
            if unknown_reuse_delta is not None
            else max(delta_new, delta_enrolled + 0.04)
        )

        self._anchors: Dict[int, np.ndarray] = {}
        self._profiles: Dict[int, np.ndarray] = {}
        self._labels: Dict[int, str] = {}
        self.last_speaker_map: Optional[SpeakerMap] = None

    # ─── Enrollment ──────────────────────────────────────────────

    def inject_centroids(self, enrolled: Dict[str, np.ndarray]) -> None:
        if not enrolled:
            return

        emb_dim = next(iter(enrolled.values())).shape[-1]
        if self.centers is None:
            self.init_centers(emb_dim)

        for name, emb in enrolled.items():
            emb = np.asarray(emb, dtype=np.float64).ravel()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            idx = self.add_center(emb)
            self._anchors[idx] = emb.copy()
            self._profiles[idx] = emb.reshape(1, -1).copy()
            self._labels[idx] = name

    def inject_profiles(self, profiles: Dict[str, np.ndarray]) -> None:
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

    def _freeze_enrolled(self) -> None:
        if self.centers is None:
            return
        for idx, anchor in self._anchors.items():
            self.centers[idx] = anchor.copy()

    def update(
        self, assignments: Iterable[Tuple[int, int]], embeddings: np.ndarray
    ) -> None:
        if self.centers is None:
            return
        for l_spk, g_spk in assignments:
            if g_spk not in self.active_centers:
                continue
            if g_spk in self._anchors:
                continue
            self.centers[g_spk] += embeddings[l_spk]

    def enrolled_distance(self, anchor_idx: int, embedding: np.ndarray) -> float:
        emb = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
        anchor = self._anchors[anchor_idx].reshape(1, -1)
        anchor_dist = float(cdist(emb, anchor, metric=self.metric)[0, 0])
        profile = self._profiles.get(anchor_idx)
        # Profile rows can refine a clearly-close anchor, but they must not
        # rescue a match the frozen anchor itself already rejects.
        if (
            profile is None
            or profile.size == 0
            or anchor_dist > self.delta_enrolled
        ):
            return anchor_dist
        profile_dist = float(cdist(emb, profile, metric=self.metric).min())
        return min(anchor_dist, profile_dist)

    # ─── Core override: enrolled-priority identify ───────────────

    def identify(
        self, segmentation: SlidingWindowFeature, embeddings: torch.Tensor
    ) -> SpeakerMap:
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

        # 1. Greedy enrolled-first matching.
        #
        # This ordering is intentional. Without it, strong local unknown
        # candidates could temporarily win before the enrolled anchor had a
        # chance to claim the same speech, which created ``Unknown-*`` onsets
        # for the enrolled speaker.
        enrolled_assignments: Dict[int, int] = {}
        enrolled_assignment_dist: Dict[int, float] = {}
        taken_enrolled: set[int] = set()
        taken_local: set[int] = set()
        candidates = []
        for anchor_idx in sorted(self._anchors):
            for local_spk in active_speakers:
                dist = self.enrolled_distance(anchor_idx, embeddings[int(local_spk)])
                if dist <= self.delta_enrolled:
                    candidates.append((float(dist), int(local_spk), int(anchor_idx)))
        for dist, local_spk, anchor_idx in sorted(candidates, key=lambda x: x[0]):
            if local_spk in taken_local or anchor_idx in taken_enrolled:
                continue
            enrolled_assignments[local_spk] = anchor_idx
            enrolled_assignment_dist[anchor_idx] = float(dist)
            taken_local.add(local_spk)
            taken_enrolled.add(anchor_idx)

        # 2. Suppress duplicate enrolled leakage locals.
        #
        # PixIT can put the same real speaker into more than one local channel
        # in a chunk. Treat those secondary locals as leakage of the matched
        # enrolled speaker, not as evidence for spawning a new ``Unknown-*``.
        enrolled_leakage: set[int] = set()
        if taken_enrolled:
            for local_spk in active_speakers:
                local_spk = int(local_spk)
                if local_spk in enrolled_assignments:
                    continue
                closest = min(
                    taken_enrolled,
                    key=lambda anchor_idx: self.enrolled_distance(
                        anchor_idx, embeddings[local_spk]
                    ),
                )
                min_dist = self.enrolled_distance(closest, embeddings[local_spk])
                if min_dist <= max(
                    self.leakage_delta,
                    self.delta_enrolled,
                    enrolled_assignment_dist.get(closest, 0.0) + 0.10,
                ):
                    enrolled_leakage.add(local_spk)

        # 3. Plain vanilla matching for the rest.
        #
        # At this point we deliberately fall back toward diart's normal online
        # behavior. The rest of the system became much more stable once we
        # stopped trying to "fix" every edge case here.
        remaining_active = np.array(
            [
                int(s)
                for s in active_speakers
                if int(s) not in enrolled_assignments and int(s) not in enrolled_leakage
            ],
            dtype=int,
        )

        dist_map = SpeakerMapBuilder.dist(embeddings, self.centers, self.metric)
        block_local = [s for s in range(num_local) if s not in remaining_active]
        block_global = list(self.inactive_centers)
        # Enrolled anchors must only be claimed through the explicit
        # enrolled-distance checks above. Leaving them in the plain diart
        # fallback map lets the one-to-one assignment solver push leftovers
        # onto an enrolled center even when the enrolled threshold failed.
        for g in self._anchors:
            if g not in block_global:
                block_global.append(g)
        valid_map = dist_map.unmap_speakers(block_local, block_global).unmap_threshold(
            self.delta_new
        )

        strict_local, _ = valid_map.valid_assignments(strict=True)
        strict_local_set = {int(s) for s in strict_local}
        missed = [int(s) for s in remaining_active if int(s) not in strict_local_set]

        new_center_speakers = []
        for spk in missed:
            min_anchor = None
            min_dist = None
            if self._anchors and not np.isnan(embeddings[spk]).any():
                min_anchor = min(
                    self._anchors,
                    key=lambda anchor_idx: self.enrolled_distance(
                        anchor_idx, embeddings[spk]
                    ),
                )
                min_dist = self.enrolled_distance(min_anchor, embeddings[spk])
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

            assigned_globals = {
                int(g) for g in valid_map.valid_assignments(strict=True)[1]
            }
            unknown_globals = [
                g
                for g in self.active_centers
                if g not in self._anchors
                and g not in taken_enrolled
                and g not in assigned_globals
            ]
            if unknown_globals:
                best_unknown = min(
                    unknown_globals,
                    key=lambda g: float(dist_map.mapping_matrix[spk, g]),
                )
                best_unknown_dist = float(dist_map.mapping_matrix[spk, best_unknown])
                if (
                    min_anchor is not None
                    and min_anchor not in taken_enrolled
                    and min_dist is not None
                    and min_dist < (self.delta_enrolled + self.enrolled_preference_margin)
                    and min_dist <= (best_unknown_dist + self.enrolled_preference_margin)
                ):
                    valid_map = valid_map.set_source_speaker(spk, min_anchor)
                    taken_enrolled.add(min_anchor)
                    continue
                if best_unknown_dist < self.unknown_reuse_delta:
                    valid_map = valid_map.set_source_speaker(spk, best_unknown)
                    continue

            if (
                min_anchor is not None
                and min_anchor not in taken_enrolled
                and min_dist is not None
                and min_dist < (self.delta_enrolled + self.enrolled_preference_margin)
            ):
                valid_map = valid_map.set_source_speaker(spk, min_anchor)
                taken_enrolled.add(min_anchor)
                continue

            if spk in long_speakers and len(new_center_speakers) < self.num_free_centers:
                new_center_speakers.append(spk)

        to_update = [
            (int(ls), int(gs))
            for ls, gs in zip(*valid_map.valid_assignments(strict=True))
            if int(ls) in long_speakers and int(gs) not in self._anchors
        ]
        self.update(to_update, embeddings)

        for spk in new_center_speakers:
            valid_map = valid_map.set_source_speaker(
                spk, self.add_center(embeddings[spk])
            )

        for local_spk, global_idx in enrolled_assignments.items():
            valid_map = valid_map.set_source_speaker(local_spk, global_idx)

        return valid_map

    # ─── Override __call__ ───────────────────────────────────────

    def __call__(
        self, segmentation: SlidingWindowFeature, embeddings: torch.Tensor
    ) -> SlidingWindowFeature:
        self._freeze_enrolled()
        speaker_map = self.identify(segmentation, embeddings)
        self.last_speaker_map = speaker_map
        return SlidingWindowFeature(
            self._apply_strict_map(speaker_map, segmentation.data),
            segmentation.sliding_window,
        )

    @staticmethod
    def _apply_strict_map(speaker_map: SpeakerMap, source_scores: np.ndarray) -> np.ndarray:
        num_frames = source_scores.shape[0]
        projected_scores = np.zeros((num_frames, speaker_map.num_target_speakers))
        for src_speaker, tgt_speaker in zip(*speaker_map.valid_assignments(strict=True)):
            projected_scores[:, tgt_speaker] = source_scores[:, src_speaker]
        return projected_scores
