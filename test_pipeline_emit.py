import unittest
from types import SimpleNamespace

import numpy as np
import torch
from pyannote.core import SlidingWindow, SlidingWindowFeature

from enrolled_clustering import EnrolledSpeakerClustering
from pipeline import RealtimePipeline


def _pipeline_stub() -> RealtimePipeline:
    pipe = RealtimePipeline.__new__(RealtimePipeline)
    pipe.sample_rate = 16000
    pipe.duration = 5.0
    pipe.chunk_samples = int(pipe.duration * pipe.sample_rate)
    pipe.step_samples = 4
    pipe.step_duration = 0.5
    pipe._enrolled_mix_gate_threshold = 0.07
    pipe.clustering = SimpleNamespace(delta_enrolled=0.7)
    return pipe


def _rescue_pipeline_stub() -> RealtimePipeline:
    pipe = _pipeline_stub()
    anchor = np.array([1.0, 0.0], dtype=np.float64)
    pipe.clustering = SimpleNamespace(
        delta_enrolled=0.7,
        _anchors={0: anchor},
        is_enrolled=lambda g: int(g) == 0,
        enrolled_distance=lambda anchor_idx, emb: float(
            1.0 - np.clip(np.dot(anchor, np.asarray(emb, dtype=np.float64)), -1.0, 1.0)
        ),
    )
    pipe._flicker_aux_dominance_ratio = 2.0
    pipe._flicker_aux_dist_margin = 0.05
    pipe._flicker_aux_min_tail_rms = 0.05
    pipe._output_latency_sec = 1.0
    pipe._enrolled_merge_block_sec = 0.25
    pipe._enrolled_merge_dist_margin = 0.05
    pipe._enrolled_merge_switch_ratio = 1.15
    pipe._enrolled_merge_min_block_rms = 0.01
    return pipe


def _aggregation_pipeline_stub() -> RealtimePipeline:
    pipe = RealtimePipeline.__new__(RealtimePipeline)
    pipe.sample_rate = 4
    pipe.duration = 2.0
    pipe.chunk_samples = 8
    pipe.step_duration = 0.5
    pipe.step_samples = 2
    pipe._output_latency_sec = 1.0
    return pipe


def _scan_pipeline_stub() -> RealtimePipeline:
    pipe = RealtimePipeline.__new__(RealtimePipeline)
    pipe.chunk_samples = 8
    return pipe


def _single_speaker_segmentation(value: float = 1.0) -> SlidingWindowFeature:
    return SlidingWindowFeature(
        np.array([[value]], dtype=np.float32),
        SlidingWindow(duration=0.5, step=0.5, start=0.0),
    )


class CandidateEmitGlobalsTest(unittest.TestCase):
    def test_recovers_enrolled_onset_without_delayed_meta(self):
        pipe = _pipeline_stub()
        delayed = {}
        current = {
            0: {
                "global_idx": 0,
                "label": "mich",
                "is_enrolled": True,
                "identity_similarity": 0.42,
            }
        }
        emit_scores = np.zeros((6, 3), dtype=np.float32)
        emit_scores[2:4, 0] = 0.25

        self.assertEqual(
            pipe._candidate_emit_globals(delayed, current, emit_scores),
            {0},
        )

    def test_does_not_backfill_weak_enrolled_rescue(self):
        pipe = _pipeline_stub()
        delayed = {}
        current = {
            0: {
                "global_idx": 0,
                "label": "mich",
                "is_enrolled": True,
                "identity_similarity": 0.22,
            }
        }
        emit_scores = np.zeros((6, 3), dtype=np.float32)
        emit_scores[:, 0] = 0.9

        self.assertEqual(
            pipe._candidate_emit_globals(delayed, current, emit_scores),
            set(),
        )

    def test_does_not_promote_unknown_without_delayed_meta(self):
        pipe = _pipeline_stub()
        delayed = {}
        current = {
            1: {
                "global_idx": 1,
                "label": "Unknown-1",
                "is_enrolled": False,
                "identity_similarity": None,
            }
        }
        emit_scores = np.zeros((6, 3), dtype=np.float32)
        emit_scores[:, 1] = 0.9

        self.assertEqual(
            pipe._candidate_emit_globals(delayed, current, emit_scores),
            set(),
        )

    def test_respects_gate_threshold(self):
        pipe = _pipeline_stub()
        delayed = {}
        current = {
            0: {
                "global_idx": 0,
                "label": "mich",
                "is_enrolled": True,
                "identity_similarity": 0.42,
            }
        }
        emit_scores = np.zeros((6, 3), dtype=np.float32)
        emit_scores[:, 0] = 0.05

        self.assertEqual(
            pipe._candidate_emit_globals(delayed, current, emit_scores),
            set(),
        )


class StepTrackGateTest(unittest.TestCase):
    def test_gate_zeroes_inactive_half_without_collar(self):
        pipe = _pipeline_stub()
        audio = np.ones(4, dtype=np.float32)
        track = np.array([1.0, 0.0], dtype=np.float32)

        np.testing.assert_allclose(
            pipe._apply_step_track_gate(
                audio,
                track,
                collar_sec=0.0,
                threshold=pipe._enrolled_mix_gate_threshold,
            ),
            np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        )

    def test_gate_with_collar_keeps_packet_open(self):
        pipe = _pipeline_stub()
        audio = np.ones(4, dtype=np.float32)
        track = np.array([1.0, 0.0], dtype=np.float32)

        np.testing.assert_allclose(
            pipe._apply_step_track_gate(
                audio,
                track,
                collar_sec=0.25,
                threshold=pipe._enrolled_mix_gate_threshold,
            ),
            np.ones(4, dtype=np.float32),
        )

class SeparatedAggregationTest(unittest.TestCase):
    def test_returns_delayed_interior_slice(self):
        pipe = _aggregation_pipeline_stub()
        chunk = np.arange(1, 9, dtype=np.float32).reshape(-1, 1)

        aggregated = pipe._aggregate_audio_chunk(1, chunk)

        expected = np.array([[5.0], [6.0]], dtype=np.float32)
        np.testing.assert_allclose(aggregated, expected, atol=1e-6)


class EnrolledFlickerRescueTest(unittest.TestCase):
    def test_swaps_to_much_stronger_nearby_local(self):
        pipe = _rescue_pipeline_stub()
        pairs = [(2, 0)]
        active_indices = [1, 2]
        step_tail_rms = np.array([0.0, 0.68, 0.056], dtype=np.float32)
        embeddings = torch.tensor(
            [
                [0.0, 1.0],
                [0.50958, (1 - 0.50958**2) ** 0.5],
                [0.55424, (1 - 0.55424**2) ** 0.5],
            ],
            dtype=torch.float32,
        )

        rescued, _ = pipe._flicker_aux_enrolled_pair(
            pairs,
            active_indices,
            step_tail_rms,
            embeddings,
        )

        self.assertEqual(rescued, [(1, 0)])

    def test_does_not_swap_when_stronger_local_is_too_far(self):
        pipe = _rescue_pipeline_stub()
        pairs = [(2, 0)]
        active_indices = [1, 2]
        step_tail_rms = np.array([0.0, 0.68, 0.056], dtype=np.float32)
        embeddings = torch.tensor(
            [
                [0.0, 1.0],
                [0.10, (1 - 0.10**2) ** 0.5],
                [0.55424, (1 - 0.55424**2) ** 0.5],
            ],
            dtype=torch.float32,
        )

        rescued, _ = pipe._flicker_aux_enrolled_pair(
            pairs,
            active_indices,
            step_tail_rms,
            embeddings,
        )

        self.assertEqual(rescued, pairs)


class EnrolledChunkMergeTest(unittest.TestCase):
    def test_prefers_stronger_nearby_local_in_delayed_blocks(self):
        pipe = _rescue_pipeline_stub()
        pipe.sample_rate = 4
        pipe.duration = 2.0
        pipe.chunk_samples = 8
        pipe.step_samples = 4
        pipe._output_latency_sec = 1.0
        pipe._enrolled_merge_block_sec = 0.5
        active_indices = [1, 2]
        embeddings = torch.tensor(
            [
                [0.0, 1.0],
                [0.92, (1 - 0.92**2) ** 0.5],
                [0.95, (1 - 0.95**2) ** 0.5],
            ],
            dtype=torch.float32,
        )
        source_stats = [
            {"has_embedding": False},
            {"has_embedding": True},
            {"has_embedding": True},
        ]
        segmentation = np.array(
            [
                [0.0, 0.8, 0.2],
                [0.0, 0.9, 0.1],
                [0.0, 0.1, 0.9],
                [0.0, 0.2, 0.8],
            ],
            dtype=np.float32,
        )
        sources_np = np.array(
            [
                [0.0, 0.1, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.2, 0.8],
                [0.0, 0.2, 0.8],
                [0.0, 0.9, 0.2],
                [0.0, 0.9, 0.2],
                [0.0, 0.3, 1.0],
                [0.0, 0.3, 1.0],
            ],
            dtype=np.float32,
        )

        merged = pipe._compose_enrolled_chunk_source(
            assigned_local=1,
            anchor_idx=0,
            active_indices=active_indices,
            embeddings=embeddings,
            source_stats=source_stats,
            segmentation=segmentation,
            sources_np=sources_np,
        )

        expected = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(merged, expected, atol=1e-6)


class ScanChunkStartsTest(unittest.TestCase):
    def test_uses_overlapping_hops_and_keeps_tail(self):
        pipe = _scan_pipeline_stub()
        self.assertEqual(
            pipe._scan_chunk_starts(total_samples=16, hop_samples=2),
            [0, 2, 4, 6, 8],
        )

    def test_short_audio_still_returns_zero_start(self):
        pipe = _scan_pipeline_stub()
        self.assertEqual(
            pipe._scan_chunk_starts(total_samples=5, hop_samples=2),
            [0],
        )


class EnrolledContinuityRescueTest(unittest.TestCase):
    def test_recent_enrolled_match_rescues_drifted_step(self):
        cluster = EnrolledSpeakerClustering(
            delta_enrolled=0.3,
            enrolled_continuity_margin=0.2,
            enrolled_continuity_max_gap=3,
            max_speakers=4,
        )
        cluster.inject_centroids({"mich": np.array([1.0, 0.0], dtype=np.float32)})

        cluster(_single_speaker_segmentation(), torch.tensor([[1.0, 0.0]]))
        cluster(
            _single_speaker_segmentation(),
            torch.tensor([[0.55, (1 - 0.55**2) ** 0.5]], dtype=torch.float32),
        )

        self.assertEqual(
            cluster.last_speaker_map.valid_assignments(strict=True),
            ([0], [0]),
        )

    def test_expired_continuity_window_allows_new_unknown(self):
        cluster = EnrolledSpeakerClustering(
            delta_enrolled=0.3,
            enrolled_continuity_margin=0.2,
            enrolled_continuity_max_gap=1,
            max_speakers=4,
        )
        cluster.inject_centroids({"mich": np.array([1.0, 0.0], dtype=np.float32)})

        cluster(_single_speaker_segmentation(), torch.tensor([[1.0, 0.0]]))
        cluster(_single_speaker_segmentation(0.0), torch.tensor([[1.0, 0.0]]))
        cluster(
            _single_speaker_segmentation(),
            torch.tensor([[0.55, (1 - 0.55**2) ** 0.5]], dtype=torch.float32),
        )

        self.assertEqual(
            cluster.last_speaker_map.valid_assignments(strict=True),
            ([0], [1]),
        )

    def test_strict_enrolled_miss_becomes_unknown(self):
        cluster = EnrolledSpeakerClustering(
            delta_enrolled=0.7,
            max_speakers=4,
        )
        cluster.inject_centroids({"mich": np.array([1.0, 0.0], dtype=np.float32)})

        # Similarity 0.29 => cosine distance 0.71, just outside the enrolled threshold.
        drift = torch.tensor([[0.29, (1 - 0.29**2) ** 0.5]], dtype=torch.float32)
        cluster(_single_speaker_segmentation(), drift)

        self.assertEqual(
            cluster.last_speaker_map.valid_assignments(strict=True),
            ([0], [1]),
        )

    def test_continuity_rescue_does_not_refresh_itself(self):
        cluster = EnrolledSpeakerClustering(
            delta_enrolled=0.3,
            enrolled_continuity_margin=0.2,
            enrolled_continuity_max_gap=1,
            max_speakers=4,
        )
        cluster.inject_centroids({"mich": np.array([1.0, 0.0], dtype=np.float32)})

        cluster(_single_speaker_segmentation(), torch.tensor([[1.0, 0.0]]))
        drift = torch.tensor([[0.55, (1 - 0.55**2) ** 0.5]], dtype=torch.float32)

        cluster(_single_speaker_segmentation(), drift)
        self.assertEqual(
            cluster.last_speaker_map.valid_assignments(strict=True),
            ([0], [0]),
        )

        cluster(_single_speaker_segmentation(), drift)
        self.assertEqual(
            cluster.last_speaker_map.valid_assignments(strict=True),
            ([0], [1]),
        )


if __name__ == "__main__":
    unittest.main()
