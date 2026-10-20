from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.core.config import Settings
from app.features.face_verification.liveness import (
    ChallengeSigner,
    InvalidChallenge,
    LivenessAnalyzer,
    _completed_challenges,
    _is_frontal,
    _uniform_sample_indices,
)
from app.features.face_verification.schemas import Challenge
from app.features.primary_photo.analyzer import FaceObservation


def observation(*, yaw: float = 0, pitch: float = 0, roll: float = 0) -> FaceObservation:
    return FaceObservation(face_count=1, yaw=yaw, pitch=pitch, roll=roll)


def test_fixed_pose_requires_front_facing_frames():
    frontal = [observation() for _ in range(8)] + [observation(yaw=30) for _ in range(4)]
    non_frontal = [observation(yaw=30) for _ in range(12)]

    assert _completed_challenges(frontal, [Challenge.LOOK_STRAIGHT]) == [Challenge.LOOK_STRAIGHT]
    assert _completed_challenges(non_frontal, [Challenge.LOOK_STRAIGHT]) == []


def test_fixed_pose_allows_leaning_but_rejects_extreme_angles():
    assert _is_frontal(observation(yaw=24, pitch=24, roll=38))
    assert not _is_frontal(observation(yaw=30))
    assert not _is_frontal(observation(pitch=30))
    assert not _is_frontal(observation(roll=50))


def test_uniform_sampling_covers_the_whole_video():
    indices = _uniform_sample_indices(60)

    assert len(indices) == 24
    assert indices[0] == 0
    assert indices[-1] == 59
    assert indices == sorted(set(indices))


def test_fixed_pose_accepts_six_clear_frontal_frames_despite_detection_dropouts(monkeypatch):
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(12)]

    class FakeCapture:
        def __init__(self, _path):
            self.index = 0

        def get(self, _property):
            return len(frames)

        def isOpened(self):
            return self.index < len(frames)

        def read(self):
            if self.index >= len(frames):
                return False, None
            frame = frames[self.index]
            self.index += 1
            return True, frame

        def release(self):
            pass

    observations = [observation() for _ in range(6)]
    observations += [observation(yaw=30)]
    observations += [FaceObservation(face_count=0) for _ in range(5)]

    calls = []

    class FakeFaceAnalyzer:
        def observe(self, _frame, **kwargs):
            calls.append(kwargs)
            return observations.pop(0)

    monkeypatch.setattr("app.features.face_verification.liveness.cv2.VideoCapture", FakeCapture)
    evaluation = LivenessAnalyzer(FakeFaceAnalyzer()).evaluate(b"video", ".mp4", [Challenge.LOOK_STRAIGHT])

    assert evaluation.result.passed
    assert evaluation.result.valid_face_frames == 7
    assert evaluation.result.frontal_face_frames == 6
    assert evaluation.result.source_frame_count == 12
    assert len(evaluation.result.frame_diagnostics) == 12
    assert [item.source_frame_index for item in evaluation.result.frame_diagnostics] == list(range(12))
    assert [item.candidate_rank for item in evaluation.result.frame_diagnostics[:3]] == [1, 2, 3]
    assert (
        calls
        == [
            {
                "allow_masked_secondary": True,
                "allow_detection_fallback": True,
                "allow_low_light_enhancement": True,
            },
        ]
        * 12
    )


def test_sampling_spans_whole_video_when_frame_count_is_missing(monkeypatch):
    # MediaRecorder WebM 처럼 길이 정보가 없어 프레임 수가 음수로 읽히는 5초·24fps 영상
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(120)]

    class FakeCapture:
        def __init__(self, _path):
            self.index = 0

        def get(self, prop):
            return 24.0 if prop == cv2.CAP_PROP_FPS else -2.2e17

        def isOpened(self):
            return True

        def grab(self):
            if self.index >= len(frames):
                return False
            self.index += 1
            return True

        def read(self):
            if not self.grab():
                return False, None
            return True, frames[self.index - 1]

        def release(self):
            pass

    class FakeFaceAnalyzer:
        def observe(self, _frame, **_kwargs):
            return observation()

    monkeypatch.setattr("app.features.face_verification.liveness.cv2.VideoCapture", FakeCapture)
    evaluation = LivenessAnalyzer(FakeFaceAnalyzer()).evaluate(b"video", ".webm", [Challenge.LOOK_STRAIGHT])

    indices = [item.source_frame_index for item in evaluation.result.frame_diagnostics]
    assert evaluation.result.sampled_frames == 24
    assert evaluation.result.source_frame_count == 120
    assert indices == _uniform_sample_indices(120)
    assert evaluation.result.frame_diagnostics[-1].timestamp_ms == 4958


def test_challenge_token_rejects_tampering():
    signer = ChallengeSigner(Settings(liveness_token_secret="x" * 32))
    challenge = signer.create()

    assert signer.verify(challenge.challenge_token) == challenge.challenges
    with pytest.raises(InvalidChallenge):
        signer.verify(challenge.challenge_token + "tampered")
