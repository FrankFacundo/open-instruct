import numpy as np

from open_instruct.grpo_local import _compute_advantages


def test_compute_advantages_standard():
    scores = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    advantages = _compute_advantages(scores, num_samples_per_prompt=2, normalization="standard")
    assert np.allclose(advantages, np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32))


def test_compute_advantages_centered():
    scores = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    advantages = _compute_advantages(scores, num_samples_per_prompt=2, normalization="centered")
    assert np.allclose(advantages, np.array([-0.5, 0.5, -0.5, 0.5], dtype=np.float32))
