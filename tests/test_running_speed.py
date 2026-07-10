"""Tests for the running-speed intermediate builder."""

import unittest

import numpy as np

from camstim_behavior_processing.load_data.running_speed import (
    compute_running_speed,
)

# A repeating pattern that contains both a positive (0.5 after 4.5) and a
# negative (4.5 after 0.5) voltage wrap, so both unwrap branches are hit.
_PATTERN = [1.0, 2.0, 3.0, 4.0, 4.5, 0.5, 4.5, 0.5, 2.0, 3.0]


def _vsig(n):
    """Return a length-``n`` encoder voltage signal with wraps."""
    return np.tile(_PATTERN, n // len(_PATTERN) + 1)[:n]


def _pkl(vsig, vin, dx):
    """Wrap encoder arrays into a minimal behavior pkl."""
    return {
        "items": {
            "behavior": {"encoders": [{"vsig": vsig, "vin": vin, "dx": dx}]}
        }
    }


def _run(n_enc, n_time, **kw):
    """Compute running speed for encoder length ``n_enc`` / time ``n_time``."""
    vsig = _vsig(n_enc)
    vin = np.full(n_enc, 5.0)
    dx = np.arange(n_enc, dtype=float)
    time = np.linspace(0, n_time / 60, n_time)
    return compute_running_speed(_pkl(vsig, vin, dx), time, **kw)


class ComputeRunningSpeedTest(unittest.TestCase):
    """Tests for :func:`compute_running_speed`."""

    def test_basic_with_wraps(self):
        """Equal-length signal with positive and negative wraps."""
        df = _run(30, 30)
        self.assertEqual(list(df.columns), ["speed", "dx", "v_sig", "v_in"])
        self.assertEqual(len(df), 30)
        self.assertEqual(df.index.name, "timestamps")

    def test_no_lowpass(self):
        """Disabling the low-pass filter still returns a full frame."""
        self.assertEqual(len(_run(30, 30, lowpass=False)), 30)

    def test_vin_one_longer(self):
        """Encoder one sample longer than the frame array is trimmed."""
        self.assertEqual(len(_run(31, 30)), 30)

    def test_vin_much_longer(self):
        """Encoder much longer than the frame array is trimmed to length."""
        self.assertEqual(len(_run(32, 30)), 30)

    def test_vin_shorter(self):
        """A frame array longer than the encoder trims the time vector."""
        self.assertEqual(len(_run(28, 30)), 28)


if __name__ == "__main__":
    unittest.main()
