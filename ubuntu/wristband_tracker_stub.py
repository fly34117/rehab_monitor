"""Minimal stub for tracker3d.IMUDisplacementTracker.

Replaces the missing module with the same API surface, returning zero-state.
If a real tracker3d.py is later placed in PYTHONPATH, the try/except import
in wristband_server.py will use it instead of this stub.
"""


class IMUDisplacementTracker:
    """Stub IMU displacement tracker — returns zero-state for all queries."""

    def __init__(self, fs=333.0):
        self.fs = fs
        self._state = {
            "px": 0.0, "py": 0.0, "pz": 0.0,
            "vx": 0.0, "vy": 0.0, "vz": 0.0,
            "stationary": True,
            "stationary_val": 0.0,
            "frames": 0,
            "confidence": 0.0,
            "drift_vx": 0.0, "drift_vy": 0.0,
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        }

    def update(self, ax, ay, az, gx, gy, gz, dt):
        """No-op update. Returns (None, None, None, None) to match expected unpacking."""
        self._state["frames"] += 1
        return None, None, None, None

    def get_state(self):
        """Return a copy of the current (zero) state dict."""
        return dict(self._state)

    def set_origin(self):
        """No-op origin set."""
        pass
