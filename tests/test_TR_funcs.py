import math
import numpy as np
import pytest

import src.TR_funcs as F


# ------------------------- Fakes / fixtures -------------------------

class FakeData:
    """Minimal stand-in for MuJoCo's data with only the fields TR_funcs reads/writes."""
    def __init__(self):
        # qpos[0:2]=pos, qpos[2]=z, qpos[3:7]=quat
        self.qpos = np.zeros(7, dtype=float)
        # qvel[:] is zeroed by place_robot
        self.qvel = np.ones(7, dtype=float)
        # sensordata layout expected: [x, y, z, qw, qx, qy, qz]
        self._sens = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=float)

    @property
    def sensordata(self):
        return self._sens

    @sensordata.setter
    def sensordata(self, arr):
        self._sens = np.array(arr, dtype=float)


class FakeModel:
    """Placeholder. TR_funcs.place_robot passes it through to mujoco.mj_forward."""
    pass


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def fake_data():
    return FakeData()


@pytest.fixture(autouse=True)
def stub_mj_forward(monkeypatch):
    """
    Auto-stub mujoco.mj_forward so tests do not depend on MuJoCo runtime.
    Exposes call count for assertions where useful.
    """
    call_counter = {"count": 0}

    def _stub_forward(model, data):
        call_counter["count"] += 1

    monkeypatch.setattr(F.mujoco, "mj_forward", _stub_forward)
    return call_counter


# ------------------------------- parse_place -------------------------------

def test_TC_001_parse_place_valid_variants():
    """TC_001: Valid PLACE parsing should return (x, y, FACING) with normalized facing."""
    assert F.parse_place("PLACE 1,2,NORTH") == (1, 2, "NORTH")
    assert F.parse_place("  PLACE   -3 , 4 , west ") == (-3, 4, "WEST")
    assert F.parse_place("PLACE 0,0,EaSt") == (0, 0, "EAST")


def test_TC_002_parse_place_invalid_facing_logs(caplog):
    """TC_002: Invalid facing should return None and log an error."""
    caplog.clear()
    assert F.parse_place("PLACE 1,2,NORTHEAST") is None
    assert any("Invalid facing" in rec.message for rec in caplog.records)


def test_TC_003_parse_place_bad_format():
    """TC_003: Invalid format (missing commas) should return None."""
    assert F.parse_place("PLACE 1 2 NORTH") is None
    assert F.parse_place("MOVE") is None


# ---------------------------- quaternion_to_yaw ----------------------------

@pytest.mark.parametrize(
    "quat, expected",
    [
        # Mapped quaternions per TR_funcs.place_robot comments
        ((1.0, 0.0, 0.0, 0.0), 0.0),                 # EAST
        ((0.7071, 0.0, 0.0, 0.7071), math.pi/2),     # NORTH
        ((0.0, 0.0, 0.0, 1.0), math.pi),             # WEST (~±π)
        ((0.7071, 0.0, 0.0, -0.7071), -math.pi/2),   # SOUTH
    ]
)
def test_TC_004_to_TC_005_quaternion_to_yaw(quat, expected):
    """TC_004/TC_005: quaternion_to_yaw returns expected yaw within tolerance."""
    qw, qx, qy, qz = quat
    yaw = F.quaternion_to_yaw(qw, qx, qy, qz)
    if abs(expected) == math.pi:
        assert abs(abs(yaw) - math.pi) < 1e-3
    else:
        assert abs(yaw - expected) < 1e-3


# -------------------------------- place_robot ------------------------------

def test_TC_006_place_robot_sets_pose_and_calls_forward(fake_model, fake_data, stub_mj_forward):
    """
    TC_006: place_robot should set position, drop height, quaternion per facing,
    zero velocities, and call mj_forward exactly once.
    """
    F.place_robot(fake_model, fake_data, pos=(1.2, -0.7), facing="WEST")
    np.testing.assert_allclose(fake_data.qpos[0:3], [1.2, -0.7, 4.25], atol=1e-9)
    np.testing.assert_allclose(fake_data.qpos[3:7], [0.0, 0.0, 0.0, 1.0], atol=1e-9)  # WEST
    assert np.all(fake_data.qvel == 0.0)
    assert stub_mj_forward["count"] == 1


# ----------------------------- robot_move_target ---------------------------

@pytest.mark.parametrize(
    "x,y,facing,expected",
    [
        (2, 2, "NORTH", (2, 3)),  # TC_007
        (2, 2, "EAST",  (3, 2)),  # TC_008
        (2, 2, "SOUTH", (2, 1)),
        (2, 2, "WEST",  (1, 2)),
    ]
)
def test_TC_007_TC_008_robot_move_target_nominal(x, y, facing, expected):
    """TC_007/TC_008 (+ extras): Moving one grid step in the current facing."""
    assert F.robot_move_target(x, y, facing) == expected


def test_TC_009_robot_move_target_invalid_facing_prints(capsys):
    """TC_009: Invalid facing should print a user-facing warning."""
    F.robot_move_target(2, 2, "BAD")
    out = capsys.readouterr().out
    assert "facing incorrect" in out


# ------------------------------ current_facing -----------------------------

@pytest.mark.parametrize(
    "yaw_deg, expected",
    [
        (0, "EAST"), (90, "NORTH"), (180, "WEST"), (270, "SOUTH"),  # TC_010
        (135, "INBETWEEN"),                                          # TC_011
        (359, "EAST"), (-1, "EAST"),
    ]
)
def test_TC_010_TC_011_current_facing(yaw_deg, expected):
    """TC_010/TC_011: Cardinal headings and in-between tolerance."""
    assert F.current_facing(yaw_deg) == expected


# -------------------------------- transforms -------------------------------

@pytest.mark.parametrize("ui", [0.0, 1.0, 2.0, 3.5, 4.0, -2.0, 10.0])
def test_TC_012_ui_map_roundtrip(ui):
    """TC_012: UI→MAP→UI round-trip should recover the original value within tolerance."""
    m = F.location_transform_UI2MAP(ui)
    ui2 = F.location_transform_MAP2UI(m)
    assert abs(ui2 - ui) < 1e-9  # exact given 0.8 and 1/0.8 are reciprocals


# -------------------------------- read_sensor ------------------------------

def test_TC_013_read_sensor(fake_data):
    """TC_013: read_sensor returns the 7-tuple in the documented order."""
    fake_data.sensordata = [1, 2, 3, 0.1, 0.2, 0.3, 0.4]
    x, y, z, qw, qx, qy, qz = F.read_sensor(fake_data)
    assert (x, y, z) == (1, 2, 3)
    assert (qw, qx, qy, qz) == (0.1, 0.2, 0.3, 0.4)


# ------------------------------ yaw_target_deg -----------------------------

@pytest.mark.parametrize(
    "curr, dir_, expected_deg",
    [
        (0,   "LEFT",  90),        # TC_014
        (0,   "RIGHT", -90),       # TC_015
        (179, "LEFT",  270),
        (181, "RIGHT", 90),
        (270, "LEFT",  360),
    ]
)
def test_TC_014_TC_015_yaw_target_deg(curr, dir_, expected_deg):
    """TC_014/TC_015: LEFT/RIGHT should snap to nearest 90-degree increments."""
    out = F.yaw_target_deg(curr, dir_)
    exp = math.radians(round(expected_deg / 90.0) * 90)
    assert abs(out - exp) < 1e-9


def test_TC_016_yaw_target_deg_invalid_logs_and_keeps(caplog):
    """
    TC_016: Invalid direction should log an error and keep current yaw, snapped to nearest 90°.
    """
    caplog.clear()
    curr = 33.0
    out = F.yaw_target_deg(curr, "BAD")
    assert any("Invalid Direction" in rec.message for rec in caplog.records)
    exp = math.radians(round(curr / 90.0) * 90)
    assert abs(out - exp) < 1e-12


# -------------------------------- wrap_diff --------------------------------

@pytest.mark.parametrize(
    "a,b,expected",
    [
        (0.0, 0.0, 0.0),                 # TC_017
        (math.pi, -math.pi, 0.0),        # TC_018: across the branch, same heading
        (math.pi/2, 0.0,  math.pi/2),    # TC_019+: positive
        (0.0, math.pi/2, -math.pi/2),    # TC_019+: negative
        (3.1, -3.1, -0.083),              # near ±π, small residual
    ]
)
def test_TC_017_TC_018_TC_019_wrap_diff(a, b, expected):
    """TC_017/TC_018/TC_019: Shortest signed angular difference within ±π."""
    out = F.wrap_diff(a, b)
    if expected == 0.0:
        assert abs(out) < 1e-9
    else:
        assert abs(out - expected) < 2e-2
