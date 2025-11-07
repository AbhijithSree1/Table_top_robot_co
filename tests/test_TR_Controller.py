import math
import queue
import logging
import numpy as np
import pytest

from src.TR_Controller import Controller


# ----------------------------- Fakes / fixtures -----------------------------

class FakeOpt:
    def __init__(self):
        self.gravity = [0.0, 0.0, 0.0]

class FakeModel:
    def __init__(self):
        self.opt = FakeOpt()

class FakeData:
    def __init__(self):
        self.marker = True


@pytest.fixture
def model():
    return FakeModel()

@pytest.fixture
def data():
    return FakeData()

@pytest.fixture
def params():
    # Mirror defaults
    return dict(
        Kp_move=9.0, Ki_move=1.2, Kd_move=7.2, Kiwd_move=0.8,
        Kp_yaw=9.6, Ki_yaw=2.05, Kd_yaw=2.15, Kiwd_yaw=0.8,
        move_ctrl_clip=5.0, yaw_ctrl_clip=10.3,
        yaw_edge_stop_low=-0.2, yaw_edge_stop_high=4.2,
        settle_debounce_count=10, dt=0.01
    )

@pytest.fixture
def controller(model, data, params):
    # Fresh controller per test
    return Controller(model=model, data=data, **params)

def q(cmds):
    """Build a Queue preloaded with string commands."""
    Q = queue.Queue()
    for c in cmds:
        Q.put(c)
    return Q


# ----------------------------- Common stubs --------------------------------

@pytest.fixture
def stubbed_io(monkeypatch):
    """
    Stub TR_funcs functions referenced inside src.TR_Controller.
    We patch INTO the module under test, not the original TR_funcs.
    """
    import src.TR_Controller as C

    monkeypatch.setattr(C, "read_sensor",
                        lambda d: (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
    monkeypatch.setattr(C, "quaternion_to_yaw", lambda *a: 0.0)

    calls = {"place": 0, "move_target": 0}

    def _place_robot(model, data, pos, facing):
        calls["place"] += 1
        # emulate a tiny state change so code paths rely on non-nones
        data.last_place = (tuple(pos), facing)

    def _robot_move_target(x, y, facing):
        calls["move_target"] += 1
        # simple forward step then clip later
        return (int(round(x)) + 1, int(round(y)) + 1)

    monkeypatch.setattr(C, "place_robot", _place_robot)
    monkeypatch.setattr(C, "robot_move_target", _robot_move_target)

    # identity transforms for easy reasoning
    monkeypatch.setattr(C, "location_transform_MAP2UI", lambda v: v)
    monkeypatch.setattr(C, "location_transform_UI2MAP", lambda v: v)

    # yaw helpers
    monkeypatch.setattr(C, "wrap_diff", lambda a, b: (a - b + math.pi) % (2*math.pi) - math.pi)
    monkeypatch.setattr(C, "yaw_target_deg",
                        lambda curr_deg, dir_: math.radians(90 if dir_ == "LEFT" else -90))

    # current_facing based on deg
    monkeypatch.setattr(C, "current_facing", lambda d: "NORTH")

    return calls


# --------------------------------- Tests ------------------------------------

def test_TC_031_queue_contract_raises_when_empty(controller):
    """
    TC_031: Documented contract today: controller_target_calc expects caller
    to catch queue.Empty when queue has no commands.
    """
    with pytest.raises(queue.Empty):
        controller.controller_target_calc(q([]))


def test_TC_032_place_flow_sets_flags_and_targets(controller, stubbed_io, caplog):
    """
    TC_032: PLACE should mark placed, set gravity, set targets to current state,
    and call place_robot exactly once.
    """

    with caplog.at_level(logging.INFO): # capture log output
         
        controller.controller_read_inputs()  
        try:
            controller.controller_target_calc(q(["PLACE 1,2,NORTH"]))
        except queue.Empty:
                    pass


    assert controller.placed_flag is True
    assert controller.model.opt.gravity[2] == -9.81
    assert stubbed_io["place"] == 1
    # targets set to current scaled positions (0,0) from stubbed read_sensor
    assert controller.x_tgt == controller.x_scaled == 0.0
    assert controller.y_tgt == controller.y_scaled == 0.0
    assert "PLACING" in caplog.text


def test_TC_033_move_updates_targets_and_clips(controller, stubbed_io):
    """
    TC_033: MOVE should update x_tgt,y_tgt via robot_move_target then clip to [0,4].
    """
    controller.placed_flag = True
    controller.facing = "NORTH"
    controller.x_scaled = 4.0
    controller.y_scaled = 4.0
    try:
        controller.controller_target_calc(q(["MOVE"]))
    except queue.Empty:
                pass
    # stub returns +1,+1 so clipping should bound to 4
    assert controller.x_tgt == 4.0
    assert controller.y_tgt == 4.0
    assert stubbed_io["move_target"] == 1

def test_TC_033_move_updates_targets(controller, stubbed_io):
    """
    TC_033: MOVE should update x_tgt,y_tgt via robot_move_target then clip to [0,4].
    """
    controller.placed_flag = True
    controller.facing = "NORTH"
    controller.x_scaled = 0.0
    controller.y_scaled = 0.0
    try:
        controller.controller_target_calc(q(["MOVE"]))
    except queue.Empty:
                pass
    assert controller.x_tgt == 1.0
    assert controller.y_tgt == 1.0
    assert stubbed_io["move_target"] == 1

def test_TC_034_left_right_update_yaw_target(controller, stubbed_io):
    """
    TC_034: LEFT then RIGHT should update yaw_tgt to stubbed values.
    """
    controller.placed_flag = True
    controller.yaw_unwrapped = 0.0
    controller.yaw_unwrapped_deg = 0.0
    try:
        controller.controller_target_calc(q(["LEFT"]))
    except queue.Empty:
                pass
    left_tgt = controller.yaw_tgt
    try:
        controller.controller_target_calc(q(["RIGHT"]))
    except queue.Empty:
                pass    
    right_tgt = controller.yaw_tgt
    assert np.isclose(left_tgt, math.radians(90))
    assert np.isclose(right_tgt, math.radians(-90))


def test_TC_035_rotation_inhibit_blocks_left(controller, stubbed_io, caplog):
    """
    TC_035: With inhibit_yaw_motion=True, LEFT must not change yaw_tgt.
    """
    controller.placed_flag = True
    controller.inhibit_yaw_motion = True
    controller.yaw_tgt = 0.0
    try:
        controller.controller_target_calc(q(["LEFT"]))
    except queue.Empty:
                pass
    assert controller.yaw_tgt == 0.0
    assert any("Cannot accept a rotation" in r.message for r in caplog.records)


def test_TC_036_move_inhibit_blocks_move(controller, stubbed_io, caplog):
    """
    TC_036: With inhibit_move_motion=True, MOVE must not change x_tgt/y_tgt.
    """
    controller.placed_flag = True
    controller.inhibit_move_motion = True
    controller.x_tgt, controller.y_tgt = 1.0, 1.0
    with caplog.at_level(logging.INFO):
        try:
            controller.controller_target_calc(q(["MOVE"]))
        except queue.Empty:
                    pass
    assert (controller.x_tgt, controller.y_tgt) == (1.0, 1.0)
    assert any("Cannot accept a move motion" in r.message for r in caplog.records)


def test_TC_037_report_prints(capsys, controller, stubbed_io):
    """
    TC_037: REPORT should print user-facing status line.
    """
    controller.placed_flag = True
    controller.x_scaled = 2.0
    controller.y_scaled = 3.0
    controller.facing = "NORTH"
    try:
        controller.controller_target_calc(q(["REPORT"]))
    except queue.Empty:
                pass
    out = capsys.readouterr().out # capture stdout
    assert "robot at x=2 y=3 facing NORTH" in out


def test_TC_038_invalid_command_prints_help(capsys, controller, stubbed_io):
    """
    TC_038: Unknown command should print help text.
    """
    try:
        controller.controller_target_calc(q(["DO_SOMETHING_ELSE"]))
    except queue.Empty:
                pass
    out = capsys.readouterr().out
    assert "Invalid command" in out
    assert "PLACE X,Y,F" in out
    assert "MOVE" in out
    assert "LEFT" in out
    assert "RIGHT" in out
    assert "REPORT" in out


def test_TC_039_edge_stop_sets_flag_and_zeroes_yaw(controller, stubbed_io):
    """
    TC_039: If position is beyond edge thresholds, yaw control must be zeroed
    and edge_evade_move flagged.
    """
    controller.placed_flag = True
    controller.facing = "EAST"
    controller.x_scaled = controller.YAW_EDGE_STOP_HIGH + 0.5
    controller.y_scaled = 0.0
    controller.yaw_tgt = 1.0
    controller.yaw_unwrapped = 0.0

    ctrl_move, ctrl_yaw = controller.controller_control_output()
    assert ctrl_yaw == 0.0
    assert controller.edge_evade_move is True


def test_TC_040_small_yaw_error_triggers_pid_reset(controller, stubbed_io):
    """
    TC_040: When |yaw_error| < 0.01 rad, PID_yaw should reset its internal state.
    """
    controller.placed_flag = True
    controller.facing = "NORTH"
    controller.yaw_tgt = 0.0
    controller.yaw_unwrapped = 0.0
    # build some integral
    controller.PID_yaw.integral = 5.0
    controller.PID_yaw.prev_error = 1.0

    controller.controller_control_output()
    assert controller.PID_yaw.integral == 0.0
    assert controller.PID_yaw.prev_error == 0.0


def test_TC_041_move_clip_applies(controller, stubbed_io):
    """
    TC_041: Large position error should be clipped to ±CTRL_CLIP_MOVE.
    """
    controller.placed_flag = True
    controller.facing = "EAST"
    controller.x_scaled = 0.0
    controller.x_tgt = 1e6  # huge error to saturate
    controller.y_scaled = controller.y_tgt = 0.0
    ctrl_move, _ = controller.controller_control_output()
    assert np.isclose(abs(ctrl_move), controller.CTRL_CLIP_MOVE)


def test_TC_042_yaw_clip_applies(controller, stubbed_io):
    """
    TC_042: Large yaw error should be clipped to ±CTRL_CLIP_YAW when not edge-stopped.
    """
    controller.placed_flag = True
    controller.facing = "NORTH"
    controller.x_scaled = controller.y_scaled = 0.0 
    controller.yaw_unwrapped = 0.0
    controller.yaw_tgt = 10.0  # big error
    _, ctrl_yaw = controller.controller_control_output()
    assert np.isclose(abs(ctrl_yaw), controller.CTRL_CLIP_YAW)


def test_TC_043_cross_motion_inhibit_move_then_release(controller, stubbed_io):
    """
    TC_043: Move controller active (>1) should set inhibit_yaw_motion, and release
    after motion settles (<=0.5) for > debounce count cycles.
    """
    controller.placed_flag = True
    controller.facing = "EAST"
    controller.x_scaled = 0.0
    controller.x_tgt = 1.0
    controller.y_scaled = controller.y_tgt = 0.0

    # Engage inhibit
    controller.controller_control_output()
    assert controller.inhibit_yaw_motion is True

    # Remove error and step through debounce
    controller.x_tgt = controller.x_scaled
    for _ in range(controller.SETTLE_DEBOUNCE_CNT + 1):
        controller.controller_control_output()  
    assert controller.inhibit_yaw_motion is False


def test_TC_044_cross_motion_inhibit_yaw_then_release(controller, stubbed_io):
    """
    TC_044: Yaw controller active (>1) should set inhibit_move_motion, and release
    after yaw settles (<=0.5) for > debounce count cycles.
    """
    controller.placed_flag = True
    controller.facing = "NORTH"
    controller.x_scaled = controller.y_scaled = 0.0
    # big yaw error
    controller.yaw_unwrapped = 0.0
    controller.yaw_tgt = 10.0

    controller.controller_control_output()
    assert controller.inhibit_move_motion is True

    # settle yaw error near zero for debounce cycles
    controller.yaw_tgt = controller.yaw_unwrapped
    for _ in range(controller.SETTLE_DEBOUNCE_CNT + 1):
         controller.controller_control_output()
    assert controller.inhibit_move_motion is False
