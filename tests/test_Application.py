import queue
import logging
import pytest
import mujoco
import time
import threading, queue
import logging
from src.ToyRobotControl_main import main

@pytest.fixture
def params():

    model = mujoco.MjModel.from_xml_path("model/Table_Top_Robot_Sim.xml")
    data = mujoco.MjData(model)

    ## PARAMETERS ##

    SETTLE_DEBOUNCE_CNT = 10

    # move control parameters
    KP_MOVE = 9
    KI_MOVE = 1.2
    KD_MOVE = 7.2
    KIWD_MOVE = 0.8
    CTRL_CLIP_MOVE = 5

    # yaw control parameters
    KP_YAW = 9.6
    KI_YAW = 2.05
    KD_YAW = 2.15
    KIWD_YAW = 0.8
    CTRL_CLIP_YAW = 10.3

    # yaw control edge detection and stopping parameters
    YAW_EDGE_STOP_LOW = -0.2
    YAW_EDGE_STOP_HIGH = 4.2


    return dict(
        model=model,
        data=data,
        KP_MOVE=KP_MOVE,
        KI_MOVE=KI_MOVE,
        KD_MOVE=KD_MOVE,
        KIWD_MOVE=KIWD_MOVE,
        KP_YAW=KP_YAW,
        KI_YAW=KI_YAW,
        KD_YAW=KD_YAW,
        KIWD_YAW=KIWD_YAW,
        CTRL_CLIP_MOVE=CTRL_CLIP_MOVE,
        CTRL_CLIP_YAW=CTRL_CLIP_YAW,
        YAW_EDGE_STOP_HIGH=YAW_EDGE_STOP_HIGH,
        YAW_EDGE_STOP_LOW=YAW_EDGE_STOP_LOW,
        SETTLE_DEBOUNCE_CNT=SETTLE_DEBOUNCE_CNT
    )


### Test Cases ###

def test_TC_045_integration_movement_1(capsys,monkeypatch,params):
    """
    TC_045: Integration test of main()
    test if model can be placed and moved correctly via command inputs.
    """

    # Build a queue that the main() function will read from
    cmd_q = queue.Queue()

    # Background thread: feeds commands with 1-second gaps
    def slow_input_feeder():
        commands = [
            "PLACE 1,1,NORTH",
            "MOVE",
            "RIGHT",
            "REPORT",
            "QUIT"
        ]
        for cmd in commands:
            cmd_q.put(cmd + "\n")  # add newline at the end of each command
            time.sleep(1.5)          # wait 1 second between commands to allow processing

    # Start feeder thread
    threading.Thread(target=slow_input_feeder, daemon=True).start()

    # Replace sys.stdin so main() reads from our queue instead of the real terminal
    class FakeStdin:
        def __iter__(self):
            while True:
                # Wait until a command is available in the queue (because of the 1 second sleep)
                yield cmd_q.get()

    fake_stdin = FakeStdin()
    monkeypatch.setattr("sys.stdin", fake_stdin)

    with pytest.raises(SystemExit):
        main(**params)

    out = capsys.readouterr().out
    assert "robot at x=1 y=2 facing EAST" in out, "Expected REPORT output not found."

def test_TC_045_integration_movement_2(capsys,monkeypatch,params):
    """
    TC_045: Integration test of main()
    test if model can be placed and moved correctly via command inputs.
    """

    # Build a queue that the main() function will read from
    cmd_q = queue.Queue()

    # Background thread: feeds commands with 1-second gaps
    def slow_input_feeder():
        commands = [
            "PLACE 2,2,SOUTH",
            "MOVE",
            "LEFT",
            "REPORT",
            "QUIT"
        ]
        for cmd in commands:
            cmd_q.put(cmd + "\n")  # add newline at the end of each command
            time.sleep(1.5)          # wait 1 second between commands to allow processing

    # Start feeder thread
    threading.Thread(target=slow_input_feeder, daemon=True).start()

    # Replace sys.stdin so main() reads from our queue instead of the real terminal
    class FakeStdin:
        def __iter__(self):
            while True:
                # Wait until a command is available in the queue (because of the 1 second sleep)
                yield cmd_q.get()

    fake_stdin = FakeStdin()
    monkeypatch.setattr("sys.stdin", fake_stdin)

    with pytest.raises(SystemExit):
        main(**params)

    out = capsys.readouterr().out
    assert "robot at x=2 y=1 facing EAST" in out, "Expected REPORT output not found."

def test_TC_045_integration_movement_3(capsys,monkeypatch,params):
    """
    TC_045: Integration test of main()
    test if model can be placed and moved correctly via command inputs.
    """

    # Build a queue that the main() function will read from
    cmd_q = queue.Queue()

    # Background thread: feeds commands with 1-second gaps
    def slow_input_feeder():
        commands = [
            "PLACE 2,2,SOUTH",
            "LEFT",
            "LEFT",
            "MOVE",
            "REPORT",
            "QUIT"
        ]
        for cmd in commands:
            cmd_q.put(cmd + "\n")  # add newline at the end of each command
            time.sleep(3)          # wait 1 second between commands to allow processing

    # Start feeder thread
    threading.Thread(target=slow_input_feeder, daemon=True).start()

    # Replace sys.stdin so main() reads from our queue instead of the real terminal
    class FakeStdin:
        def __iter__(self):
            while True:
                # Wait until a command is available in the queue (because of the 1 second sleep)
                yield cmd_q.get()

    fake_stdin = FakeStdin()
    monkeypatch.setattr("sys.stdin", fake_stdin)

    with pytest.raises(SystemExit):
        main(**params)

    out = capsys.readouterr().out
    assert "robot at x=2 y=3 facing NORTH" in out, "Expected REPORT output not found."

def test_TC_046_first_command_not_place(caplog,monkeypatch,params):
    """
    test if the first command is not PLACE, appropriate error is logged.
    """

    # Build a queue that the main() function will read from
    cmd_q = queue.Queue()

    # Background thread: feeds commands with 1-second gaps
    def slow_input_feeder():
        commands = [
            "LEFT",
            "QUIT"
        ]
        for cmd in commands:
            cmd_q.put(cmd + "\n")  # add newline at the end of each command
            time.sleep(1)          # wait 1 second between commands to allow processing

    # Start feeder thread
    threading.Thread(target=slow_input_feeder, daemon=True).start()

    # Replace sys.stdin so main() reads from our queue instead of the real terminal
    class FakeStdin:
        def __iter__(self):
            while True:
                # Wait until a command is available in the queue (because of the 1 second sleep)
                yield cmd_q.get()

    fake_stdin = FakeStdin()
    monkeypatch.setattr("sys.stdin", fake_stdin)
    with caplog.at_level(logging.INFO):
        with pytest.raises(SystemExit):
            main(**params)
    last_log = caplog.records[-1]
    assert "Please PLACE the robot first using the PLACE X,Y,FACING command" in last_log.getMessage(), "Expected output not found."

def test_TC_047_invalid_command(capsys,monkeypatch,params):
    """
    test if command is invalid, appropriate error is printed.
    """

    # Build a queue that the main() function will read from
    cmd_q = queue.Queue()

    # Background thread: feeds commands with 1-second gaps
    def slow_input_feeder():
        commands = [
            "PLACE 2,2,SOUTH",
            "NORTHWEST",
            "QUIT"
        ]
        for cmd in commands:
            cmd_q.put(cmd + "\n")  # add newline at the end of each command
            time.sleep(1)          # wait 1 second between commands to allow processing

    # Start feeder thread
    threading.Thread(target=slow_input_feeder, daemon=True).start()

    # Replace sys.stdin so main() reads from our queue instead of the real terminal
    class FakeStdin:
        def __iter__(self):
            while True:
                # Wait until a command is available in the queue (because of the 1 second sleep)
                yield cmd_q.get()

    fake_stdin = FakeStdin()
    monkeypatch.setattr("sys.stdin", fake_stdin)
   
    with pytest.raises(SystemExit):
        main(**params)
    out = capsys.readouterr().out
    assert "Invalid command" in out
    assert "PLACE X,Y,F" in out
    assert "MOVE" in out
    assert "LEFT" in out
    assert "RIGHT" in out
    assert "REPORT" in out

def test_TC_048_moving_left(caplog,monkeypatch,params):
    """
    test when moving left command is given, appropriate log is printed.
    """

    # Build a queue that the main() function will read from
    cmd_q = queue.Queue()

    # Background thread: feeds commands with 1-second gaps
    def slow_input_feeder():
        commands = [
            "PLACE 0,0,NORTH",
            "LEFT",
            "QUIT"
        ]
        for cmd in commands:
            cmd_q.put(cmd + "\n")  # add newline at the end of each command
            time.sleep(1)          # wait 1 second between commands to allow processing

    # Start feeder thread
    threading.Thread(target=slow_input_feeder, daemon=True).start()

    # Replace sys.stdin so main() reads from our queue instead of the real terminal
    class FakeStdin:
        def __iter__(self):
            while True:
                # Wait until a command is available in the queue (because of the 1 second sleep)
                yield cmd_q.get()

    fake_stdin = FakeStdin()
    monkeypatch.setattr("sys.stdin", fake_stdin)
    with caplog.at_level(logging.INFO):
        with pytest.raises(SystemExit):
            main(**params)
    last_log = caplog.records[-1]
    assert "Moving left" in last_log.getMessage(), "Expected output not found."

def test_TC_049_moving_right(caplog,monkeypatch,params):
    """
    test when moving right command is given, appropriate log is printed.
    """

    # Build a queue that the main() function will read from
    cmd_q = queue.Queue()

    # Background thread: feeds commands with 1-second gaps
    def slow_input_feeder():
        commands = [
            "PLACE 4,4,SOUTH",
            "RIGHT",
            "QUIT"
        ]
        for cmd in commands:
            cmd_q.put(cmd + "\n")  # add newline at the end of each command
            time.sleep(1)          # wait 1 second between commands to allow processing

    # Start feeder thread
    threading.Thread(target=slow_input_feeder, daemon=True).start()

    # Replace sys.stdin so main() reads from our queue instead of the real terminal
    class FakeStdin:
        def __iter__(self):
            while True:
                # Wait until a command is available in the queue (because of the 1 second sleep)
                yield cmd_q.get()

    fake_stdin = FakeStdin()
    monkeypatch.setattr("sys.stdin", fake_stdin)
    with caplog.at_level(logging.INFO):
        with pytest.raises(SystemExit):
            main(**params)
    last_log = caplog.records[-1]
    assert "Moving right" in last_log.getMessage(), "Expected output not found."

def test_TC_050_moving_foward(caplog,monkeypatch,params):
    """
    test when moving forward command is given, appropriate log is printed.
    """

    # Build a queue that the main() function will read from
    cmd_q = queue.Queue()

    # Background thread: feeds commands with 1-second gaps
    def slow_input_feeder():
        commands = [
            "PLACE 2,2,SOUTH",
            "MOVE",
            "QUIT"
        ]
        for cmd in commands:
            cmd_q.put(cmd + "\n")  # add newline at the end of each command
            time.sleep(1)          # wait 1 second between commands to allow processing

    # Start feeder thread
    threading.Thread(target=slow_input_feeder, daemon=True).start()

    # Replace sys.stdin so main() reads from our queue instead of the real terminal
    class FakeStdin:
        def __iter__(self):
            while True:
                # Wait until a command is available in the queue (because of the 1 second sleep)
                yield cmd_q.get()

    fake_stdin = FakeStdin()
    monkeypatch.setattr("sys.stdin", fake_stdin)
    with caplog.at_level(logging.INFO):
        with pytest.raises(SystemExit):
            main(**params)
    last_log = caplog.records[-1]
    assert "moving to coordinates 2,1" in last_log.getMessage(), "Expected output not found."

def test_TC_051_moving_foward_clip(caplog,monkeypatch,params):
    """
    test when moving forward command is given at edge motion is inhibited and appropriate log is printed.
    """

    # Build a queue that the main() function will read from
    cmd_q = queue.Queue()

    # Background thread: feeds commands with 1-second gaps
    def slow_input_feeder():
        commands = [
            "PLACE 4,4,NORTH",
            "MOVE",
            "QUIT"
        ]
        for cmd in commands:
            cmd_q.put(cmd + "\n")  # add newline at the end of each command
            time.sleep(1)          # wait 1 second between commands to allow processing

    # Start feeder thread
    threading.Thread(target=slow_input_feeder, daemon=True).start()

    # Replace sys.stdin so main() reads from our queue instead of the real terminal
    class FakeStdin:
        def __iter__(self):
            while True:
                # Wait until a command is available in the queue (because of the 1 second sleep)
                yield cmd_q.get()

    fake_stdin = FakeStdin()
    monkeypatch.setattr("sys.stdin", fake_stdin)
    with caplog.at_level(logging.INFO):
        with pytest.raises(SystemExit):
            main(**params)
    last_log = caplog.records[-1]
    assert "Cannot move forward. Robot will fall and get hurt! Please rotate. " in last_log.getMessage(), "Expected output not found."

def test_TC_052_place_outside_table(caplog,monkeypatch,params):
    """
    test the robot cannot be placed outside the table and appropriate error is logged.
    """

    # Build a queue that the main() function will read from
    cmd_q = queue.Queue()

    # Background thread: feeds commands with 1-second gaps
    def slow_input_feeder():
        commands = [
            "PLACE -1,4,NORTH",
            "QUIT"
        ]
        for cmd in commands:
            cmd_q.put(cmd + "\n")  # add newline at the end of each command
            time.sleep(1)          # wait 1 second between commands to allow processing

    # Start feeder thread
    threading.Thread(target=slow_input_feeder, daemon=True).start()

    # Replace sys.stdin so main() reads from our queue instead of the real terminal
    class FakeStdin:
        def __iter__(self):
            while True:
                # Wait until a command is available in the queue (because of the 1 second sleep)
                yield cmd_q.get()

    fake_stdin = FakeStdin()
    monkeypatch.setattr("sys.stdin", fake_stdin)
    with caplog.at_level(logging.INFO):
        with pytest.raises(SystemExit):
            main(**params)
    last_log = caplog.records[-1]
    assert "Place coordinates outside table, Robot will fall and get hurt! Allowed coordinates from 0 to 4" in last_log.getMessage(), "Expected output not found."

def test_TC_053_move_inhibit(caplog,monkeypatch,params):
    """
    test that forward movement is inhibited when inhibit flag is set and appropriate log is printed.
    """

    # Build a queue that the main() function will read from
    cmd_q = queue.Queue()

    # Background thread: feeds commands with 1-second gaps
    def slow_input_feeder():
        commands = [
            "PLACE 2,2,NORTH",
            "LEFT",
            "MOVE",
            "QUIT"
        ]
        for cmd in commands:
            cmd_q.put(cmd + "\n")  # add newline at the end of each command
            time.sleep(0.5)          # wait 1 second between commands to allow processing

    # Start feeder thread
    threading.Thread(target=slow_input_feeder, daemon=True).start()

    # Replace sys.stdin so main() reads from our queue instead of the real terminal
    class FakeStdin:
        def __iter__(self):
            while True:
                # Wait until a command is available in the queue (because of the 1 second sleep)
                yield cmd_q.get()

    fake_stdin = FakeStdin()
    monkeypatch.setattr("sys.stdin", fake_stdin)
    with caplog.at_level(logging.INFO):
        with pytest.raises(SystemExit):
            main(**params)
    last_log = caplog.records[-1]
    assert "Cannot accept a move motion, another motion in progress" in last_log.getMessage(), "Expected output not found."

def test_TC_054_yaw_inhibit(caplog,monkeypatch,params):
    """
    test that yaw movement is inhibited when inhibit flag is set and appropriate log is printed.
    """

    # Build a queue that the main() function will read from
    cmd_q = queue.Queue()

    # Background thread: feeds commands with 1-second gaps
    def slow_input_feeder():
        commands = [
            "PLACE 2,2,NORTH",
            "MOVE",
            "LEFT",
            "QUIT"
        ]
        for cmd in commands:
            cmd_q.put(cmd + "\n")  # add newline at the end of each command
            time.sleep(0.2)          # wait 1 second between commands to allow processing

    # Start feeder thread
    threading.Thread(target=slow_input_feeder, daemon=True).start()

    # Replace sys.stdin so main() reads from our queue instead of the real terminal
    class FakeStdin:
        def __iter__(self):
            while True:
                # Wait until a command is available in the queue (because of the 1 second sleep)
                yield cmd_q.get()

    fake_stdin = FakeStdin()
    monkeypatch.setattr("sys.stdin", fake_stdin)
    with caplog.at_level(logging.INFO):
        with pytest.raises(SystemExit):
            main(**params)
    last_log = caplog.records[-1]
    assert "Cannot accept a rotation motion, another motion in progress" in last_log.getMessage(), "Expected output not found."