import numpy as np
import pytest
from src.TR_PID import PID  # adjust import if src path differs


# ---------------------------- Fixtures ----------------------------

@pytest.fixture
def basic_pid():
    """Reusable PID with moderate gains and dt=0.1."""
    return PID(Kp=1.0, Ki=0.5, Kd=0.2, dt=0.1, winduplim=10.0)


# ---------------------------- Tests ----------------------------

def test_TC_001_proportional_term_only():
    """
    TC_001: With only proportional gain, output should equal Kp * error.
    Integral and derivative effects should be zero.
    """
    pid = PID(Kp=2.0, Ki=0.0, Kd=0.0, dt=0.1)
    result = pid.step(5.0)
    assert np.isclose(result, 10.0), "Proportional term mismatch."


def test_TC_002_integral_accumulation(basic_pid):
    """
    TC_002: The integral term should accumulate over multiple steps
    and affect output proportionally to Ki * integral.
    """
    basic_pid.Kd = 0.0  # isolate integral and proportional
    e = 2.0
    first = basic_pid.step(e)
    second = basic_pid.step(e)
    # second output should be greater due to accumulated integral
    assert second > first, "Integral term did not accumulate as expected."


def test_TC_003_derivative_response(basic_pid):
    """
    TC_003: When error changes rapidly, derivative term should dominate.
    The sign of derivative should follow the sign of (error change).
    """
    basic_pid.step(0.0)
    out = basic_pid.step(10.0)
    # derivative term = (10 - 0) / 0.1 * Kd = 100 * 0.2 = 20 contribution
    assert out > 10.0, "Derivative did not amplify sudden error increase."


def test_TC_004_reset_function_resets_state(basic_pid):
    """
    TC_004: reset() must clear previous error and integral.
    Ensures fresh start for new control sequences.
    """
    basic_pid.step(5.0)
    basic_pid.reset()
    assert basic_pid.prev_error == 0.0 and basic_pid.integral == 0.0


def test_TC_005_integral_windup_clamping():
    """
    TC_005: The integral term must not exceed the configured windup limit.
    """
    pid = PID(Kp=0.0, Ki=1.0, Kd=0.0, dt=1.0, winduplim=5.0)
    for _ in range(20):
        pid.step(10.0)
    assert abs(pid.integral) <= 5.0, "Integral exceeded windup limit."


def test_TC_006_derivative_sign(basic_pid):
    """
    TC_006: Derivative term should reduce output when error decreases.
    Simulates overshoot correction.
    """
    basic_pid.step(10.0)
    out = basic_pid.step(5.0)
    assert out < 10.0, "Derivative term failed to reduce output on falling error."


def test_TC_007_zero_error_steady_state(basic_pid):
    """
    TC_007: For zero error, derivative=0 and proportional term=0;
    output should be only integral term contribution (which may persist).
    """
    # build up some integral
    for _ in range(3):
        basic_pid.step(2.0)
    for _ in range(2): # left derivative zero out
        out = basic_pid.step(0.0)
    assert np.isclose(out, basic_pid.Ki * basic_pid.integral, atol=1e-9)


def test_TC_008_dt_influence():
    """
    TC_008: Smaller dt increases derivative magnitude (inverse relation).
    """
    pid_fast = PID(Kp=0.0, Ki=0.0, Kd=1.0, dt=0.01)
    pid_slow = PID(Kp=0.0, Ki=0.0, Kd=1.0, dt=0.1)
    pid_slow.step(0.0)
    pid_fast.step(0.0)
    d_fast = pid_fast.step(10.0)
    d_slow = pid_slow.step(10.0)
    assert d_fast > d_slow, "Derivative scaling with dt is incorrect."


def test_TC_009_output_sign_consistency():
    """
    TC_009: PID output should have the same sign as error
    for positive gains and non-negative integrals.
    """
    pid = PID(Kp=1.0, Ki=0.2, Kd=0.1, dt=0.1)
    out = pid.step(-5.0)
    assert out < 0.0, "Output sign mismatch for negative error."
