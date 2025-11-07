import numpy as np

class PID:

    """
    This class creates a PID controller
    Args:
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            dt: time step
            winduplim: integral windup limit (optional)
    """

    def __init__(self, Kp, Ki, Kd, dt, winduplim = 100):
        self.Kp = Kp # Proportional gain
        self.Ki = Ki # Integral gain
        self.Kd = Kd # Derivative gain
        self.winduplim = winduplim
        self.prev_error = 0.0
        self.integral = 0.0
        self.dt = dt

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

    def step(self, tracking_error):
        """
        PID controller step
        Args:
            Tracking_error: The error you want to minimise

        Returns:
            output: control output from PID
        """
        self.integral = np.clip(self.integral+ tracking_error * self.dt,-self.winduplim,self.winduplim) # integral anti wind up by clipping
        derivative = (tracking_error - self.prev_error) / self.dt
        output = self.Kp * tracking_error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = tracking_error
        return output
