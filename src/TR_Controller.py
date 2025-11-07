import numpy as np
import logging
from .TR_funcs import parse_place, quaternion_to_yaw, current_facing, location_transform_MAP2UI
from .TR_funcs import place_robot, robot_move_target, location_transform_UI2MAP, read_sensor
from .TR_funcs import yaw_target_deg, wrap_diff
from .TR_PID import PID


class Controller:
    """
    This class creates the controller for the Table Top Robot. It perform tasks such as reading the current state
    of the robot from mujoco, parsing user commands, calculating control actions using PID controllers and implementing
    safety features such as edge fall prevention and cross motion prevention.
    Args:
        model: mujoco model object
        data: mujoco data object
        Kp_move: Proportional gain for movement PID
        Ki_move: Integral gain for movement PID
        Kd_move: Derivative gain for movement PID
        Kp_yaw: Proportional gain for yaw PID
        Ki_yaw: Integral gain for yaw PID
        Kd_yaw: Derivative gain for yaw PID
        Kiwd_move: Integral windup limit for movement PID
        Kiwd_yaw: Integral windup limit for yaw PID
        move_ctrl_clip: control output clipping value for movement PID
        yaw_ctrl_clip: control output clipping value for yaw PID
        yaw_edge_stop_high: high edge stop threshold for yaw motion
        yaw_edge_stop_low: low edge stop threshold for yaw motion
        settle_debounce_count: debounce count for settling detection
        dt: time step
        
    """

    def __init__(self,model,data,Kp_move,Ki_move,Kd_move,Kp_yaw,Ki_yaw,Kd_yaw,
                 Kiwd_move,Kiwd_yaw,move_ctrl_clip,yaw_ctrl_clip,yaw_edge_stop_high,
                 yaw_edge_stop_low,settle_debounce_count,dt):
        self.model = model
        self.data = data
        self.KP_MOVE = Kp_move
        self.KI_MOVE = Ki_move
        self.KD_MOVE = Kd_move
        self.KP_YAW = Kp_yaw
        self.KI_YAW = Ki_yaw
        self.KD_YAW = Kd_yaw
        self.KIWD_MOVE = Kiwd_move
        self.KIWD_YAW = Kiwd_yaw
        self.dt = dt
        self.CTRL_CLIP_MOVE = move_ctrl_clip
        self.CTRL_CLIP_YAW = yaw_ctrl_clip
        self.YAW_EDGE_STOP_HIGH = yaw_edge_stop_high
        self.YAW_EDGE_STOP_LOW = yaw_edge_stop_low
        self.SETTLE_DEBOUNCE_CNT = settle_debounce_count

        # PID #
        self.PID_move_X = PID(self.KP_MOVE,self.KI_MOVE,self.KD_MOVE,self.dt,winduplim=self.KIWD_MOVE)
        self.PID_move_Y = PID(self.KP_MOVE,self.KI_MOVE,self.KD_MOVE,self.dt,winduplim=self.KIWD_MOVE)
        self.PID_yaw = PID(self.KP_YAW,self.KI_YAW,self.KD_YAW,self.dt,winduplim=self.KIWD_YAW)

        # Control Targets # 
        self.x_tgt = 2                           # Initalised to 2 as
        self.y_tgt = 2
        self.yaw_tgt = 0                         # radians

        # Edge fall over prevention # 
        self.inhibit_yaw_motion = False
        self.inhibit_move_motion = False
        self.edge_evade_move = False

        # cross motion prevention # 
        self.yaw_settle_cntr = 0
        self.motion_settle_cntr = 0

        # continuous yaw calculation # 
        self.yaw_wrapped_prev = None              # radians
        self.yaw_unwrapped     = 0.0              # radians
        self.yaw_unwrapped_deg = 0.0              # degrees
        self.placed_flag = False
        self.facing = None
        self.x_scaled = 0.0
        self.y_scaled = 0.0

    def controller_read_inputs(self):
        """
        Reads the current state of the robot from mujoco and updates the internal states
        such as x,y coordinates, continuous yaw angle and facing direction.
        """
        x,y,_,q_w,q_x,q_y,q_z = read_sensor(self.data)                      # Read current model state
        self.x_scaled = location_transform_MAP2UI(x)                        # Transform MAP coordinates to UI
        self.y_scaled = location_transform_MAP2UI(y)
        yaw_wrapped = quaternion_to_yaw(q_w,q_x,q_y,q_z)                    # Calculate yaw angle in radians from quaternions, wrapped between 0-2pi

        if self.yaw_wrapped_prev is None:                                   # On first timestep set the unwrapped yaw angles to the wrapped angles
            self.yaw_wrapped_prev  = yaw_wrapped
            self.yaw_unwrapped     = yaw_wrapped
            
        else:                                                               # On future steps calculate the continuous yaw angle, avoids discontinuities, smooth control
            dy = wrap_diff(yaw_wrapped, self.yaw_wrapped_prev)              # calculate delta between previous wrapped angle and current angle
            self.yaw_unwrapped   += dy                                        # continuously add to unwrapped yaw to obtain continuous yaw angle 
            self.yaw_wrapped_prev = yaw_wrapped

        self.yaw_unwrapped_deg = np.rad2deg(self.yaw_unwrapped)
        self.facing = current_facing(self.yaw_unwrapped_deg)                # obtain the current facing of the bot

    def controller_target_calc(self,cmd_q):
        """
        Reads the user commands from standard input and updates that internal states and targets
        such as x,y coordinates, new robot spwan, facing direction, x,y and yaw targets. 
        Args:
            cmd_q: command queue from standard input thread
        """
        
        
        while True:
            line = cmd_q.get_nowait()
            s = line.strip() # Requirement ID SR-026
            if not s:
                continue

            # If user types QUIT exit the simulation
            if s.upper() == "QUIT": # Requirement ID SR-032                            
                raise SystemExit
            
            # check if placed before accepting moving or rotating commands
            # Requirement ID SR-015 SR-026
            if s.upper() in ["LEFT","RIGHT","MOVE","REPORT"] and self.placed_flag == False:
                logging.info("Please PLACE the robot first using the PLACE X,Y,FACING command")
                continue

            # Requirement ID SR-018 SR-023 SR-026
            if s.upper() == "LEFT" and self.placed_flag == True and self.inhibit_yaw_motion == False:
                self.yaw_tgt = yaw_target_deg(self.yaw_unwrapped_deg,"LEFT")  # calculate yaw target for LEFT movement
                logging.info("Moving left") # Requirement ID SR-030
                continue
            
            # Requirement ID SR-019 SR-023 SR-026
            elif s.upper() == "RIGHT" and self.placed_flag == True and self.inhibit_yaw_motion == False:
                self.yaw_tgt = yaw_target_deg(self.yaw_unwrapped_deg,"RIGHT") # calculate yaw target for RIGHT movement
                logging.info("Moving right") # Requirement ID SR-029
                continue

            # Requirement ID SR-023
            if s.upper() in ["LEFT","RIGHT"] and self.inhibit_yaw_motion == True:
                logging.warning("Cannot accept a rotation motion, another motion in progress")
                continue
            
            # Requirement ID SR-017 SR-022
            if s.upper() == "MOVE" and self.placed_flag == True and self.inhibit_move_motion == False:
                self.x_tgt,self.y_tgt = robot_move_target(self.x_scaled,self.y_scaled,self.facing)   # calculate target x and y movement
                
                # Requirement ID SR-021
                if self.x_tgt < 0 or self.x_tgt > 4 or self.y_tgt < 0 or self.y_tgt > 4:
                    logging.warning("Cannot move forward. Robot will fall and get hurt! Please rotate. ")
                else:
                    logging.info(f"moving to coordinates {self.x_tgt},{self.y_tgt}") # Requirement ID SR-028
                self.x_tgt = np.clip(self.x_tgt,0,4)      # clip x and y target to ensure they are bounded in the table space
                self.y_tgt = np.clip(self.y_tgt,0,4)
                continue
            
            # Requirement ID SR-022
            elif s.upper() == "MOVE" and self.inhibit_move_motion == True:
                logging.warning("Cannot accept a move motion, another motion in progress")
                continue
            
            # Requirement ID SR-020
            if s.upper() == "REPORT" and self.placed_flag == True:   # report current position when "REPORT" is entered
                print(f"robot at x={int(np.round(self.x_scaled))} y={int(np.round(self.y_scaled))} facing {self.facing}")
                continue
            
            # Requirement ID SR-007
            place = parse_place(s) # check for valid PLACE command

            # Requirement ID SR-027
            if s.upper() not in ["LEFT","RIGHT","MOVE","REPORT"] and place is None:
                print("Invalid command, Valid Commands are:")
                print("     PLACE X,Y,F   X,Y: bot initial coordinates, integers from 0-4 F: Target facing in [NORTH,SOUTH,EAST,WEST]")
                print("     MOVE    move robot forward in current facing direction")
                print("     LEFT    rotate robot anti clockwise")
                print("     RIGHT   rotate robot clockwise")
                print("     REPORT  report the robots current X,Y and Facing ")
                continue

            if place is not None: # Requirement ID SR-001 SR-006 SR-016
                x_place, y_place, tgt_facing = place
                # Requirement ID SR-015
                if x_place < 0 or x_place > 4 or y_place < 0 or y_place > 4:
                    logging.warning("Place coordinates outside table, Robot will fall and get hurt! Allowed coordinates from 0 to 4")
                    continue
                else:
                    logging.info(f"PLACING with x={x_place} y={y_place} facing={tgt_facing}")
                    x_place = location_transform_UI2MAP(np.clip(x_place,0,4))   # transform UI frame X coordinate to MAP frame
                    y_place = location_transform_UI2MAP(np.clip(y_place,0,4))   # transform UI frame Y coordinate to MAP frame
                    place_robot(self.model,self.data,pos=[x_place,y_place],facing=tgt_facing)       # place the robot
                    self.placed_flag = True                                          # set placed flag as true
                    self.model.opt.gravity[2] = -9.81                                # enable gravity to allow the robot to fall and continue further movements
                    x,y,_,q_w,q_x,q_y,q_z = read_sensor(self.data)                   # read updates states after PLACING
                    self.x_scaled = location_transform_MAP2UI(x)                     # transform MAP frame X coordinate to UI frame
                    self.y_scaled = location_transform_MAP2UI(y)                     # transform MAP frame Y coordinate to UI frame
                    self.x_tgt = self.x_scaled                                            # reset x_target as the new PLACED x coordinate
                    self.y_tgt = self.y_scaled                                            # reset y_target as the new PLACED y coordinate
                    yaw_wrapped  = quaternion_to_yaw(q_w,q_x,q_y,q_z)           # calculate current rotation after PLACING
                    # Requirement ID SR-013
                    self.yaw_wrapped_prev = yaw_wrapped                              # reset yaw continuous measurements
                    self.yaw_unwrapped     = yaw_wrapped
                    self.yaw_unwrapped_deg = np.rad2deg(self.yaw_unwrapped)
                    self.yaw_tgt = yaw_wrapped                                       # reset yaw target to the new PLACED yaw angle
                    self.facing = current_facing(self.yaw_unwrapped_deg)                   # determine the current facing
                    # Requirement ID SR-031
                    logging.info(f"robot placed at x={self.x_scaled:.2f} y={self.y_scaled:.2f} facing {self.facing}")

    def controller_control_output(self):
        """
        Calculates the control actions for movement and yaw using PID controllers based on the current
        state and target state. Implements safety features such as edge fall prevention and cross motion prevention.
        Returns:
            ctrl_move: control action for movement (Nm)
            ctrl_yaw: control action for yaw (Nm)
        """
        # Calculate yaw error
        yaw_error = wrap_diff(self.yaw_tgt, self.yaw_unwrapped)

        # Calculate movement error
        y_error = self.y_tgt  - self.y_scaled
        x_error = self.x_tgt  - self.x_scaled

        ## Control for longitudinal motion ##
        """
        This section of code controls the robots motion in longitudinal direction depending on facing of the robot. The controller
        outputs the total torque demand acorss the entire chassis to read the x and y target. This is further arbitrated below to
        individual torque demands. Control is only executed when the robot is not yawing significantly (identified using the inhibit_move_motion flag).
        Control is clipped between -5Nm and 5Nm to avoid excessive torque. Higher torque can cause the robot to wheeliee!
        Control direction is inverted (by inverting the error) when the robot is facing the SOUTH or WEST direction to allow the robot to move forward
        and reduce target errors. Integral windup is used to avoid unwanted integral action. 
        """

        # Requirement ID SR-017
        if self.facing == "NORTH" and self.inhibit_move_motion ==False:  # Facing North               
            ctrl_move = self.PID_move_Y.step(y_error)
        elif self.facing == "SOUTH" and self.inhibit_move_motion ==False:  # Facing South               
            ctrl_move = self.PID_move_Y.step(-y_error) # Reverse control direction
        elif self.facing == "EAST" and self.inhibit_move_motion ==False:  # Facing East               
            ctrl_move = self.PID_move_X.step(x_error)
        elif self.facing == "WEST" and self.inhibit_move_motion ==False:  # Facing West               
            ctrl_move = self.PID_move_X.step(-x_error) # Reverse control direction       
        else:
            ctrl_move = 0

        ctrl_move = np.clip(ctrl_move,-self.CTRL_CLIP_MOVE,self.CTRL_CLIP_MOVE)  
        ## Control for rotational motion ##
        """
        This section of the code controls the robots motion around the Z axis. The robot has 4 wheel each with its own motor - 
        This allows the robot to do a TANK TURN on the spot to rotate. Rotation is faciliated using this control. The controller 
        reduces the error between the continual yaw target and actual. If the clearance to perform the yaw motion is not available, 
        the controller hold the current yaw angle until clearance becomes available. The yaw motion is also inhibited when 
        a move motion is active indicated using the inhibit_yaw_motion flag. This controller provides the delta torque required per axle
        between the left and right side to faciliate rotation. Control action is clipped to 10.3Nm to allow smooth control and prevent excessive rotation. 

        If during a rotation control action, the x or y coordinates gets too close to the edges, the yaw motion is stopped untill 
        the longitudinal controller can make space before continuing with the yaw motion. This is signified with an edge_evade_move flag. 

        The PID controllers internal states are reset when the yaw error is less than a value (0.1).
        """

        # Requirement ID SR-018 SR-019    
        if self.inhibit_yaw_motion==False:
            
            ctrl_yaw = self.PID_yaw.step(yaw_error)
            ctrl_yaw = np.clip(ctrl_yaw,-self.CTRL_CLIP_YAW,self.CTRL_CLIP_YAW)  # clip control

            # Requirement ID SR-024   # Edge fall prevention during yaw motion
            if (self.x_scaled < self.YAW_EDGE_STOP_LOW or self.x_scaled > self.YAW_EDGE_STOP_HIGH or 
                self.y_scaled < self.YAW_EDGE_STOP_LOW or self.y_scaled > self.YAW_EDGE_STOP_HIGH):
                ctrl_yaw = 0
                self.edge_evade_move = True
            else:
                self.edge_evade_move = False
        else:
            ctrl_yaw = 0

        if np.abs(yaw_error)<0.01:
            self.PID_yaw.reset()                  # reset integral and error when the yaw error is almost zero to reduce impact on furture movements


        ## prevent cross motion between rotation and translation
        """
        This section of code identifies which controller is currently active and inhibits the other. The robot uses 4 motor, one on each
        wheel to facilitate translational and rotational motion. This ensures that the robots action are predictable and the robot dosent
        try to rotate when moving forward, which would result in unwanted pose.The code checks if either of the controllers are active and 
        inhibits the other one. The inhibition is only removed after a debounce implemented with the help of the SETTLE_DEBOUNCE_CNT
        variable. To release the inhibit the control action from the current controller has to be less than a value for SETTLE_DEBOUNCE_CNTs.
        """
        # Requirement ID SR-023  #Inhibit yaw motion when move motion is active
        if abs(ctrl_move)>1:
            self.motion_settle_cntr=0
            self.inhibit_yaw_motion = True

        if abs(ctrl_move)<= 0.5 and self.inhibit_yaw_motion==True:
            self.motion_settle_cntr+=1
            if self.motion_settle_cntr>=self.SETTLE_DEBOUNCE_CNT:
                self.inhibit_yaw_motion = False

        # Requirement ID SR-022  #Inhibit move motion when yaw motion is active
        if abs(ctrl_yaw)>1:
            self.yaw_settle_cntr=0
            self.inhibit_move_motion = True

        if abs(ctrl_yaw)<=0.5 and self.inhibit_move_motion==True:
            self.yaw_settle_cntr+=1
            if self.yaw_settle_cntr>=self.SETTLE_DEBOUNCE_CNT or self.edge_evade_move == True:  # remove inhibit_move_motion immediately when edge_evade_move flag is true
                self.inhibit_move_motion = False
        return ctrl_move, ctrl_yaw