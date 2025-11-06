import mujoco
import mujoco.viewer
import numpy as np
import time
import sys, threading, queue, re
import logging

## LOAD MODEL ##

model = mujoco.MjModel.from_xml_path("model/Table_Top_Robot_Sim.xml")
data = mujoco.MjData(model)

## PARAMETERS ##

UITOMAP_SCALING_FACTOR = 0.8
UITOMAP_OFFSET_FACTOR = 2
MAPTOUI_SCALING_FACTOR = 1.25
MAPTOUI_OFFSET_FACTOR = 2
SETTLE_DEBOUNCE_CNT = 10

# compile a regular expression to match PLACE commands
PLACE_RE = re.compile(r'^\s*PLACE\s+(-?\d+)\s*,\s*(-?\d+)\s*,\s*([A-Za-z]+)\s*$')
FACING_OK = {"NORTH","SOUTH","EAST","WEST"}

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


## HELPERS ##

def pid_control(error, prev_error, integral, Kp, Ki, Kd, dt, winduplim=100):
    integral = np.clip(integral+ error * dt,-winduplim,winduplim) # integral anti wind up by clipping
    derivative = (error - prev_error) / dt
    output = Kp * error + Ki * integral + Kd * derivative
    return output, error, integral


def parse_place(s):
    m = PLACE_RE.match(s) # match for the format PLACE X,Y,F where F in {"NORTH","SOUTH","EAST","WEST"}
    if not m:
        return None
    x, y = int(m.group(1)), int(m.group(2))
    facing = m.group(3).upper()
    if facing not in FACING_OK:
        logging.error(f"Invalid facing: {facing}")
        return None
    else:
        return x, y, facing

def quaternion_to_yaw(q_w, q_x, q_y, q_z):
    """
    Convert quaternion to yaw angle
    Args:
        q_w, q_x, q_y, q_z: quaternion components
    Returns:
        yaw angle (radians)
    """
    siny_cosp = 2 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw_rad = np.arctan2(siny_cosp, cosy_cosp)
    return yaw_rad

def place_robot(model, data, pos=(0.0, 0.0), facing = "NORTH"):
    """
    place toyrobot to specified position and orientation
    Args:
        model: mujoco model object
        data: mujoco data object
        pos: [x, y] position coordinates in map frame (after UI2MAP conversion)
        facing: ["NORTH","SOUTH","EAST","WEST"] 
    """
    # position
    data.qpos[0:2] = pos
    data.qpos[2] = 4.25  # always drop the robot from 25cm above the table top. 

    # orientation 
    if facing == "EAST":
        quat = [1, 0, 0, 0]  # No rotation
    elif facing == "WEST":
        quat = [0, 0, 0, 1]  # 180 degrees around Z
    elif facing == "SOUTH":
        quat = [0.7071, 0, 0, -0.7071]  # -90 degrees around Z
    elif facing == "NORTH":
        quat = [0.7071, 0, 0, 0.7071]  # 90 degrees around Z
    data.qpos[3:7] = quat
    # Reset velocities to reinitialize robot
    data.qvel[:] = 0
    # Forward simulation to perform place
    mujoco.mj_forward(model, data)

def robot_move_target (x,y,facing):
    """
    Calculates the x,y target coordinates when the robot gets a "MOVE" command
    Args:
        x: current x coordinate (in UI frame)
        y: current y coordinate (in UI frame)
        facing: ["NORTH","SOUTH","EAST","WEST"] 
    Returns:
        x_target: target for movement in x coordinate (in UI frame)
        y_target: target for movement in y coordinate (in UI frame)
    """
    if facing=="NORTH": # if facing NORTH move foward in the y direction
        y+=1
    elif facing=="EAST": # if facing EAST move right in the x direction
        x+=1
    elif facing=="SOUTH": # if facing SOUTH move backward in the y direction
        y-=1
    elif facing=="WEST": # if facing WEST move left in the x direction
        x-=1
    else: 
        print("facing incorrect, cannot decide where to move!")
    return int(np.round(x)),int(np.round(y)) # round to the nearest and returns as an integer to allow 5 unit steps

def current_facing (yaw):
    """
    Calculates current facing based on yaw angle
    Args:
        yaw: current yaw angle in degrees
    Returns:
        facing: current orientation of the robot in ["NORTH","SOUTH","EAST","WEST"]
    """
    yaw = yaw % 360                     # wrap yaw angle between 0 and 360 degs
    yaw_tol = 45                        # threshold for direction calculation
    if yaw > 360 - yaw_tol or yaw < 0 + yaw_tol:
        facing = "EAST"
    elif np.abs(yaw - 90) < yaw_tol:
        facing = "NORTH"
    elif np.abs(yaw - 180) < yaw_tol:
        facing = "WEST"
    elif np.abs(yaw - 270) < yaw_tol:
        facing = "SOUTH"
    else: 
        facing = "INBETWEEN"
    return facing

def location_transform_UI2MAP(val):
    """
    Transforms the UI frame inputs (x or y coordinate) into the MAP frame
    Args:
        val: current x or y coordinate in UI frame
    Returns:
        val_scaled: transformed x or y coordinate in MAP frame
    """
    val_scaled =  (val - UITOMAP_OFFSET_FACTOR)*UITOMAP_SCALING_FACTOR
    return val_scaled

def location_transform_MAP2UI(val):
    """
    Transforms the MAP frame inputs (x or y coordinate) into the UI frame
    Args:
        val: current x or y coordinate in MAP frame
    Returns:
        val_scaled: transformed x or y coordinate in UI frame
    """
    val_scaled =  val*MAPTOUI_SCALING_FACTOR + MAPTOUI_OFFSET_FACTOR
    return val_scaled

def read_sensor(data):
    """
    Reads the sensor information from the simulation
    Args: 
        data: mujoco data object
    Returns:
        x: x coordinate in MAP frame
        y: y coordinate in MAP frame
        z: z coordinate in MAP frame
        q_w: quaternion w component
        q_x: quaternion x component
        q_y: quaternion y component
        q_z: quaternion z component
    """
    x = data.sensordata[0]
    y = data.sensordata[1]
    z = data.sensordata[2]
    q_w = data.sensordata[3]
    q_x = data.sensordata[4]
    q_y = data.sensordata[5]
    q_z = data.sensordata[6]
    return x,y,z,q_w,q_x,q_y,q_z

def yaw_target_deg(current_yaw_deg, direction):
    """
    Calculate target yaw rotation in radians based on "LEFT" or "RIGHT" commands.
    Targets the controller to rotate the robot anti clockwise if "LEFT" command is provided.
    Targets the controller to rotate the robot clockwise if "RIGHT" command is provided.
    Args:
        current_yaw_deg: current yaw angle in degrees
        direction: target direction command ["LEFT","RIGHT"]
    Returns:
        target_yaw_rad: target yaw rate rounded to the nearest 90 degs converted to radians
    """
    if direction == "LEFT":
        target_yaw_deg = current_yaw_deg + 90   # if "LEFT" rotate anti clockwise
    elif direction == "RIGHT":
        target_yaw_deg = current_yaw_deg - 90   # if "LEFT" rotate clockwise
    else:
        logging.error("Invalid Direction, cannot calculate yaw target, setting current value")
        target_yaw_deg = current_yaw_deg        # if direction command is invalid, keep current orientation as the target
    target_yaw_rad= np.deg2rad(round(target_yaw_deg / 90.0) * 90) # round to the nearest 90 degress (0,90,180,270)
    return target_yaw_rad

def wrap_diff(a, b):
    """
    Calculate the shortest signed angular distance betweens two angles
    Args:
        a: first angle in radians
        b: second angle in radians
    Returns:
        omega_delta: short angular distance between the two angles in radians
    """
    return np.arctan2(np.sin(a - b), np.cos(a - b))
        
## MAIN ##

def main():

    ## INITIALISATIONS ##
    dt = model.opt.timestep             # simulation timestep
    model.opt.gravity[2] = 0            # prevent the model from being placed untill the first PLACE command is issued
    placed_flag = False

    # PID #
    prev_error_y = 0
    prev_error_x = 0
    prev_integral_x = 0
    prev_integral_y = 0
    prev_error_yaw = 0
    prev_integral_yaw = 0

    # Control Targets # 
    x_tgt = 2                           # Initalised to 2 as the model spwans in the centre of the table
    y_tgt = 2
    yaw_tgt = 0                         # radians

    # Control Actions # 
    ctrl_move = 0
    ctrl_yaw = 0

    # Edge fall over prevention # 
    inhibit_yaw_motion = False
    inhibit_move_motion = False
    edge_evade_move = False

    # cross motion prevention # 
    yaw_settle_cntr = 0
    motion_settle_cntr = 0

    # continuous yaw calculation # 
    yaw_wrapped_prev = None              # radians
    yaw_unwrapped     = 0.0              # radians
    yaw_unwrapped_deg = 0.0              # degrees
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # continuous stdin reader 
        cmd_q = queue.Queue()               # FIFO queue

        def stdin_reader():
            for line in sys.stdin:          # Reads standard input and stores in queue
                cmd_q.put(line)

        threading.Thread(target=stdin_reader, daemon=True).start()  # Create a new thread that reads stdin
        while viewer.is_running():

            x,y,_,q_w,q_x,q_y,q_z = read_sensor(data)               # Read current model state
            x_scaled = location_transform_MAP2UI(x)                 # Transform MAP coordinates to UI
            y_scaled = location_transform_MAP2UI(y)
            yaw_wrapped = quaternion_to_yaw(q_w,q_x,q_y,q_z)        # Calculate yaw angle in radians from quaternions, wrapped between 0-2pi

            if yaw_wrapped_prev is None:                            # On first timestep set the unwrapped yaw angles to the wrapped angles
                yaw_wrapped_prev = yaw_wrapped
                yaw_unwrapped     = yaw_wrapped
            else:                                                   # On future steps calculate the continuous yaw angle, avoids discontinuities, smooth control
                dy = wrap_diff(yaw_wrapped, yaw_wrapped_prev)       # calculate delta between previous wrapped angle and current angle
                yaw_unwrapped += dy                                 # continuously add to unwrapped yaw to obtain continuous yaw angle 
                yaw_wrapped_prev = yaw_wrapped
                yaw_unwrapped_deg = np.rad2deg(yaw_unwrapped)
            
            facing = current_facing(yaw_unwrapped_deg)               # obtain the current facing of the bot

            try:
                while True:
                    line = cmd_q.get_nowait()
                    s = line.strip()
                    if not s:
                        continue

                    # If user types QUIT exit the simulation
                    if s.upper() == "QUIT":                             
                        raise SystemExit
                    
                    # check if placed before accepting moving or rotating commands

                    if s.upper() in ["LEFT","RIGHT","MOVE","REPORT"] and placed_flag == False:
                        logging.info("Please PLACE the robot first using the PLACE X,Y,FACING command")

                    if s.upper() == "LEFT" and placed_flag == True and inhibit_yaw_motion == False:
                        yaw_tgt = yaw_target_deg(yaw_unwrapped_deg,"LEFT")  # calculate yaw target for LEFT movement
                        logging.info("Moving left")

                    elif s.upper() == "RIGHT" and placed_flag == True and inhibit_yaw_motion == False:
                        yaw_tgt = yaw_target_deg(yaw_unwrapped_deg,"RIGHT") # calculate yaw target for RIGHT movement
                        logging.info("Moving right")

                    if s.upper() in ["LEFT","RIGHT"] and inhibit_yaw_motion == True:
                        logging.warning("Cannot accept a rotation motion, another motion in progress")

                    if s.upper() == "MOVE" and placed_flag == True and inhibit_move_motion == False:
                        x_tgt,y_tgt = robot_move_target(x_scaled,y_scaled,facing)   # calculate target x and y movement
                        if x_tgt < 0 or x_tgt > 4 or y_tgt < 0 or y_tgt > 4:
                            logging.warning("Cannot move forward. Robot will fall and get hurt! Please rotate. ")
                        else:
                            logging.info(f"moving to coordinates {x_tgt},{y_tgt}")
                        x_tgt = np.clip(x_tgt,0,4)      # clip x and y target to ensure they are bounded in the table space
                        y_tgt = np.clip(y_tgt,0,4)
                    elif s.upper() == "MOVE" and inhibit_move_motion == True:
                        logging.warning("Cannot accept a move motion, another motion in progress")

                    if s.upper() == "REPORT" and placed_flag == True:   # report current position when "REPORT" is entered
                        print(f"robot at x={np.round(x_scaled)} y={np.round(y_scaled)} facing {facing}")
                    
                    if s.upper() not in ["LEFT","RIGHT","MOVE","REPORT"] and not PLACE_RE.match(s):
                        print("Invalid command, Valid Commands are:")
                        print("     PLACE X,Y,F   X,Y: bot initial coordinates, integers from 0-4 F: Target facing in [NORTH,SOUTH,EAST,WEST]")
                        print("     MOVE    move robot forward in current facing direction")
                        print("     LEFT    rotate robot anti clockwise")
                        print("     RIGHT   rotate robot clockwise")
                        print("     REPORT  report the robots current X,Y and Facing ")

                    if PLACE_RE.match(s):
                        place = parse_place(s)    # check for valid PLACE command
                        if place is None:
                            logging.warning("Invalid PLACE. Use PLACE X,Y,F with F in {NORTH,SOUTH,EAST,WEST}.")
                            continue
                        x_place, y_place, facing = place
                        if x_place < 0 or x_place > 4 or y_place < 0 or y_place > 4:
                            logging.warning("Place coordinates outside table, Robot will fall and get hurt! Allowed coordinates from 0 to 4")
                            continue
                        else:
                            logging.info(f"PLACING with x={x_place} y={y_place} facing={facing}")
                            x_place = location_transform_UI2MAP(np.clip(x_place,0,4))   # transform UI frame X coordinate to MAP frame
                            y_place = location_transform_UI2MAP(np.clip(y_place,0,4))   # transform UI frame Y coordinate to MAP frame
                            place_robot(model,data,pos=[x_place,y_place],facing=facing)       # place the robot
                            placed_flag = True                                          # set placed flag as true
                            model.opt.gravity[2] = -9.81                                # enable gravity to allow the robot to fall and continue further movements
                            x,y,_,q_w,q_x,q_y,q_z = read_sensor(data)                   # read updates states after PLACING
                            x_scaled = location_transform_MAP2UI(x)                     # transform MAP frame X coordinate to UI frame
                            y_scaled = location_transform_MAP2UI(y)                     # transform MAP frame Y coordinate to UI frame
                            x_tgt = x_scaled                                            # reset x_target as the new PLACED x coordinate
                            y_tgt = y_scaled                                            # reset y_target as the new PLACED y coordinate
                            yaw_wrapped  = quaternion_to_yaw(q_w,q_x,q_y,q_z)           # calculate current rotation after PLACING
                            yaw_wrapped_prev = yaw_wrapped                              # reset yaw continuous measurements
                            yaw_unwrapped     = yaw_wrapped
                            yaw_unwrapped_deg = np.rad2deg(yaw_unwrapped)
                            yaw_tgt = yaw_wrapped                                       # reset yaw target to the new PLACED yaw angle
                            facing = current_facing(yaw_unwrapped_deg)                   # determine the current facing
                            logging.info(f"robot placed at x={x_scaled:.2f} y={y_scaled:.2f} facing {facing}")
            except queue.Empty:
                pass

            # Calculate yaw error
            yaw_error = wrap_diff(yaw_tgt, yaw_unwrapped)

            # Calculate movement error
            y_error = y_tgt  - y_scaled
            x_error = x_tgt  - x_scaled

            ## Control for longitudinal motion ##
            """
            This section of code controls the robots motion in longitudinal direction depending on facing of the robot. The controller
            outputs the total torque demand acorss the entire chassis to read the x and y target. This is further arbitrated below to
            individual torque demands. Control is only executed when the robot is not yawing significantly (identified using the inhibit_move_motion flag).
            Control is clipped between -5Nm and 5Nm to avoid excessive torque. Higher torque can cause the robot to wheeliee!
            Control direction is inverted (by inverting the error) when the robot is facing the SOUTH or WEST direction to allow the robot to move forward
            and reduce target errors. Integral windup is used to avoid unwanted integral action. 
            """

            if facing == "NORTH" and inhibit_move_motion ==False:  # Facing North
                ctrl_move,prev_error_y,prev_integral_y = pid_control(y_error,prev_error_y,prev_integral_y,KP_MOVE,KI_MOVE,KD_MOVE,dt,winduplim=KIWD_MOVE)
            elif facing == "SOUTH" and inhibit_move_motion ==False:  # Facing South
                ctrl_move,prev_error_y,prev_integral_y = pid_control(-y_error,prev_error_y,prev_integral_y,KP_MOVE,KI_MOVE,KD_MOVE,dt,winduplim=KIWD_MOVE) # Reverse control direction
            elif facing == "EAST" and inhibit_move_motion ==False:  # Facing East
                ctrl_move,prev_error_x,prev_integral_x = pid_control(x_error,prev_error_x,prev_integral_x,KP_MOVE,KI_MOVE,KD_MOVE,dt,winduplim=KIWD_MOVE)
            elif facing == "WEST" and inhibit_move_motion ==False:  # Facing West
                ctrl_move,prev_error_x,prev_integral_x = pid_control(-x_error,prev_error_x,prev_integral_x,KP_MOVE,KI_MOVE,KD_MOVE,dt,winduplim=KIWD_MOVE) # Reverse control direction             
            else:
                ctrl_move = 0
            
            ctrl_move = np.clip(ctrl_move,-CTRL_CLIP_MOVE,CTRL_CLIP_MOVE)  
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
                
            if inhibit_yaw_motion==False:
                
                ctrl_yaw,prev_error_yaw,prev_integral_yaw = pid_control(yaw_error,prev_error_yaw,prev_integral_yaw,Kp=KP_YAW,Ki=KI_YAW,Kd=KD_YAW,dt=dt,winduplim=KIWD_YAW)
                ctrl_yaw = np.clip(ctrl_yaw,-CTRL_CLIP_YAW,CTRL_CLIP_YAW)  # clip control

                if (x_scaled < YAW_EDGE_STOP_LOW or x_scaled > YAW_EDGE_STOP_HIGH or 
                    y_scaled < YAW_EDGE_STOP_LOW or y_scaled > YAW_EDGE_STOP_HIGH):
                    ctrl_yaw = 0
                    edge_evade_move = True
                else:
                    edge_evade_move = False
            else:
                ctrl_yaw = 0

            if np.abs(yaw_error)<0.01:
                prev_error_yaw = 0
                prev_integral_yaw = 0                   # reset integral and error when the yaw error is almost zero to reduce impact on furture movements


            ## prevent cross motion between rotation and translation
            """
            This section of code identifies which controller is currently active and inhibits the other. The robot uses 4 motor, one on each
            wheel to facilitate translational and rotational motion. This ensures that the robots action are predictable and the robot dosent
            try to rotate when moving forward, which would result in unwanted pose.The code checks if either of the controllers are active and 
            inhibits the other one. The inhibition is only removed after a debounce implemented with the help of the SETTLE_DEBOUNCE_CNT
            variable. To release the inhibit the control action from the current controller has to be less than a value for SETTLE_DEBOUNCE_CNTs.
            """
            if abs(ctrl_move)>1:
                motion_settle_cntr=0
                inhibit_yaw_motion = True
            
            if abs(ctrl_move)<= 0.5 and inhibit_yaw_motion==True:
                motion_settle_cntr+=1
                if motion_settle_cntr>SETTLE_DEBOUNCE_CNT:
                    inhibit_yaw_motion = False


            if abs(ctrl_yaw)>1:
                yaw_settle_cntr=0
                inhibit_move_motion = True
            
            if abs(ctrl_yaw)<=0.5 and inhibit_move_motion==True:
                yaw_settle_cntr+=1
                if yaw_settle_cntr>SETTLE_DEBOUNCE_CNT or edge_evade_move == True:  # remove inhibit_move_motion immediately when edge_evade_move flag is true
                    inhibit_move_motion = False

            ## MOTOR FINAL TORQUE DEMANDS ##
            """
            The final torque demand arbitration is perfomed by quartering the total torque demand from the translational controller
            and applying a positive delta from the yaw controller on the right side and a negative delta from the yaw controller on the 
            left side.
            """
            
            data.ctrl[0] = ctrl_move*0.25 - ctrl_yaw/2      #FL demand
            data.ctrl[1] = ctrl_move*0.25 + ctrl_yaw/2      #FR demand
            data.ctrl[2] = ctrl_move*0.25 - ctrl_yaw/2      #RL demand
            data.ctrl[3] = ctrl_move*0.25 + ctrl_yaw/2      #RR demand

            mujoco.mj_step(model, data)                     # move simulation forward
            viewer.sync()                                   
            time.sleep(dt)                                  # sleep to convert the simulation into realtime(ish)
    

if __name__ == "__main__":
    ## logging config ##
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    ## run main
    main()
