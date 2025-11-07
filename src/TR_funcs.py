import re
import logging
import numpy as np
import mujoco

## PARAMETERS ##

# compile a regular expression to match PLACE commands
PLACE_RE = re.compile(r'^\s*PLACE\s+(-?\d+)\s*,\s*(-?\d+)\s*,\s*([A-Za-z]+)\s*$')
FACING_OK = {"NORTH","SOUTH","EAST","WEST"}

UITOMAP_SCALING_FACTOR = 0.8
UITOMAP_OFFSET_FACTOR = 2
MAPTOUI_SCALING_FACTOR = 1.25
MAPTOUI_OFFSET_FACTOR = 2

def parse_place(s): # Requirement ID SR-007 SR-008
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

def place_robot(model, data, pos=(0.0, 0.0), facing = "NORTH"): # Requirement ID SR-006
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
    # Requirement ID SR-009
    if facing == "EAST":
        quat = [1, 0, 0, 0]  # No rotation
    # Requirement ID SR-011
    elif facing == "WEST":
        quat = [0, 0, 0, 1]  # 180 degrees around Z
    # Requirement ID SR-012
    elif facing == "SOUTH":
        quat = [0.7071, 0, 0, -0.7071]  # -90 degrees around Z
    # Requirement ID SR-010
    elif facing == "NORTH":
        quat = [0.7071, 0, 0, 0.7071]  # 90 degrees around Z
    data.qpos[3:7] = quat
    # Reset velocities to reinitialise robot
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