import mujoco
import mujoco.viewer
import numpy as np
import time
import sys, threading, queue, re
import logging
from TR_funcs import parse_place, quaternion_to_yaw, current_facing, location_transform_MAP2UI
from TR_funcs import place_robot, robot_move_target, location_transform_UI2MAP, read_sensor
from TR_funcs import yaw_target_deg, wrap_diff
from TR_PID import PID

## LOAD MODEL ##

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
        
## MAIN ##

def main():

    ## INITIALISATIONS ##
    dt = model.opt.timestep             # simulation timestep
    model.opt.gravity[2] = 0            # prevent the model from being placed untill the first PLACE command is issued
    placed_flag = False

    # PID #
    PID_move_north = PID(KP_MOVE,KI_MOVE,KD_MOVE,dt,winduplim=KIWD_MOVE)
    PID_move_south = PID(KP_MOVE,KI_MOVE,KD_MOVE,dt,winduplim=KIWD_MOVE)
    PID_move_east = PID(KP_MOVE,KI_MOVE,KD_MOVE,dt,winduplim=KIWD_MOVE)
    PID_move_west = PID(KP_MOVE,KI_MOVE,KD_MOVE,dt,winduplim=KIWD_MOVE)
    PID_yaw = PID(KP_YAW,KI_YAW,KD_YAW,dt,winduplim=KIWD_YAW)

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
                    
                    if s.upper() not in ["LEFT","RIGHT","MOVE","REPORT"] and parse_place(s) is None:
                        print("Invalid command, Valid Commands are:")
                        print("     PLACE X,Y,F   X,Y: bot initial coordinates, integers from 0-4 F: Target facing in [NORTH,SOUTH,EAST,WEST]")
                        print("     MOVE    move robot forward in current facing direction")
                        print("     LEFT    rotate robot anti clockwise")
                        print("     RIGHT   rotate robot clockwise")
                        print("     REPORT  report the robots current X,Y and Facing ")

                    if parse_place(s) is not None:
                        place = parse_place(s)    # check for valid PLACE command
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
                ctrl_move = PID_move_north.step(y_error)
            elif facing == "SOUTH" and inhibit_move_motion ==False:  # Facing South               
                ctrl_move = PID_move_south.step(-y_error) # Reverse control direction
            elif facing == "EAST" and inhibit_move_motion ==False:  # Facing East               
                ctrl_move = PID_move_east.step(x_error)
            elif facing == "WEST" and inhibit_move_motion ==False:  # Facing West               
                ctrl_move = PID_move_west.step(-x_error) # Reverse control direction       
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
                
                ctrl_yaw = PID_yaw.step(yaw_error)
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
                PID_yaw.reset()                  # reset integral and error when the yaw error is almost zero to reduce impact on furture movements


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
