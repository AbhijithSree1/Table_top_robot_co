import mujoco
import mujoco.viewer
import time
import sys, threading, queue
import logging
from .TR_Controller import Controller
 
## MAIN ##

def main(model, data, KP_MOVE, KI_MOVE, KD_MOVE, KIWD_MOVE,
         KP_YAW, KI_YAW, KD_YAW, KIWD_YAW,
         CTRL_CLIP_MOVE, CTRL_CLIP_YAW,
         YAW_EDGE_STOP_HIGH, YAW_EDGE_STOP_LOW,
         SETTLE_DEBOUNCE_CNT):

    """
    This is the main function that runs the mujoco simulation and the robot controller
    It initialises the model, data and controller parameters
    It also sets up a continuous stdin reader to read commands from the user
    The main loop runs the simulation, reads inputs, updates controller targets and applies control outputs to the robot

    Commands:
        PLACE X,Y,FACING  - Places the robot at the specified X,Y coordinates and facing direction {"NORTH","SOUTH","EAST","WEST"}
        LEFT              - Rotates the robot 90 degrees to the left
        RIGHT             - Rotates the robot 90 degrees to the right
        MOVE              - Moves the robot one unit forward in the direction it is currently facing
        REPORT            - Reports the current X,Y coordinates and facing direction of the robot
    """

    ## MODEL INITIALISATIONS ##
    dt = model.opt.timestep             # simulation timestep
    model.opt.gravity[2] = 0            # prevent the model from being placed until the first PLACE command is issued

    # INITIALISE CONTROLLER
    TR_controller = Controller(model=model,data=data,Kp_move=KP_MOVE,Ki_move=KI_MOVE,Kd_move=KD_MOVE,
                                Kiwd_move=KIWD_MOVE,Kp_yaw=KP_YAW,Ki_yaw=KI_YAW,Kd_yaw=KD_YAW,
                                Kiwd_yaw=KIWD_YAW, move_ctrl_clip=CTRL_CLIP_MOVE,yaw_ctrl_clip=CTRL_CLIP_YAW,
                                yaw_edge_stop_high=YAW_EDGE_STOP_HIGH, yaw_edge_stop_low=YAW_EDGE_STOP_LOW,
                                settle_debounce_count=SETTLE_DEBOUNCE_CNT,dt=dt)
   
    with mujoco.viewer.launch_passive(model, data) as viewer:

        # Requirement ID SR-025
        # continuous stdin reader 
        cmd_q = queue.Queue()               # FIFO queue

        def stdin_reader():
            for line in sys.stdin:          # Reads standard input and stores in queue
                cmd_q.put(line)

        threading.Thread(target=stdin_reader, daemon=True).start()  # Create a new thread that reads stdin

        # set initial camera view
        viewer.cam.lookat[:] = [0, 0, 2]
        viewer.cam.distance = 10
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -40

        while viewer.is_running():
            
            # Read inputs and instantiate controller targets
            TR_controller.controller_read_inputs()

            try:
                # Read command from stdin queue and update internal control targets
                TR_controller.controller_target_calc(cmd_q=cmd_q)
            except queue.Empty:
                pass

            ctrl_move, ctrl_yaw = TR_controller.controller_control_output()

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

    ## LOAD MODEL ##

    # Requirement ID SR-002 SR-003 SR-004
    model = mujoco.MjModel.from_xml_path("model/Table_Top_Robot_Sim.xml") # load the mujoco model
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

    ## run main
    main(model, data, KP_MOVE, KI_MOVE, KD_MOVE, KIWD_MOVE,
         KP_YAW, KI_YAW, KD_YAW, KIWD_YAW,
         CTRL_CLIP_MOVE, CTRL_CLIP_YAW,
         YAW_EDGE_STOP_HIGH, YAW_EDGE_STOP_LOW,
         SETTLE_DEBOUNCE_CNT)
