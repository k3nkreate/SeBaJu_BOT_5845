"""
Author: Kennedy Chukwuma
--Design step and improvements
1. Designed specifical for balancing with the hip and knee joints fixed in xml
2. Uncommented the knee and hip joint in the XML, using actuator-position to lock joint with Kp and Kd in XML
3. Also added the actuator-motor for the hip and knee, while it appears that the actuator-position is easier to
control for a jump step, the actuator-motor has been added to the XML (commented out); to be tested in future.
    Mode 1: LEG_ACTUATOR_MODE = "motor_pd"
        XML uses hip/knee <motor> actuators
        Python computes hip/knee PD commands

    Mode 2: LEG_ACTUATOR_MODE = "position_servo"
        XML uses hip/knee <position> actuators
        Python sends desired hip/knee angles directly
"""

import mujoco  # Import MuJoCo so Python can load the XML model and step the physics.
import mujoco.viewer  # Import the passive viewer so we can see the simulation while the controller runs.
import math  # Import math for sin/cos/atan2/degrees functions.
import time  # Import time so SLOW_FACTOR can control the visible simulation speed.
import os  # Import os so the XML can be found in the same folder as this controller.
import numpy as np  # Import NumPy for clipping and small vector calculations.


#==========================================================
# XML and SIM parameters
# ============================================================
XML_FILE = "SeBaJu_BOT(II).xml"  # XML file to load; keep it in the same folder as this controller.
SLOW_FACTOR = 0.50  # 0.25 = quarter speed, 0.50 = half speed, 1.0 = real time, 2.0 = faster than real time.
BASE_HEIGHT = 0.2439  # Camera look-at height, chosen near the horizontal torso height.
WHEEL_RADIUS = 0.06  # Wheel radius from the XML cylinder size="0.10 0.045".

#==========================================================
GYRO_FILTER = 0.70  # Low-pass filter for pitch rate; higher value = smoother but more delay.
VEL_FILTER = 0.70  # Low-pass filter for wheel velocity; higher value = smoother but more delay.
ACCEL_BLEND = 0.0  # Small blend from accelerometer tilt; keep low because acceleration is noisy during balancing.

# ============================================================
# CONTROLLER TUNING CONSTANTS
# ============================================================
PITCH_OFFSET =  math.radians(4.73) #0.0
MAX_WHEEL = 0.35

K_THETA   = 1.3 #1.3    # forward lean correction
K_THETA_D = 0.17 #0.23  # lean-rate damping
K_X_D     = 0.16 #0.07  # velocity damping #4.0    # forward lean correction
K_X       = 0.8 #0.85 #0.5 #0.035    # position-hold stiffness

COM_RATE_FILTER = 0.85
VEL_FILTER = 0.70
WHEEL_MOTOR_SIGN = -1

# =========================================================
# LEG ACTUATOR EXPERIMENT MODE SELECTOR
# Choose one:
# "motor_pd"        -> Python calculates hip/knee PD lock
# "position_servo"  -> MuJoCo position actuators lock/move hip/knee (Kv AND Kv)
# For each leg actuation mode selected always remember to make the appropriate actuation change in the XML
# =======================================================
# LEG_ACTUATOR_MODE = "motor_pd"
LEG_ACTUATOR_MODE = "position_servo"

# =========================================================
# =========================================================
# EXPERIMENT SELECTOR
# =========================================================
EXPERIMENT_MODE = "SLANT_RIGHT"  # Choose one of the following experiment modes.
# Options:
# "BASE_JUMP"
# "INDIVIDUAL_LEG_TEST"
# "SLANT_LEFT"
# "SLANT_RIGHT"
# "ROLL_FORWARD"
# "ROLL_BACKWARD"
# "STEP_LEFT"
# "STEP_RIGHT"
# "JUMP_FORWARD"
# "JUMP_BACKWARD"
# =========================================================

#===========================================================================================
# JUMP TEST
JUMP_ENABLE = JUMP_ENABLE = EXPERIMENT_MODE in ["BASE_JUMP", "JUMP_FORWARD", "JUMP_BACKWARD"]
JUMP_START_TIME = 5.0
JUMP_ONCE = True

# JUMP STATE MACHINE STATES
STATE_BALANCE = "BALANCE"
STATE_PRELOAD = "PRELOAD"
STATE_THRUST = "THRUST"
STATE_FLIGHT = "FLIGHT"
STATE_LANDING = "LANDING"
STATE_RECOVERY = "RECOVERY"
STATE_SETTLE = "SETTLE"

# =========================================================
# HIGH-LEVEL MOTION MODES
# =========================================================
MODE_NONE = "NONE"
MODE_INDIVIDUAL_LEG = "INDIVIDUAL_LEG"
MODE_SLANT = "SLANT"
MODE_ROLL_MOVE = "ROLL_MOVE"
MODE_STEP = "STEP"
MODE_JUMP_X = "JUMP_X"

# =========================================================
# STEP SUB-STATES
# =========================================================
STEP_IDLE = "STEP_IDLE"
STEP_PREP = "STEP_PREP"
STEP_UNLOAD = "STEP_UNLOAD"
STEP_SWING = "STEP_SWING"
STEP_PLACE = "STEP_PLACE"
STEP_TRANSFER = "STEP_TRANSFER"

# JUMP TIMING AND DETECTION SETTINGS
T_PRELOAD = 0.80                   # Slow crouch time, seconds.
# T_THRUST should ot be smaller than dt, to achieve a controlled jump with mujoco timestep of 0.002s, T_THRUST = 0.05 is a good value for a controlled jump
T_THRUST = 0.02 #0.035 #0.04 #0.14                    # Fast extension time, seconds.
T_THRUST_MAX = 0.16 #0.12                # Safety timeout if wheels never leave ground.
T_LANDING = 0.80 #0.35                   # Time to absorb landing.
T_RECOVERY = 2.00 #1.00                  # Time to return from landing pose to standing pose.

# CONTACT FORCE HYSTERESIS SETTINGS
G = 9.81

# These are ratios. The actual force thresholds will be calculated
# inside run(), after MuJoCo has loaded the model.
F_CONTACT_OFF_RATIO = 0.03
F_CONTACT_ON_RATIO  = 0.15

F_CONTACT_OFF_MIN = 1.5
F_CONTACT_ON_MIN  = 6.0

CONTACT_LOSS_DEBOUNCE = 0.004     # 2 steps at dt=0.002
CONTACT_GAIN_DEBOUNCE = 0.012     # 6 steps at dt=0.002

TAKEOFF_VZ_MIN = 0.08           # m/s

COMZ_VEL_FILTER = 0.40            # Less filtering so take-off velocity is not hidden.

CONTACT_FORCE_MIN = 1.0           # Newton; tune later

# =========================================================
# PRINT INTERVALS
# =========================================================
PRINT_DT_BALANCE  = 0.50
PRINT_DT_PRELOAD  = 0.40 #0.10
PRINT_DT_THRUST   = 0.01
PRINT_DT_FLIGHT   = 0.04
PRINT_DT_LANDING  = 0.08
PRINT_DT_RECOVERY = 0.20
PRINT_DT_SETTLE   = 0.25

# =========================================================
# WHEEL CLEARANCE SETTINGS
# =========================================================
# Absolute clearance: true geometric clearance above ground.
WHEEL_CLEARANCE_TAKEOFF_ABS = 0.0005   # 0.5 mm, for detecting micro take-off

# Relative clearance: lift compared with normal loaded wheel position.
WHEEL_CLEARANCE_TAKEOFF_REL = 0.0030   # 3 mm relative lift

# Target for a visible jump, not for state switching.
VISIBLE_WHEEL_CLEARANCE_TARGET = 0.010 # 10 mm visible wheel clearance

# ------------------------------------------------------------------------
# First safe joint offsets. If crouch/extension moves in the wrong physical direction, flip these signs.
HIP_PRELOAD_DELTA  = +0.12 #+0.20         # rad, crouch hip target relative to standing pose.
KNEE_PRELOAD_DELTA = -0.22 #-0.35         # rad, crouch knee target relative to standing pose.

HIP_THRUST_DELTA   = -0.27 #-0.26 #-0.24            # rad, extension hip target relative to standing pose.
KNEE_THRUST_DELTA  = +0.44 #+0.42 #+0.40           # rad, extension knee target relative to standing pose.

HIP_LAND_DELTA     = +0.08 #+0.10         # rad, landing absorption hip target relative to standing pose.
KNEE_LAND_DELTA    = -0.20 #-0.25         # rad, landing absorption knee target relative to standing pose.

# Safety limits matching your XML ctrlrange for position actuators.
HIP_CTRL_MIN, HIP_CTRL_MAX = -0.60, 0.60
KNEE_CTRL_MIN, KNEE_CTRL_MAX = -0.80, 0.80


# =========================================================
# SETTLING TIME MEASUREMENT AFTER LANDING
SETTLE_LEAN_ERR = math.radians(2.0) # This is a tolerance around the non-zero equilibrium angle
SETTLE_RATE = math.radians(8.0) # CoM lean-rate limit
SETTLE_VEL = 0.04 # Forward/backward wheel velocity limit.
SETTLE_WHEEL = 0.06 # Wheel command should be small if the controller is no longer fighting
SETTLE_X_ERR = 0.015 # Position error relative to landing location
SETTLE_HOLD = 0.30 # Robot must satisfy all conditions continuously for this long
MAX_SETTLE_WAIT = 12.0 # Safety timeout so the simulation does not wait forever

# =========================================================
# LANDING POSITION REFERENCE
# False means: balance where the robot lands.
# True means: after landing, try to return to the original start position.
RETURN_TO_START_AFTER_JUMP = False

# =========================================================
# ROLL / SIDE-SLANT CONTROL
# =========================================================
ROLL_REF_MAX = math.radians(6.0)
ROLL_SAFETY_MAX = math.radians(10.0)

K_ROLL_P = 0.70
K_ROLL_D = 0.08

HIP_ROLL_TO_DIFF = 0.14
KNEE_ROLL_TO_DIFF = 0.20
LEG_DIFF_MAX = 0.22

# =========================================================
# POINT-TO-POINT ROLLING
# =========================================================
X_CMD_RATE = 0.10 #0.20          # m/s reference movement speed
X_GOAL_TOL = 0.015         # m
X_I_MAX = 0.08
K_X_I = 0.10

# =========================================================
# STEPPING TEST
# =========================================================
STEP_PREP_TIME = 0.35
STEP_UNLOAD_TIME = 0.35
STEP_SWING_TIME = 0.45
STEP_PLACE_TIME = 0.30
STEP_TRANSFER_TIME = 0.40

STEP_ROLL_BIAS = math.radians(4.0)

STEP_HIP_LIFT = +0.10
STEP_KNEE_LIFT = -0.20
STEP_HIP_SWING = -0.10
STEP_KNEE_SWING = +0.08

# =========================================================
# TRANSLATIONAL JUMP
# =========================================================
JUMP_X_PITCH_BIAS = math.radians(1.8)
JUMP_X_WHEEL_FF = 0.08
JUMP_X_MAX = 0.18

# =========================================================
# COMMAND SLEW LIMITS
# =========================================================
LEG_DQ_MAX = 0.020
WHEEL_DU_MAX = 0.020

# =============================================================
# LEG LOCK GAINS SPECIFICALLY FOR HIP AND KNEE "ACTUATOR-MOTOR" [OPTIONAL]
# =============================================================
HIP_KP = 75.0
HIP_KD = 7.5

KNEE_KP = 85.0
KNEE_KD = 8.5

HIP_GEAR = 30.0
KNEE_GEAR = 30.0

MAX_HIP = 1.0
MAX_KNEE = 1.0

HIP_MOTOR_SIGN = 1.0
KNEE_MOTOR_SIGN = 1.0
# ===============================================================

#=====================================================================================================
# HELPER FUNCTIONS
# =====================================================================================================
def clamp(value, lo, hi):  # Define a helper that limits a number between lo and hi.
    return float(np.clip(value, lo, hi))  # Clip the value and convert it back to a normal Python float.


def sensor_data(model, data, name):  # Define a helper that reads a named MuJoCo sensor.
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)  # Find the numeric sensor id from its name.
    if sid < 0:  # Check whether the sensor exists in the XML.
        raise RuntimeError(f"Sensor not found: {name}")  # Stop with a clear error if the XML is missing the sensor.
    adr = model.sensor_adr[sid]  # Get the start address of this sensor inside data.sensordata.
    dim = model.sensor_dim[sid]  # Get how many numbers this sensor returns.
    return np.array(data.sensordata[adr:adr + dim], dtype=float)  # Return the sensor values as a NumPy array.


def quat_to_pitch(qw, qx, qy, qz):  # Convert a MuJoCo quaternion into pitch angle about the robot's side-to-side Y axis.
    sinp = 2.0 * (qw * qy - qz * qx)  # Compute the sine of pitch from the quaternion components.
    return math.asin(clamp(sinp, -1.0, 1.0))  # Clamp numerical error and return pitch in radians.


def joint_maps(model, joint_names):  # Build dictionaries that let us read joint position and velocity by joint name.
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names}  # Convert each joint name to its MuJoCo id.
    qadr = {n: model.jnt_qposadr[jid[n]] for n in joint_names}  # Store where each joint position lives in data.qpos.
    vadr = {n: model.jnt_dofadr[jid[n]] for n in joint_names}  # Store where each joint velocity lives in data.qvel.
    return jid, qadr, vadr  # Return all three maps to the main controller.

# This motor_lock function is used alongside hip and knee lock gains when the an actuator-motor is used to actuate in the XLM
def pd_motor_lock(data, qadr, vadr, joint_name, q_ref, kp, kd, gear, max_ctrl, motor_sign=1.0):
    q = float(data.qpos[qadr[joint_name]])
    qd = float(data.qvel[vadr[joint_name]])

    tau = kp * (q_ref - q) - kd * qd
    ctrl = tau / gear

    return motor_sign * clamp(ctrl, -max_ctrl, max_ctrl)

# This get the actuator ID for any selected respective LEG_ACTUATION_MODE
def get_actuator_id(model, name, required=True):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    if required and aid < 0:
        raise RuntimeError(f"Actuator not found: {name}")

    return aid

# Leg command dependant on the leg actuation mode selector
def command_legs(
    data, qadr, vadr, mode, act_lhip, act_rhip, act_lknee, act_rknee, hip_target_L, hip_target_R, knee_target_L, knee_target_R,
):
    """
    Sends hip/knee targets using either:
    1. Python motor-PD control, or
    2. MuJoCo position-servo control.

    In motor_pd mode:
        hip_target and knee_target are reference angles.
        Python converts angle error into motor command.

    In position_servo mode:
        hip_target and knee_target are sent directly to the MuJoCo
        position actuators as desired joint angles in radians.
    """

    if mode == "motor_pd":
        data.ctrl[act_lhip] = pd_motor_lock(data, qadr, vadr, "left_hip", hip_target_L, HIP_KP, HIP_KD, HIP_GEAR, MAX_HIP, HIP_MOTOR_SIGN)
        data.ctrl[act_rhip] = pd_motor_lock(data, qadr, vadr, "right_hip", hip_target_R, HIP_KP, HIP_KD, HIP_GEAR, MAX_HIP, HIP_MOTOR_SIGN)
        data.ctrl[act_lknee] = pd_motor_lock(data, qadr, vadr, "left_knee", knee_target_L, KNEE_KP, KNEE_KD, KNEE_GEAR, MAX_KNEE, KNEE_MOTOR_SIGN)
        data.ctrl[act_rknee] = pd_motor_lock(data, qadr, vadr, "right_knee", knee_target_R, KNEE_KP, KNEE_KD, KNEE_GEAR, MAX_KNEE, KNEE_MOTOR_SIGN)

    elif mode == "position_servo":
        # In position-servo mode, ctrl is the desired joint angle in radians.
        data.ctrl[act_lhip] = hip_target_L
        data.ctrl[act_rhip] = hip_target_R
        data.ctrl[act_lknee] = knee_target_L
        data.ctrl[act_rknee] = knee_target_R

    else:
        raise RuntimeError("Invalid leg actuator mode.")

def smoothstep01(t, T):
    """
    Smooth 0-to-1 transition.
    Purpose: avoid instant jumps in hip/knee targets.
    sigma(0)=0, sigma(1)=1, and slope is zero at both ends.
    """
    if T <= 0.0:
        return 1.0
    tau = clamp(t / T, 0.0, 1.0)
    return 3.0 * tau**2 - 2.0 * tau**3

def fast_thrust01(t, T):
    """
    Faster thrust transition than smoothstep.
    It gives more leg extension early in the thrust phase,
    which improves upward impulse before the wheels unload.
    """
    if T <= 0.0:
        return 1.0

    tau = clamp(t / T, 0.0, 1.0)

    # Fast ease-out curve:
    # rises quickly at the start, then slows near the end.
    return 1.0 - (1.0 - tau)**3


def blend(a, b, alpha):
    """
    Interpolate between two joint targets.
    alpha=0 returns a, alpha=1 returns b.
    """
    return a + (b - a) * alpha


def wheel_ground_contact_info(model, data, ground_gid, left_wheel_gid, right_wheel_gid):
    """
    Returns:
        left_pair, right_pair, left_fn, right_fn

    left_pair/right_pair:
        True if MuJoCo reports a contact pair.

    left_fn/right_fn:
        Normal contact force at each wheel.
    """
    left_pair = False
    right_pair = False

    left_fn = 0.0
    right_fn = 0.0

    contact_force = np.zeros(6)

    for i in range(data.ncon):
        con = data.contact[i]
        pair = {con.geom1, con.geom2}

        mujoco.mj_contactForce(model, data, i, contact_force)
        normal_force = max(0.0, float(contact_force[0]))

        if ground_gid in pair and left_wheel_gid in pair:
            left_pair = True
            left_fn += normal_force

        if ground_gid in pair and right_wheel_gid in pair:
            right_pair = True
            right_fn += normal_force

    return left_pair, right_pair, left_fn, right_fn


def balance_gains_for_state(state):
    """
    Stage-aware balance gains.
    During thrust, position hold is weakened so the wheels do not fight the leg extension.
    During landing, damping is increased to help recovery.
    """
    if state == STATE_BALANCE:
        return K_THETA, K_THETA_D, K_X_D, K_X, MAX_WHEEL
    if state == STATE_PRELOAD:
        return K_THETA, 0.22, 0.20, 0.50 * K_X, MAX_WHEEL
    if state == STATE_THRUST:
        return K_THETA, 0.25, 0.18, 0.10 * K_X, MAX_WHEEL
    if state == STATE_FLIGHT:
        return K_THETA, 0.10, 0.00, 0.00, 0.10
    elif state == STATE_LANDING:
        # During landing, prioritise pitch damping.
        # Do not aggressively recover x-position immediately after impact.
        return K_THETA, 0.18, 0.10, 0.00, 0.22
    elif state == STATE_RECOVERY:
        # Gradually recover position after landing.
        return K_THETA, 0.20, 0.12, 0.20 * K_X, 0.28
    elif state == STATE_SETTLE:
        # Settling: restore more normal balance but avoid aggressive position pull.
        return K_THETA, K_THETA_D, 0.22, 0.15 * K_X, MAX_WHEEL
    return K_THETA, K_THETA_D, K_X_D, K_X, MAX_WHEEL


def quat_to_roll_pitch(qw, qx, qy, qz):
    """
    Convert quaternion to roll and pitch.
    Roll is needed for side-slant control.
    """
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.asin(clamp(sinp, -1.0, 1.0))

    return roll, pitch


def slew(target, previous, step_max):
    """
    Limit how fast a command can change.
    This prevents sudden leg or wheel commands.
    """
    change = clamp(target - previous, -step_max, +step_max)
    return previous + change


def leg_common_diff_to_targets(hip_common, knee_common, hip_diff, knee_diff):
    """
    Convert common + differential leg commands into left/right targets.

    Common command affects both legs equally.
    Differential command makes the two legs move differently.
    """
    hip_L = hip_common + hip_diff
    hip_R = hip_common - hip_diff

    knee_L = knee_common - knee_diff
    knee_R = knee_common + knee_diff

    return hip_L, hip_R, knee_L, knee_R


def swing_profile01(t, T):
    """
    Smooth swing profile for one-leg stepping.
    prog moves from 0 to 1.
    lift goes 0 -> 1 -> 0.
    """
    s = smoothstep01(t, T)
    prog = 0.5 - 0.5 * math.cos(math.pi * s)
    lift = math.sin(math.pi * s)
    return prog, lift



#============================================================================================================================================
# MAIN FUNCTION
#============================================================================================================================================
def run():  # Main function that loads the robot and runs the feedback loop.
    xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), XML_FILE)  # Build full path to the XML file beside this script.
    if not os.path.exists(xml_path):  # Check that the XML file is really there.
        print(f"ERROR: XML not found: {xml_path}")  # Print a helpful path if the XML is missing.
        raise SystemExit(1)  # Stop the program because MuJoCo cannot run without the XML.

    print(f"\nLoading: {xml_path}")  # Tell the user which XML is being loaded.
    model = mujoco.MjModel.from_xml_path(xml_path)  # Load the MuJoCo model from the XML file.
    data = mujoco.MjData(model)  # Create the live simulation data structure for this model.

    # =========================================================
    # Robot mass and contact thresholds
    # =========================================================
    robot_mass = float(np.sum(model.body_mass))
    robot_weight = robot_mass * G

    F_CONTACT_OFF = max(F_CONTACT_OFF_MIN, F_CONTACT_OFF_RATIO * robot_weight)
    F_CONTACT_ON  = max(F_CONTACT_ON_MIN,  F_CONTACT_ON_RATIO  * robot_weight)

    print(f"  Robot mass   = {robot_mass:.3f} kg")
    print(f"  Robot weight = {robot_weight:.3f} N")
    print(f"  Contact OFF threshold = {F_CONTACT_OFF:.3f} N")
    print(f"  Contact ON threshold  = {F_CONTACT_ON:.3f} N")
    print(f"  EXPERIMENT_MODE = {EXPERIMENT_MODE}")

    #This block needs to be changed
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")  # Look for the standing keyframe in the XML.
    if key_id >= 0:  # Check if the keyframe exists.
        mujoco.mj_resetDataKeyframe(model, data, key_id)  # Reset the robot to the standing keyframe.
        print("Keyframe 'stand' applied.")  # Confirm that the standing pose was applied.
    else:  # If no keyframe exists, continue from the XML default pose.
        mujoco.mj_forward(model, data)  # Compute all derived positions and sensor values from the current pose.

    # This block needs to be delected since position is not determined from the torso free joint position again but from the wheels
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")  # Find the freejoint id for the floating torso.
    root_qadr = model.jnt_qposadr[root_id]  # Store where torso x/y/z position begins inside data.qpos.

    # ============================================================================================
    # SeBaJu joint names
    jnames = ["left_hip","right_hip","left_knee","right_knee","left_wheel_joint", "right_wheel_joint"]
    jid, qadr, vadr = joint_maps(model, jnames)

    # ============================================================================================
    # Actuator IDs, factoring the two leg actuation mode selector
   
    # Wheel actuators are always required
    # -------------------------------------------------
    act_lwheel = get_actuator_id(model, "left_wheel_motor", required=True)
    act_rwheel = get_actuator_id(model, "right_wheel_motor", required=True)

    # -------------------------------------------------
    # Hip/knee actuators depend on experiment mode
    # -------------------------------------------------
    if LEG_ACTUATOR_MODE == "motor_pd":
        act_lhip = get_actuator_id(model, "left_hip_motor", required=True)
        act_rhip = get_actuator_id(model, "right_hip_motor", required=True)
        act_lknee = get_actuator_id(model, "left_knee_motor", required=True)
        act_rknee = get_actuator_id(model, "right_knee_motor", required=True)

    elif LEG_ACTUATOR_MODE == "position_servo":
        act_lhip = get_actuator_id(model, "left_hip_servo", required=True)
        act_rhip = get_actuator_id(model, "right_hip_servo", required=True)
        act_lknee = get_actuator_id(model, "left_knee_servo", required=True)
        act_rknee = get_actuator_id(model, "right_knee_servo", required=True)

    else:
        raise RuntimeError(
            "Unknown LEG_ACTUATOR_MODE. Use 'motor_pd' or 'position_servo'."
        )

    # ============================================================================================
    # Contact geom IDs for take-off / landing detection
    # ============================================================================================
    ground_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    left_wheel_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_geom")
    right_wheel_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_wheel_geom")

    if min(ground_gid, left_wheel_gid, right_wheel_gid) < 0:
        raise RuntimeError("Ground or wheel geom not found. Check XML geom names.")
    
    # Obtaining initial data from SeBaJu Model 
    mujoco.mj_forward(model, data)  # Update sensors before taking calibration measurements.
    imu_q0 = sensor_data(model, data, "imu_quat")  # Read the initial IMU quaternion.
    pitch_bias = quat_to_pitch(imu_q0[0], imu_q0[1], imu_q0[2], imu_q0[3])  # Treat initial pitch as zero balance reference.
    com0 = sensor_data(model, data, "robot_com")  # Read initial whole-robot center of mass.
    wl0 = sensor_data(model, data, "left_wheel_pos")  # Read initial left wheel center position.
    wr0 = sensor_data(model, data, "right_wheel_pos")  # Read initial right wheel center position.
    wheel_mid0 = 0.5 * (wl0 + wr0)  # Compute midpoint between wheel centers.
    com_x_ref = float(com0[0] - wheel_mid0[0])  # Save initial CoM-to-wheel x offset as desired offset.
    wheel_s0 = 0.5 * (data.qpos[qadr["left_wheel_joint"]] + data.qpos[qadr["right_wheel_joint"]]) * WHEEL_RADIUS  # Save initial wheel travel.

    dt = float(model.opt.timestep)  # Read physics timestep from the XML.
    step = 0  # Count simulation steps for printing diagnostics.
    pitch_rate_filtered = 0.0  # Initialize filtered pitch rate.
    velocity_filtered = 0.0  # Initialize filtered forward velocity.
    
    prev_com_lean = 0.0
    com_lean_rate_filtered = 0.0

    # Initial Hip and Knee reference positions from XML
    hip_ref_L = float(data.qpos[qadr["left_hip"]])
    hip_ref_R = float(data.qpos[qadr["right_hip"]])
    knee_ref_L = float(data.qpos[qadr["left_knee"]])
    knee_ref_R = float(data.qpos[qadr["right_knee"]])

    # =========================================================
    # Jump state variables initialization
    # =========================================================
    # States and Boolean flags for jump state machine
    jump_state = STATE_BALANCE
    jump_done = False
    loaded_state = True        # True means robot is load-bearing on the ground
    
    # Velocity and position variables for jump state machine
    prev_com_z = float(com0[2])
    com_z_vel_filtered = 0.0
    preload_min_com_z = float(com0[2]) # lowest CoM during crouch
    takeoff_com_z = None
    takeoff_com_z_vel = None
    landing_com_z = None
    jump_height_from_takeoff = 0.0
    jump_height_from_stand = 0.0
    v_takeoff_from_height = 0.0
    stand_com_z = None
    takeoff_com_z = None
    landing_com_z = None
    peak_com_z = -np.inf
    peak_air_com_z = -np.inf
    jump_height_airborne = 0.0
    jump_height_total = 0.0
    v_takeoff_from_air = 0.0
    jump_height_from_crouch = 0.0
    jump_height_above_stand_air = 0.0
    wheel_gap_ref = None
    gap_rel = 0.0
    peak_gap_abs = -np.inf
    peak_gap_rel = -np.inf
    
    # Timers for jump phases
    air_timer = 0.0 # DELETE
    contact_timer = 0.0 # DELETE
    unloaded_timer = 0.0
    loaded_timer = 0.0
    state_t0 = 0.0
    takeoff_time = None
    landing_time = None
    thrust_start_time = None
    thrust_duration_actual = None
    flight_time = 0.0
    
    # Force and thrust variables for jump state machine
    total_normal_force = 0.0
    thrust_force_sum = 0.0
    thrust_force_count = 0
    average_thrust_force = 0.0
  
    # THRUST AND JOINT TRACKING DIAGNOSTICS
    thrust_alpha_now = 0.0
    thrust_alpha_takeoff = 0.0

    hip_q_L = 0.0
    hip_q_R = 0.0
    knee_q_L = 0.0
    knee_q_R = 0.0

    hip_err_L = 0.0
    hip_err_R = 0.0
    knee_err_L = 0.0
    knee_err_R = 0.0

    flight_hold_valid = False
    hip_flight_hold_L = 0.0
    hip_flight_hold_R = 0.0
    knee_flight_hold_L = 0.0
    knee_flight_hold_R = 0.0

    # LANDING CONTINUITY VARIABLES
    landing_start_hip_L = hip_ref_L
    landing_start_hip_R = hip_ref_R
    landing_start_knee_L = knee_ref_L
    landing_start_knee_R = knee_ref_R

    
    # SETTLING TIME MEASUREMENT AFTER LANDING
    settle_timer = 0.0
    settle_time = None
    settle_failed = False

    # LANDING POSITION REFERENCE
    landing_x_ref = 0.0
    x_ref_active = 0.0

        # =========================================================
    # HIGH-LEVEL MOTION VARIABLES
    # =========================================================
    motion_mode = MODE_NONE
    motion_started = False
    motion_t0 = 0.0

    x_goal = 0.0
    x_cmd = 0.0
    x_i = 0.0

    slant_ref = 0.0

    step_phase = STEP_IDLE
    step_side = "L"
    step_dir = +1.0

    jump_direction = 0.0
    launch_x_ref = 0.0

    prev_wheel_u = 0.0

    prev_hip_L_cmd = hip_ref_L
    prev_hip_R_cmd = hip_ref_R
    prev_knee_L_cmd = knee_ref_L
    prev_knee_R_cmd = knee_ref_R

    # WHEEL AND LEG SATURATION TRACKING
    wheel_sat_count = 0
    leg_sat_count = 0
    max_abs_wheel_u = 0.0
    max_abs_xdot_after_landing = 0.0
    max_abs_theta_dot_after_landing = 0.0


    print(f"  SLOW_FACTOR  = {SLOW_FACTOR:.2f}  (edit this to change simulation speed)")  # Print simulation speed setting.
    print("-" * 100)  # Print separator line.
    print(f"{'Time':>6} {'Pitch':>9} {'CoMLean':>9} {'Rate':>9} {'Vel':>9} {'WheelU':>8} {'X':>8} {'COMx':>8} {'COMz':>8} {'IMUx':>8} {'IMUz':>8} {'OffX':>8} {'OffZ':>8} {'MoveDir':>9} {'Status'}")
    print("-" * 100)  # Print separator line.


    with mujoco.viewer.launch_passive(model, data) as viewer:  # Open MuJoCo passive viewer.
        
        viewer.cam.type = 0  # Use free camera so we can set viewpoint manually.
        viewer.cam.azimuth = 160  # Look from the side so X motion appears left-to-right.
        viewer.cam.elevation = -15  # Look slightly downward at the robot.
        viewer.cam.distance = 2.5  # Set camera distance from robot.

        while viewer.is_running():  # Keep running until the viewer window is closed.
            # Sensed values while simulation keeps running, feedback to control for constant update.
            imu_q = sensor_data(model, data, "imu_quat")  # Read IMU orientation quaternion.
            imu_g = sensor_data(model, data, "imu_gyro")  # Read IMU angular velocity.
            imu_a = sensor_data(model, data, "imu_accel")  # Read IMU acceleration; corrected below using IMU-to-CoM offset.
            imu_pos = sensor_data(model, data, "imu_pos") # Read IMU position
            com = sensor_data(model, data, "robot_com")  # Read whole-robot center of mass.

            # =========================================================
            # ROLL ESTIMATION FOR SIDE-SLANT AND LEG ASYMMETRY
            # =========================================================
            roll, pitch_quat_full = quat_to_roll_pitch(
                imu_q[0], imu_q[1], imu_q[2], imu_q[3]
            )

            roll_rate = float(imu_g[0])

            if abs(roll) > ROLL_SAFETY_MAX:
                motion_mode = MODE_NONE
                step_phase = STEP_IDLE
            
            # -------------------------------------------------
            # Vertical CoM velocity for jump monitoring
            # -------------------------------------------------
            com_z = float(com[2])
            com_z_vel_raw = (com_z - prev_com_z) / dt
            prev_com_z = com_z
            # Filtered velocity is only for display, not event detection
            com_z_vel_filtered = COMZ_VEL_FILTER * com_z_vel_filtered + (1.0 - COMZ_VEL_FILTER) * com_z_vel_raw
            com_z_vel = com_z_vel_filtered

            # Track lowest CoM during preload/crouch.
            if jump_state == STATE_PRELOAD:
                preload_min_com_z = min(preload_min_com_z, com_z)

            

            wl = sensor_data(model, data, "left_wheel_pos")  # Read left wheel center.
            wr = sensor_data(model, data, "right_wheel_pos")  # Read right wheel center.

            # =========================================================
            # WHEEL HEIGHT / CLEARANCE ABOVE GROUND
            # =========================================================
            left_wheel_gap = float(wl[2]) - WHEEL_RADIUS
            right_wheel_gap = float(wr[2]) - WHEEL_RADIUS
            min_wheel_gap = min(left_wheel_gap, right_wheel_gap)

            # =========================================================
            # WHEEL CLEARANCE: ABSOLUTE AND RELATIVE
            # =========================================================
            if wheel_gap_ref is None:
                gap_rel = 0.0
            else:
                gap_rel = min_wheel_gap - wheel_gap_ref

            # Track peak wheel clearance during the jump attempt.
            if jump_state in [
                STATE_PRELOAD,
                STATE_THRUST,
                STATE_FLIGHT,
                STATE_LANDING,
                STATE_RECOVERY
            ]:
                peak_gap_abs = max(peak_gap_abs, min_wheel_gap)
                peak_gap_rel = max(peak_gap_rel, gap_rel)

            # Take-off is accepted when the wheels are both:
            # 1. physically just above the ground, and
            # 2. lifted relative to their loaded baseline.
            wheel_clear_air = (
                min_wheel_gap > WHEEL_CLEARANCE_TAKEOFF_ABS
                and gap_rel > WHEEL_CLEARANCE_TAKEOFF_REL
            )

            imu_to_com = com - imu_pos                          # Offset from IMU to real whole-robot CoM.
            imu_to_com_x = float(imu_to_com[0])                 # Forward/backward offset.
            imu_to_com_y = float(imu_to_com[1])                 # Side offset.
            imu_to_com_z = float(imu_to_com[2])                 # Vertical offset.

            # At the moment IMU sensed values are not being used, rather the robot_com values are used.
            pitch_imu = quat_to_pitch(imu_q[0], imu_q[1], imu_q[2], imu_q[3]) - pitch_bias  # Body pitch from IMU, bias removed.
            pitch_acc = math.atan2(float(imu_a[0]) - imu_to_com_x, max(float(imu_a[2]) - imu_to_com_z, 1e-6))  # Approximate pitch from gravity direction at the IMU.
            pitch = pitch_imu #(1.0 - ACCEL_BLEND) * pitch_imu + ACCEL_BLEND * pitch_acc  # Mostly trust quaternion pitch, lightly blend accelerometer.
            #======================================================================================
            pitch_rate_raw = float(imu_g[1])  # Read pitch rate around Y axis from the gyro.
            pitch_rate_filtered = GYRO_FILTER * pitch_rate_filtered + (1.0 - GYRO_FILTER) * pitch_rate_raw  # Smooth the gyro signal.
            pitch_rate = pitch_rate_filtered  # Use the filtered pitch rate for control.

            # Values required for robot position computation
            wheel_mid = 0.5 * (wl + wr)  # Compute wheel axle midpoint in world coordinates.
            dx = float(com[0] - wheel_mid[0] - com_x_ref)  # Compute CoM horizontal error relative to the wheel axle.
            dz = float(com[2] - wheel_mid[2])  # Compute CoM height above the wheel axle.
            theta = -math.atan2(dx, max(dz, 1e-6))  # Compute actual inverted-pendulum lean angle from wheels to CoM (rad).
            # Calculating pitch/tilt rate from the robot_com 
            theta_dot_raw = (theta - prev_com_lean) / dt  # How fast the CoM lean is changing.
            prev_com_lean = theta  # Save current lean for next timestep.
            com_lean_rate_filtered = COM_RATE_FILTER * com_lean_rate_filtered + (1.0 - COM_RATE_FILTER) * theta_dot_raw
            theta_dot = com_lean_rate_filtered

            # Calculating robot velocity
            omega_L = float(data.qvel[vadr["left_wheel_joint"]])  # Read left wheel angular velocity.
            omega_R = float(data.qvel[vadr["right_wheel_joint"]])  # Read right wheel angular velocity.
            velocity_raw = 0.5 * (omega_L + omega_R) * WHEEL_RADIUS  # Convert average wheel spin into forward speed.
            velocity_filtered = VEL_FILTER * velocity_filtered + (1.0 - VEL_FILTER) * velocity_raw  # Smooth the velocity signal.
            xdot = velocity_filtered  # Use filtered velocity in the controller.

            # Specifical for status indication
            if xdot > 0.05:
                move_dir = "FORWARD"
            elif xdot < -0.05:
                move_dir = "BACKWARD"
            else:   
                move_dir = "STOPPED"
    
            # Wheel travel distance using wheel position instead of torso
            wheel_s = 0.5 * (data.qpos[qadr["left_wheel_joint"]] + data.qpos[qadr["right_wheel_joint"]]) * WHEEL_RADIUS  # Estimate wheel travel distance.
            x = wheel_s - wheel_s0  # Compute how far the robot has drifted from its start position.

            # Wheel current position
            x_curr = wheel_mid[0]
            
            # x_current and y_current specifical used for the camera positioning not the robot's
            x_current = float(data.qpos[root_qadr])  # Read the torso world X position for printing and camera tracking.
            y_current = float(data.qpos[root_qadr + 1])  # Read the torso world Y position for camera tracking.
            viewer.cam.lookat[:] = [x_current, y_current, BASE_HEIGHT]  # Keep the camera centered on the robot.


            # =========================================================
            # FORCE-BASED WHEEL CONTACT DETECTION WITH HYSTERESIS
            # =========================================================
            left_pair, right_pair, left_fn, right_fn = wheel_ground_contact_info(
                model, data,
                ground_gid,
                left_wheel_gid,
                right_wheel_gid
            )

            Fn_total = left_fn + right_fn

            # Track contact force during thrust.
            if jump_state == STATE_THRUST:
                thrust_force_sum += Fn_total
                thrust_force_count += 1

            # Schmitt trigger / hysteresis logic
            if loaded_state:
                # We are currently considered on the ground.
                # Only declare airborne if force stays very low.
                if Fn_total < F_CONTACT_OFF:
                    unloaded_timer += dt
                else:
                    unloaded_timer = 0.0

                if unloaded_timer >= CONTACT_LOSS_DEBOUNCE:
                    loaded_state = False
                    loaded_timer = 0.0

            else:
                # We are currently considered airborne.
                # Only declare landed if force becomes clearly load-bearing.
                if Fn_total > F_CONTACT_ON:
                    loaded_timer += dt
                else:
                    loaded_timer = 0.0

                if loaded_timer >= CONTACT_GAIN_DEBOUNCE:
                    loaded_state = True
                    unloaded_timer = 0.0

            airborne = not loaded_state

            # =========================================================
            # PEAK HEIGHT TRACKING
            # =========================================================

            # Overall peak during the whole jump attempt
            if jump_state in [
                STATE_PRELOAD,
                STATE_THRUST,
                STATE_FLIGHT,
                STATE_LANDING,
                STATE_RECOVERY
            ]:
                peak_com_z = max(peak_com_z, com_z)

            # Peak during confirmed flight
            if jump_state == STATE_FLIGHT:
                peak_air_com_z = max(peak_air_com_z, com_z)

            # ===============================================
            # JUMP STAGE-AWARE DIRECT STATE-FEEDBACK BALANCER
            # =====================================================
            theta_ref = PITCH_OFFSET
            
            # =========================================================
            # MODE-AWARE X REFERENCE
            # =========================================================
            if motion_mode == MODE_ROLL_MOVE and jump_state == STATE_BALANCE:
                x_cmd = slew(x_goal, x_cmd, X_CMD_RATE * dt)
                x_ref = x_cmd

            elif jump_state in [STATE_LANDING, STATE_RECOVERY, STATE_SETTLE]:
                x_ref = x_ref_active

            elif jump_state == STATE_BALANCE and jump_done:
                x_ref = x_ref_active

            elif motion_mode == MODE_JUMP_X and jump_state in [STATE_PRELOAD, STATE_THRUST]:
                x_ref = launch_x_ref

            else:
                x_ref = 0.0
            # ---------------------------------------------------------

            # =========================================================
            # SMALL INTEGRAL ONLY FOR POINT-TO-POINT ROLLING
            # =========================================================
            if motion_mode == MODE_ROLL_MOVE and jump_state == STATE_BALANCE:
                x_i = clamp(x_i + (x_ref - x) * dt, -X_I_MAX, +X_I_MAX)
            else:
                x_i *= 0.98
            # --------------------------------------------------------

            # =========================================================
            # BALANCE GAINS FOR CURRENT JUMP STATE
            # =========================================================
            Kth, Kthd, Kxd, Kx, max_wheel_state = balance_gains_for_state(jump_state)

            # =========================================================
            # MODE-AWARE WHEEL CONTROL
            # =========================================================
            theta_ref_local = theta_ref
            wheel_ff = 0.0

            # For translational jump, lean slightly in desired direction.
            if motion_mode == MODE_JUMP_X and jump_state in [STATE_PRELOAD, STATE_THRUST]:
                theta_ref_local = theta_ref + jump_direction * JUMP_X_PITCH_BIAS

                if jump_state == STATE_PRELOAD:
                    a_ff = smoothstep01(elapsed, T_PRELOAD)
                else:
                    a_ff = fast_thrust01(elapsed, T_THRUST)

                wheel_ff = jump_direction * JUMP_X_WHEEL_FF * a_ff

            wheel_u_unsat = (
                Kth   * (theta - theta_ref_local)
                + Kthd * theta_dot
                - Kxd  * xdot
                - Kx   * (x - x_ref)
                - K_X_I * x_i
                + wheel_ff
            )

            wheel_u_target = WHEEL_MOTOR_SIGN * clamp(
                wheel_u_unsat,
                -max_wheel_state,
                +max_wheel_state
            )

            wheel_u = slew(wheel_u_target, prev_wheel_u, WHEEL_DU_MAX)
            prev_wheel_u = wheel_u

            data.ctrl[act_lwheel] = wheel_u
            data.ctrl[act_rwheel] = wheel_u

            # =========================================================
            # WHEEL SATURATION CHECK
            # =========================================================
            wheel_sat = abs(wheel_u) >= 0.98 * max_wheel_state

            # =========================================================
            # SETTLING TIME CHECK
            # =========================================================
            settled_now = False

            if landing_time is not None and jump_state in [STATE_RECOVERY, STATE_SETTLE, STATE_BALANCE]:

                x_settle_error = x - x_ref_active

                settled_now = (
                    loaded_state
                    and abs(theta - PITCH_OFFSET) < SETTLE_LEAN_ERR
                    and abs(theta_dot) < SETTLE_RATE
                    and abs(xdot) < SETTLE_VEL
                    and abs(wheel_u) < SETTLE_WHEEL
                    and abs(x_settle_error) < SETTLE_X_ERR
                )

                if settled_now:
                    settle_timer += dt
                else:
                    settle_timer = 0.0
            else:
                settle_timer = 0.0

            # ============================================================================================================
            # JUMP STATE MACHINE + HIP/KNEE TARGET GENERATION
            # ============================================================================================================

            # Standing targets are the initial joint angles measured after mj_forward().
            hip_stand_L = hip_ref_L
            hip_stand_R = hip_ref_R
            knee_stand_L = knee_ref_L
            knee_stand_R = knee_ref_R

            # Preload/crouch targets.
            hip_preload_L = hip_ref_L + HIP_PRELOAD_DELTA
            hip_preload_R = hip_ref_R + HIP_PRELOAD_DELTA
            knee_preload_L = knee_ref_L + KNEE_PRELOAD_DELTA
            knee_preload_R = knee_ref_R + KNEE_PRELOAD_DELTA

            # Thrust/extension targets.
            hip_thrust_L = hip_ref_L + HIP_THRUST_DELTA
            hip_thrust_R = hip_ref_R + HIP_THRUST_DELTA
            knee_thrust_L = knee_ref_L + KNEE_THRUST_DELTA
            knee_thrust_R = knee_ref_R + KNEE_THRUST_DELTA

            # Landing absorption targets.
            hip_land_L = hip_ref_L + HIP_LAND_DELTA
            hip_land_R = hip_ref_R + HIP_LAND_DELTA
            knee_land_L = knee_ref_L + KNEE_LAND_DELTA
            knee_land_R = knee_ref_R + KNEE_LAND_DELTA

            stable_for_jump = (
                abs(theta - theta_ref) < math.radians(2.5)
                and abs(xdot) < 0.08
                and loaded_state
            )

            # =========================================================
            # ARM SELECTED EXPERIMENT ONCE ROBOT IS STABLE
            # =========================================================
            stable_for_motion = (
                jump_state == STATE_BALANCE
                and not motion_started
                and data.time > 5.0
                and abs(theta - theta_ref) < math.radians(2.5)
                and abs(xdot) < 0.08
                and loaded_state
            )

            if stable_for_motion:

                if EXPERIMENT_MODE == "INDIVIDUAL_LEG_TEST":
                    motion_mode = MODE_INDIVIDUAL_LEG
                    motion_t0 = data.time
                    motion_started = True

                elif EXPERIMENT_MODE == "SLANT_LEFT":
                    motion_mode = MODE_SLANT
                    slant_ref = +math.radians(4.0)
                    motion_t0 = data.time
                    motion_started = True

                elif EXPERIMENT_MODE == "SLANT_RIGHT":
                    motion_mode = MODE_SLANT
                    slant_ref = -math.radians(4.0)
                    motion_t0 = data.time
                    motion_started = True

                elif EXPERIMENT_MODE == "ROLL_FORWARD":
                    motion_mode = MODE_ROLL_MOVE
                    x_goal = x + 0.20
                    x_cmd = x
                    motion_t0 = data.time
                    motion_started = True

                elif EXPERIMENT_MODE == "ROLL_BACKWARD":
                    motion_mode = MODE_ROLL_MOVE
                    x_goal = x - 0.20
                    x_cmd = x
                    motion_t0 = data.time
                    motion_started = True

                elif EXPERIMENT_MODE == "STEP_LEFT":
                    motion_mode = MODE_STEP
                    step_side = "L"
                    step_dir = +1.0
                    step_phase = STEP_PREP
                    motion_t0 = data.time
                    motion_started = True

                elif EXPERIMENT_MODE == "STEP_RIGHT":
                    motion_mode = MODE_STEP
                    step_side = "R"
                    step_dir = +1.0
                    step_phase = STEP_PREP
                    motion_t0 = data.time
                    motion_started = True

                elif EXPERIMENT_MODE == "JUMP_FORWARD":
                    motion_mode = MODE_JUMP_X
                    jump_direction = +1.0
                    launch_x_ref = x
                    motion_t0 = data.time
                    motion_started = True

                elif EXPERIMENT_MODE == "JUMP_BACKWARD":
                    motion_mode = MODE_JUMP_X
                    jump_direction = -1.0
                    launch_x_ref = x
                    motion_t0 = data.time
                    motion_started = True

            elapsed = data.time - state_t0

            # =========================================================
            # THRUST PROGRESS DIAGNOSTIC
            # =========================================================
            if thrust_start_time is not None:
                thrust_alpha_now = fast_thrust01(data.time - thrust_start_time, T_THRUST)
            else:
                thrust_alpha_now = 0.0

            # -----------------------------
            # State transitions
            # -----------------------------
            if jump_state == STATE_BALANCE:
                if (
                    JUMP_ENABLE
                    and not jump_done
                    and data.time > JUMP_START_TIME
                    and stable_for_jump
                ):
                    jump_state = STATE_PRELOAD
                    state_t0 = data.time

                    # Reset jump measurements at the actual start of the jump attempt.
                    stand_com_z = com_z
                    preload_min_com_z = com_z

                    # Wheel gap reference at the start of the jump attempt.
                    # This captures the normal loaded wheel penetration/contact offset.
                    wheel_gap_ref = min_wheel_gap

                    peak_gap_abs = min_wheel_gap
                    peak_gap_rel = 0.0

                    peak_com_z = com_z
                    peak_air_com_z = -np.inf

                    takeoff_com_z = None
                    landing_com_z = None

                    jump_height_airborne = 0.0
                    jump_height_total = 0.0
                    jump_height_from_crouch = 0.0
                    jump_height_above_stand_air = 0.0
                    v_takeoff_from_air = 0.0
                    flight_time = 0.0

                    # Settling time measurement after landing is reset at the start of the jump attempt.
                    settle_timer = 0.0
                    settle_time = None
                    settle_failed = False

                    # WHEEL AND LEG SATURATION TRACKING RESET
                    wheel_sat_count = 0
                    leg_sat_count = 0
                    max_abs_wheel_u = 0.0
                    max_abs_xdot_after_landing = 0.0
                    max_abs_theta_dot_after_landing = 0.0

                    print(
                        f"\n>>> JUMP STATE: PRELOAD "
                        f"t={data.time:.4f}s "
                        f"StandZ={stand_com_z:.4f}"
                    )


            # =============================================================================
            elif jump_state == STATE_PRELOAD:
                if elapsed >= T_PRELOAD:
                    jump_state = STATE_THRUST
                    state_t0 = data.time
                    thrust_start_time = data.time

                    # Reset thrust force measurement.
                    thrust_force_sum = 0.0
                    thrust_force_count = 0

                    print(
                        f"\n>>> JUMP STATE: THRUST "
                        f"t={data.time:.4f}s "
                        f"COMz={com_z:.4f} "
                        f"PreloadMinZ={preload_min_com_z:.4f}"
                    )


            # =============================================================================
            elif jump_state == STATE_THRUST:

                takeoff_detected = (
                    airborne
                    and Fn_total < F_CONTACT_OFF
                    and wheel_clear_air
                    and com_z_vel_raw > TAKEOFF_VZ_MIN
                )

                if takeoff_detected:
                    jump_state = STATE_FLIGHT
                    state_t0 = data.time

                    takeoff_time = data.time
                    takeoff_com_z = com_z
                    takeoff_com_z_vel = com_z_vel_raw
                    peak_air_com_z = com_z


                    # =========================================================
                    # FREEZE ACTUAL LEG POSE AT TAKE-OFF
                    # =========================================================
                    thrust_alpha_takeoff = thrust_alpha_now

                    hip_flight_hold_L = float(data.qpos[qadr["left_hip"]])
                    hip_flight_hold_R = float(data.qpos[qadr["right_hip"]])

                    knee_flight_hold_L = float(data.qpos[qadr["left_knee"]])
                    knee_flight_hold_R = float(data.qpos[qadr["right_knee"]])

                    flight_hold_valid = True


                    if thrust_force_count > 0:
                        average_thrust_force = thrust_force_sum / thrust_force_count
                    else:
                        average_thrust_force = 0.0

                    print(
                        f"\n>>> TAKE-OFF DETECTED "
                        f"t={takeoff_time:.4f}s "
                        f"COMz={takeoff_com_z:.4f} "
                        f"Vz_raw={takeoff_com_z_vel:.4f} "
                        f"Fn={Fn_total:.2f}N "
                        f"GapAbs={min_wheel_gap:.4f}m "
                        f"GapRel={gap_rel:.4f}m "
                        f"pair=({int(left_pair)},{int(right_pair)}) "
                        f"loaded={int(loaded_state)}"
                    )

                elif elapsed >= T_THRUST_MAX:
                    jump_state = STATE_LANDING
                    state_t0 = data.time

                    print(
                        f"\n>>> NO TAKE-OFF: SAFE LANDING "
                        f"t={data.time:.4f}s "
                        f"COMz={com_z:.4f} "
                        f"Fn={Fn_total:.2f}N"
                    )
            # =============================================================================


            elif jump_state == STATE_FLIGHT:

                # Continue updating airborne peak during FLIGHT
                peak_air_com_z = max(peak_air_com_z, com_z)

                landing_detected = loaded_state

                if landing_detected:
                    # Capture actual pose before entering LANDING
                    landing_start_hip_L = float(data.qpos[qadr["left_hip"]])
                    landing_start_hip_R = float(data.qpos[qadr["right_hip"]])
                    landing_start_knee_L = float(data.qpos[qadr["left_knee"]])
                    landing_start_knee_R = float(data.qpos[qadr["right_knee"]])

                    jump_state = STATE_LANDING
                    state_t0 = data.time

                    landing_time = data.time
                    landing_com_z = com_z

                    # Capture where the robot landed.
                    landing_x_ref = x

                    if RETURN_TO_START_AFTER_JUMP:
                        x_ref_active = 0.0
                    else:
                        x_ref_active = landing_x_ref

                    # Reset settling measurement at landing.
                    settle_timer = 0.0
                    settle_time = None
                    settle_failed = False

                    if takeoff_time is not None:
                        flight_time = landing_time - takeoff_time
                    else:
                        flight_time = 0.0

                    if takeoff_com_z is not None and np.isfinite(peak_air_com_z):
                        jump_height_airborne = max(0.0, peak_air_com_z - takeoff_com_z)
                    else:
                        jump_height_airborne = 0.0

                    # Height of the airborne peak relative to standing height.
                    # This can be zero if the robot takes off from a deep crouch
                    # but does not rise above the original standing CoM height before first contact returns.
                    if stand_com_z is not None and np.isfinite(peak_air_com_z):
                        jump_height_above_stand_air = max(0.0, peak_air_com_z - stand_com_z)
                    else:
                        jump_height_above_stand_air = 0.0

                    if jump_height_airborne > 0.0:
                        v_takeoff_from_air = math.sqrt(2.0 * G * jump_height_airborne)
                    else:
                        v_takeoff_from_air = 0.0

                    print(
                        f"\n>>> LANDING DETECTED "
                        f"t={landing_time:.4f}s "
                        f"FlightTime={flight_time:.4f}s "
                        f"TakeoffZ={takeoff_com_z:.4f} "
                        f"PeakAirZ={peak_air_com_z:.4f} "
                        f"PeakZ={peak_com_z:.4f} "
                        f"LandingZ={landing_com_z:.4f} "
                        f"H_airborne={jump_height_airborne:.4f}m "
                        f"H_above_stand_air={jump_height_above_stand_air:.4f}m "
                        f"Vto_air_est={v_takeoff_from_air:.4f}m/s "
                        f"Favg_thrust={average_thrust_force:.2f}N"
                    )
            # ============================================================================================================


            elif jump_state == STATE_LANDING:
                if elapsed >= T_LANDING:
                    jump_state = STATE_RECOVERY
                    state_t0 = data.time
                    print("\n>>> JUMP STATE: RECOVERY")


            # ============================================================================================================
            elif jump_state == STATE_RECOVERY:
                if elapsed >= T_RECOVERY:

                    jump_state = STATE_SETTLE
                    state_t0 = data.time

                    print("\n>>> JUMP STATE: SETTLE / WAITING FOR STABLE BALANCE")

            # ============================================================================================================
            elif jump_state == STATE_SETTLE:

                # Success: robot has satisfied settling conditions continuously.
                if settle_time is None and settle_timer >= SETTLE_HOLD:
                    settle_time = data.time - landing_time

                # Timeout should be measured from landing, not from the start of SETTLE.
                settle_timeout = (
                    landing_time is not None
                    and (data.time - landing_time) >= MAX_SETTLE_WAIT
                )

                if settle_time is not None or settle_timeout:

                    settle_failed = settle_time is None

                    # Final full-attempt measurements.
                    if stand_com_z is not None and np.isfinite(peak_com_z):
                        jump_height_total = max(0.0, peak_com_z - stand_com_z)
                    else:
                        jump_height_total = 0.0

                    if np.isfinite(peak_com_z):
                        jump_height_from_crouch = max(0.0, peak_com_z - preload_min_com_z)
                    else:
                        jump_height_from_crouch = 0.0

                    visible_jump = (
                        peak_gap_abs >= 0.010
                        and jump_height_above_stand_air >= 0.010
                    )

                    jump_state = STATE_BALANCE
                    state_t0 = data.time
                    jump_done = JUMP_ONCE

                    if settle_failed:
                        settle_time_text = "NOT_SETTLED"
                        settle_time_numeric = -1.0
                    else:
                        settle_time_text = f"{settle_time:.4f}s"
                        settle_time_numeric = settle_time

                    print("\n>>> JUMP STATE: BALANCE / SETTLED")
                    print(
                        f">>> FINAL JUMP SUMMARY "
                        f"FlightTime={flight_time:.4f}s "
                        f"H_airborne={jump_height_airborne:.4f}m "
                        f"H_total={jump_height_total:.4f}m "
                        f"H_from_crouch={jump_height_from_crouch:.4f}m "
                        f"PeakGapAbs={peak_gap_abs:.4f}m "
                        f"PeakGapRel={peak_gap_rel:.4f}m "
                        f"StandZ={stand_com_z:.4f} "
                        f"PreloadMinZ={preload_min_com_z:.4f} "
                        f"PeakZ={peak_com_z:.4f} "
                        f"Vto_air_est={v_takeoff_from_air:.4f}m/s "
                        f"Favg_thrust={average_thrust_force:.2f}N "
                        #f"SettleTime={settle_time_text} "
                        f"SettleTimeNum={settle_time_numeric:.4f} "
                        f"TimedOut={int(settle_failed)} "
                        f"WheelSatCount={wheel_sat_count} "
                        f"LegSatCount={leg_sat_count} "
                        f"MaxWheelU={max_abs_wheel_u:.3f} "
                        f"MaxPostLandVel={max_abs_xdot_after_landing:.3f} "
                        f"MaxPostLandRate={math.degrees(max_abs_theta_dot_after_landing):.1f}deg/s "
                        f"SUCCESS_VISIBLE={int(visible_jump)} "
                    )

            elapsed = data.time - state_t0

            # =========================================================
            # STEP SUB-STATE MACHINE
            # Only active during normal BALANCE, not during jump.
            # =========================================================
            if motion_mode == MODE_STEP and jump_state == STATE_BALANCE:

                step_elapsed = data.time - motion_t0

                if step_side == "L":
                    swing_fn = left_fn
                    support_fn = right_fn
                else:
                    swing_fn = right_fn
                    support_fn = left_fn

                if step_phase == STEP_PREP:
                    if step_elapsed >= STEP_PREP_TIME:
                        step_phase = STEP_UNLOAD
                        motion_t0 = data.time

                elif step_phase == STEP_UNLOAD:
                    if step_elapsed >= STEP_UNLOAD_TIME or swing_fn < STEP_UNLOAD_FN_MAX:
                        step_phase = STEP_SWING
                        motion_t0 = data.time

                elif step_phase == STEP_SWING:
                    if step_elapsed >= STEP_SWING_TIME:
                        step_phase = STEP_PLACE
                        motion_t0 = data.time

                elif step_phase == STEP_PLACE:
                    if step_elapsed >= STEP_PLACE_TIME:
                        step_phase = STEP_TRANSFER
                        motion_t0 = data.time

                elif step_phase == STEP_TRANSFER:
                    if step_elapsed >= STEP_TRANSFER_TIME:
                        step_phase = STEP_IDLE
                        motion_mode = MODE_NONE
                        motion_t0 = data.time

            # =========================================================
            # END ROLLING / SLANT MODES
            # =========================================================
            if motion_mode == MODE_ROLL_MOVE and jump_state == STATE_BALANCE:
                if abs(x - x_goal) < X_GOAL_TOL and abs(xdot) < 0.04:
                    motion_mode = MODE_NONE
                    motion_t0 = data.time

            if motion_mode == MODE_SLANT and jump_state == STATE_BALANCE:
                if data.time - motion_t0 > 3.0:
                    motion_mode = MODE_NONE
                    motion_t0 = data.time

            if motion_mode == MODE_INDIVIDUAL_LEG and jump_state == STATE_BALANCE:
                if data.time - motion_t0 > 3.0:
                    motion_mode = MODE_NONE
                    motion_t0 = data.time

            # -----------------------------
            # Leg target generation for each state
            # -----------------------------
            if jump_state in [STATE_BALANCE, STATE_SETTLE]:
                hip_target_L = hip_stand_L
                hip_target_R = hip_stand_R
                knee_target_L = knee_stand_L
                knee_target_R = knee_stand_R

            elif jump_state == STATE_PRELOAD:
                a = smoothstep01(elapsed, T_PRELOAD)
                hip_target_L = blend(hip_stand_L, hip_preload_L, a)
                hip_target_R = blend(hip_stand_R, hip_preload_R, a)
                knee_target_L = blend(knee_stand_L, knee_preload_L, a)
                knee_target_R = blend(knee_stand_R, knee_preload_R, a)

            elif jump_state == STATE_THRUST:
                a = fast_thrust01(elapsed, T_THRUST)
                hip_target_L = blend(hip_preload_L, hip_thrust_L, a)
                hip_target_R = blend(hip_preload_R, hip_thrust_R, a)
                knee_target_L = blend(knee_preload_L, knee_thrust_L, a)
                knee_target_R = blend(knee_preload_R, knee_thrust_R, a)

            elif jump_state == STATE_FLIGHT:
                if flight_hold_valid:
                    hip_target_L = hip_flight_hold_L
                    hip_target_R = hip_flight_hold_R
                    knee_target_L = knee_flight_hold_L
                    knee_target_R = knee_flight_hold_R
                else:
                    hip_target_L = hip_thrust_L
                    hip_target_R = hip_thrust_R
                    knee_target_L = knee_thrust_L
                    knee_target_R = knee_thrust_R

            elif jump_state == STATE_LANDING:
                a = smoothstep01(elapsed, T_LANDING)

                # Start landing recovery from the actual leg pose at touchdown.
                # This avoids a sudden jump from flight pose to thrust pose.
                hip_target_L = blend(landing_start_hip_L, hip_land_L, a)
                hip_target_R = blend(landing_start_hip_R, hip_land_R, a)

                knee_target_L = blend(landing_start_knee_L, knee_land_L, a)
                knee_target_R = blend(landing_start_knee_R, knee_land_R, a)

            elif jump_state == STATE_RECOVERY:
                a = smoothstep01(elapsed, T_RECOVERY)
                hip_target_L = blend(hip_land_L, hip_stand_L, a)
                hip_target_R = blend(hip_land_R, hip_stand_R, a)
                knee_target_L = blend(knee_land_L, knee_stand_L, a)
                knee_target_R = blend(knee_land_R, knee_stand_R, a)

            else:
                hip_target_L = hip_stand_L
                hip_target_R = hip_stand_R
                knee_target_L = knee_stand_L
                knee_target_R = knee_stand_R

            # =========================================================
            # COMMON + DIFFERENTIAL LEG CONTROL
            # =========================================================

            # The existing state machine produced these as symmetric targets.
            # Treat them as the common posture first.
            hip_common = 0.5 * (hip_target_L + hip_target_R)
            knee_common = 0.5 * (knee_target_L + knee_target_R)

            hip_diff = 0.0
            knee_diff = 0.0

            # ---------------------------------------------------------
            # 1. Individual leg actuation test
            # ---------------------------------------------------------
            if motion_mode == MODE_INDIVIDUAL_LEG and jump_state == STATE_BALANCE:
                t = data.time - motion_t0

                hip_diff = 0.06 * math.sin(2.0 * math.pi * 0.5 * t) #0.28
                knee_diff = 0.08 * math.sin(2.0 * math.pi * 0.5 * t) #0.30

            # ---------------------------------------------------------
            # 2. Controlled sideways slant
            # ---------------------------------------------------------
            elif motion_mode == MODE_SLANT and jump_state == STATE_BALANCE:

                roll_error = slant_ref - roll
                d_roll = K_ROLL_P * roll_error - K_ROLL_D * roll_rate
                d_roll = clamp(d_roll, -LEG_DIFF_MAX, +LEG_DIFF_MAX)

                hip_diff = HIP_ROLL_TO_DIFF * d_roll
                knee_diff = KNEE_ROLL_TO_DIFF * d_roll

            # ---------------------------------------------------------
            # 3. Step-by-step one-leg movement
            # ---------------------------------------------------------
            elif motion_mode == MODE_STEP and jump_state == STATE_BALANCE:

                # Shift body slightly toward support leg.
                if step_side == "L":
                    # left is swing, right is support
                    roll_ref = -STEP_ROLL_BIAS
                else:
                    # right is swing, left is support
                    roll_ref = +STEP_ROLL_BIAS

                roll_error = roll_ref - roll
                d_roll = K_ROLL_P * roll_error - K_ROLL_D * roll_rate
                d_roll = clamp(d_roll, -LEG_DIFF_MAX, +LEG_DIFF_MAX)

                hip_diff = HIP_ROLL_TO_DIFF * d_roll
                knee_diff = KNEE_ROLL_TO_DIFF * d_roll

            # Convert common + differential into left/right targets.
            hip_target_L, hip_target_R, knee_target_L, knee_target_R = leg_common_diff_to_targets(
                hip_common,
                knee_common,
                hip_diff,
                knee_diff
            )

            # =========================================================
            # ADD SWING-LEG MOTION FOR STEP MODE
            # =========================================================
            if motion_mode == MODE_STEP and jump_state == STATE_BALANCE:

                phase_elapsed = data.time - motion_t0

                swing_hip_offset = 0.0
                swing_knee_offset = 0.0

                if step_phase == STEP_UNLOAD:
                    a = smoothstep01(phase_elapsed, STEP_UNLOAD_TIME)
                    swing_hip_offset = STEP_HIP_LIFT * a
                    swing_knee_offset = STEP_KNEE_LIFT * a

                elif step_phase == STEP_SWING:
                    prog, lift = swing_profile01(phase_elapsed, STEP_SWING_TIME)

                    swing_hip_offset = STEP_HIP_LIFT * lift + step_dir * STEP_HIP_SWING * (prog - 0.5)
                    swing_knee_offset = STEP_KNEE_LIFT * lift + step_dir * STEP_KNEE_SWING * (prog - 0.5)

                elif step_phase == STEP_PLACE:
                    a = 1.0 - smoothstep01(phase_elapsed, STEP_PLACE_TIME)
                    swing_hip_offset = STEP_HIP_LIFT * a
                    swing_knee_offset = STEP_KNEE_LIFT * a

                if step_side == "L":
                    hip_target_L += swing_hip_offset
                    knee_target_L += swing_knee_offset
                else:
                    hip_target_R += swing_hip_offset
                    knee_target_R += swing_knee_offset

            # =========================================================
            # LEG COMMAND SLEW LIMITING
            # =========================================================
            hip_target_L = slew(hip_target_L, prev_hip_L_cmd, LEG_DQ_MAX)
            hip_target_R = slew(hip_target_R, prev_hip_R_cmd, LEG_DQ_MAX)
            knee_target_L = slew(knee_target_L, prev_knee_L_cmd, LEG_DQ_MAX)
            knee_target_R = slew(knee_target_R, prev_knee_R_cmd, LEG_DQ_MAX)

            prev_hip_L_cmd = hip_target_L
            prev_hip_R_cmd = hip_target_R
            prev_knee_L_cmd = knee_target_L
            prev_knee_R_cmd = knee_target_R

            # Clamp to XML actuator ctrlranges.
            hip_target_L = clamp(hip_target_L, HIP_CTRL_MIN, HIP_CTRL_MAX)
            hip_target_R = clamp(hip_target_R, HIP_CTRL_MIN, HIP_CTRL_MAX)
            knee_target_L = clamp(knee_target_L, KNEE_CTRL_MIN, KNEE_CTRL_MAX)
            knee_target_R = clamp(knee_target_R, KNEE_CTRL_MIN, KNEE_CTRL_MAX)

            command_legs(data, qadr, vadr, LEG_ACTUATOR_MODE, act_lhip, act_rhip, act_lknee, act_rknee, hip_target_L, hip_target_R, knee_target_L, knee_target_R,)

            # =========================================================
            # ACTUAL JOINT POSITION AND TRACKING ERROR
            # =========================================================
            hip_q_L = float(data.qpos[qadr["left_hip"]])
            hip_q_R = float(data.qpos[qadr["right_hip"]])

            knee_q_L = float(data.qpos[qadr["left_knee"]])
            knee_q_R = float(data.qpos[qadr["right_knee"]])

            hip_err_L = hip_target_L - hip_q_L
            hip_err_R = hip_target_R - hip_q_R

            knee_err_L = knee_target_L - knee_q_L
            knee_err_R = knee_target_R - knee_q_R

            # =========================================================
            # LEG ACTUATOR FORCE MONITORING
            # =========================================================
            hip_tau_L = float(data.actuator_force[act_lhip])
            hip_tau_R = float(data.actuator_force[act_rhip])
            knee_tau_L = float(data.actuator_force[act_lknee])
            knee_tau_R = float(data.actuator_force[act_rknee])

            LEG_FORCE_LIMIT = 60.0

            leg_sat = (
                abs(hip_tau_L) > 0.98 * LEG_FORCE_LIMIT or
                abs(hip_tau_R) > 0.98 * LEG_FORCE_LIMIT or
                abs(knee_tau_L) > 0.98 * LEG_FORCE_LIMIT or
                abs(knee_tau_R) > 0.98 * LEG_FORCE_LIMIT
            )

            # =========================================================
            # WHEEL AND LEG SATURATION TRACKING
            if jump_state in [STATE_LANDING, STATE_RECOVERY, STATE_SETTLE]:
                if wheel_sat:
                    wheel_sat_count += 1

                if leg_sat:
                    leg_sat_count += 1

                max_abs_wheel_u = max(max_abs_wheel_u, abs(wheel_u))
                max_abs_xdot_after_landing = max(max_abs_xdot_after_landing, abs(xdot))
                max_abs_theta_dot_after_landing = max(max_abs_theta_dot_after_landing, abs(theta_dot))
            # ==================================================================================================================================
                
                
            mujoco.mj_step(model, data)  # Advance the physics by one timestep using the current controls.
            time.sleep(dt * SLOW_FACTOR)  # Slow down or speed up the visible simulation rate.
            viewer.sync()  # Update the viewer window with the latest simulation state.

            if jump_state == STATE_BALANCE:
                print_dt = PRINT_DT_BALANCE
            elif jump_state == STATE_PRELOAD:
                print_dt = PRINT_DT_PRELOAD
            elif jump_state == STATE_THRUST:
                print_dt = PRINT_DT_THRUST
            elif jump_state == STATE_FLIGHT:
                print_dt = PRINT_DT_FLIGHT
            elif jump_state == STATE_LANDING:
                print_dt = PRINT_DT_LANDING
            elif jump_state == STATE_RECOVERY:
                print_dt = PRINT_DT_RECOVERY
            elif jump_state == STATE_SETTLE:
                print_dt = PRINT_DT_SETTLE
            else:
                print_dt = 0.20

            print_every = max(1, int(print_dt / dt))

            if step % print_every == 0:  # During balance: print every 0.50 s; During jump: print every 0.02 s
                if wheel_sat:
                    status = "WHEEL_SAT / reduce wheel gain"
                elif leg_sat:
                    status = "LEG_SAT / soften landing or raise force"
                elif abs(theta - theta_ref) < 0.035 and abs(xdot) < 0.08:
                    status = "BALANCING"
                elif abs(xdot) > 1.0:
                    status = "FAST MOTION"
                else:
                    status = "correcting"
                print(
                        f"{data.time:6.2f} "
                        f"{math.degrees(pitch):+8.2f} "
                        f"{math.degrees(theta):+8.2f} "
                        f"{math.degrees(theta_dot):+8.1f} "
                        f"{xdot:+8.3f} "
                        f"{wheel_u:+7.3f} "
                        f"{x_curr:+7.3f} "
                        f"{com[0]:+7.3f} "
                        f"{com[2]:+7.3f} "
                        f"{imu_pos[0]:+7.3f} "
                        f"{imu_pos[2]:+7.3f} "
                        f"{imu_to_com_x:+7.3f} "
                        f"{imu_to_com_z:+7.3f} "
                        f"{move_dir:>9} "
                        f"{jump_state:>9} "
                        f"COMzV={com_z_vel:+6.3f} "
                        f"VzRaw={com_z_vel_raw:+6.3f} "
                        f"pair=({int(left_pair)},{int(right_pair)}) "
                        f"Fn={Fn_total:6.2f}N "
                        f"GapAbs={min_wheel_gap:+7.4f}m "
                        f"GapRel={gap_rel:+7.4f}m "
                        f"Gap={min_wheel_gap:+6.4f}m "
                        f"loaded={int(loaded_state)} "
                        f"air={int(airborne)} "
                        f"Hip={hip_target_L:+5.2f} Knee={knee_target_L:+5.2f} "
                        f"PeakZ={peak_com_z:+6.3f} "
                        f"TauH={hip_tau_L:+6.1f} "
                        f"TauK={knee_tau_L:+6.1f} "
                        f"WheelSat={int(wheel_sat)} "
                        f"LegSat={int(leg_sat)} "
                        f"ThrA={thrust_alpha_now:4.2f} "
                        f"Hq={hip_q_L:+5.2f} Kq={knee_q_L:+5.2f} "
                        f"He={hip_err_L:+5.2f} Ke={knee_err_L:+5.2f} "
                        f"Mode={motion_mode:>7} "
                        f"Step={step_phase:>10} "
                        f"Roll={math.degrees(roll):+5.2f} "
                        f"RollRate={math.degrees(roll_rate):+7.1f} "
                        f"LFn={left_fn:6.2f} "
                        f"RFn={right_fn:6.2f} "
                        f"XGoal={x_goal:+6.3f} "
                        f"XErr={(x_goal - x):+6.3f} "
                        f"{status}"
)

            step += 1  # Increase the step counter.

    print("\nViewer closed.")  # Print message after the viewer is closed.


if __name__ == "__main__":  # Only run automatically when this file is executed directly.
    run()  # Start the simulation and controller loop.