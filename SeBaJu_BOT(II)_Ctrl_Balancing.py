"""
Author: Kennedy Chukwuma
Designed specifical for balancing with the hip and knee joints fixed in xml
"""

import mujoco  # Import MuJoCo so Python can load the XML model and step the physics.
import mujoco.viewer  # Import the passive viewer so we can see the simulation while the controller runs.
import math  # Import math for sin/cos/atan2/degrees functions.
import time  # Import time so SLOW_FACTOR can control the visible simulation speed.
import os  # Import os so the XML can be found in the same folder as this controller.
import numpy as np  # Import NumPy for clipping and small vector calculations.


#==========================================================
#
XML_FILE = "SeBaJu_BOT(II).xml"  # XML file to load; keep it in the same folder as this controller.
SLOW_FACTOR = 0.50  # 0.25 = quarter speed, 0.50 = half speed, 1.0 = real time, 2.0 = faster than real time.
BASE_HEIGHT = 0.2439  # Camera look-at height, chosen near the horizontal torso height.
WHEEL_RADIUS = 0.06  # Wheel radius from the XML cylinder size="0.10 0.045".



#==========================================================
GYRO_FILTER = 0.70  # Low-pass filter for pitch rate; higher value = smoother but more delay.
VEL_FILTER = 0.70  # Low-pass filter for wheel velocity; higher value = smoother but more delay.
ACCEL_BLEND = 0.0  # Small blend from accelerometer tilt; keep low because acceleration is noisy during balancing.

#============================================================
# -------------------------------
# TUNING CONSTANTS
# -------------------------------
PITCH_OFFSET = 0.0
MAX_WHEEL = 0.35

K_THETA   = 1.3 #1.3    # forward lean correction
K_THETA_D = 0.17 #0.23  # lean-rate damping
K_X_D     = 0.16 #0.07   # velocity damping #4.0    # forward lean correction
K_X       = 0.85 #0.8 #0.5 #0.035    # position-hold stiffness


COM_RATE_FILTER = 0.85
VEL_FILTER = 0.70
WHEEL_MOTOR_SIGN = -1

#================================================================
# HELPER FUNCTIONS
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

#============================================================================================================================================

def run():  # Main function that loads the robot and runs the feedback loop.
    xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), XML_FILE)  # Build full path to the XML file beside this script.
    if not os.path.exists(xml_path):  # Check that the XML file is really there.
        print(f"ERROR: XML not found: {xml_path}")  # Print a helpful path if the XML is missing.
        raise SystemExit(1)  # Stop the program because MuJoCo cannot run without the XML.

    print(f"\nLoading: {xml_path}")  # Tell the user which XML is being loaded.
    model = mujoco.MjModel.from_xml_path(xml_path)  # Load the MuJoCo model from the XML file.
    data = mujoco.MjData(model)  # Create the live simulation data structure for this model.

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


    # wheel joints only for the balance-first phase
    jnames = ["left_wheel_joint", "right_wheel_joint"]
    jid, qadr, vadr = joint_maps(model, jnames)

    # actuator ids by name, so re-enabling hip/knee later does not break indexing
    act_left  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_motor")
    act_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_motor")
    if act_left < 0 or act_right < 0:
        raise RuntimeError("Wheel actuators not found")


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

    vel_integral = 0.0
    pitch_target = PITCH_OFFSET
    inner_integral = 0.0


    
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
            wl = sensor_data(model, data, "left_wheel_pos")  # Read left wheel center.
            wr = sensor_data(model, data, "right_wheel_pos")  # Read right wheel center.

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

            theta_dot_raw = (theta - prev_com_lean) / dt  # How fast the CoM lean is changing.
            prev_com_lean = theta  # Save current lean for next timestep.
            com_lean_rate_filtered = COM_RATE_FILTER * com_lean_rate_filtered + (1.0 - COM_RATE_FILTER) * theta_dot_raw
            theta_dot = com_lean_rate_filtered

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

            # -----------------------------------------
            # DIRECT STATE-FEEDBACK BALANCER
            # -----------------------------------------
            theta_ref = PITCH_OFFSET
            x_ref = 0.0

            wheel_u = clamp(
                            K_THETA   * (theta - theta_ref)
                            + K_THETA_D * theta_dot
                            - K_X_D   * xdot
                            - K_X     * (x - x_ref),
                            -MAX_WHEEL, MAX_WHEEL
                        )

            wheel_u = WHEEL_MOTOR_SIGN * wheel_u
            data.ctrl[act_left] = wheel_u
            data.ctrl[act_right] = wheel_u

            mujoco.mj_step(model, data)  # Advance the physics by one timestep using the current controls.
            time.sleep(dt * SLOW_FACTOR)  # Slow down or speed up the visible simulation rate.
            viewer.sync()  # Update the viewer window with the latest simulation state.

            if step % 250 == 0:  # Print diagnostics roughly twice per second at 500 Hz.
                if abs(wheel_u) >= MAX_WHEEL * 0.98:  # Check whether wheel command is saturating.
                    status = "SATURATED / reduce gain"  # Explain that gains may be too high.
                elif abs(theta - theta_ref) < 0.035 and abs(xdot) < 0.08:  # Check if balance and speed are both close to zero.
                    status = "BALANCING"  # Report good standing balance.
                elif abs(xdot) > 1.0:  # Check if robot is running away.
                    status = "FAST MOTION"  # Tell user the likely fix.
                else:  # Otherwise the controller is still correcting.
                    status = "correcting"  # Report normal recovery behavior.
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
                        f"{status}"
)

            step += 1  # Increase the step counter.

    print("\nViewer closed.")  # Print message after the viewer is closed.


if __name__ == "__main__":  # Only run automatically when this file is executed directly.
    run()  # Start the simulation and controller loop.
