import mujoco
import mujoco.viewer
import math
import time
import os
import numpy as np

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

XML_FILE    = "simple_box_robot.xml"
SLOW_FACTOR = 1.0   # 0.5 = half speed, 1.0 = real-time

BASE_HEIGHT = 0.330  # box CoM height above ground (for camera tracking)

# ── Filters ───────────────────────────────────────────────────────
GYRO_FILTER = 0.80   # 0 = no filter (raw), 1 = frozen; 0.80 is a good start
VEL_FILTER  = 0.60   # same scale as GYRO_FILTER

# ── Pitch (balance) loop ──────────────────────────────────────────
KP_PITCH    = 0.325  # ctrl / rad      — start here, increase if falls too easily
KD_PITCH    = 0.80   # ctrl / (rad/s)  — damps oscillation

# ── Velocity trim ─────────────────────────────────────────────────
KP_VEL      = 0.005  # ctrl / (m/s)   — slow drift correction

# ── Pitch offset ─────────────────────────────────────────────────
# Box is symmetric so CoM is exactly over axle → 0.0 is correct.
# If robot drifts forward  → make slightly negative
# If robot drifts backward → make slightly positive
PITCH_OFFSET = 0.001   # rad

# ── Actuator limits ───────────────────────────────────────────────
MAX_WHEEL = 1.0

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def quat_to_pitch(qw, qx, qy, qz):
    """Extract pitch (Y-axis rotation) from quaternion. + = lean forward."""
    return math.asin(float(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)))

def clamp(value, lo, hi):
    return float(np.clip(value, lo, hi))

# ══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════

def run():
    xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), XML_FILE)
    if not os.path.exists(xml_path):
        print(f"ERROR: XML not found: {xml_path}")
        raise SystemExit(1)

    print(f"\nLoading: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    # ── Reset to keyframe ─────────────────────────────────────────
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kid)
        print("Keyframe 'stand' applied.")

    # ── Root joint address (for position tracking) ────────────────
    root_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_qadr = model.jnt_qposadr[root_id]

    # ── Actuator indices ──────────────────────────────────────────
    left_wheel_ctrl  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_motor")
    right_wheel_ctrl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_motor")

    # ── Sensor addresses ──────────────────────────────────────────
    quat_adr       = int(model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_orientation")])
    gyro_adr       = int(model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")])
    lwheel_vel_adr = int(model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "left_wheel_vel")])
    rwheel_vel_adr = int(model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "right_wheel_vel")])

    # ── Wheel radius from geom ────────────────────────────────────
    geom_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_geom")
    wheel_radius = float(model.geom_size[geom_id, 0])

    # ── Sim parameters ────────────────────────────────────────────
    dt   = model.opt.timestep
    step = 0

    # ── Filter state ──────────────────────────────────────────────
    pitch_rate_filtered = 0.0
    velocity_filtered   = 0.0

    print(f"  dt           = {dt*1000:.1f} ms")
    print(f"  wheel_radius = {wheel_radius*100:.1f} cm")
    print(f"  PITCH_OFFSET = {PITCH_OFFSET:.4f} rad ({math.degrees(PITCH_OFFSET):.2f}°)")
    print("─" * 90)
    print(f"{'Time':>6}  {'Pitch':>9}  {'Rate':>9}  {'Vel':>11}  {'WheelU':>9}  {'X':>8}  {'Y':>8}  {'Status'}")
    print("─" * 90)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type      = 0
        viewer.cam.azimuth   = 160
        viewer.cam.elevation = -20
        viewer.cam.distance  = 3.5

        while viewer.is_running():

            # ── READ SENSORS ──────────────────────────────────────
            sd = data.sensordata

            qw = sd[quat_adr + 0]
            qx = sd[quat_adr + 1]
            qy = sd[quat_adr + 2]
            qz = sd[quat_adr + 3]
            pitch = quat_to_pitch(qw, qx, qy, qz)

            # Gyro Y = pitch rate
            pitch_rate_raw      = float(sd[gyro_adr + 1])
            pitch_rate_filtered = GYRO_FILTER * pitch_rate_filtered + (1 - GYRO_FILTER) * pitch_rate_raw
            pitch_rate          = pitch_rate_filtered

            # Wheel velocities → chassis speed
            omega_L           = float(sd[lwheel_vel_adr])
            omega_R           = float(sd[rwheel_vel_adr])
            velocity_raw      = (omega_L + omega_R) / 2.0 * wheel_radius   # m/s
            velocity_filtered = VEL_FILTER * velocity_filtered + (1 - VEL_FILTER) * velocity_raw
            velocity          = velocity_filtered

            # Robot XY position (from free joint)
            x_current = float(data.qpos[root_qadr])
            y_current = float(data.qpos[root_qadr + 1])

            # ── CAMERA TRACKING ───────────────────────────────────
            #viewer.cam.lookat[:] = [x_current, y_current, BASE_HEIGHT]

            # ── PITCH CONTROL → WHEEL COMMAND ────────────────────
            pitch_error = pitch - PITCH_OFFSET

            wheel_u = clamp(
                KP_PITCH * pitch_error
              + KD_PITCH * pitch_rate
              + KP_VEL   * velocity,
                -MAX_WHEEL, MAX_WHEEL
            )

            data.ctrl[left_wheel_ctrl]  = wheel_u
            data.ctrl[right_wheel_ctrl] = wheel_u

            # ── STEP PHYSICS ──────────────────────────────────────
            mujoco.mj_step(model, data)
            time.sleep(dt * SLOW_FACTOR)
            viewer.sync()

            # ── DIAGNOSTICS ───────────────────────────────────────
            if step % 250 == 0:
                direction = "FWD" if pitch > 0 else "BCK"
                if abs(wheel_u) >= 0.99:
                    status = "SATURATED"
                elif abs(pitch_error) < 0.025 and abs(velocity) < 0.05:
                    status = "STATIONARY ✓"
                elif abs(pitch_error) < 0.04:
                    status = "balanced (moving)"
                elif abs(wheel_u) < 0.15:
                    status = "correcting..."
                else:
                    status = "recovering..."

                print(
                    f"{data.time:6.2f}s  "
                    f"{math.degrees(pitch):+7.2f}°{direction}  "
                    f"{math.degrees(pitch_rate):+7.1f}°/s  "
                    f"vel={velocity:+7.3f}m/s  "
                    f"{wheel_u:+6.3f}  "
                    f"x={x_current:+6.3f}m  "
                    f"y={y_current:+6.3f}m  "
                    f"{status}"
                )

            step += 1

    print("\nViewer closed.")


if __name__ == "__main__":
    run()
