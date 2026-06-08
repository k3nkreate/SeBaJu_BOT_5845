"""
╔══════════════════════════════════════════════════════════════════╗
║  SeBaJu — CASCADE PID  (3-layer: position → velocity → pitch)  ║
║  Author: Kennedy CHUKWUMA                                        ║
╚══════════════════════════════════════════════════════════════════╝
 
ARCHITECTURE:
  OUTER  : position error  → velocity command   (PD)
  MIDDLE : velocity error  → pitch target        (PI)
  INNER  : pitch error     → wheel torque        (PID)
  YAW    : y-position error → differential wheel torque (PD)
"""

import mujoco
import mujoco.viewer
import math
import time
import os
import numpy as np

#=═════════════════════════════════════════════════════════════════
# SIMULATION: CONFIGURATION ON HOW THE XML MODEL IS INTERPRETED AND CONTROLLED
#=═════════════════════════════════════════════════════════════════
XML_FILE    = "SeBaJu_BOT.xml"
SLOW_FACTOR = 0.5   # 0.5 = half speed for easier observation, 1.0 = real-time, 2.0 = double speed, etc.


#=═════════════════════════════════════════════════════════════════
# SENSOR (GYRO, VELOCITY, DERIVATIVE) FILTERING CONFIGURATION
#=═════════════════════════════════════════════════════════════════
GYRO_FILTER = 0.80 # pitch rate — used for display only
VEL_FILTER = 0.60 # 0.20 = more smoothing, less noise (but more lag); 0.0 = no filtering, raw velocity data (more noise), 1.0 = full filtering, velocity is always 0 (not useful, frozen in time)
DERIV_FILTER = 0.92   # heavier filter used only for KD term, reduces chatter/jerking


# ══════════════════════════════════════════════════════════════════
# TARGET POSITION
# ══════════════════════════════════════════════════════════════════
X_TARGET_FINAL = 0.00   # m  — where robot should stand
MOVE_VELOCITY  = 0.10   # m/s — ramp speed toward X_TARGET_FINAL
Y_TARGET       = 0.00   # m  — lateral centre line


# ══════════════════════════════════════════════════════════════════
# OUTER LOOP (PD) — position error → velocity target
# ══════════════════════════════════════════════════════════════════
POS_KP               = 0.88 #0.40    # m/s per metre of position error
POS_KD               = 0.56 #0.10    # m/s per m/s of velocity (damping)
MAX_VEL_CMD          = 0.15 #0.30    # m/s — max velocity the position loop can command


# ══════════════════════════════════════════════════════════════════
# MIDDLE LOOP (PI) — velocity error → pitch target (using velocity target from position (OUTER) loop )
# ══════════════════════════════════════════════════════════════════
VEL_KP               = 0.076 #0.06    # rad per m/s of velocity error
VEL_KI               = 0.034 #0.008   # rad/(m/s·s) — eliminates residual velocity offset
VEL_MAX_INTEGRAL     = 0.04    # rad·s — anti-windup cap
MAX_PITCH_CORRECTION = 0.10    # rad — clamp on pitch target offset
MAX_TARGET_RATE      = math.radians(0.50) #1.0  # rad/step — rate limiter


# ══════════════════════════════════════════════════════════════════
# INNER LOOP — BALANCE CONTROLLER (PID):
# Converts pitch error → wheel torque (using pitch target from middle loop)
# ══════════════════════════════════════════════════════════════════
PITCH_OFFSET       = -0.033 #-0.0333 #-0.025 #-0.035  # rad — natural equilibrium lean [was -0.04]
INNER_KP           = 0.44 #0.50    # confirmed working — do not change
INNER_KD           = 0.035 #0.025 #0.03 #0.018   # confirmed working — do not change
INNER_KI           = 0.005   # REDUCED from 0.02 — was causing integral windup
INNER_MAX_INTEGRAL = 0.03    # REDUCED from 0.05 — tighter anti-windup


# ══════════════════════════════════════════════════════════════════
# YAW CORRECTION — y-position error → differential wheel torque (PD)
# ══════════════════════════════════════════════════════════════════
KP_YAW       = 0.04 #0.06 #0.02 #0.08 #0.15 #0.8   # wheel torque delta per m of y error
YAW_KD       = 0.012 #0.008 #0.02 #0.05   # wheel torque delta per rad/s of yaw rate
Y_DEADBAND   = 0.03 #0.10    # m  — no correction within this radius
MAX_YAW_TURN = 0.015 #0.04 #0.03 #0.05  # max differential torque added to wheels


# ══════════════════════════════════════════════════════════════════
# HIP AND KNEE POSTURE
# ══════════════════════════════════════════════════════════════════
HIP_KP  = 28.0 #50.0 #25.0 
HIP_KD  = 4.0 #6.0 #3.8
KNEE_KP = 25.0 #45.0 #22.0 
KNEE_KD = 3.0 #5.0 #2.5

MAX_WHEEL = 1.0
MAX_HIP   = 1.0
MAX_KNEE  = 1.0


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def quat_to_pitch(qw, qx, qy, qz):
    return math.asin(float(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)))

def clamp(value, lo, hi):
    return float(np.clip(value, lo, hi))


# ══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════

def run():
    xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), XML_FILE)
    if not os.path.exists(xml_path):
        print(f"ERROR: XML not found: {xml_path}"); raise SystemExit(1)

    print(f"\nLoading: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    

    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kid)
        print("Keyframe 'stand' applied.")


    # Root joint is free-floating base, not actuated. We read its position to track the robot's position.
    root_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_qadr = model.jnt_qposadr[root_id]


    jnames = ["left_hip","right_hip","left_knee","right_knee",
              "left_wheel_joint","right_wheel_joint"]
    jid  = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jnames}
    qadr = {n: model.jnt_qposadr[jid[n]] for n in jnames}
    vadr = {n: model.jnt_dofadr [jid[n]] for n in jnames}
    

    #=═════════════════════════════════════════════════════════════════
    # INITIAL STATES AND CONTROL VARIABLES
    #=═════════════════════════════════════════════════════════════════
    dt                  = model.opt.timestep
    step                = 0
    x_target            = 0.0
    pitch_target        = PITCH_OFFSET
    pitch_rate_filtered = 0.0
    pitch_rate_deriv    = 0.0
    velocity_filtered   = 0.0
    vel_integral        = 0.0
    inner_integral      = 0.0
    yaw_heading = 0.0 
    
    #=═════════════════════════════════════════════════════════════════
    # PRINT CONFIGURATION SUMMARY
    #=═════════════════════════════════════════════════════════════════
    print(f"\nCASCADE PID - 3-LAYER: position → velocity → pitch")
    print(f"  Outer: Kp={POS_KP}, Kd={POS_KD}  (position error → velocity target)")
    print(f"  Middle: Kp={VEL_KP}, Ki={VEL_KI}  (velocity error → pitch target)")
    print(f"  Inner: Kp={INNER_KP}, Kd={INNER_KD}, Ki={INNER_KI}  (pitch error → wheel torque)")
    print(f"  PITCH_OFFSET = {PITCH_OFFSET:.4f} rad ({math.degrees(PITCH_OFFSET):.2f}°)")
    print("─" * 88)
    print(f"{'Time':>6}  {'Pitch':>9}  {'Rate':>9}  {'Vel':>11}  {'VelCmd':>8}  {'PitchTgt':>11}  {'WheelU':>9}  {'XCurrent':>10}  {'YCurrent':>10}  {'Status'}")
    print("─" * 88)

    #=═════════════════════════════════════════════════════════════════
    # LAUNCHING THE VIEWER AND STARTING THE CONTROL LOOP
    #=═════════════════════════════════════════════════════════════════
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth   = 90
        viewer.cam.elevation = -10
        viewer.cam.distance  = 1.8

        while viewer.is_running():

            # ── READ SENSORS ──────────────────────────────────────────
            sd = data.sensordata
            qw, qx, qy, qz = sd[0], sd[1], sd[2], sd[3]
            pitch      = quat_to_pitch(qw, qx, qy, qz)

            #pitch_rate = float(sd[5]) # replaced by filtered version below
            #==═════════════════════════════════════════════════════════════════
            # GYRO FILTERING IMPLEMENTATION [reference: https://en.wikipedia.org/wiki/Low-pass_filter#Exponential_moving_average]
            #==═════════════════════════════════════════════════════════════════
            pitch_rate_raw = float(sd[5])
            pitch_rate_filtered = GYRO_FILTER * pitch_rate_filtered + (1 - GYRO_FILTER) * pitch_rate_raw
            pitch_rate_deriv    = DERIV_FILTER * pitch_rate_deriv   + (1 - DERIV_FILTER) * pitch_rate_raw
            pitch_rate = pitch_rate_filtered  # The filtered gyro rate is used for console display only, the heavily-filtered version is used for the KD term in the inner loop to reduce chatter/jerking

            yaw_rate = float(sd[6]) # rad/s, positive = turning left, negative = turning right

            omega_L    = float(sd[11])
            omega_R    = float(sd[13])
            # velocity   = (omega_L + omega_R) / 2.0 * 0.06  # m/s, positive=forward # replaced by filtered version below
            velocity_raw = (omega_L + omega_R) / 2.0 * 0.06  # m/s, positive=forward
            velocity_filtered = VEL_FILTER * velocity_filtered + (1 - VEL_FILTER) * velocity_raw
            velocity = velocity_filtered  # Use the filtered velocity for control

            #=═════════════════════════════════════════════════════════════════
            # WITHIN MAIN LOOP: UPDATED VARIABLES AND CONTROL LOGIC
            #=═════════════════════════════════════════════════════════════════
            x_current = float(data.qpos[root_qadr])  # Current x position of the robot's base
            y_current = float(data.qpos[root_qadr + 1])  # Current y position of the robot's base

            #=═════════════════════════════════════════════════════════════════
            # RAMP x_target TOWARD X_TARGET_FINAL
            #=═════════════════════════════════════════════════════════════════
            # Slowly advance x_target toward X_TARGET_FINAL at the specified MOVE_VELOCITY

            if abs(x_target - X_TARGET_FINAL) > 0.001: # Yet to arrive at target
                step_size = MOVE_VELOCITY * dt         # How much to move x_target this timestep
                if X_TARGET_FINAL > x_target:
                    x_target = min(x_target + step_size, X_TARGET_FINAL)  # Move but don't overshoot
                else:
                    x_target = max(x_target - step_size, X_TARGET_FINAL)  # Move but don't overshoot
                    #Once x_target reaches X_TARGET_FINAL, it will stay there and we can observe how the robot maintains balance at the new position.


            # ══════════════════════════════════════════════════════════════════
            # OUTER LOOP — position error → velocity command (PD)
            # ══════════════════════════════════════════════════════════════════
            pos_error    = x_target - x_current
            vel_cmd      = clamp(POS_KP * pos_error - POS_KD * velocity,
                                 -MAX_VEL_CMD, MAX_VEL_CMD)

            
            # ══════════════════════════════════════════════════════════════════
            # MIDDLE LOOP — velocity error → pitch target (PI)
            # ══════════════════════════════════════════════════════════════════════════════
            vel_error = vel_cmd - velocity
            
            vel_integral = clamp(
                vel_integral + vel_error * dt,
                -VEL_MAX_INTEGRAL, VEL_MAX_INTEGRAL
            )
            # Anti-windup bleed when settled
            if abs(pos_error) < 0.04 and abs(velocity) < 0.04:
                vel_integral *= 0.97

            pitch_correction = clamp(
                VEL_KP * vel_error + VEL_KI * vel_integral, 
                -MAX_PITCH_CORRECTION, MAX_PITCH_CORRECTION
                )

            # Rate limiter on pitch target
            raw_target   = PITCH_OFFSET + pitch_correction
            if raw_target > pitch_target + MAX_TARGET_RATE:
                pitch_target = pitch_target + MAX_TARGET_RATE
            elif raw_target < pitch_target - MAX_TARGET_RATE:
                pitch_target = pitch_target - MAX_TARGET_RATE
            else:
                pitch_target = raw_target
            
            
            # ══════════════════════════════════════════════════════════
            # INNER LOOP — pitch error → wheel torque
            # ══════════════════════════════════════════════════════════
            pitch_err = pitch - pitch_target

            inner_integral = clamp(
                inner_integral + pitch_err * dt,
                -INNER_MAX_INTEGRAL, INNER_MAX_INTEGRAL
            )

            wheel_u = clamp(
                INNER_KP * pitch_err
              + INNER_KD * pitch_rate_deriv    # ← was pitch_rate, but using a separate heavily-filtered version for the derivative term to reduce chatter/jerking
              + INNER_KI * inner_integral,
                -MAX_WHEEL, MAX_WHEEL
            )

            # ══════════════════════════════════════════════════════
            # YAW HEADING HOLD — integrate yaw rate to track heading
            # Controls heading angle (not Y position) to avoid leg twist
            # ══════════════════════════════════════════════════════
            yaw_heading += yaw_rate * dt
            yaw_correction = clamp(
                -0.03 * yaw_heading - 0.01 * yaw_rate, 
                -0.020, 0.020
                )

            data.ctrl[4] = clamp(wheel_u + yaw_correction, -MAX_WHEEL, MAX_WHEEL)
            data.ctrl[5] = clamp(wheel_u - yaw_correction, -MAX_WHEEL, MAX_WHEEL)

            # ── HIP POSTURE ───────────────────────────────────────────
            for jname, ctrl_idx in [("left_hip", 0), ("right_hip", 1)]:
                err = data.qpos[qadr[jname]]
                vel = data.qvel[vadr[jname]]
                hip_cmd = -HIP_KP*err - HIP_KD*vel 
                data.ctrl[ctrl_idx] = clamp(hip_cmd, -MAX_HIP, MAX_HIP)

            # ── KNEE POSTURE ──────────────────────────────────────────
            for jname, ctrl_idx in [("left_knee", 2), ("right_knee", 3)]:
                err = data.qpos[qadr[jname]]
                vel = data.qvel[vadr[jname]]
                data.ctrl[ctrl_idx] = clamp(-KNEE_KP*err - KNEE_KD*vel, -MAX_KNEE, MAX_KNEE)

            # ── STEP PHYSICS ──────────────────────────────────────────
            mujoco.mj_step(model, data)
            time.sleep(dt * SLOW_FACTOR)
            viewer.sync()

            # ── DIAGNOSTICS ───────────────────────────────────────────
            if step % 250 == 0:
                direction = "FWD" if pitch > 0 else "BCK"
                if abs(wheel_u) >= 0.99:
                    status = "SATURATED"
                elif abs(pitch_err) < 0.025 and abs(velocity) < 0.05:
                    status = "STATIONARY ✓"
                elif abs(pitch_err) < 0.04:
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
                    f"vc={vel_cmd:+6.3f}  "
                    f"tgt={math.degrees(pitch_target):+6.2f}°  "
                    f"{wheel_u:+6.3f}  "
                    f"x={x_current:+6.3f}m  "
                    f"y={y_current:+6.3f}m  "
                    f"{status}"
                )

            step += 1

    print("\nViewer closed.")


if __name__ == "__main__":
    run()