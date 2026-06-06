"""
╔══════════════════════════════════════════════════════════════════╗
║  SeBaJu — CASCADE PID  
║  Author: Kennedy CHUKWUMA
╚══════════════════════════════════════════════════════════════════╝
"""

import mujoco
import mujoco.viewer
import math
import time
import os
import numpy as np

XML_FILE    = "SeBaJu_BOT.xml"
SLOW_FACTOR = 0.5   # 0.5 = half speed for easier observation, 1.0 = real-time, 2.0 = double speed, etc.

GYRO_FILTER = 0.80 #0.70  # 0.15 = more smoothing, less noise (but more lag); 0.0 = no filtering, raw gyro data (more noise), 1.0 = full filtering, gyro rate is always 0 (not useful, frozen in time)
VEL_FILTER = 0.60 # 0.20 = more smoothing, less noise (but more lag); 0.0 = no filtering, raw velocity data (more noise), 1.0 = full filtering, velocity is always 0 (not useful, frozen in time)

#=═════════════════════════════════════════════════════════════════
# MOVEMENT PROFILE CONFIGURATION
#=═════════════════════════════════════════════════════════════════
X_TARGET_FINAL = 0.00    # m — target position to reach
MOVE_VELOCITY   = 0.10  # m/s — how fast to move forward (positive) or backward (negative)
#x_target = 0.0  # Desired position (m) — we want to stay at x=0 until we decide to move
Y_TARGET = 0.0   # metres - want to stay centered on y=0 (no sideways movement)


#==═════════════════════════════════════════════════════════════════
# CASCADE PID CONTROLLER GAINS AND CONFIG
#==═════════════════════════════════════════════════════════════════
KP_YAW = 0.0 #0.15 #0.8  # turning correction gain
MAX_YAW_TURN = 0.05 #0.15  # max additional wheel command for yaw correction (to prevent over-correction that could destabilize the robot)
Y_DEADBAND = 0.03  # m - if within this distance from y=0, consider it "close enough" and don't apply yaw correction to avoid unnecessary oscillations
YAW_FILTER = 0.85

# ══════════════════════════════════════════════════════════════════
# OUTER LOOP — VELOCITY CONTROLLER
# Converts velocity error → desired pitch correction
# ══════════════════════════════════════════════════════════════════
V_REF                = 0.0    # m/s — target velocity (0 = stationary), increase or decrease to test forward or backward movement
OUTER_KP             = 0.03 #0.03   # rad lean per m/s error  [was 0.04 — doubled]
OUTER_KD             = 0.07 #0.07   # slow integral for persistent drift [was 0.002]
#OUTER_MAX_INTEGRAL   = 0.05
MAX_PITCH_CORRECTION = 0.07 #0.10   # rad — max lean outer loop can command [was 0.15]

# ══════════════════════════════════════════════════════════════════
# INNER LOOP — BALANCE CONTROLLER
# Converts pitch error → wheel torque
# ══════════════════════════════════════════════════════════════════
PITCH_OFFSET       = -0.033 #-0.0333 #-0.025 #-0.035  # rad — natural equilibrium lean [was -0.04]
INNER_KP           = 0.50    # confirmed working — do not change
INNER_KD           = 0.025 #0.03 #0.018   # confirmed working — do not change
INNER_KI           = 0.005   # REDUCED from 0.02 — was causing integral windup
INNER_MAX_INTEGRAL = 0.03    # REDUCED from 0.05 — tighter anti-windup

# ══════════════════════════════════════════════════════════════════
# HIP AND KNEE POSTURE
# ══════════════════════════════════════════════════════════════════
HIP_KP  = 50.0 #25.0 
HIP_KD  = 6.0 #3.8
KNEE_KP = 45.0 #22.0 
KNEE_KD = 5.0 #2.5

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
    
    #===================================================
    # INITIALISING KEY PARAMETERS FOR THE CONTROL LOOPS
    #===================================================
    dt             = model.opt.timestep
    #outer_integral = 0.0
    inner_integral = 0.0
    step           = 0
    x_target = 0.0  # Desired position (m) — we want to stay at x=0
    pitch_target = PITCH_OFFSET  # Initial target pitch (rad) — will be updated by outer loop
    pitch_rate_filtered = 0.0  # For gyro filtering
    velocity_filtered = 0.0  # For velocity filtering
    yaw_filtered = 0.0 
    

    print(f"\nCASCADE PID (FIXED v2):")
    print(f"  Outer: Kp={OUTER_KP}, Kd={OUTER_KD}  (position + velocity → pitch target)")
    print(f"  Inner: Kp={INNER_KP}, Kd={INNER_KD}, Ki={INNER_KI}  (pitch → wheel torque)")
    print(f"  PITCH_OFFSET = {PITCH_OFFSET:.4f} rad ({math.degrees(PITCH_OFFSET):.2f}°)")
    print("─" * 80)
    print(f"{'Time':>6}  {'Pitch':>9}  {'Rate':>9}  {'Vel':>11}  {'PitchTgt':>11}  {'WheelU':>9}  {'XCurrent':>10}  {'YCurrent':>10}  {'Status'}")
    print("─" * 80)

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
            pitch_rate = pitch_rate_filtered  # Use the filtered gyro rate for control

            omega_L    = float(sd[11])
            omega_R    = float(sd[13])
            # velocity   = (omega_L + omega_R) / 2.0 * 0.06  # m/s, positive=forward # replaced by filtered version below
            velocity_raw = (omega_L + omega_R) / 2.0 * 0.06  # m/s, positive=forward
            velocity_filtered = VEL_FILTER * velocity_filtered + (1 - VEL_FILTER) * velocity_raw
            velocity = velocity_filtered  # Use the filtered velocity for control

            #=═════════════════════════════════════════════════════════════════
            # WITHIN MAIN LOOP: UPDATED VARIABLES AND CONTROL LOGIC
            #=═════════════════════════════════════════════════════════════════
            active_v_ref = V_REF 
            yaw_turn  = 0.00 
            #x_target = 0.0  # Desired position (m) — we want to stay at x=0
            x_current = float(data.qpos[root_qadr])  # Current x position of the robot's base
            y_current = float(data.qpos[root_qadr + 1])  # Current y position of the robot's base

            #=═════════════════════════════════════════════════════════════════
            # MOVEMENT PROFILE CONFIGURATION IMPLEMENTATION
            #=═════════════════════════════════════════════════════════════════
            # Slowly advance x_target toward X_TARGET_FINAL at the specified MOVE_VELOCITY

            if abs(x_target - X_TARGET_FINAL) > 0.001: # Yet to arrive at target
                step_size = MOVE_VELOCITY * dt         # How much to move x_target this timestep
                if X_TARGET_FINAL > x_target:
                    x_target = min(x_target + step_size, X_TARGET_FINAL)  # Move but don't overshoot
                else:
                    x_target = max(x_target - step_size, X_TARGET_FINAL)  # Move but don't overshoot
                    #Once x_target reaches X_TARGET_FINAL, it will stay there and we can observe how the robot maintains balance at the new position.

            
            # Y correction - Yaw control to keep the robot centered on the y-axis

            y_error = Y_TARGET - y_current
            if abs(y_error) < Y_DEADBAND:
                yaw_turn = 0.0  # Within deadband, no correction
            else:   
                yaw_turn_raw = clamp(KP_YAW * y_error, -MAX_YAW_TURN, MAX_YAW_TURN)  # Simple proportional control to keep the robot centered on the y-axis
                # Apply filtering to the yaw turn command to prevent oscillations
                yaw_filtered = YAW_FILTER * yaw_filtered + (1 - YAW_FILTER) * yaw_turn_raw
                yaw_turn = yaw_filtered  # Use the filtered yaw turn command
            
            
            # ══════════════════════════════════════════════════════════
            # OUTER LOOP — position + velocity error → pitch target
            # Changed Outer PI to PD
            # ══════════════════════════════════════════════════════════
            vel_error = active_v_ref - velocity
            # vel_error < 0 means moving backward faster than desired
            # → pitch_correction negative → lean forward → wheels brake backward drift

            pos_error = x_target - x_current

            #outer_integral = clamp(
            #    outer_integral + vel_error * dt,
            #    -OUTER_MAX_INTEGRAL, OUTER_MAX_INTEGRAL
            #)

            #pitch_correction = clamp(
            #    OUTER_KP * vel_error + OUTER_KI * outer_integral,
            #    -MAX_PITCH_CORRECTION, MAX_PITCH_CORRECTION
            #)

            pitch_correction = clamp(
                OUTER_KP * pos_error + OUTER_KD * vel_error,
                -MAX_PITCH_CORRECTION, MAX_PITCH_CORRECTION
            )
            #=═════════════════════════════════════════════════════════
            # pitch_target = PITCH_OFFSET + pitch_correction
            # Adding a rate limiter to the pitch_target to prevent sudden large changes that could destabilize the robot
            #=═════════════════════════════════════════════════════════
            
            raw_target = PITCH_OFFSET + pitch_correction

            # rate limiter - pitch_target can only change by a certain amount per timestep = 0.002s (0.5 deg)
            MAX_TARGET_RATE = math.radians(0.5)  # max change in target pitch per second
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
              + INNER_KD * pitch_rate
              + INNER_KI * inner_integral,
                -MAX_WHEEL, MAX_WHEEL
            )

            #Remove yaw_turn from wheel commands to isolate the effect of the cascade PID controller on balance first. We can reintroduce yaw_turn later once the balance control is stable.
            data.ctrl[4] = clamp(wheel_u, -MAX_WHEEL, MAX_WHEEL)
            data.ctrl[5] = clamp(wheel_u, -MAX_WHEEL, MAX_WHEEL)

            # ── HIP POSTURE ───────────────────────────────────────────
            for jname, ctrl_idx in [("left_hip", 0), ("right_hip", 1)]:
                err = data.qpos[qadr[jname]]
                vel = data.qvel[vadr[jname]]
                data.ctrl[ctrl_idx] = clamp(-HIP_KP*err - HIP_KD*vel, -MAX_HIP, MAX_HIP)

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


# ══════════════════════════════════════════════════════════════════
# WHAT TO WATCH IN THE OUTPUT & WHAT TO DO NEXT
# ══════════════════════════════════════════════════════════════════
#
# HEALTHY SIGNS after t=3s:
#   Pitch:    between -4° and +1°  (small oscillation around offset)
#   Rate:     < ±30°/s             (was ±120°/s before — that was bad)
#   Vel:      drifting toward 0, stabilising within ±0.15 m/s by t=8s
#   PitchTgt: close to -2.0° and slowly converging (not clamping at -10°!)
#   WheelU:   < ±0.3               (not saturating)
#
# IF ROBOT STILL FALLS BACKWARD (pitch going to -10° or worse):
#   → PITCH_OFFSET is too negative. Change to -0.02 and retry.
#
# IF ROBOT OSCILLATES PITCH ±5° rapidly:
#   → INNER_KD too low. Try 0.022 then 0.025.
#
# IF VELOCITY DRIFTS TO -0.5 m/s AND STAYS THERE:
#   → OUTER_KP too low. Try 0.12.
#
# IF VELOCITY OSCILLATES (positive then negative repeatedly):
#   → OUTER_KP too high. Try 0.05.
#
# TUNING SEQUENCE FOR YOUR MEETING:
#   Run as-is first. If pitch stays within ±5° at t=3s → good start.
#   Only change ONE gain at a time. Wait 5 seconds before judging.
