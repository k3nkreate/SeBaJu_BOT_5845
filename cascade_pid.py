"""
╔══════════════════════════════════════════════════════════════════╗
║  SeBaJu — CASCADE PID  (FIXED v2)                              ║
║  Key fixes vs previous version:                                  ║
║    1. INNER_KI reduced from 0.02 → 0.005  (was winding up)     ║
║    2. OUTER_KP increased from 0.04 → 0.08  (was too weak)      ║
║    3. OUTER_KI reduced from 0.002 → 0.001  (slow and safe)     ║
║    4. MAX_PITCH_CORRECTION reduced 0.15 → 0.10  (safer limit)  ║
║    5. PITCH_OFFSET tuned to -0.035 (slightly less backward)     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import mujoco
import mujoco.viewer
import math
import time
import os
import numpy as np

XML_FILE    = "SeBaJu_BOT.xml"
SLOW_FACTOR = 0.5   # 0.5 = half speed for easier observation

# ══════════════════════════════════════════════════════════════════
# OUTER LOOP — VELOCITY CONTROLLER
# Converts velocity error → desired pitch correction
# ══════════════════════════════════════════════════════════════════
V_REF                = 0.0    # m/s — target velocity (0 = stationary)
OUTER_KP             = 0.08   # rad lean per m/s error  [was 0.04 — doubled]
OUTER_KI             = 0.003 #0.005 #0.001  # slow integral for persistent drift [was 0.002]
OUTER_MAX_INTEGRAL   = 0.05
MAX_PITCH_CORRECTION = 0.10   # rad — max lean outer loop can command [was 0.15]

# ══════════════════════════════════════════════════════════════════
# INNER LOOP — BALANCE CONTROLLER
# Converts pitch error → wheel torque
# ══════════════════════════════════════════════════════════════════
PITCH_OFFSET       = -0.033 #-0.0333 #-0.025 #-0.035  # rad — natural equilibrium lean [was -0.04]
INNER_KP           = 0.50    # confirmed working — do not change
INNER_KD           = 0.025 #0.018   # confirmed working — do not change
INNER_KI           = 0.005   # REDUCED from 0.02 — was causing integral windup
INNER_MAX_INTEGRAL = 0.03    # REDUCED from 0.05 — tighter anti-windup

# ══════════════════════════════════════════════════════════════════
# HIP AND KNEE POSTURE
# ══════════════════════════════════════════════════════════════════
HIP_KP  = 25.0;  HIP_KD  = 3.8
KNEE_KP = 22.0;  KNEE_KD = 2.5

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


class KeyboardController:
    V_CMD     = 0.3
    TURN_RATE = 0.15

    def __init__(self):
        self.v_ref = 0.0
        self.turn  = 0.0

    def update(self, viewer):
        ks = getattr(viewer, 'key_state', {})
        if   ks.get('W', False) or ks.get('w', False): self.v_ref = +self.V_CMD
        elif ks.get('S', False) or ks.get('s', False): self.v_ref = -self.V_CMD
        else:                                           self.v_ref =  0.0
        if   ks.get('A', False) or ks.get('a', False): self.turn  = +self.TURN_RATE
        elif ks.get('D', False) or ks.get('d', False): self.turn  = -self.TURN_RATE
        else:                                           self.turn  =  0.0


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

    jnames = ["left_hip","right_hip","left_knee","right_knee",
              "left_wheel_joint","right_wheel_joint"]
    jid  = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jnames}
    qadr = {n: model.jnt_qposadr[jid[n]] for n in jnames}
    vadr = {n: model.jnt_dofadr [jid[n]] for n in jnames}

    dt             = model.opt.timestep
    outer_integral = 0.0
    inner_integral = 0.0
    step           = 0
    keyboard       = KeyboardController()

    print(f"\nCASCADE PID (FIXED v2):")
    print(f"  Outer: Kp={OUTER_KP}, Ki={OUTER_KI}  (velocity → pitch target)")
    print(f"  Inner: Kp={INNER_KP}, Kd={INNER_KD}, Ki={INNER_KI}  (pitch → wheel torque)")
    print(f"  PITCH_OFFSET = {PITCH_OFFSET:.4f} rad ({math.degrees(PITCH_OFFSET):.2f}°)")
    print(f"\nControls: W=Forward  S=Backward  A=TurnLeft  D=TurnRight")
    print("─" * 80)
    print(f"{'Time':>6}  {'Pitch':>9}  {'Rate':>9}  {'Vel':>11}  {'PitchTgt':>9}  {'WheelU':>7}  {'Status'}")
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
            pitch_rate = float(sd[5])
            omega_L    = float(sd[11])
            omega_R    = float(sd[13])
            velocity   = (omega_L + omega_R) / 2.0 * 0.06  # m/s, positive=forward

            keyboard.update(viewer)
            active_v_ref = V_REF + keyboard.v_ref
            active_turn  = keyboard.turn

            # ══════════════════════════════════════════════════════════
            # OUTER LOOP — velocity → pitch target
            # ══════════════════════════════════════════════════════════
            vel_error = active_v_ref - velocity
            # vel_error < 0 means moving backward faster than desired
            # → pitch_correction negative → lean forward → wheels brake backward drift

            outer_integral = clamp(
                outer_integral + vel_error * dt,
                -OUTER_MAX_INTEGRAL, OUTER_MAX_INTEGRAL
            )

            pitch_correction = clamp(
                OUTER_KP * vel_error + OUTER_KI * outer_integral,
                -MAX_PITCH_CORRECTION, MAX_PITCH_CORRECTION
            )

            pitch_target = PITCH_OFFSET + pitch_correction

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

            data.ctrl[4] = clamp(wheel_u + active_turn, -MAX_WHEEL, MAX_WHEEL)
            data.ctrl[5] = clamp(wheel_u - active_turn, -MAX_WHEEL, MAX_WHEEL)

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
