"""
Contestant control software skeleton.

Session handshake order matters with udpin/udpout:
  1. Send one heartbeat so the sim's udpin socket learns our return address.
  2. Block on wait_heartbeat() to confirm the sim is up before commanding.
  3. Keep heartbeats alive in the background throughout the session.
  4. Drain telemetry in a dedicated thread — never block the control loop on recv.
"""

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, cast

os.environ["MAVLINK20"] = "1"

from pymavlink import mavutil  # noqa: E402
from pymavlink.dialects.v20 import common as mavlink_dialect  # noqa: E402

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------

HB_RATE_HZ: float = 2.0  # minimum; increase if the sim drops connection
COMMAND_RATE_HZ: float = 120.0  # sim physics runs at 120 Hz — match it
HEARTBEAT_TIMEOUT_S: float = 10.0


# ---------------------------------------------------------------------------
# Shared vehicle state
# ---------------------------------------------------------------------------


@dataclass
class VehicleState:
    # ATTITUDE
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    rollspeed: float = 0.0
    pitchspeed: float = 0.0
    yawspeed: float = 0.0

    # ODOMETRY — NED (m, m/s)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # HIGHRES_IMU
    xacc: float = 0.0
    yacc: float = 0.0
    zacc: float = 0.0
    xgyro: float = 0.0
    ygyro: float = 0.0
    zgyro: float = 0.0

    # Signed offset: sim_clock_ns - local_clock_ns. Use to correlate
    # sim timestamps in telemetry with your own timing if needed.
    time_offset_ns: Optional[int] = None

    # Flip these to False after consuming if you want edge-triggered logic.
    attitude_updated: bool = False
    odometry_updated: bool = False
    imu_updated: bool = False

    _lock: threading.Lock = field(default_factory=threading.Lock)


# ---------------------------------------------------------------------------
# Heartbeat thread
# ---------------------------------------------------------------------------


def start_heartbeat(master: mavutil.mavfile, rate_hz: float = HB_RATE_HZ) -> None:
    """
    Keeps the sim from considering us disconnected.

    Sleeps one full interval before the first send — the caller already sent
    the bootstrap heartbeat in main(), so firing immediately would double-up.
    """
    interval = 1.0 / rate_hz

    def _loop() -> None:
        while True:
            time.sleep(interval)
            master.mav.heartbeat_send(
                mavlink_dialect.MAV_TYPE_GCS,
                mavlink_dialect.MAV_AUTOPILOT_INVALID,
                0,
                0,
                mavlink_dialect.MAV_STATE_ACTIVE,
            )

    threading.Thread(target=_loop, daemon=True, name="heartbeat").start()


# ---------------------------------------------------------------------------
# Telemetry thread
# ---------------------------------------------------------------------------


def start_telemetry(master: mavutil.mavfile, state: VehicleState) -> None:
    """
    Blocking recv loop in its own thread — keeps the control loop free of I/O.

    All message types the sim emits are handled here. Unrecognised types are
    dropped silently; add cases as needed.
    """

    def _loop() -> None:
        while True:
            msg = master.recv_match(blocking=True, timeout=1.0)
            if msg is None:
                continue

            msg_type = msg.get_type()

            if msg_type == "ATTITUDE":
                with state._lock:
                    state.roll = msg.roll
                    state.pitch = msg.pitch
                    state.yaw = msg.yaw
                    state.rollspeed = msg.rollspeed
                    state.pitchspeed = msg.pitchspeed
                    state.yawspeed = msg.yawspeed
                    state.attitude_updated = True

            elif msg_type == "ODOMETRY":
                with state._lock:
                    state.x = msg.x
                    state.y = msg.y
                    state.z = msg.z
                    state.vx = msg.vx
                    state.vy = msg.vy
                    state.vz = msg.vz
                    state.odometry_updated = True

            elif msg_type == "HIGHRES_IMU":
                with state._lock:
                    state.xacc = msg.xacc
                    state.yacc = msg.yacc
                    state.zacc = msg.zacc
                    state.xgyro = msg.xgyro
                    state.ygyro = msg.ygyro
                    state.zgyro = msg.zgyro
                    state.imu_updated = True

            elif msg_type == "TIMESYNC":
                # ts1 is sim time in nanoseconds. Store the offset in case
                # you need to correlate your timestamps with sim-relative time.
                local_ns = int(time.time() * 1e9)
                with state._lock:
                    state.time_offset_ns = msg.ts1 - local_ns

            elif msg_type in ("HEARTBEAT", "BAD_DATA"):
                pass

    threading.Thread(target=_loop, daemon=True, name="telemetry").start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    master = cast(
        mavutil.mavfile,
        mavutil.mavlink_connection(
            "udpout:localhost:14540",
            source_system=255,
        ),
    )

    # udpin won't know where to send until it receives a datagram from us —
    # send one heartbeat to punch the return path open before blocking on recv.
    master.mav.heartbeat_send(
        mavlink_dialect.MAV_TYPE_GCS,
        mavlink_dialect.MAV_AUTOPILOT_INVALID,
        0,
        0,
        mavlink_dialect.MAV_STATE_ACTIVE,
    )

    print(f"[CTRL] Waiting for simulator heartbeat (timeout={HEARTBEAT_TIMEOUT_S}s)...")
    hb = master.wait_heartbeat(timeout=HEARTBEAT_TIMEOUT_S)
    if hb is None:
        raise RuntimeError(
            f"No heartbeat from simulator within {HEARTBEAT_TIMEOUT_S}s — "
            "is mock_vehicle.py running?"
        )
    print(
        f"[CTRL] Connected — "
        f"sysid={master.target_system} compid={master.target_component}"
    )

    _boot_time = time.time()

    def time_boot_ms() -> int:
        return int((time.time() - _boot_time) * 1000) & 0xFFFFFFFF

    start_heartbeat(master, rate_hz=HB_RATE_HZ)

    state = VehicleState()
    start_telemetry(master, state)

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------
    loop_interval = 1.0 / COMMAND_RATE_HZ

    print(f"[CTRL] Control loop running at {COMMAND_RATE_HZ:.0f} Hz.")

    while True:
        loop_start = time.time()

        # Snapshot telemetry — hold the lock only for the copy, not across
        # the send call below.
        with state._lock:
            roll = state.roll
            pitch = state.pitch
            yaw = state.yaw
            x = state.x
            y = state.y
            z = state.z

        # ----------------------------------------------------------------
        # Your pipeline goes here:
        #   Vision + Telemetry → Perception → Planning → Control
        # ----------------------------------------------------------------
        target_x = 10.0
        target_y = 0.0
        target_z = -5.0  # NED: negative z = up

        master.mav.set_position_target_local_ned_send(
            time_boot_ms=time_boot_ms(),
            target_system=master.target_system,
            target_component=master.target_component,
            coordinate_frame=mavlink_dialect.MAV_FRAME_LOCAL_NED,
            type_mask=0b0000111111111000,  # position only; ignore velocity/accel/yaw
            x=target_x,
            y=target_y,
            z=target_z,
            vx=0.0,
            vy=0.0,
            vz=0.0,
            afx=0.0,
            afy=0.0,
            afz=0.0,
            yaw=0.0,
            yaw_rate=0.0,
        )

        elapsed = time.time() - loop_start
        remaining = loop_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)


if __name__ == "__main__":
    main()
