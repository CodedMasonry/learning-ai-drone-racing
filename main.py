"""
Contestant control software skeleton.

UDP handshake order:
  1. send_once() so the sim's udpin learns our return address.
  2. wait_heartbeat() before commanding.
  3. hb.start() to maintain the session.
  4. Telemetry runs in a dedicated thread — control loop never blocks on recv.
"""

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, cast

os.environ["MAVLINK20"] = "1"

from pymavlink import mavutil  # noqa: E402
from pymavlink.dialects.v20 import common as mavlink_dialect  # noqa: E402

HB_RATE_HZ: float = 2.0
COMMAND_RATE_HZ: float = 120.0
HEARTBEAT_TIMEOUT_S: float = 10.0


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

    # sim_clock_ns - local_clock_ns
    time_offset_ns: Optional[int] = None

    _lock: threading.Lock = field(
        default_factory=threading.Lock, compare=False, repr=False
    )


class HeartbeatSender:
    def __init__(self, master: mavutil.mavfile, rate_hz: float = HB_RATE_HZ) -> None:
        self._master = master
        self._interval = 1.0 / rate_hz
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def send_once(self) -> None:
        self._send()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="heartbeat"
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _send(self) -> None:
        self._master.mav.heartbeat_send(
            mavlink_dialect.MAV_TYPE_GCS,
            mavlink_dialect.MAV_AUTOPILOT_INVALID,
            0,
            0,
            mavlink_dialect.MAV_STATE_ACTIVE,
        )

    def _loop(self) -> None:
        while not self._stop_event.wait(timeout=self._interval):
            try:
                self._send()
            except Exception as exc:  # noqa: BLE001
                print(f"[HEARTBEAT] Send failed: {exc}")


def start_telemetry(master: mavutil.mavfile, state: VehicleState) -> None:
    def _loop() -> None:
        while True:
            try:
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

                elif msg_type == "ODOMETRY":
                    with state._lock:
                        state.x = msg.x
                        state.y = msg.y
                        state.z = msg.z
                        state.vx = msg.vx
                        state.vy = msg.vy
                        state.vz = msg.vz

                elif msg_type == "HIGHRES_IMU":
                    with state._lock:
                        state.xacc = msg.xacc
                        state.yacc = msg.yacc
                        state.zacc = msg.zacc
                        state.xgyro = msg.xgyro
                        state.ygyro = msg.ygyro
                        state.zgyro = msg.zgyro

                elif msg_type == "TIMESYNC":
                    local_ns = int(time.time() * 1e9)
                    with state._lock:
                        state.time_offset_ns = msg.ts1 - local_ns

                elif msg_type in ("HEARTBEAT", "BAD_DATA"):
                    pass

            except Exception as exc:  # noqa: BLE001
                print(f"[TELEMETRY] Recv error: {exc}")

    threading.Thread(target=_loop, daemon=True, name="telemetry").start()


def main() -> None:
    master = cast(
        mavutil.mavfile,
        mavutil.mavlink_connection("udpout:localhost:14540", source_system=255),
    )

    hb = HeartbeatSender(master, rate_hz=HB_RATE_HZ)
    hb.send_once()

    print(f"[CTRL] Waiting for simulator heartbeat (timeout={HEARTBEAT_TIMEOUT_S}s)...")
    hb_msg = master.wait_heartbeat(timeout=HEARTBEAT_TIMEOUT_S)
    if hb_msg is None:
        raise RuntimeError(
            f"No heartbeat from simulator within {HEARTBEAT_TIMEOUT_S}s — "
            "is mock_vehicle.py running?"
        )
    print(
        f"[CTRL] Connected — sysid={master.target_system} compid={master.target_component}"
    )

    hb.start()

    _boot_time = time.time()

    def time_boot_ms() -> int:
        return int((time.time() - _boot_time) * 1000) % (2**32)

    state = VehicleState()
    start_telemetry(master, state)

    loop_interval = 1.0 / COMMAND_RATE_HZ
    print(f"[CTRL] Control loop running at {COMMAND_RATE_HZ:.0f} Hz.")

    while True:
        loop_start = time.time()

        with state._lock:
            roll = state.roll
            pitch = state.pitch
            yaw = state.yaw
            x = state.x
            y = state.y
            z = state.z
            vx = state.vx
            vy = state.vy
            vz = state.vz
            xacc = state.xacc
            yacc = state.yacc
            zacc = state.zacc
            xgyro = state.xgyro
            ygyro = state.ygyro
            zgyro = state.zgyro

        # ----------------------------------------------------------------
        # Vision + Telemetry → Perception → Planning → Control
        # ----------------------------------------------------------------
        target_x = 10.0
        target_y = 0.0
        target_z = -5.0  # NED: negative z = up

        master.mav.set_position_target_local_ned_send(
            time_boot_ms=time_boot_ms(),
            target_system=master.target_system,
            target_component=master.target_component,
            coordinate_frame=mavlink_dialect.MAV_FRAME_LOCAL_NED,
            type_mask=0b0000111111111000,  # position only
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
