"""
Mock MAVLink Vehicle Simulator
Protocol: VADR-TS-001 (Issue 00.01, 2026-03-09)

Outbound (sim → client):  HEARTBEAT (≥2 Hz), ATTITUDE, HIGHRES_IMU, ODOMETRY, TIMESYNC
Inbound  (client → sim):  SET_POSITION_TARGET_LOCAL_NED, SET_ATTITUDE_TARGET

HEARTBEAT is sent unconditionally so client wait_heartbeat() unblocks on transport ready.
Telemetry is gated on _client_connected (udpin learns remote addr from first rx datagram).
"""

import argparse
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, cast

os.environ["MAVLINK20"] = "1"

from pymavlink import mavutil  # noqa: E402
from pymavlink.dialects.v20 import common as mavlink_dialect  # noqa: E402

CLIENT_HB_TIMEOUT_S: float = 5.0
MIN_HB_HZ: int = 2


@dataclass
class VehicleState:
    # Attitude (rad)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    rollspeed: float = 0.0
    pitchspeed: float = 0.0
    yawspeed: float = 0.0

    # Position NED (m)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # Velocity NED (m/s)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # IMU
    xacc: float = 0.0
    yacc: float = 0.0
    zacc: float = -9.81
    xgyro: float = 0.0
    ygyro: float = 0.0
    zgyro: float = 0.0
    xmag: float = 0.3
    ymag: float = 0.0
    zmag: float = 0.5
    abs_pressure: float = 101325.0
    temperature: float = 25.0

    # Command targets — written by recv thread, read by telemetry thread
    target_x: float = 0.0
    target_y: float = 0.0
    target_z: float = 0.0
    target_roll: float = 0.0
    target_pitch: float = 0.0
    target_yaw: float = 0.0
    target_thrust: float = 0.5

    last_cmd_type: str = "none"
    last_cmd_time: float = field(default_factory=time.time)

    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, dt: float):
        """First-order dynamics: state drifts toward targets each tick."""
        with self._lock:
            alpha = min(1.0, dt * 2.0)

            self.x += (self.target_x - self.x) * alpha * 0.1
            self.y += (self.target_y - self.y) * alpha * 0.1
            self.z += (self.target_z - self.z) * alpha * 0.1

            self.vx = (self.target_x - self.x) * 0.5
            self.vy = (self.target_y - self.y) * 0.5
            self.vz = (self.target_z - self.z) * 0.5

            self.roll += (self.target_roll - self.roll) * alpha * 0.3
            self.pitch += (self.target_pitch - self.pitch) * alpha * 0.3
            self.yaw += (self.target_yaw - self.yaw) * alpha * 0.1

            self.rollspeed = (self.target_roll - self.roll) * 0.3
            self.pitchspeed = (self.target_pitch - self.pitch) * 0.3
            self.yawspeed = (self.target_yaw - self.yaw) * 0.1

            import random

            self.xacc = random.gauss(0.0, 0.01)
            self.yacc = random.gauss(0.0, 0.01)
            self.zacc = random.gauss(-9.81, 0.02)
            self.xgyro = random.gauss(self.rollspeed, 0.001)
            self.ygyro = random.gauss(self.pitchspeed, 0.001)
            self.zgyro = random.gauss(self.yawspeed, 0.001)


class HeartbeatMonitor:
    """Client heartbeat liveness tracker (§5.2). Thread-safe."""

    def __init__(self, timeout_s: float = CLIENT_HB_TIMEOUT_S):
        self._timeout_s = timeout_s
        self._last_hb_time: Optional[float] = None
        self._ever_received: bool = False
        self._stale_warned: bool = False
        self._lock = threading.Lock()

    def record(self, msg) -> None:
        with self._lock:
            self._last_hb_time = time.time()
            self._stale_warned = False
            if not self._ever_received:
                self._ever_received = True
                print(
                    f"[SIM] Client heartbeat received — "
                    f"type={msg.type} autopilot={msg.autopilot} "
                    f"base_mode={msg.base_mode} system_status={msg.system_status}"
                )

    @property
    def ever_received(self) -> bool:
        with self._lock:
            return self._ever_received

    @property
    def is_alive(self) -> bool:
        with self._lock:
            if self._last_hb_time is None:
                return False
            return (time.time() - self._last_hb_time) < self._timeout_s

    def check_and_warn(self) -> None:
        """Emit a once-per-stale-period warning; resets on next heartbeat."""
        with self._lock:
            if not self._ever_received:
                return
            age = time.time() - (self._last_hb_time or 0.0)
            if age >= self._timeout_s and not self._stale_warned:
                print(
                    f"[SIM] WARNING: No client HEARTBEAT for {age:.1f}s "
                    f"(timeout={self._timeout_s}s). "
                    "Client may have disconnected (§5.2)."
                )
                self._stale_warned = True


class MockVehicle:
    def __init__(
        self,
        bind_addr: str = "udpin:localhost:14540",
        send_hz: int = 120,
        heartbeat_hz: int = 2,
        timesync_hz: int = 1,
        verbose: bool = True,
    ):
        if heartbeat_hz < MIN_HB_HZ:
            print(
                f"[SIM] WARNING: heartbeat_hz={heartbeat_hz} below spec minimum "
                f"{MIN_HB_HZ} Hz (§4.4). Clamping."
            )
            heartbeat_hz = MIN_HB_HZ

        self.bind_addr = bind_addr
        self.send_hz = send_hz
        self.hb_hz = heartbeat_hz
        self.ts_hz = timesync_hz
        self.verbose = verbose
        self.state = VehicleState()
        self._running = False
        self._conn: mavutil.mavfile

        self._cmd_count = 0
        self._stats_time = time.time()

        # Gates telemetry until udpin learns the client's remote address.
        # HEARTBEAT bypasses this gate — see _heartbeat_loop.
        self._client_connected = threading.Event()
        self._hb_monitor = HeartbeatMonitor()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        print(f"[SIM] Binding on {self.bind_addr} — waiting for client...")
        self._conn = cast(
            mavutil.mavfile,
            mavutil.mavlink_connection(
                self.bind_addr,
                source_system=1,
                source_component=1,
            ),
        )
        self._running = True

        threading.Thread(target=self._recv_loop, daemon=True, name="recv").start()
        threading.Thread(target=self._telemetry_loop, daemon=True, name="telem").start()
        threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="heartbeat"
        ).start()
        threading.Thread(target=self._stats_loop, daemon=True, name="stats").start()

        try:
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[SIM] Shutting down.")
            self._running = False

    # ------------------------------------------------------------------
    # Recv
    # ------------------------------------------------------------------

    def _recv_loop(self):
        while self._running:
            msg = self._conn.recv_match(blocking=True, timeout=1.0)
            if msg is None:
                continue

            if not self._client_connected.is_set():
                print("[SIM] Client address learned — telemetry stream starting.")
                self._client_connected.set()

            msg_type = msg.get_type()

            if msg_type == "HEARTBEAT":
                self._handle_heartbeat(msg)
            elif msg_type == "SET_POSITION_TARGET_LOCAL_NED":
                self._handle_position_target(msg)
            elif msg_type == "SET_ATTITUDE_TARGET":
                self._handle_attitude_target(msg)
            elif msg_type == "BAD_DATA":
                pass
            else:
                if self.verbose:
                    print(f"[SIM] Unhandled message: {msg_type}")

    def _handle_heartbeat(self, msg):
        self._hb_monitor.record(msg)
        if self.verbose:
            print(
                f"[SIM] ← HEARTBEAT  "
                f"type={msg.type} autopilot={msg.autopilot} "
                f"base_mode={msg.base_mode:#04x} "
                f"custom_mode={msg.custom_mode} "
                f"system_status={msg.system_status}"
            )

    def _handle_position_target(self, msg):
        with self.state._lock:
            self.state.target_x = msg.x
            self.state.target_y = msg.y
            self.state.target_z = msg.z
            self.state.last_cmd_type = "POSITION"
            self.state.last_cmd_time = time.time()
        self._cmd_count += 1
        if self.verbose:
            print(
                f"[SIM] ← SET_POSITION_TARGET  x={msg.x:.2f}  y={msg.y:.2f}  z={msg.z:.2f}  "
                f"mask={bin(msg.type_mask)}"
            )

    def _handle_attitude_target(self, msg):
        # Quaternion [w, x, y, z] → Euler for state update and logging
        q = msg.q
        roll = math.atan2(
            2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)
        )
        pitch = math.asin(max(-1, min(1, 2 * (q[0] * q[2] - q[3] * q[1]))))
        yaw = math.atan2(
            2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)
        )
        with self.state._lock:
            self.state.target_roll = roll
            self.state.target_pitch = pitch
            self.state.target_yaw = yaw
            self.state.target_thrust = msg.thrust
            self.state.last_cmd_type = "ATTITUDE"
            self.state.last_cmd_time = time.time()
        self._cmd_count += 1
        if self.verbose:
            print(
                f"[SIM] ← SET_ATTITUDE_TARGET  "
                f"r={math.degrees(roll):.1f}°  p={math.degrees(pitch):.1f}°  "
                f"y={math.degrees(yaw):.1f}°  thrust={msg.thrust:.2f}"
            )

    # ------------------------------------------------------------------
    # Heartbeat loop — runs independently of telemetry, no addr gate
    # ------------------------------------------------------------------

    def _heartbeat_loop(self):
        interval = 1.0 / self.hb_hz
        print(f"[SIM] Heartbeat loop started ({self.hb_hz} Hz).")
        while self._running:
            loop_start = time.time()
            self._send_heartbeat()
            self._hb_monitor.check_and_warn()
            sleep_t = interval - (time.time() - loop_start)
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ------------------------------------------------------------------
    # Telemetry loop — gated on _client_connected
    # ------------------------------------------------------------------

    def _telemetry_loop(self):
        dt = 1.0 / self.send_hz
        ts_every = max(1, self.send_hz // self.ts_hz)
        tick = 0

        print("[SIM] Telemetry loop waiting for client address...")
        self._client_connected.wait()
        print("[SIM] Telemetry loop running.")

        while self._running:
            loop_start = time.time()
            self.state.update(dt)

            self._send_attitude()
            self._send_highres_imu()
            self._send_odometry()

            if tick % ts_every == 0:
                self._send_timesync()

            tick += 1
            sleep_t = dt - (time.time() - loop_start)
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ------------------------------------------------------------------
    # Senders
    # ------------------------------------------------------------------

    def _send_heartbeat(self):
        self._conn.mav.heartbeat_send(
            type=mavlink_dialect.MAV_TYPE_QUADROTOR,
            autopilot=mavlink_dialect.MAV_AUTOPILOT_GENERIC,
            base_mode=mavlink_dialect.MAV_MODE_FLAG_GUIDED_ENABLED,
            custom_mode=0,
            system_status=mavlink_dialect.MAV_STATE_ACTIVE,
        )

    def _send_attitude(self):
        s = self.state
        self._conn.mav.attitude_send(
            time_boot_ms=self._time_boot_ms(),
            roll=s.roll,
            pitch=s.pitch,
            yaw=s.yaw,
            rollspeed=s.rollspeed,
            pitchspeed=s.pitchspeed,
            yawspeed=s.yawspeed,
        )

    def _send_highres_imu(self):
        s = self.state
        self._conn.mav.highres_imu_send(
            time_usec=int(time.time() * 1e6),
            xacc=s.xacc,
            yacc=s.yacc,
            zacc=s.zacc,
            xgyro=s.xgyro,
            ygyro=s.ygyro,
            zgyro=s.zgyro,
            xmag=s.xmag,
            ymag=s.ymag,
            zmag=s.zmag,
            abs_pressure=s.abs_pressure,
            diff_pressure=0.0,
            pressure_alt=0.0,
            temperature=s.temperature,
            fields_updated=0x1FFF,
            id=0,
        )

    def _send_odometry(self):
        s = self.state
        cr, sr = math.cos(s.roll / 2), math.sin(s.roll / 2)
        cp, sp = math.cos(s.pitch / 2), math.sin(s.pitch / 2)
        cy, sy = math.cos(s.yaw / 2), math.sin(s.yaw / 2)
        q = [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
        nan = float("nan")
        self._conn.mav.odometry_send(
            time_usec=int(time.time() * 1e6),
            frame_id=mavlink_dialect.MAV_FRAME_LOCAL_NED,
            child_frame_id=mavlink_dialect.MAV_FRAME_BODY_NED,
            x=s.x,
            y=s.y,
            z=s.z,
            q=q,
            vx=s.vx,
            vy=s.vy,
            vz=s.vz,
            rollspeed=s.rollspeed,
            pitchspeed=s.pitchspeed,
            yawspeed=s.yawspeed,
            pose_covariance=[nan] * 21,
            velocity_covariance=[nan] * 21,
            reset_counter=0,
            estimator_type=mavlink_dialect.MAV_ESTIMATOR_TYPE_NAIVE,
            quality=100,
        )

    def _send_timesync(self):
        self._conn.mav.timesync_send(
            tc1=0,
            ts1=int(time.time() * 1e9),
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _stats_loop(self):
        while self._running:
            time.sleep(5.0)
            now = time.time()
            dt = now - self._stats_time
            rate = self._cmd_count / dt if dt > 0 else 0
            s = self.state
            client_hb_status = (
                "alive"
                if self._hb_monitor.is_alive
                else ("never seen" if not self._hb_monitor.ever_received else "STALE")
            )
            print(
                f"[SIM] Stats | cmd_rate={rate:.1f} Hz | "
                f"last_cmd={s.last_cmd_type} | "
                f"client_hb={client_hb_status} | "
                f"pos=({s.x:.2f},{s.y:.2f},{s.z:.2f}) | "
                f"att=({math.degrees(s.roll):.1f}°,"
                f"{math.degrees(s.pitch):.1f}°,"
                f"{math.degrees(s.yaw):.1f}°)"
            )
            self._cmd_count = 0
            self._stats_time = now

    def _time_boot_ms(self) -> int:
        return int(time.time() * 1000) & 0xFFFFFFFF


def main():
    parser = argparse.ArgumentParser(description="Mock MAVLink vehicle simulator")
    parser.add_argument(
        "--bind",
        default="udpin:localhost:14540",
        help="MAVLink connection string (default: udpin:localhost:14540)",
    )
    parser.add_argument(
        "--hz", type=int, default=120, help="Telemetry send rate in Hz (default: 120)"
    )
    parser.add_argument(
        "--hb-hz",
        type=int,
        default=2,
        help=f"Heartbeat rate in Hz (default: 2, min per spec: {MIN_HB_HZ})",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-command log lines"
    )
    args = parser.parse_args()

    sim = MockVehicle(
        bind_addr=args.bind,
        send_hz=args.hz,
        heartbeat_hz=args.hb_hz,
        verbose=not args.quiet,
    )
    sim.start()


if __name__ == "__main__":
    main()
