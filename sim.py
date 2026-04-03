"""
Mock MAVLink Vehicle Simulator
Implements the protocol defined in VADR-TS-001 (Issue 00.01, 2026-03-09):

  HEARTBEAT             Simulator → Client   Connection status (≥ 2 Hz)
  ATTITUDE              Simulator → Client   Vehicle attitude
  HIGHRES_IMU           Simulator → Client   IMU data
  ODOMETRY              Simulator → Client   Position/velocity
  TIMESYNC              Simulator → Client   Timing
  SET_POSITION_TARGET_LOCAL_NED  Client → Simulator  Position control
  SET_ATTITUDE_TARGET            Client → Simulator  Attitude control

Heartbeat protocol (§4.4, §5.2, §6):
  - Simulator begins sending HEARTBEAT immediately on bind (≥ 2 Hz), before
    any client packet arrives, so wait_heartbeat() on the client side unblocks
    as soon as the transport is ready.
  - Client is required to send its own HEARTBEAT (§5.2 "maintain heartbeat
    messages"). The simulator tracks client heartbeat liveness and emits a
    warning when the client goes silent beyond CLIENT_HB_TIMEOUT_S seconds.
  - Telemetry (ATTITUDE, HIGHRES_IMU, ODOMETRY) is withheld until the
    simulator has learnt the client's UDP address (i.e. received at least one
    packet). HEARTBEAT is exempt from this gate so the client can discover
    the sim at any time.

Usage:
    python mock_vehicle.py

Start this BEFORE your controller.
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Warn if no client HEARTBEAT has been received for this many seconds.
# Per spec §5.2 the client must maintain heartbeats; we allow some slack.
CLIENT_HB_TIMEOUT_S: float = 5.0

# Minimum simulator heartbeat rate required by spec §4.4.
MIN_HB_HZ: int = 2


# ---------------------------------------------------------------------------
# Simulated vehicle state
# ---------------------------------------------------------------------------


@dataclass
class VehicleState:
    # Attitude (radians)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    rollspeed: float = 0.0
    pitchspeed: float = 0.0
    yawspeed: float = 0.0

    # Position in NED (metres)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # Velocity in NED (m/s)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # IMU
    xacc: float = 0.0
    yacc: float = 0.0
    zacc: float = -9.81  # gravity (m/s²)
    xgyro: float = 0.0
    ygyro: float = 0.0
    zgyro: float = 0.0
    xmag: float = 0.3
    ymag: float = 0.0
    zmag: float = 0.5
    abs_pressure: float = 101325.0
    temperature: float = 25.0

    # Control targets (written by incoming command handler)
    target_x: float = 0.0
    target_y: float = 0.0
    target_z: float = 0.0
    target_roll: float = 0.0
    target_pitch: float = 0.0
    target_yaw: float = 0.0
    target_thrust: float = 0.5

    # Bookkeeping
    last_cmd_type: str = "none"
    last_cmd_time: float = field(default_factory=time.time)

    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, dt: float):
        """Very simple first-order dynamics so telemetry looks alive."""
        with self._lock:
            alpha = min(1.0, dt * 2.0)  # smoothing factor

            # Position control: drift toward target
            self.x += (self.target_x - self.x) * alpha * 0.1
            self.y += (self.target_y - self.y) * alpha * 0.1
            self.z += (self.target_z - self.z) * alpha * 0.1

            self.vx = (self.target_x - self.x) * 0.5
            self.vy = (self.target_y - self.y) * 0.5
            self.vz = (self.target_z - self.z) * 0.5

            # Attitude control
            self.roll += (self.target_roll - self.roll) * alpha * 0.3
            self.pitch += (self.target_pitch - self.pitch) * alpha * 0.3
            self.yaw += (self.target_yaw - self.yaw) * alpha * 0.1

            self.rollspeed = (self.target_roll - self.roll) * 0.3
            self.pitchspeed = (self.target_pitch - self.pitch) * 0.3
            self.yawspeed = (self.target_yaw - self.yaw) * 0.1

            # IMU: add tiny noise
            import random

            self.xacc = random.gauss(0.0, 0.01)
            self.yacc = random.gauss(0.0, 0.01)
            self.zacc = random.gauss(-9.81, 0.02)
            self.xgyro = random.gauss(self.rollspeed, 0.001)
            self.ygyro = random.gauss(self.pitchspeed, 0.001)
            self.zgyro = random.gauss(self.yawspeed, 0.001)


# ---------------------------------------------------------------------------
# Heartbeat liveness tracker
# ---------------------------------------------------------------------------


class HeartbeatMonitor:
    """
    Tracks client heartbeat liveness per §5.2.

    The client is considered *alive* once its first HEARTBEAT has been
    received and *stale* if more than CLIENT_HB_TIMEOUT_S seconds have
    elapsed since the last one.
    """

    def __init__(self, timeout_s: float = CLIENT_HB_TIMEOUT_S):
        self._timeout_s = timeout_s
        self._last_hb_time: Optional[float] = None
        self._ever_received: bool = False
        self._stale_warned: bool = False
        self._lock = threading.Lock()

    def record(self, msg) -> None:
        """Call when a HEARTBEAT is received from the client."""
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
        """True if a heartbeat has been seen recently."""
        with self._lock:
            if self._last_hb_time is None:
                return False
            return (time.time() - self._last_hb_time) < self._timeout_s

    def check_and_warn(self) -> None:
        """
        Emit a warning (once per stale period) if the client heartbeat has
        timed out.  Resets when a new heartbeat arrives.
        """
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


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class MockVehicle:
    def __init__(
        self,
        bind_addr: str = "udpin:localhost:14540",
        send_hz: int = 120,
        heartbeat_hz: int = 2,
        timesync_hz: int = 1,
        verbose: bool = True,
    ):
        # Clamp heartbeat rate to spec minimum (§4.4)
        if heartbeat_hz < MIN_HB_HZ:
            print(
                f"[SIM] WARNING: heartbeat_hz={heartbeat_hz} is below the "
                f"spec minimum of {MIN_HB_HZ} Hz (§4.4). Clamping."
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

        # Rate counters
        self._cmd_count = 0
        self._stats_time = time.time()

        # Set when the first packet arrives from the client, gates telemetry.
        # HEARTBEAT is NOT gated — it is sent immediately so the client's
        # wait_heartbeat() can unblock as soon as the transport path exists.
        self._client_connected = threading.Event()

        # Heartbeat liveness monitor
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

        # Threads
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
    # Receive loop — handles incoming client messages
    # ------------------------------------------------------------------

    def _recv_loop(self):
        while self._running:
            msg = self._conn.recv_match(blocking=True, timeout=1.0)
            if msg is None:
                continue

            # First packet received — sim now knows the client's UDP address,
            # which unlocks the telemetry gate.
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
                pass  # ignore framing noise
            else:
                if self.verbose:
                    print(f"[SIM] Unhandled message: {msg_type}")

    def _handle_heartbeat(self, msg):
        """
        Process an incoming client HEARTBEAT (§5.2).

        Records liveness and logs on first contact and on reconnect after a
        stale period.
        """
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
        # Quaternion → Euler (simple approximation for display)
        q = msg.q  # [w, x, y, z]
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
    # Heartbeat loop — dedicated thread for simulator → client heartbeats
    #
    # Per §6 the simulator sends HEARTBEAT *before* the client connects so
    # that wait_heartbeat() on the client unblocks as soon as the path is
    # open.  This runs independently of _telemetry_loop and does NOT wait
    # on _client_connected.
    # ------------------------------------------------------------------

    def _heartbeat_loop(self):
        interval = 1.0 / self.hb_hz
        print(
            f"[SIM] Heartbeat loop started ({self.hb_hz} Hz, interval={interval:.3f}s)."
        )
        while self._running:
            loop_start = time.time()
            self._send_heartbeat()
            # Also check client liveness each heartbeat cycle (cheap)
            self._hb_monitor.check_and_warn()
            elapsed = time.time() - loop_start
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ------------------------------------------------------------------
    # Telemetry loop — sends all outbound telemetry at configured rate
    # (gated on having learnt the client's UDP address)
    # ------------------------------------------------------------------

    def _telemetry_loop(self):
        dt = 1.0 / self.send_hz
        ts_every = max(1, self.send_hz // self.ts_hz)
        tick = 0

        # Wait until the client's first packet arrives so the UDP stack
        # knows where to send.  udpin learns the remote address from the
        # first received datagram.
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
            elapsed = time.time() - loop_start
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ------------------------------------------------------------------
    # Individual senders
    # ------------------------------------------------------------------

    def _send_heartbeat(self):
        """
        Send simulator HEARTBEAT (§4.3, §4.4).

        Sent at ≥ 2 Hz unconditionally — does not wait for client connection
        so that the client's wait_heartbeat() works before any command is
        sent (§6 session sequence).
        """
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
            fields_updated=0x1FFF,  # all fields valid
            id=0,
        )

    def _send_odometry(self):
        s = self.state
        # Quaternion from Euler
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
            ts1=int(time.time() * 1e9),  # nanoseconds
        )

    # ------------------------------------------------------------------
    # Stats loop
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _time_boot_ms(self) -> int:
        return int(time.time() * 1000) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
