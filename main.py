import os
import time
from typing import cast

from ai_gp_protocol import MAV_FRAME_LOCAL_NED

# Required for MAVLink v2
os.environ["MAVLINK20"] = "1"
from pymavlink import mavutil

# Mav connection
master = cast(
    mavutil.mavfile,
    mavutil.mavlink_connection(
        "udpout:localhost:14540",
        dialect="ai_gp_protocol",
        source_system=255,
    ),
)

master.wait_heartbeat()
print("Connection Established")

# Aim for 100hz command cycle
loop_interval = 1.0 / 100.0
while True:
    start_time = time.time()

    # Telemetry
    msg = master.recv_match(
        type=["ATTITUDE", "HIGHRES_IMU", "ODOMETRY"], blocking=False
    )
    if msg:
        if msg.get_type() == "ATTITUDE":
            # Handle orientation/velocities
            pass

    # Control
    master.mav.set_position_target_local_ned_send(
        time_boot_ms=int(time.time() * 1000),
        target_system=master.target_system,
        target_component=master.target_component,
        coordinate_frame=MAV_FRAME_LOCAL_NED,
        type_mask=0b0000111111111000,  # Position only
        x=10.0,
        y=0.0,
        z=-5.0,  # NED coordinates (m)
        vx=0,
        vy=0,
        vz=0,
        afx=0,
        afy=0,
        afz=0,
        yaw=0,
        yaw_rate=0,
    )

    # Timing
    elapsed = time.time() - start_time
    if elapsed < loop_interval:
        time.sleep(loop_interval - elapsed)
