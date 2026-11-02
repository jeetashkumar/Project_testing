
#!/usr/bin/env python3

import time
import platform
import threading
import logging
from datetime import datetime

import requests
import psutil
import cv2
import numpy as np

from flask import Flask, Response
from requests.auth import HTTPDigestAuth


# ============================================================
# CONFIGURATION
# ============================================================

# Hikvision camera configuration
CAMERA_IP = "192.168.137.83"
CAMERA_USER = "admin"
CAMERA_PASS = "NCAI#456"

CAMERA_URL = (
    f"http://{CAMERA_IP}/ISAPI/Streaming/channels/101/picture"
)

# Cattle-monitoring FastAPI backend
BACKEND_BASE_URL = "http://192.168.137.219:8000"

# Camera ID registered in the cattle-monitoring database
CAMERA_ID = "camera_test_01"

# Snapshot upload interval (seconds)
SNAPSHOT_INTERVAL = 5

# Raspberry Pi heartbeat interval (seconds)
STATUS_INTERVAL = 60

# HTTP request timeouts
CAMERA_TIMEOUT = 10
SESSION_TIMEOUT = 20
HEARTBEAT_TIMEOUT = 10

# Frame upload timeout: (connection timeout, read timeout)
FRAME_UPLOAD_TIMEOUT = (10, 60)

# Flask local preview configuration
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5001


# ============================================================
# API ENDPOINTS
# ============================================================

HEALTH_URL = (
    f"{BACKEND_BASE_URL}/health"
)

CREATE_SESSION_URL = (
    f"{BACKEND_BASE_URL}/api/v1/sessions"
)

HEARTBEAT_URL = (
    f"{BACKEND_BASE_URL}/api/pi_status"
)


# ============================================================
# GLOBAL STATE
# ============================================================

app = Flask(__name__)

latest_snapshot = None
snapshot_lock = threading.Lock()

camera_connection_failed = False
snapshot_upload_failed = False

session_id = None
sequence_number = 0

running = True

# Reuse HTTP connections to the backend
backend_session = requests.Session()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("cattle-pi-sender")


# ============================================================
# 1. CHECK BACKEND CONNECTION
# ============================================================

def check_backend():

    try:

        response = backend_session.get(
            HEALTH_URL,
            timeout=10
        )

        response.raise_for_status()

        result = response.json()

        log.info(
            "Backend connection successful: %s",
            result
        )

        if not result.get("models_loaded", False):

            log.warning(
                "Backend is running, but ML models are not loaded."
            )

        return True

    except requests.RequestException as exc:

        log.error(
            "Backend connection failed: %s",
            exc
        )

        return False


# ============================================================
# 2. CREATE CATTLE MONITORING SESSION
# ============================================================

def create_monitoring_session():

    global session_id
    global sequence_number

    try:

        payload = {
            "camera_id": CAMERA_ID
        }

        response = backend_session.post(
            CREATE_SESSION_URL,
            json=payload,
            timeout=SESSION_TIMEOUT
        )

        if response.status_code not in (200, 201):

            log.error(
                "Session creation failed | HTTP %s | %s",
                response.status_code,
                response.text
            )

            return False

        result = response.json()

        session_id = result.get("session_id")

        if not session_id:

            log.error(
                "Backend did not return session_id: %s",
                result
            )

            return False

        sequence_number = 0

        log.info(
            "Monitoring session created successfully: %s",
            session_id
        )

        return True

    except requests.RequestException as exc:

        log.error(
            "Session creation error: %s",
            exc
        )

        return False


# ============================================================
# 3. GET RASPBERRY PI HEALTH INFORMATION
# ============================================================

def get_pi_status():

    try:

        # CPU temperature
        cpu_temp = "N/A"

        try:

            with open(
                "/sys/class/thermal/thermal_zone0/temp",
                "r"
            ) as file:

                cpu_temp = round(
                    int(file.read()) / 1000,
                    1
                )

        except (OSError, ValueError):

            pass

        status = {

            "camera_id": CAMERA_ID,

            "hostname": platform.node(),

            "status": "online",

            "uptime_minutes": round(
                (time.time() - psutil.boot_time()) / 60,
                1
            ),

            "cpu_usage_percent": psutil.cpu_percent(
                interval=1
            ),

            "ram_usage_percent": psutil.virtual_memory().percent,

            "disk_usage_percent": psutil.disk_usage("/").percent,

            "temperature_c": cpu_temp,

            "timestamp": datetime.now().isoformat(),

            "camera_connection_failed": camera_connection_failed,

            "snapshot_upload_failed": snapshot_upload_failed,

            "session_id": session_id,

        }

        return status

    except Exception as exc:

        log.exception(
            "Unable to collect Raspberry Pi status"
        )

        return {

            "camera_id": CAMERA_ID,

            "status": "error",

            "error": str(exc)

        }


# ============================================================
# 4. SEND RASPBERRY PI HEARTBEAT
# ============================================================

def send_pi_status():

    try:

        status = get_pi_status()

        response = backend_session.post(
            HEARTBEAT_URL,
            json=status,
            timeout=HEARTBEAT_TIMEOUT
        )

        if response.status_code == 200:

            log.info(
                "Raspberry Pi heartbeat sent successfully"
            )

        else:

            log.warning(
                "Heartbeat failed | HTTP %s | %s",
                response.status_code,
                response.text
            )

    except requests.RequestException as exc:

        log.error(
            "Heartbeat connection error: %s",
            exc
        )


# ============================================================
# 5. CAPTURE HIKVISION CAMERA SNAPSHOT
# ============================================================

def capture_snapshot():

    global latest_snapshot
    global camera_connection_failed

    try:

        response = requests.get(
            CAMERA_URL,

            auth=HTTPDigestAuth(
                CAMERA_USER,
                CAMERA_PASS
            ),

            timeout=CAMERA_TIMEOUT
        )

        response.raise_for_status()

        image_bytes = response.content

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            camera_connection_failed = True

            log.warning(
                "Invalid image received from Hikvision camera"
            )

            return None

        camera_connection_failed = False

        # Store the latest snapshot for local Flask preview
        with snapshot_lock:

            latest_snapshot = image_bytes

        log.info(
            "Camera snapshot captured: %s x %s",
            frame.shape[1],
            frame.shape[0]
        )

        return image_bytes

    except requests.RequestException as exc:

        camera_connection_failed = True

        log.error(
            "Camera connection error: %s",
            exc
        )

        return None

    except Exception:

        camera_connection_failed = True

        log.exception(
            "Snapshot capture error"
        )

        return None


# ============================================================
# 6. UPLOAD FRAME TO CATTLE MONITORING BACKEND
# ============================================================

def upload_frame(image_bytes):

    global sequence_number
    global snapshot_upload_failed

    if not session_id:

        log.error(
            "No active cattle-monitoring session"
        )

        snapshot_upload_failed = True

        return False

    frame_timestamp = time.time()

    upload_url = (
        f"{BACKEND_BASE_URL}/api/v1/sessions/"
        f"{session_id}/frames"
    )

    files = {

        "file": (

            "hikvision_snapshot.jpg",

            image_bytes,

            "image/jpeg"

        )

    }

    data = {

        "timestamp": str(frame_timestamp),

        "sequence_number": str(sequence_number)

    }

    try:

        response = backend_session.post(
            upload_url,
            files=files,
            data=data,
            timeout=FRAME_UPLOAD_TIMEOUT
        )

        # Backend accepted the frame
        if response.status_code == 202:

            snapshot_upload_failed = False

            log.info(
                "Frame accepted | Camera: %s | Session: %s | Sequence: %s",
                CAMERA_ID,
                session_id,
                sequence_number
            )

            sequence_number += 1

            return True

        snapshot_upload_failed = True

        log.error(
            "Frame upload failed | HTTP %s | Response: %s",
            response.status_code,
            response.text
        )

        return False

    except requests.RequestException as exc:

        snapshot_upload_failed = True

        log.error(
            "Frame upload connection error: %s",
            exc
        )

        return False


# ============================================================
# 7. SNAPSHOT BACKGROUND LOOP
# ============================================================

def snapshot_loop():

    while running:

        try:

            image_bytes = capture_snapshot()

            if image_bytes is not None:

                upload_frame(image_bytes)

        except Exception:

            log.exception(
                "Snapshot background loop error"
            )

        time.sleep(SNAPSHOT_INTERVAL)


# ============================================================
# 8. RASPBERRY PI STATUS BACKGROUND LOOP
# ============================================================

def status_loop():

    while running:

        try:

            send_pi_status()

        except Exception:

            log.exception(
                "Status background loop error"
            )

        time.sleep(STATUS_INTERVAL)


# ============================================================
# 9. FINISH MONITORING SESSION
# ============================================================

def finish_monitoring_session():

    if not session_id:

        return

    finish_url = (
        f"{BACKEND_BASE_URL}/api/v1/sessions/"
        f"{session_id}/finish"
    )

    try:

        response = backend_session.post(
            finish_url,
            timeout=SESSION_TIMEOUT
        )

        if response.status_code in (200, 202):

            log.info(
                "Monitoring session finalization requested: %s",
                session_id
            )

        else:

            log.error(
                "Session finalization failed | HTTP %s | %s",
                response.status_code,
                response.text
            )

    except requests.RequestException as exc:

        log.error(
            "Session finalization error: %s",
            exc
        )


# ============================================================
# 10. FLASK LOCAL SNAPSHOT PREVIEW
# ============================================================

@app.route("/snapshot")
def get_snapshot():

    with snapshot_lock:

        image = latest_snapshot

    if image is None:

        return "No snapshot available", 404

    return Response(
        image,
        mimetype="image/jpeg"
    )


# ============================================================
# 11. FLASK LOCAL SENDER STATUS
# ============================================================

@app.route("/status")
def local_status():

    return {

        "camera_id": CAMERA_ID,

        "session_id": session_id,

        "sequence_number": sequence_number,

        "camera_connection_failed": camera_connection_failed,

        "snapshot_upload_failed": snapshot_upload_failed,

    }


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    log.info(
        "Starting Hikvision cattle-monitoring sender"
    )

    log.info(
        "Camera IP: %s",
        CAMERA_IP
    )

    log.info(
        "Camera ID: %s",
        CAMERA_ID
    )

    log.info(
        "Backend URL: %s",
        BACKEND_BASE_URL
    )

    log.info(
        "Snapshot interval: %s seconds",
        SNAPSHOT_INTERVAL
    )

    log.info(
        "Heartbeat interval: %s seconds",
        STATUS_INTERVAL
    )

    if not CAMERA_PASS or CAMERA_PASS == "YOUR_CAMERA_PASSWORD":

        raise RuntimeError(
            "Please enter your Hikvision camera password "
            "in CAMERA_PASS."
        )

    # Check backend connectivity
    if not check_backend():

        raise RuntimeError(
            "Cattle backend is not reachable. "
            "Check the backend URL and server."
        )

    # Create a monitoring session
    if not create_monitoring_session():

        raise RuntimeError(
            "Unable to create monitoring session. "
            "Verify that the camera is registered and enabled."
        )

    # Start snapshot sender thread
    snapshot_thread = threading.Thread(
        target=snapshot_loop,
        daemon=True
    )

    # Start heartbeat thread
    status_thread = threading.Thread(
        target=status_loop,
        daemon=True
    )

    snapshot_thread.start()

    status_thread.start()

    log.info(
        "Snapshot sender and heartbeat threads started"
    )

    try:

        app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            threaded=True,
            use_reloader=False
        )

    except KeyboardInterrupt:

        log.info(
            "Stopping Raspberry Pi cattle-monitoring sender"
        )

    finally:

        running = False

        # Allow background loops to stop before closing HTTP session
        snapshot_thread.join(timeout=10)

        status_thread.join(timeout=10)

        finish_monitoring_session()

        backend_session.close()

        log.info(
            "Raspberry Pi sender stopped"
        )
