import asyncio
import time
import threading

import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

from ultralytics import YOLO


# ============================================================
# APP
# ============================================================

app = FastAPI(title="Wild AI Cloud")


# ============================================================
# YOLO MODEL
# ============================================================

print("==========================================")
print("Loading YOLO model...")
print("==========================================")

# IMPORTANT:
# This automatically downloads YOLO11n the first time
# if the model is not already present.
#
# If you later train your own model, replace this with:
#
# model = YOLO("best.pt")
#

model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully")


# ============================================================
# GLOBAL FRAME STORAGE
# ============================================================

latest_raw_frame = None
latest_processed_frame = None

frame_lock = threading.Lock()

last_frame_time = 0
detection_count = 0


# ============================================================
# YOLO DETECTION
# ============================================================

def process_frame(jpeg_bytes):

    global detection_count

    try:

        # ----------------------------------------------------
        # JPEG bytes -> NumPy array
        # ----------------------------------------------------

        np_data = np.frombuffer(
            jpeg_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            np_data,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            print("ERROR: Could not decode JPEG")

            return None


        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = model(
            frame,
            imgsz=320,
            conf=0.35,
            verbose=False
        )


        # ----------------------------------------------------
        # Draw detections
        # ----------------------------------------------------

        annotated_frame = results[0].plot()

        detection_count = len(
            results[0].boxes
        )


        # ----------------------------------------------------
        # Add information overlay
        # ----------------------------------------------------

        cv2.putText(
            annotated_frame,
            "WILD AI - CLOUD YOLO",
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            annotated_frame,
            f"Detections: {detection_count}",
            (5, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # Encode processed frame as JPEG
        # ----------------------------------------------------

        success, encoded = cv2.imencode(
            ".jpg",
            annotated_frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                75
            ]
        )

        if not success:

            print("ERROR: JPEG encoding failed")

            return None


        return encoded.tobytes()


    except Exception as e:

        print(
            "YOLO processing error:",
            str(e)
        )

        return None


# ============================================================
# ESP32 WEBSOCKET
# ============================================================

@app.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):

    global latest_raw_frame
    global latest_processed_frame
    global last_frame_time

    await websocket.accept()

    print()
    print("==========================================")
    print("ESP32 CAMERA CONNECTED")
    print("==========================================")

    try:

        while True:

            # ------------------------------------------------
            # Receive binary JPEG frame
            # ------------------------------------------------

            data = await websocket.receive_bytes()

            if not data:

                continue


            print(
                f"Received frame: {len(data)} bytes"
            )


            # ------------------------------------------------
            # Store raw frame
            # ------------------------------------------------

            with frame_lock:

                latest_raw_frame = data


            # ------------------------------------------------
            # Run YOLO
            # ------------------------------------------------

            processed = await asyncio.to_thread(
                process_frame,
                data
            )


            if processed is not None:

                with frame_lock:

                    latest_processed_frame = processed

                    last_frame_time = time.time()


                print(
                    f"YOLO processed frame | "
                    f"Detections: {detection_count}"
                )


            # ------------------------------------------------
            # Small delay prevents CPU overload
            # ------------------------------------------------

            await asyncio.sleep(0.01)


    except WebSocketDisconnect:

        print()
        print("==========================================")
        print("ESP32 CAMERA DISCONNECTED")
        print("==========================================")


    except Exception as e:

        print(
            "WebSocket error:",
            str(e)
        )


# ============================================================
# MJPEG VIDEO STREAM
# ============================================================

def generate_video():

    while True:

        frame = None

        with frame_lock:

            if latest_processed_frame is not None:

                frame = latest_processed_frame


        if frame is not None:

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )


        time.sleep(0.05)


# ============================================================
# VIDEO URL
# ============================================================

@app.get("/video_feed")
def video_feed():

    return StreamingResponse(
        generate_video(),
        media_type=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# STATUS
# ============================================================

@app.get("/status")
def status():

    with frame_lock:

        camera_connected = (
            latest_raw_frame is not None
        )

        video_available = (
            latest_processed_frame is not None
        )


    return JSONResponse({

        "status": "running",

        "camera_frame_received":
            camera_connected,

        "processed_video_available":
            video_available,

        "detections":
            detection_count,

        "video_url":
            "/video_feed"

    })


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
def home():

    html = """

    <!DOCTYPE html>

    <html>

    <head>

        <title>Wild AI Cloud</title>

        <meta
            name="viewport"
            content="width=device-width,
                     initial-scale=1"
        >

        <style>

            body {
                background: #111;
                color: white;
                font-family: Arial;
                text-align: center;
                margin: 0;
                padding: 20px;
            }

            h1 {
                margin-bottom: 20px;
            }

            img {
                width: 640px;
                max-width: 95%;
                border: 2px solid white;
                border-radius: 8px;
            }

            .status {
                margin-top: 15px;
                font-size: 18px;
            }

        </style>

    </head>


    <body>

        <h1>
            WILD AI - CLOUD ANIMAL DETECTION
        </h1>

        <img
            src="/video_feed"
            alt="Live camera feed"
        >

        <div class="status">

            Live ESP32-CAM → YOLO → Cloud

        </div>

    </body>

    </html>

    """

    return HTMLResponse(html)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    print()
    print("==========================================")
    print("WILD AI CLOUD STARTED")
    print("==========================================")

    print(
        "Video feed:"
    )

    print(
        "/video_feed"
    )

    print(
        "WebSocket:"
    )

    print(
        "/ws/camera"
    )

    print(
        "=========================================="
    )
