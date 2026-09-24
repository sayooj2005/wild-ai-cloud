from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from ultralytics import YOLO

import cv2
import numpy as np
import asyncio
import time
import os


# =====================================================
# FASTAPI
# =====================================================

app = FastAPI(title="Wild AI Cloud")


# =====================================================
# YOLO MODEL
# =====================================================

MODEL_PATH = "best.pt"

print("======================================")
print("        WILD AI CLOUD")
print("======================================")

print("Loading YOLO model...")

if not os.path.exists(MODEL_PATH):
    print("ERROR: best.pt not found!")
    model = None
else:
    model = YOLO(MODEL_PATH)

    print("YOLO model loaded successfully")

    print("Classes:")

    try:
        print(model.names)
    except Exception:
        print("Could not read class names")


# =====================================================
# GLOBAL FRAME STORAGE
# =====================================================

latest_frame = None
latest_annotated_frame = None

frame_lock = asyncio.Lock()


# =====================================================
# DETECTION INFORMATION
# =====================================================

latest_detections = []


# =====================================================
# HOME PAGE
# =====================================================

@app.get("/")
async def home():

    return HTMLResponse(
        """
        <!DOCTYPE html>

        <html>

        <head>

        <title>Wild AI</title>

        <style>

        body {
            background: #111;
            color: white;
            font-family: Arial;
            text-align: center;
        }

        h1 {
            margin-top: 20px;
        }

        #camera {
            width: 640px;
            max-width: 95%;
            border: 2px solid white;
            margin-top: 20px;
        }

        #status {
            margin-top: 15px;
            font-size: 18px;
        }

        </style>

        </head>

        <body>

        <h1>WILD AI</h1>

        <div id="status">
            Waiting for camera...
        </div>

        <img id="camera">

        <script>

        const image = document.getElementById("camera");
        const status = document.getElementById("status");

        async function updateFrame() {

            try {

                const response =
                    await fetch("/latest");

                if (response.ok) {

                    const blob =
                        await response.blob();

                    image.src =
                        URL.createObjectURL(blob);

                    status.innerText =
                        "Camera + YOLO active";

                }

            }

            catch (error) {

                status.innerText =
                    "Waiting for camera...";

            }

        }

        setInterval(updateFrame, 1000);

        </script>

        </body>

        </html>
        """
    )


# =====================================================
# CAMERA WEBSOCKET
# =====================================================

@app.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):

    global latest_frame
    global latest_annotated_frame
    global latest_detections

    await websocket.accept()

    print()
    print("======================================")
    print("ESP32 CAMERA CONNECTED")
    print("======================================")

    try:

        while True:

            # -----------------------------------------
            # RECEIVE BINARY JPEG
            # -----------------------------------------

            data = await websocket.receive_bytes()

            if not data:
                continue

            print(
                f"Received frame: {len(data)} bytes"
            )


            # -----------------------------------------
            # JPEG → NUMPY
            # -----------------------------------------

            np_array = np.frombuffer(
                data,
                dtype=np.uint8
            )


            # -----------------------------------------
            # NUMPY → OPENCV IMAGE
            # -----------------------------------------

            frame = cv2.imdecode(
                np_array,
                cv2.IMREAD_COLOR
            )


            if frame is None:

                print(
                    "ERROR: Could not decode JPEG"
                )

                continue


            # -----------------------------------------
            # SAVE ORIGINAL FRAME
            # -----------------------------------------

            async with frame_lock:

                latest_frame = frame.copy()


            # -----------------------------------------
            # YOLO DETECTION
            # -----------------------------------------

            if model is not None:

                try:

                    results = model.predict(
                        source=frame,
                        imgsz=320,
                        conf=0.25,
                        verbose=False
                    )


                    # ---------------------------------
                    # ANNOTATED FRAME
                    # ---------------------------------

                    annotated =
                        results[0].plot()


                    # ---------------------------------
                    # DETECTIONS
                    # ---------------------------------

                    detections = []


                    result = results[0]


                    if result.boxes is not None:

                        for box in result.boxes:

                            # Class ID
                            class_id = int(
                                box.cls[0].item()
                            )


                            # Confidence
                            confidence = float(
                                box.conf[0].item()
                            )


                            # Class name
                            class_name = result.names[
                                class_id
                            ]


                            # Bounding box
                            x1, y1, x2, y2 = (
                                box.xyxy[0]
                                .tolist()
                            )


                            detections.append(
                                {
                                    "class": class_name,

                                    "confidence":
                                        round(
                                            confidence,
                                            3
                                        ),

                                    "x1":
                                        int(x1),

                                    "y1":
                                        int(y1),

                                    "x2":
                                        int(x2),

                                    "y2":
                                        int(y2)
                                }
                            )


                            print(
                                f"DETECTED: "
                                f"{class_name} "
                                f"({confidence:.2f})"
                            )


                    # ---------------------------------
                    # STORE RESULTS
                    # ---------------------------------

                    async with frame_lock:

                        latest_annotated_frame = (
                            annotated.copy()
                        )

                        latest_detections = (
                            detections
                        )


                except Exception as e:

                    print(
                        "YOLO ERROR:",
                        e
                    )

                    async with frame_lock:

                        latest_annotated_frame = (
                            frame.copy()
                        )

            else:

                async with frame_lock:

                    latest_annotated_frame = (
                        frame.copy()
                    )


    except WebSocketDisconnect:

        print()
        print(
            "ESP32 CAMERA DISCONNECTED"
        )


    except Exception as e:

        print(
            "WebSocket error:",
            e
        )


# =====================================================
# LATEST ANNOTATED IMAGE
# =====================================================

@app.get("/latest")
async def latest():

    async with frame_lock:

        if latest_annotated_frame is None:

            return Response(
                content=b"",
                media_type="image/jpeg",
                status_code=404
            )

        frame = latest_annotated_frame.copy()


    # ---------------------------------------------
    # ENCODE JPEG
    # ---------------------------------------------

    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            85
        ]
    )


    if not success:

        return Response(
            content=b"",
            status_code=500
        )


    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg"
    )


# =====================================================
# DETECTION API
# =====================================================

@app.get("/detections")
async def detections():

    async with frame_lock:

        return {
            "detections":
                latest_detections
        }


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
async def health():

    return {
        "status": "online",

        "camera":
            latest_frame is not None,

        "yolo":
            model is not None,

        "detections":
            latest_detections
    }
