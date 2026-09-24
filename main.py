import asyncio
import json
import time
from typing import Set

import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from ultralytics import YOLO


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Wild AI Cloud",
    version="1.0"
)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

latest_frame = None

latest_processed_frame = None

latest_detection_data = {
    "timestamp": 0,
    "detections": [],
    "count": 0
}


# Website WebSocket clients
website_clients: Set[WebSocket] = set()


# ============================================================
# YOLO MODEL
# ============================================================

MODEL_NAME = "yolo11n.pt"

print()
print("==============================================")
print("             WILD AI CLOUD")
print("==============================================")

print("Loading YOLO model:")
print(MODEL_NAME)

try:

    model = YOLO(MODEL_NAME)

    print("YOLO model loaded successfully.")

except Exception as e:

    print("ERROR loading YOLO:")
    print(str(e))

    model = None


# ============================================================
# JPEG CLEANING
# ============================================================

def clean_jpeg(data: bytes):

    if not data:
        return None

    # JPEG start marker
    start_marker = b"\xff\xd8"

    # JPEG end marker
    end_marker = b"\xff\xd9"

    start = data.find(start_marker)

    end = data.rfind(end_marker)

    if start == -1:
        print("JPEG start marker not found.")
        return None

    if end == -1:
        print("JPEG end marker not found.")
        return None

    end += 2

    cleaned = data[start:end]

    return cleaned


# ============================================================
# DECODE JPEG
# ============================================================

def decode_jpeg(data: bytes):

    cleaned = clean_jpeg(data)

    if cleaned is None:
        return None

    try:

        array = np.frombuffer(
            cleaned,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            print("OpenCV could not decode JPEG.")

            return None

        return frame

    except Exception as e:

        print(
            "JPEG decode error:",
            str(e)
        )

        return None


# ============================================================
# YOLO DETECTION
# ============================================================

def detect_objects(frame):

    global model

    detections = []

    if model is None:

        return detections

    try:

        results = model.predict(
            source=frame,
            conf=0.35,
            verbose=False,
            imgsz=640
        )

        if not results:

            return detections

        result = results[0]

        if result.boxes is None:

            return detections

        names = result.names

        for box in result.boxes:

            # Bounding box
            xyxy = box.xyxy[0].cpu().numpy()

            x1, y1, x2, y2 = map(
                int,
                xyxy
            )

            # Confidence
            confidence = float(
                box.conf[0].cpu().numpy()
            )

            # Class ID
            class_id = int(
                box.cls[0].cpu().numpy()
            )

            class_name = names[class_id]

            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(
                    confidence,
                    3
                ),
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            }

            detections.append(
                detection
            )

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label
            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )

            # Text background
            text_size = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )[0]

            text_width = text_size[0]
            text_height = text_size[1]

            cv2.rectangle(
                frame,
                (
                    x1,
                    max(
                        0,
                        y1 - text_height - 10
                    )
                ),
                (
                    x1 + text_width + 6,
                    y1
                ),
                (0, 255, 0),
                -1
            )

            # Label text
            cv2.putText(
                frame,
                label,
                (
                    x1 + 3,
                    y1 - 5
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return detections

    except Exception as e:

        print(
            "YOLO inference error:",
            str(e)
        )

        return detections


# ============================================================
# PROCESS FRAME
# ============================================================

def process_frame(frame):

    global latest_detection_data

    # Make a copy for drawing
    processed = frame.copy()

    # Run YOLO
    detections = detect_objects(
        processed
    )

    # Current time
    timestamp = time.time()

    # Store detection information
    latest_detection_data = {
        "timestamp": timestamp,
        "detections": detections,
        "count": len(detections)
    }

    # Display detection count
    status = (
        f"Detections: {len(detections)}"
    )

    cv2.rectangle(
        processed,
        (0, 0),
        (260, 35),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        processed,
        status,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Encode processed image
    success, encoded = cv2.imencode(
        ".jpg",
        processed,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            80
        ]
    )

    if not success:

        return None

    return encoded.tobytes()


# ============================================================
# SEND FRAME TO WEBSITE
# ============================================================

async def send_to_websites(
    processed_frame,
    detection_data
):

    disconnected = []

    # Send detection metadata first
    metadata = json.dumps(
        {
            "type": "detection",
            **detection_data
        }
    )

    for client in list(
        website_clients
    ):

        try:

            await client.send_text(
                metadata
            )

            await client.send_bytes(
                processed_frame
            )

        except Exception:

            disconnected.append(
                client
            )

    for client in disconnected:

        website_clients.discard(
            client
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return HTMLResponse(
        """
        <!DOCTYPE html>

        <html>

        <head>

            <title>Wild AI Cloud</title>

            <style>

                body {
                    background: #101010;
                    color: white;
                    font-family: Arial;
                    text-align: center;
                }

                img {
                    max-width: 90%;
                    border: 2px solid #444;
                }

            </style>

        </head>

        <body>

            <h1>Wild AI Cloud</h1>

            <p>
                ESP32-CAM → Render → YOLO11n
            </p>

            <img
                id="camera"
                width="640"
            >

            <h2 id="status">
                Waiting for camera...
            </h2>

            <script>

                const image =
                    document.getElementById(
                        "camera"
                    );

                const status =
                    document.getElementById(
                        "status"
                    );

                const protocol =
                    location.protocol === "https:"
                    ? "wss://"
                    : "ws://";

                const socket =
                    new WebSocket(
                        protocol +
                        location.host +
                        "/ws/website"
                    );

                socket.binaryType =
                    "blob";

                socket.onopen = function() {

                    status.innerText =
                        "Connected to Wild AI Cloud";

                };

                socket.onmessage = function(event) {

                    if (
                        typeof event.data ===
                        "string"
                    ) {

                        const data =
                            JSON.parse(
                                event.data
                            );

                        if (
                            data.type ===
                            "detection"
                        ) {

                            status.innerText =
                                "Detections: " +
                                data.count;
                        }

                        return;
                    }

                    const blob =
                        event.data;

                    const url =
                        URL.createObjectURL(
                            blob
                        );

                    image.src = url;

                };

                socket.onclose = function() {

                    status.innerText =
                        "Connection closed";

                };

            </script>

        </body>

        </html>
        """
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return JSONResponse(
        {
            "status": "ok",
            "yolo_loaded": model is not None,
            "camera_connected":
                latest_frame is not None,
            "detections":
                latest_detection_data
        }
    )


# ============================================================
# ESP32 CAMERA WEBSOCKET
# ============================================================

@app.websocket("/ws/camera")
async def camera_websocket(
    websocket: WebSocket
):

    global latest_frame
    global latest_processed_frame

    await websocket.accept()

    print()
    print(
        "=============================================="
    )

    print(
        "ESP32 CAMERA CONNECTED"
    )

    print(
        "=============================================="
    )

    try:

        while True:

            message = await websocket.receive()

            # ------------------------------------------------
            # BINARY CAMERA FRAME
            # ------------------------------------------------

            if "bytes" in message:

                data = message["bytes"]

                if not data:

                    continue

                print(
                    f"Frame received: "
                    f"{len(data)} bytes"
                )

                # Clean JPEG
                cleaned = clean_jpeg(
                    data
                )

                if cleaned is None:

                    print(
                        "Invalid JPEG frame."
                    )

                    continue

                print(
                    f"Clean JPEG: "
                    f"{len(cleaned)} bytes"
                )

                # Decode
                frame = decode_jpeg(
                    cleaned
                )

                if frame is None:

                    continue

                height, width = (
                    frame.shape[:2]
                )

                print(
                    "JPEG decoded successfully: "
                    f"({height}, {width}, 3)"
                )

                latest_frame = frame

                # ------------------------------------------------
                # YOLO
                # ------------------------------------------------

                start_time = time.time()

                processed = process_frame(
                    frame
                )

                inference_time = (
                    time.time()
                    - start_time
                )

                if processed is None:

                    continue

                latest_processed_frame = (
                    processed
                )

                # ------------------------------------------------
                # DETECTIONS
                # ------------------------------------------------

                detections = (
                    latest_detection_data[
                        "detections"
                    ]
                )

                if detections:

                    print(
                        "DETECTIONS:"
                    )

                    for detection in detections:

                        print(
                            "  "
                            f"{detection['class_name']} "
                            f""
                            f"{detection['confidence']:.2f}"
                        )

                else:

                    print(
                        "No objects detected."
                    )

                print(
                    f"YOLO processing time: "
                    f"{inference_time:.2f}s"
                )

                # ------------------------------------------------
                # WEBSITE
                # ------------------------------------------------

                await send_to_websites(
                    processed,
                    latest_detection_data
                )

            # ------------------------------------------------
            # TEXT MESSAGE
            # ------------------------------------------------

            elif "text" in message:

                text = message["text"]

                print(
                    "Camera message:",
                    text
                )

    except WebSocketDisconnect:

        print(
            "ESP32 camera disconnected."
        )

    except Exception as e:

        print(
            "Camera WebSocket error:",
            str(e)
        )


# ============================================================
# WEBSITE WEBSOCKET
# ============================================================

@app.websocket("/ws/website")
async def website_websocket(
    websocket: WebSocket
):

    await websocket.accept()

    website_clients.add(
        websocket
    )

    print(
        "Website connected."
    )

    try:

        # Send latest detection state
        await websocket.send_text(
            json.dumps(
                {
                    "type": "detection",
                    **latest_detection_data
                }
            )
        )

        while True:

            await websocket.receive_text()

    except WebSocketDisconnect:

        print(
            "Website disconnected."
        )

    except Exception as e:

        print(
            "Website WebSocket error:",
            str(e)
        )

    finally:

        website_clients.discard(
            websocket
        )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    print()
    print(
        "=============================================="
    )

    print(
        "       WILD AI CLOUD SERVER STARTED"
    )

    print(
        "=============================================="
    )

    print(
        "Camera WebSocket:"
    )

    print(
        "/ws/camera"
    )

    print(
        "Website WebSocket:"
    )

    print(
        "/ws/website"
    )

    print(
        "Health:"
    )

    print(
        "/health"
    )

    print(
        "YOLO model:"
    )

    print(
        MODEL_NAME
    )

    print(
        "=============================================="
    )
