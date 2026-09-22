from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
import cv2
import numpy as np
from ultralytics import YOLO
import time

app = FastAPI(title="Wild AI Cloud")

# =========================================================
# YOLO MODEL
# =========================================================

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully")


# =========================================================
# GLOBAL VARIABLES
# =========================================================

latest_frame = None
latest_processed_frame = None

frame_count = 0
detection_count = 0

# Process only every Nth frame.
# This helps the Render Free CPU.
PROCESS_EVERY = 3


# =========================================================
# HOME
# =========================================================

@app.get("/")
async def home():
    return {
        "project": "Wild AI",
        "status": "online",
        "message": "Wild AI cloud server is running"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }


# =========================================================
# VIEW CAMERA
# =========================================================

@app.get("/view", response_class=HTMLResponse)
async def view_camera():

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Wild AI Camera</title>

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

            img {
                width: 90%;
                max-width: 800px;
                border: 3px solid white;
                margin-top: 20px;
            }

        </style>

        <script>

            function refreshImage() {

                const image = document.getElementById("camera");

                image.src = "/latest.jpg?t=" + new Date().getTime();

            }

            setInterval(refreshImage, 1000);

        </script>

    </head>

    <body>

        <h1>Wild AI - Live Camera</h1>

        <img id="camera" src="/latest.jpg">

    </body>
    </html>
    """

    return HTMLResponse(content=html)


# =========================================================
# LATEST PROCESSED IMAGE
# =========================================================

@app.get("/latest.jpg")
async def latest_image():

    global latest_processed_frame

    if latest_processed_frame is None:

        return Response(
            content=b"",
            media_type="image/jpeg",
            status_code=404
        )

    return Response(
        content=latest_processed_frame,
        media_type="image/jpeg"
    )


# =========================================================
# ESP32 CAMERA WEBSOCKET
# =========================================================

@app.websocket("/ws/camera")
async def camera_stream(websocket: WebSocket):

    global latest_frame
    global latest_processed_frame
    global frame_count
    global detection_count

    await websocket.accept()

    print()
    print("========================================")
    print("ESP32-CAM CONNECTED")
    print("========================================")

    try:

        while True:

            # -------------------------------------------------
            # RECEIVE JPEG FRAME
            # -------------------------------------------------

            data = await websocket.receive_bytes()

            frame_count += 1

            # -------------------------------------------------
            # CONVERT JPEG TO OPENCV IMAGE
            # -------------------------------------------------

            image_array = np.frombuffer(
                data,
                dtype=np.uint8
            )

            frame = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )

            if frame is None:

                print("Invalid image received")

                continue

            latest_frame = frame

            # -------------------------------------------------
            # REDUCE CPU USAGE
            # -------------------------------------------------

            if frame_count % PROCESS_EVERY != 0:

                continue

            # -------------------------------------------------
            # YOLO DETECTION
            # -------------------------------------------------

            results = model.predict(
                source=frame,
                imgsz=320,
                conf=0.35,
                verbose=False
            )

            detections = []

            # -------------------------------------------------
            # READ YOLO RESULTS
            # -------------------------------------------------

            for result in results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    class_id = int(box.cls[0])

                    confidence = float(box.conf[0])

                    class_name = model.names[class_id]

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    detections.append({
                        "class": class_name,
                        "confidence": round(
                            confidence,
                            2
                        ),
                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2
                        ]
                    })

                    detection_count += 1

                    # -------------------------------------------------
                    # DRAW DETECTION BOX
                    # -------------------------------------------------

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    label = (
                        f"{class_name} "
                        f"{confidence * 100:.1f}%"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            # -------------------------------------------------
            # PRINT DETECTIONS
            # -------------------------------------------------

            if detections:

                print()
                print(
                    f"Frame {frame_count} - "
                    f"DETECTION:"
                )

                for detection in detections:

                    print(
                        f"  {detection['class']} "
                        f"({detection['confidence'] * 100:.1f}%)"
                    )

            else:

                print(
                    f"Frame {frame_count} - "
                    f"No objects detected"
                )

            # -------------------------------------------------
            # ADD STATUS TEXT
            # -------------------------------------------------

            cv2.putText(
                frame,
                "WILD AI - YOLO",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # -------------------------------------------------
            # ENCODE PROCESSED IMAGE
            # -------------------------------------------------

            success, encoded_image = cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    70
                ]
            )

            if success:

                latest_processed_frame = (
                    encoded_image.tobytes()
                )

    except WebSocketDisconnect:

        print()
        print("========================================")
        print("ESP32-CAM DISCONNECTED")
        print("========================================")

    except Exception as e:

        print()
        print("========================================")
        print("WEBSOCKET ERROR")
        print("========================================")

        print(e)

        print("========================================")


# =========================================================
# SERVER STARTUP
# =========================================================

@app.on_event("startup")
async def startup_event():

    print()
    print("========================================")
    print("        WILD AI CLOUD SERVER")
    print("========================================")

    print("FastAPI: READY")
    print("YOLO: READY")
    print("Camera WebSocket: /ws/camera")
    print("Camera View: /view")

    print("========================================")
