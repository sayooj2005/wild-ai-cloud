```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import asyncio
from ultralytics import YOLO

# =====================================================
# APP
# =====================================================

app = FastAPI(title="Wild AI Cloud")


# =====================================================
# LOAD YOLO MODEL
# =====================================================

print("Loading YOLO model...")

# YOLO will automatically download this model
# the first time the server starts.
model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully!")


# =====================================================
# GLOBAL VARIABLES
# =====================================================

latest_frame = None


# =====================================================
# HOME
# =====================================================

@app.get("/")
async def home():

    return {
        "project": "Wild AI",
        "status": "online",
        "ai": "YOLO enabled"
    }


# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "ai": "YOLO enabled"
    }


# =====================================================
# CAMERA PAGE
# =====================================================

@app.get("/camera")
async def camera_page():

    html = """
    <!DOCTYPE html>

    <html>

    <head>

        <title>Wild AI Camera</title>

        <meta name="viewport"
              content="width=device-width, initial-scale=1.0">

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

                margin-bottom: 5px;

            }

            h2 {

                margin-top: 5px;

                font-weight: normal;

            }

            #camera {

                width: 640px;

                max-width: 95%;

                border: 3px solid white;

                border-radius: 10px;

                display: block;

                margin: 20px auto;

            }

            #status {

                font-size: 18px;

                margin-top: 10px;

            }

        </style>

    </head>


    <body>

        <h1>WILD AI</h1>

        <h2>LIVE CAMERA + YOLO</h2>

        <img id="camera">

        <div id="status">

            Connecting...

        </div>


        <script>

            const camera =
                document.getElementById("camera");

            const status =
                document.getElementById("status");


            const protocol =
                location.protocol === "https:"
                ? "wss://"
                : "ws://";


            const ws = new WebSocket(
                protocol +
                location.host +
                "/ws/viewer"
            );


            ws.binaryType = "blob";


            ws.onopen = function() {

                status.innerText =
                    "LIVE - YOLO PROCESSING";

            };


            ws.onmessage = function(event) {

                const imageURL =
                    URL.createObjectURL(event.data);


                camera.onload = function() {

                    URL.revokeObjectURL(imageURL);

                };


                camera.src = imageURL;

            };


            ws.onclose = function() {

                status.innerText =
                    "Camera disconnected";

            };


            ws.onerror = function() {

                status.innerText =
                    "WebSocket error";

            };

        </script>

    </body>

    </html>
    """

    return HTMLResponse(content=html)


# =====================================================
# YOLO PROCESSING FUNCTION
# =====================================================

def detect_animals(frame):

    # Run YOLO
    results = model(
        frame,
        conf=0.40,
        verbose=False
    )

    result = results[0]

    # Number of detections
    detection_count = 0

    # =================================================
    # PROCESS EACH DETECTION
    # =================================================

    for box in result.boxes:

        detection_count += 1

        # Bounding box coordinates
        coordinates = box.xyxy[0].cpu().numpy()

        x1 = int(coordinates[0])
        y1 = int(coordinates[1])
        x2 = int(coordinates[2])
        y2 = int(coordinates[3])

        # Confidence
        confidence = float(box.conf[0])

        # Class
        class_id = int(box.cls[0])

        # Class name
        class_name = model.names[class_id]

        # =================================================
        # DRAW BOX
        # =================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # =================================================
        # LABEL
        # =================================================

        label = (
            f"{class_name} "
            f"{confidence * 100:.1f}%"
        )

        # Get label size
        (text_width, text_height), baseline = \
            cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

        # Label background
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width + 5, y1),
            (0, 255, 0),
            -1
        )

        # Label text
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

        # =================================================
        # PRINT DETECTION TO RENDER LOG
        # =================================================

        print(
            f"DETECTED: {class_name} "
            f"| CONFIDENCE: "
            f"{confidence * 100:.1f}%"
        )


    # =================================================
    # STATUS TEXT
    # =================================================

    if detection_count > 0:

        cv2.putText(
            frame,
            "ANIMAL DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "NO ANIMAL DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    return frame


# =====================================================
# ESP32 CAMERA WEBSOCKET
# =====================================================

@app.websocket("/ws/camera")
async def camera_stream(websocket: WebSocket):

    global latest_frame

    await websocket.accept()

    print("================================")
    print("ESP32-CAM CONNECTED")
    print("================================")


    frame_count = 0


    try:

        while True:

            # =================================================
            # RECEIVE JPEG FROM ESP32
            # =================================================

            data = await websocket.receive_bytes()

            frame_count += 1


            # =================================================
            # JPEG → NUMPY
            # =================================================

            image_array = np.frombuffer(
                data,
                dtype=np.uint8
            )


            # =================================================
            # NUMPY → OPENCV IMAGE
            # =================================================

            frame = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )


            if frame is None:

                print("Invalid JPEG frame")

                continue


            # =================================================
            # RUN YOLO
            # =================================================

            processed_frame = detect_animals(frame)


            # =================================================
            # OPENCV IMAGE → JPEG
            # =================================================

            success, encoded_image = cv2.imencode(
                ".jpg",
                processed_frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    80
                ]
            )


            if not success:

                print("JPEG encoding failed")

                continue


            # =================================================
            # STORE PROCESSED FRAME
            # =================================================

            latest_frame = encoded_image.tobytes()


            # =================================================
            # LOG EVERY 30 FRAMES
            # =================================================

            if frame_count % 30 == 0:

                height, width = frame.shape[:2]

                print(
                    f"Frames processed: "
                    f"{frame_count} | "
                    f"{width}x{height}"
                )


    except WebSocketDisconnect:

        print("================================")
        print("ESP32-CAM DISCONNECTED")
        print("================================")


    except Exception as e:

        print("Camera WebSocket error:")

        print(e)


# =====================================================
# VIEWER WEBSOCKET
# =====================================================

@app.websocket("/ws/viewer")
async def viewer_stream(websocket: WebSocket):

    global latest_frame

    await websocket.accept()

    print("Browser viewer connected")


    try:

        while True:

            if latest_frame is not None:

                await websocket.send_bytes(
                    latest_frame
                )


            # Send approximately 10 frames/sec
            await asyncio.sleep(0.1)


    except WebSocketDisconnect:

        print("Browser viewer disconnected")


    except Exception as e:

        print("Viewer WebSocket error:")

        print(e)
```
