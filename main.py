from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import asyncio

app = FastAPI(title="Wild AI Cloud")

# Store the latest camera frame
latest_frame = None

# =====================================================
# HOME
# =====================================================

@app.get("/")
async def home():
    return {
        "project": "Wild AI",
        "status": "online",
        "message": "Cloud server is running"
    }


# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy"
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

            #camera {
                width: 640px;
                max-width: 95%;
                border: 3px solid white;
                border-radius: 10px;
            }

            #status {
                margin-top: 15px;
                font-size: 18px;
            }

        </style>

    </head>

    <body>

        <h1>WILD AI - LIVE CAMERA</h1>

        <img id="camera" />

        <div id="status">
            Connecting to camera...
        </div>

        <script>

            const camera = document.getElementById("camera");
            const status = document.getElementById("status");

            const wsProtocol =
                location.protocol === "https:" ? "wss://" : "ws://";

            const ws = new WebSocket(
                wsProtocol +
                location.host +
                "/ws/viewer"
            );

            ws.binaryType = "blob";

            ws.onopen = function() {
                status.innerText = "LIVE";
            };

            ws.onmessage = function(event) {

                const url = URL.createObjectURL(event.data);

                camera.onload = function() {
                    URL.revokeObjectURL(url);
                };

                camera.src = url;
            };

            ws.onclose = function() {
                status.innerText = "Camera disconnected";
            };

            ws.onerror = function() {
                status.innerText = "Camera connection error";
            };

        </script>

    </body>

    </html>
    """

    return HTMLResponse(content=html)


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

            # Receive JPEG frame
            data = await websocket.receive_bytes()

            frame_count += 1

            # Convert JPEG → OpenCV image
            image_array = np.frombuffer(
                data,
                dtype=np.uint8
            )

            frame = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )

            if frame is None:

                print("Invalid JPEG frame")

                continue

            # Store latest JPEG
            latest_frame = data

            height, width = frame.shape[:2]

            print(
                f"Frame received: "
                f"{frame_count} | "
                f"{width}x{height} | "
                f"{len(data)} bytes"
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

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:

        print("Browser viewer disconnected")

    except Exception as e:

        print("Viewer WebSocket error:")
        print(e)
