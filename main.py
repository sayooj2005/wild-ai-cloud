from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np

app = FastAPI(title="Wild AI Cloud")


# ==========================================
# HOME
# ==========================================

@app.get("/")
async def home():
    return {
        "project": "Wild AI",
        "status": "online",
        "message": "Cloud server is running"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }


# ==========================================
# ESP32-CAM WEBSOCKET
# ==========================================

@app.websocket("/ws/camera")
async def camera_stream(websocket: WebSocket):

    await websocket.accept()

    print("================================")
    print("ESP32-CAM CONNECTED")
    print("================================")

    frame_count = 0

    try:

        while True:

            # Receive binary camera frame
            data = await websocket.receive_bytes()

            frame_count += 1

            # Convert received bytes into image
            image_array = np.frombuffer(
                data,
                dtype=np.uint8
            )

            frame = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )

            # Check whether image decoding worked
            if frame is None:

                print("Invalid image received")

                continue

            # Get image dimensions
            height, width = frame.shape[:2]

            print(
                f"Frame received: "
                f"{frame_count} | "
                f"Size: {width}x{height} | "
                f"Bytes: {len(data)}"
            )

    except WebSocketDisconnect:

        print("================================")
        print("ESP32-CAM DISCONNECTED")
        print("================================")

    except Exception as e:

        print("WebSocket error:")
        print(e)
