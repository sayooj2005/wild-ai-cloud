from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np

app = FastAPI(title="Wild AI Cloud")


@app.get("/")
async def home():
    return {
        "project": "Wild AI",
        "status": "online",
        "message": "Cloud server is running"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }


@app.websocket("/ws/camera")
async def camera_stream(websocket: WebSocket):

    await websocket.accept()

    print("================================")
    print("ESP32-CAM CONNECTED")
    print("================================")

    frame_count = 0

    try:

        while True:

            # Receive JPEG frame from ESP32
            data = await websocket.receive_bytes()

            frame_count += 1

            print(
                f"Frame received: "
                f"{frame_count} | "
                f"Bytes: {len(data)}"
            )

            # Convert JPEG bytes to NumPy array
            image_array = np.frombuffer(
                data,
                dtype=np.uint8
            )

            # Decode JPEG
            frame = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )

            if frame is None:
                print("Invalid JPEG frame")
                continue

            # Get resolution
            height, width = frame.shape[:2]

            print(
                f"Frame {frame_count} decoded: "
                f"{width}x{height}"
            )

            # =========================================
            # THIS IS WHERE YOLO WILL GO LATER
            # =========================================

            # results = model(frame)

    except WebSocketDisconnect:

        print("================================")
        print("ESP32-CAM DISCONNECTED")
        print("================================")

    except Exception as e:

        print("WebSocket error:")
        print(e)
