from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import cv2
import numpy as np
import time

app = FastAPI(title="Wild AI Cloud")


@app.get("/")
def home():
    return {
        "project": "Wild AI",
        "status": "online",
        "message": "Cloud server is working"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.websocket("/ws/camera")
async def camera_stream(websocket: WebSocket):

    await websocket.accept()

    print("ESP32-CAM connected")

    frame_count = 0

    try:

        while True:

            # Receive camera frame
            data = await websocket.receive_bytes()

            frame_count += 1

            # Convert received bytes to image
            np_array = np.frombuffer(
                data,
                dtype=np.uint8
            )

            frame = cv2.imdecode(
                np_array,
                cv2.IMREAD_COLOR
            )

            if frame is None:
                print("Invalid frame received")
                continue

            height, width = frame.shape[:2]

            print(
                f"Frame received: {frame_count} "
                f"| Size: {width}x{height}"
            )

    except WebSocketDisconnect:

        print("ESP32-CAM disconnected")

