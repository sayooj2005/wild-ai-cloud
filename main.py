import asyncio
import json
import time

import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO


# ============================================================
# WILD AI CLOUD
# ESP32-CAM -> RENDER -> YOLO
# ============================================================

app = FastAPI(
    title="Wild AI Cloud",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

CAMERA_ID = "CAM_01"

# Pretrained YOLO model.
# NO best.pt is required.
MODEL_NAME = "yolo11n.pt"

# Detection confidence
CONFIDENCE = 0.40

# Smaller image = less CPU usage on Render Free
IMAGE_SIZE = 416

# Start with 2 FPS
MAX_FPS = 2

MIN_INTERVAL = 1.0 / MAX_FPS


# ============================================================
# CLIENT STORAGE
# ============================================================

camera_clients = set()

website_clients = set()


# ============================================================
# CAMERA STATUS
# ============================================================

camera_connected = False


# ============================================================
# YOLO MODEL
# ============================================================

print("")
print("==========================================")
print("          WILD AI CLOUD")
print("==========================================")

print("Starting server...")
print("")
print("Loading pretrained YOLO11n...")
print("")


try:

    model = YOLO(MODEL_NAME)

    print(
        "YOLO11n loaded successfully."
    )

except Exception as error:

    print("")
    print(
        "ERROR: YOLO could not be loaded."
    )

    print(error)

    model = None


print("")
print("==========================================")
print("          YOLO READY")
print("==========================================")
print("")


# ============================================================
# HOME
# ============================================================

@app.get("/")
async def home():

    return {

        "project": "Wild AI",

        "status": "running",

        "camera_id": CAMERA_ID,

        "camera_connected":
            camera_connected,

        "yolo_model":
            MODEL_NAME,

        "yolo_ready":
            model is not None
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {

        "status": "ok",

        "camera_connected":
            camera_connected,

        "yolo_ready":
            model is not None,

        "model":
            MODEL_NAME
    }


# ============================================================
# SEND JSON TO WEBSITE
# ============================================================

async def broadcast_json(data):

    if not website_clients:

        return


    message = json.dumps(
        data
    )


    disconnected = []


    for client in list(
        website_clients
    ):

        try:

            await client.send_text(
                message
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
# SEND PROCESSED IMAGE TO WEBSITE
# ============================================================

async def broadcast_frame(frame):

    if not website_clients:

        return


    try:

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                70
            ]
        )


        if not success:

            return


        image_bytes = encoded.tobytes()


    except Exception as error:

        print(
            "Frame encoding error:",
            error
        )

        return


    disconnected = []


    for client in list(
        website_clients
    ):

        try:

            await client.send_bytes(
                image_bytes
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
# YOLO DETECTION
# ============================================================

def detect_objects(frame):

    detections = []


    if model is None:

        return frame, detections


    try:

        results = model.predict(

            source=frame,

            conf=CONFIDENCE,

            imgsz=IMAGE_SIZE,

            device="cpu",

            verbose=False
        )


        result = results[0]


        if result.boxes is None:

            return frame, detections


        # ====================================================
        # PROCESS EVERY DETECTION
        # ====================================================

        for box in result.boxes:

            try:

                # Bounding box
                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )


                x1, y1, x2, y2 = map(
                    int,
                    coordinates
                )


                # Confidence
                confidence = float(
                    box.conf[0]
                    .cpu()
                    .numpy()
                )


                # Class ID
                class_id = int(
                    box.cls[0]
                    .cpu()
                    .numpy()
                )


                # Class name
                class_name = (
                    result.names[
                        class_id
                    ]
                )


                # =================================================
                # DETECTION OBJECT
                # =================================================

                detection = {

                    "animal":
                        class_name,

                    "confidence":
                        round(
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


                # =================================================
                # DRAW BOUNDING BOX
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
                    f"{confidence:.2f}"
                )


                cv2.putText(

                    frame,

                    label,

                    (
                        x1,

                        max(
                            25,
                            y1 - 10
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (0, 255, 0),

                    2
                )


            except Exception as error:

                print(
                    "Detection parsing error:",
                    error
                )


    except Exception as error:

        print(
            "YOLO inference error:",
            error
        )


    return frame, detections


# ============================================================
# ESP32-CAM WEBSOCKET
# ============================================================

@app.websocket("/ws/camera")
async def camera_websocket(
    websocket: WebSocket
):

    global camera_connected


    # ========================================================
    # ACCEPT
    # ========================================================

    await websocket.accept()


    camera_clients.add(
        websocket
    )


    camera_connected = True


    print("")
    print("==========================================")
    print("ESP32-CAM CONNECTED")
    print("==========================================")
    print("")


    await broadcast_json({

        "type":
            "camera_status",

        "camera":
            CAMERA_ID,

        "status":
            "connected",

        "timestamp":
            time.time()
    })


    # ========================================================
    # FRAME RATE CONTROL
    # ========================================================

    last_processing_time = 0


    try:

        while True:

            message = await websocket.receive()


            # ==================================================
            # TEXT MESSAGE
            # ==================================================

            if "text" in message:

                text = message["text"]


                print(
                    "ESP32:",
                    text
                )


                continue


            # ==================================================
            # BINARY JPEG FRAME
            # ==================================================

            if "bytes" in message:

                jpeg_data = message[
                    "bytes"
                ]


                if not jpeg_data:

                    continue


                # =================================================
                # LIMIT PROCESSING FPS
                # =================================================

                current_time = time.time()


                if (
                    current_time
                    - last_processing_time
                    <
                    MIN_INTERVAL
                ):

                    continue


                last_processing_time = (
                    current_time
                )


                # =================================================
                # JPEG -> NUMPY
                # =================================================

                array = np.frombuffer(

                    jpeg_data,

                    dtype=np.uint8
                )


                # =================================================
                # NUMPY -> OPENCV
                # =================================================

                frame = cv2.imdecode(

                    array,

                    cv2.IMREAD_COLOR
                )


                if frame is None:

                    print(
                        "Invalid JPEG frame."
                    )

                    continue


                print(
                    "Frame received:",
                    len(jpeg_data),
                    "bytes"
                )


                # =================================================
                # RUN YOLO
                # =================================================

                processed_frame, detections = (
                    await asyncio.to_thread(
                        detect_objects,
                        frame
                    )
                )


                # =================================================
                # RESULT JSON
                # =================================================

                result = {

                    "type":
                        "detection",

                    "camera":
                        CAMERA_ID,

                    "timestamp":
                        time.time(),

                    "detections":
                        detections
                }


                # =================================================
                # PRINT RESULTS
                # =================================================

                if detections:

                    print("")
                    print(
                        "================================"
                    )

                    print(
                        "        OBJECT DETECTED"
                    )

                    print(
                        "================================"
                    )


                    for detection in detections:

                        print(
                            "Animal/Object:",
                            detection[
                                "animal"
                            ]
                        )


                        print(
                            "Confidence:",
                            detection[
                                "confidence"
                            ]
                        )


                        print(
                            "Bounding box:",
                            detection[
                                "bbox"
                            ]
                        )


                    print(
                        "================================"
                    )


                else:

                    print(
                        "No object detected."
                    )


                # =================================================
                # SEND DETECTION TO WEBSITE
                # =================================================

                await broadcast_json(
                    result
                )


                # =================================================
                # SEND PROCESSED FRAME
                # =================================================

                await broadcast_frame(
                    processed_frame
                )


    except WebSocketDisconnect:

        print("")
        print(
            "ESP32-CAM disconnected."
        )


    except Exception as error:

        print(
            "Camera WebSocket error:",
            error
        )


    finally:

        camera_clients.discard(
            websocket
        )


        camera_connected = (
            len(camera_clients) > 0
        )


        await broadcast_json({

            "type":
                "camera_status",

            "camera":
                CAMERA_ID,

            "status":
                "connected"
                if camera_connected
                else "disconnected",

            "timestamp":
                time.time()
        })


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


    # ========================================================
    # SEND INITIAL STATUS
    # ========================================================

    await websocket.send_text(

        json.dumps({

            "type":
                "status",

            "cloud":
                "connected",

            "camera":
                "connected"
                if camera_connected
                else "waiting",

            "ai":
                "ready"
                if model is not None
                else "error",

            "model":
                MODEL_NAME
        })
    )


    try:

        while True:

            await websocket.receive_text()


    except WebSocketDisconnect:

        print(
            "Website disconnected."
        )


    except Exception as error:

        print(
            "Website WebSocket error:",
            error
        )


    finally:

        website_clients.discard(
            websocket
        )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup():

    print("")
    print("==========================================")
    print("       WILD AI CLOUD IS ONLINE")
    print("==========================================")

    print("")
    print(
        "Camera WebSocket:"
    )

    print(
        "/ws/camera"
    )

    print("")
    print(
        "Website WebSocket:"
    )

    print(
        "/ws/website"
    )

    print("")
    print(
        "Health:"
    )

    print(
        "/health"
    )

    print("")
    print(
        "YOLO:"
    )

    print(
        MODEL_NAME
    )

    print("")
    print("==========================================")
