from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, JSONResponse

from ultralytics import YOLO

import cv2
import numpy as np
import asyncio
import os
import time


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Wild AI Cloud"
)


# ============================================================
# YOLO
# ============================================================

MODEL_NAME = "yolo11n.pt"

print()
print("==========================================")
print("          WILD AI CLOUD")
print("==========================================")
print()

print("Loading YOLO model...")

try:

    # If yolo11n.pt is not present,
    # Ultralytics downloads it automatically.
    model = YOLO(MODEL_NAME)

    print("YOLO model loaded successfully")

    print("Available classes:")
    print(model.names)

except Exception as e:

    print("YOLO MODEL ERROR:")
    print(e)

    model = None


# ============================================================
# GLOBAL CAMERA DATA
# ============================================================

latest_raw_frame = None

latest_annotated_frame = None

latest_detections = []

last_frame_time = 0

total_frames = 0


# ============================================================
# LOCK
# ============================================================

frame_lock = asyncio.Lock()


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
async def home():

    return HTMLResponse(
        """
<!DOCTYPE html>

<html>

<head>

    <meta charset="UTF-8">

    <title>Wild AI</title>

    <style>

        body {
            background: #111;
            color: white;
            font-family: Arial, sans-serif;
            text-align: center;
            margin: 0;
            padding: 20px;
        }

        h1 {
            margin-bottom: 5px;
        }

        #status {
            margin: 10px;
            font-size: 18px;
        }

        #camera {
            width: 640px;
            max-width: 95%;
            image-rendering: auto;
            border: 2px solid white;
            margin-top: 15px;
        }

        #detections {
            margin-top: 20px;
            font-size: 18px;
        }

        .animal {
            margin: 8px;
            padding: 8px;
            border: 1px solid #555;
            display: inline-block;
        }

    </style>

</head>


<body>

    <h1>WILD AI</h1>

    <div id="status">
        Waiting for ESP32 camera...
    </div>

    <img
        id="camera"
        alt="Camera feed"
    >

    <div id="detections">
        No detections
    </div>


<script>

const camera =
    document.getElementById("camera");

const status =
    document.getElementById("status");

const detections =
    document.getElementById("detections");


// ========================================================
// GET PROCESSED CAMERA FRAME
// ========================================================

async function updateImage()
{
    try
    {
        const response =
            await fetch(
                "/latest?t=" + Date.now()
            );

        if (!response.ok)
        {
            status.innerText =
                "Waiting for camera frame...";

            return;
        }


        const blob =
            await response.blob();


        const imageURL =
            URL.createObjectURL(blob);


        camera.src =
            imageURL;


        status.innerText =
            "ESP32 camera + YOLO active";
    }

    catch(error)
    {
        status.innerText =
            "Connection error";
    }
}


// ========================================================
// GET DETECTIONS
// ========================================================

async function updateDetections()
{
    try
    {
        const response =
            await fetch(
                "/detections?t=" + Date.now()
            );


        if (!response.ok)
        {
            return;
        }


        const data =
            await response.json();


        if (
            !data.detections ||
            data.detections.length === 0
        )
        {
            detections.innerHTML =
                "No animals detected";

            return;
        }


        let html = "";


        data.detections.forEach(
            function(item)
            {
                html +=
                    '<div class="animal">' +
                    '<b>' +
                    item.class +
                    '</b>' +
                    ' — ' +
                    (item.confidence * 100).toFixed(1) +
                    '%' +
                    '</div>';
            }
        );


        detections.innerHTML =
            html;

    }

    catch(error)
    {
        console.log(error);
    }
}


// ========================================================
// UPDATE
// ========================================================

setInterval(
    updateImage,
    1000
);


setInterval(
    updateDetections,
    1000
);


updateImage();

updateDetections();

</script>


</body>

</html>
"""
    )


# ============================================================
# CAMERA WEBSOCKET
# ============================================================

@app.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):

    global latest_raw_frame
    global latest_annotated_frame
    global latest_detections
    global last_frame_time
    global total_frames


    # --------------------------------------------------------
    # ACCEPT ESP32 CONNECTION
    # --------------------------------------------------------

    await websocket.accept()


    print()
    print("==========================================")
    print("ESP32 CAMERA CONNECTED")
    print("==========================================")
    print()


    try:

        while True:

            # =================================================
            # RECEIVE BINARY JPEG
            # =================================================

            data = await websocket.receive_bytes()


            if not data:
                continue


            total_frames += 1


            print(
                f"FRAME RECEIVED: "
                f"{len(data)} bytes"
            )


            # =================================================
            # JPEG -> NUMPY
            # =================================================

            np_array = np.frombuffer(
                data,
                dtype=np.uint8
            )


            # =================================================
            # NUMPY -> OPENCV
            # =================================================

            frame = cv2.imdecode(
                np_array,
                cv2.IMREAD_COLOR
            )


            if frame is None:

                print(
                    "ERROR: JPEG decode failed"
                )

                continue


            print(
                f"IMAGE DECODED: "
                f"{frame.shape[1]}x"
                f"{frame.shape[0]}"
            )


            # =================================================
            # SAVE RAW FRAME
            # =================================================

            async with frame_lock:

                latest_raw_frame = frame.copy()

                last_frame_time = time.time()


            # =================================================
            # YOLO
            # =================================================

            if model is None:

                print(
                    "YOLO model is not available"
                )

                async with frame_lock:

                    latest_annotated_frame = (
                        frame.copy()
                    )

                continue


            try:

                # ------------------------------------------------
                # YOLO DETECTION
                # ------------------------------------------------

                results = model.predict(
                    source=frame,
                    imgsz=320,
                    conf=0.25,
                    verbose=False
                )


                result = results[0]


                # ------------------------------------------------
                # DRAW BOUNDING BOXES
                # ------------------------------------------------

                annotated_frame =
                    result.plot()


                # ------------------------------------------------
                # DETECTION LIST
                # ------------------------------------------------

                detections = []


                if result.boxes is not None:

                    for box in result.boxes:

                        try:

                            class_id = int(
                                box.cls[0].item()
                            )


                            confidence = float(
                                box.conf[0].item()
                            )


                            class_name = (
                                result.names[
                                    class_id
                                ]
                            )


                            coordinates = (
                                box.xyxy[0]
                                .tolist()
                            )


                            x1 = int(
                                coordinates[0]
                            )

                            y1 = int(
                                coordinates[1]
                            )

                            x2 = int(
                                coordinates[2]
                            )

                            y2 = int(
                                coordinates[3]
                            )


                            detection = {

                                "class":
                                    class_name,

                                "confidence":
                                    round(
                                        confidence,
                                        3
                                    ),

                                "x1": x1,

                                "y1": y1,

                                "x2": x2,

                                "y2": y2
                            }


                            detections.append(
                                detection
                            )


                            print(
                                "ANIMAL DETECTED: "
                                f"{class_name} "
                                f"confidence="
                                f"{confidence:.2f}"
                            )


                        except Exception as e:

                            print(
                                "Detection parsing error:",
                                e
                            )


                # ------------------------------------------------
                # SAVE RESULTS
                # ------------------------------------------------

                async with frame_lock:

                    latest_annotated_frame = (
                        annotated_frame.copy()
                    )

                    latest_detections = (
                        detections
                    )


            except Exception as e:

                print(
                    "YOLO PROCESSING ERROR:"
                )

                print(e)


                async with frame_lock:

                    latest_annotated_frame = (
                        frame.copy()
                    )

                    latest_detections = []


    except WebSocketDisconnect:

        print()
        print("==========================================")
        print("ESP32 CAMERA DISCONNECTED")
        print("==========================================")


    except Exception as e:

        print()
        print("WEBSOCKET ERROR:")
        print(e)


# ============================================================
# LATEST PROCESSED IMAGE
# ============================================================

@app.get("/latest")
async def latest():

    async with frame_lock:

        if latest_annotated_frame is None:

            return Response(
                content=b"",
                media_type="image/jpeg",
                status_code=404
            )


        frame =
            latest_annotated_frame.copy()


    # ========================================================
    # OPENCV -> JPEG
    # ========================================================

    success, encoded =
        cv2.imencode(
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
            media_type="image/jpeg",
            status_code=500
        )


    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg"
    )


# ============================================================
# DETECTIONS API
# ============================================================

@app.get("/detections")
async def get_detections():

    async with frame_lock:

        return JSONResponse(
            content={
                "detections":
                    latest_detections,

                "total_frames":
                    total_frames,

                "camera_active":
                    latest_raw_frame is not None,

                "yolo_active":
                    model is not None
            }
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    async with frame_lock:

        camera_active =
            latest_raw_frame is not None

        detection_count =
            len(latest_detections)


    return {
        "status": "online",

        "camera": camera_active,

        "yolo": model is not None,

        "detections": detection_count,

        "frames_received":
            total_frames
    }


# ============================================================
# ROOT STATUS
# ============================================================

@app.get("/status")
async def status():

    return {
        "service":
            "Wild AI Cloud",

        "model":
            MODEL_NAME,

        "model_loaded":
            model is not None,

        "frames_received":
            total_frames,

        "camera_connected":
            latest_raw_frame is not None,

        "detections":
            latest_detections
    }
