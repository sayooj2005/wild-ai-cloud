import time
import threading

import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from ultralytics import YOLO


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Wild AI Cloud",
    version="1.0.0"
)


# ============================================================
# YOLO
# ============================================================

print("==========================================")
print("Loading YOLO model...")
print("==========================================")

# Pretrained YOLO model.
#
# This is only for testing the complete pipeline.
#
# Later replace with:
#
# model = YOLO("best.pt")
#
# after training your wildlife model.

model = YOLO("yolo26n.pt")

print("==========================================")
print("YOLO MODEL LOADED")
print("==========================================")


# ============================================================
# GLOBAL FRAME STORAGE
# ============================================================

latest_raw_frame = None
latest_annotated_frame = None

latest_detections = []

frame_lock = threading.Lock()


# ============================================================
# PROCESSING CONTROL
# ============================================================

last_process_time = 0.0

PROCESS_INTERVAL = 0.2

CONFIDENCE_THRESHOLD = 0.25


# ============================================================
# CAMERA STATUS
# ============================================================

camera_connected = False

last_frame_time = 0.0


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

    <title>Wild AI Cloud</title>

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1"
    >

    <style>

        * {
            box-sizing: border-box;
        }

        body {

            margin: 0;

            padding: 20px;

            background: #111;

            color: white;

            font-family: Arial, sans-serif;

            text-align: center;
        }

        h1 {

            margin-bottom: 5px;
        }

        .subtitle {

            color: #aaa;

            margin-bottom: 20px;
        }

        .status {

            margin: 15px auto;

            padding: 10px;

            max-width: 600px;

            border-radius: 8px;

            background: #222;
        }

        .online {

            color: #00ff88;
        }

        .offline {

            color: #ff5555;
        }

        .camera-container {

            width: 100%;

            max-width: 800px;

            margin: auto;
        }

        #camera {

            width: 100%;

            max-width: 640px;

            height: auto;

            background: #000;

            border: 2px solid #444;

            border-radius: 8px;
        }

        .detections {

            max-width: 800px;

            margin: 20px auto;

            padding: 15px;

            background: #1c1c1c;

            border-radius: 8px;

            text-align: left;
        }

        .animal {

            padding: 10px;

            margin: 5px 0;

            background: #292929;

            border-radius: 5px;
        }

        .empty {

            color: #aaa;
        }

    </style>

</head>


<body>

    <h1>🦌 Wild AI Cloud</h1>

    <div class="subtitle">

        ESP32-CAM → Render → YOLO

    </div>


    <div class="status">

        Camera:

        <span id="cameraStatus">
            Checking...
        </span>

    </div>


    <div class="camera-container">

        <img
            id="camera"
            src="/video"
            alt="Waiting for camera..."
        >

    </div>


    <div class="detections">

        <h2>Detected Objects</h2>

        <div id="detections">

            Waiting for camera...

        </div>

    </div>


<script>


// ==========================================================
// REFRESH CAMERA IMAGE
// ==========================================================

function refreshCamera()
{

    const image =
        document.getElementById("camera");


    image.src =
        "/video?t=" +
        Date.now();
}


setInterval(
    refreshCamera,
    300
);


// ==========================================================
// UPDATE STATUS AND DETECTIONS
// ==========================================================

async function updateStatus()
{

    try
    {

        const response =
            await fetch("/status");


        const data =
            await response.json();


        const statusElement =
            document.getElementById(
                "cameraStatus"
            );


        if (data.camera_connected)
        {

            statusElement.innerHTML =
                "🟢 CONNECTED";

            statusElement.className =
                "online";

        }
        else
        {

            statusElement.innerHTML =
                "🔴 WAITING";

            statusElement.className =
                "offline";
        }


        const detectionElement =
            document.getElementById(
                "detections"
            );


        if (
            !data.detections ||
            data.detections.length === 0
        )
        {

            detectionElement.innerHTML =
                '<div class="empty">' +
                'No objects detected' +
                '</div>';

            return;
        }


        detectionElement.innerHTML =
            data.detections.map(
                function(d)
                {

                    return `
                    <div class="animal">

                        <strong>
                            ${d.class}
                        </strong>

                        <br>

                        Confidence:
                        ${(d.confidence * 100).toFixed(1)}%

                        <br>

                        Bounding box:
                        ${d.bbox.join(", ")}

                    </div>
                    `;

                }
            ).join("");

    }

    catch(error)
    {

        console.log(
            "Status error:",
            error
        );

    }

}


setInterval(
    updateStatus,
    500
);


updateStatus();


</script>


</body>

</html>
"""
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {

        "status": "online",

        "camera_connected":
            camera_connected,

        "yolo_loaded":
            model is not None,

        "detections":
            latest_detections
    }


# ============================================================
# STATUS API
# ============================================================

@app.get("/status")
async def status():

    global camera_connected
    global last_frame_time

    # Consider camera disconnected if no frame
    # has arrived for 5 seconds.

    if (
        time.time() - last_frame_time > 5
    ):

        camera_connected = False


    return {

        "camera_connected":
            camera_connected,

        "last_frame":
            last_frame_time,

        "detections":
            latest_detections
    }


# ============================================================
# CURRENT ANNOTATED FRAME
# ============================================================

@app.get("/video")
async def video():

    global latest_annotated_frame


    if latest_annotated_frame is None:

        return Response(

            content=b"",

            media_type="image/jpeg"
        )


    with frame_lock:

        frame_copy =
            latest_annotated_frame


    return Response(

        content=frame_copy,

        media_type="image/jpeg"
    )


# ============================================================
# YOLO PROCESSING
# ============================================================

def process_frame(
    jpeg_bytes
):

    global latest_detections


    # --------------------------------------------------------
    # JPEG → NumPy
    # --------------------------------------------------------

    np_array = np.frombuffer(

        jpeg_bytes,

        dtype=np.uint8
    )


    # --------------------------------------------------------
    # NumPy → OpenCV
    # --------------------------------------------------------

    frame = cv2.imdecode(

        np_array,

        cv2.IMREAD_COLOR
    )


    if frame is None:

        print(
            "❌ JPEG decode failed"
        )

        return None


    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    try:

        results = model.predict(

            source=frame,

            conf=CONFIDENCE_THRESHOLD,

            imgsz=320,

            verbose=False
        )

    except Exception as error:

        print(
            "❌ YOLO ERROR:",
            error
        )

        return None


    detections = []


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    for result in results:

        if result.boxes is None:

            continue


        boxes = result.boxes


        for i in range(
            len(boxes)
        ):

            class_id = int(

                boxes.cls[i].item()
            )


            confidence = float(

                boxes.conf[i].item()
            )


            x1, y1, x2, y2 = map(

                int,

                boxes.xyxy[i].tolist()
            )


            class_name = model.names.get(

                class_id,

                str(class_id)
            )


            # ------------------------------------------------
            # STORE DETECTION
            # ------------------------------------------------

            detection = {

                "class":
                    class_name,

                "confidence":
                    confidence,

                "bbox":
                    [
                        x1,
                        y1,
                        x2,
                        y2
                    ]
            }


            detections.append(
                detection
            )


            # ------------------------------------------------
            # DRAW BOX
            # ------------------------------------------------

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2
            )


            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

            label = (

                f"{class_name} "

                f"{confidence * 100:.1f}%"
            )


            cv2.putText(

                frame,

                label,

                (
                    x1,
                    max(
                        y1 - 10,
                        20
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (0, 255, 0),

                2
            )


    latest_detections = detections


    return frame


# ============================================================
# ESP32-CAM WEBSOCKET
# ============================================================

@app.websocket("/ws/camera")
async def camera_websocket(
    websocket: WebSocket
):

    global latest_raw_frame
    global latest_annotated_frame

    global latest_detections

    global camera_connected

    global last_frame_time

    global last_process_time


    await websocket.accept()


    camera_connected = True


    print()
    print(
        "=========================================="
    )

    print(
        "ESP32-CAM CONNECTED"
    )

    print(
        "=========================================="
    )


    try:

        while True:

            # ------------------------------------------------
            # RECEIVE BINARY DATA
            # ------------------------------------------------

            data = (
                await websocket.receive_bytes()
            )


            if not data:

                print(
                    "⚠️ Empty frame received"
                )

                continue


            # ------------------------------------------------
            # CAMERA STATUS
            # ------------------------------------------------

            camera_connected = True

            last_frame_time = time.time()


            # ------------------------------------------------
            # SAVE RAW FRAME
            # ------------------------------------------------

            latest_raw_frame = data


            print(

                f"FRAME RECEIVED: "
                f"{len(data)} bytes"
            )


            # ------------------------------------------------
            # PROCESSING RATE LIMIT
            # ------------------------------------------------

            current_time =
                time.time()


            if (
                current_time -
                last_process_time
                < PROCESS_INTERVAL
            ):

                continue


            last_process_time =
                current_time


            # ------------------------------------------------
            # YOLO
            # ------------------------------------------------

            frame =
                process_frame(data)


            if frame is None:

                continue


            # ------------------------------------------------
            # ENCODE ANNOTATED IMAGE
            # ------------------------------------------------

            success, encoded =
                cv2.imencode(

                    ".jpg",

                    frame,

                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        80
                    ]
                )


            if not success:

                print(
                    "❌ JPEG encode failed"
                )

                continue


            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            with frame_lock:

                latest_annotated_frame =
                    encoded.tobytes()


            # ------------------------------------------------
            # LOG DETECTIONS
            # ------------------------------------------------

            if latest_detections:

                print(
                    "🦌 DETECTION:",
                    latest_detections
                )

            else:

                print(
                    "No objects detected"
                )


    except WebSocketDisconnect:

        camera_connected = False


        print()

        print(
            "=========================================="
        )

        print(
            "ESP32-CAM DISCONNECTED"
        )

        print(
            "=========================================="
        )


    except Exception as error:

        camera_connected = False


        print(
            "❌ WebSocket error:",
            error
        )
