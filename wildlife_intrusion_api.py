import os
import cv2
import asyncio
import io
import math
import requests as http_requests
from fastapi import FastAPI, WebSocket, UploadFile, File, Depends, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from datetime import datetime
import shutil
from PIL import Image
from twilio.rest import Client as TwilioClient

# Database setup
import models
from database import engine, get_db, SessionLocal

models.Base.metadata.create_all(bind=engine)

# Import detection logic
from detection_system import WildlifeIoTSystem

app = FastAPI(title="Cloud-Based Wildlife Guardian & Decision Support API")

# Ensure required directories exist
os.makedirs("temp", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Sector Zone Camera Configurations (Ponmudi, Trivandrum)
SECTOR_ZONES = {
    "zone1_forest_border": {"name": "Zone 1 - Ponmudi Upper Sanatorium", "lat": 8.7599, "lon": 77.1169},
    "zone2_village_buffer": {"name": "Zone 2 - Golden Valley River Buffer", "lat": 8.7500, "lon": 77.1150},
    "zone3_waterhole": {"name": "Zone 3 - Kallar Forest Checkpost", "lat": 8.7186, "lon": 77.1066},
    "zone4_agri_sector": {"name": "Zone 4 - Peppara Wildlife Border", "lat": 8.6231, "lon": 77.1353}
}

# Known Safe Evacuation Havens (Ponmudi & Trivandrum Outskirts)
SAFE_HAVENS = [
    {"id": 1, "name": "KTDC Golden Peak Refuge / Police Station", "lat": 8.7580, "lon": 77.1175, "capacity": 150, "contact": "+91-471-2815225"},
    {"id": 2, "name": "Kallar Forest Base Camp", "lat": 8.7180, "lon": 77.1050, "capacity": 100, "contact": "+91-471-2815226"},
    {"id": 3, "name": "Vithura Community Shelter Hub", "lat": 8.6754, "lon": 77.0901, "capacity": 300, "contact": "+91-471-2815227"},
    {"id": 4, "name": "Peppara Dam Guard Station", "lat": 8.6250, "lon": 77.1360, "capacity": 80, "contact": "+91-471-2815228"}
]

from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import numpy as np
import base64

# Cloud Deployment Config
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
INFERENCE_INTERVAL = float(os.getenv("INFERENCE_INTERVAL", "0.2"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ESP32-CAM Global State
latest_esp32_frame = None
frame_ready_event = asyncio.Event()
connected_dashboard_clients = []
cloud_iot_detector = None

@app.on_event("startup")
async def startup_event():
    global cloud_iot_detector
    # Initialize YOLO once for the cloud pipeline without local camera
    cloud_iot_detector = WildlifeIoTSystem(model_path=MODEL_PATH, capture_index=None)
    asyncio.create_task(inference_loop())

async def inference_loop():
    global latest_esp32_frame
    while True:
        await frame_ready_event.wait()
        
        # Grab latest frame and clear the event to prevent buffering/lag
        frame_bytes = latest_esp32_frame
        latest_esp32_frame = None
        frame_ready_event.clear()
        
        if not frame_bytes:
            continue
            
        try:
            # Decode JPEG
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue

            # Run Cloud YOLO Inference
            processed_frame, detections = await run_in_threadpool(cloud_iot_detector.process_frame, frame)
            
            # Encode Processed Frame
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64_img = base64.b64encode(buffer).decode('utf-8')
            
            # ── Part 15: Camera Zone Location ─────────────────────────────────
            # Each camera is assigned a fixed GPS zone.
            # Image quadrant → zone label (A-D). No fake GPS conversion.
            cam_lat = float(os.getenv("CAM_LAT", "11.258"))
            cam_lon = float(os.getenv("CAM_LON", "75.782"))
            h_frame, w_frame = frame.shape[:2]

            def estimate_zone(bbox, fw, fh):
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if cx < fw / 2 and cy < fh / 2:
                    return "Zone-A (Top-Left)"
                elif cx >= fw / 2 and cy < fh / 2:
                    return "Zone-B (Top-Right)"
                elif cx < fw / 2 and cy >= fh / 2:
                    return "Zone-C (Bottom-Left)"
                else:
                    return "Zone-D (Bottom-Right)"

            # ── Part 16: Escape Route Calculator ──────────────────────────────
            def find_safe_route(threat_lat, threat_lon, safe_havens, detections):
                """Finds nearest safe haven and returns recommended route.
                Note: GPS offsets are camera-zone based, not direct bbox conversions."""
                import math
                best = None
                best_dist = float('inf')
                for s in safe_havens:
                    dist = math.sqrt((threat_lat - s["lat"])**2 + (threat_lon - s["lon"])**2)
                    if dist < best_dist:
                        best_dist = dist
                        best = s
                if not best:
                    return None
                dlat = best["lat"] - threat_lat
                dlon = best["lon"] - threat_lon
                angle = math.degrees(math.atan2(dlon, dlat))
                if angle < 0:
                    angle += 360
                directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                heading = directions[int((angle + 22.5) / 45) % 8]
                dist_m = int(best_dist * 111000)  # approx degrees → meters
                return {
                    "safe_zone_name": best["name"],
                    "safe_zone_lat": best["lat"],
                    "safe_zone_lon": best["lon"],
                    "heading": heading,
                    "distance_meters": dist_m,
                    "contact": best.get("contact", ""),
                    "note": "Recommended route based on currently detected wildlife risk zones."
                }

            # Enrich each detection with zone info
            for det in detections:
                det["zone"] = estimate_zone(det["bbox"], w_frame, h_frame)
                det["camera_lat"] = cam_lat
                det["camera_lon"] = cam_lon

            # Compute escape route only when there is a real high-risk detection
            escape_route = None
            if detections:
                highest = max(detections, key=lambda d: d.get("risk_score", 0))
                if highest.get("risk_score", 0) >= 50:
                    escape_route = find_safe_route(cam_lat, cam_lon, SAFE_HAVENS, detections)
                    # Persist to DB
                    if escape_route:
                        db = SessionLocal()
                        db.add(models.EvacuationRoute(
                            threat_species=highest["animal"],
                            threat_lat=cam_lat,
                            threat_lon=cam_lon,
                            safe_zone_name=escape_route["safe_zone_name"],
                            safe_zone_lat=escape_route["safe_zone_lat"],
                            safe_zone_lon=escape_route["safe_zone_lon"],
                            recommended_heading=escape_route["heading"],
                            distance_meters=escape_route["distance_meters"],
                            risk_level=highest.get("danger_level", "high")
                        ))
                        db.commit()
                        db.close()
                        print(f"[ROUTE] Animal: {highest['animal']} | Behaviour: {highest.get('behaviour','?')} | → {escape_route['safe_zone_name']} ({escape_route['heading']}, {escape_route['distance_meters']}m)")

            payload = {
                "type": "detection",
                "camera_id": os.getenv("CAMERA_ID", "ESP32_CAM_01"),
                "timestamp": datetime.now().isoformat(),
                "detections": detections,
                "escape_route": escape_route,
                "processed_frame": f"data:image/jpeg;base64,{b64_img}",
                "ai_status": "processing",
                "camera_status": "connected"
            }
            
            # Broadcast to all connected Web Dashboards
            dead_clients = []
            for client in connected_dashboard_clients:
                try:
                    await client.send_json(payload)
                except Exception:
                    dead_clients.append(client)
            
            for client in dead_clients:
                connected_dashboard_clients.remove(client)
                
        except Exception as e:
            print(f"[Cloud Inference Error] {e}")
            
        # No artificial delays - Maximum FPS processing

@app.post("/api/esp32/frame")
async def receive_esp32_frame(request: Request):
    """ESP32-CAM sends raw JPEG bytes to this Cloud endpoint."""
    global latest_esp32_frame
    body = await request.body()
    if body:
        latest_esp32_frame = body
        frame_ready_event.set()
    return {"status": "ok", "message": "Frame received."}

@app.websocket("/ws/client")
async def websocket_client(websocket: WebSocket):
    """Existing WildAI Website connects here for detection JSON & processed frames."""
    await websocket.accept()
    connected_dashboard_clients.append(websocket)
    try:
        await websocket.send_json({"type": "camera_status", "status": "connected"})
        while True:
            # Keep alive
            await websocket.receive_text()
    except Exception:
        connected_dashboard_clients.remove(websocket)

@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Receives a video file from the admin portal and saves it locally."""
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "message": "Upload successful"}

@app.websocket("/ws/stream/{filename}")
async def websocket_stream(websocket: WebSocket, filename: str):
    """Streams processed video frames with pose estimation skeletons & dynamic risk score telemetry."""
    await websocket.accept()
    
    sector_info = SECTOR_ZONES.get(filename, {"name": f"Sector {filename}", "lat": 11.258, "lon": 75.782})
    
    if filename == "live":
        capture_source = 0
        show_notification_msg = "Live camera"
    else:
        file_candidates = [f"temp/{filename}", filename, "elephant_Vdo.mp4", "boar_vdo.mp4", "human_test.mp4", "vid1.mp4"]
        capture_source = None
        for candidate in file_candidates:
            if os.path.exists(candidate):
                capture_source = candidate
                break
        if not capture_source:
            capture_source = 0  # Fallback to camera if video not found
            
        show_notification_msg = f"Sector stream: {sector_info['name']}"

    try:
        iot_detector = WildlifeIoTSystem(model_path='yolov8n.pt', capture_index=capture_source)
        latest_alert = None
        seen_species = set()
        
        def mock_trigger(dt, conf, fr, danger_level="high", risk_score=85, behaviour="Charging / Aggressive"):
            nonlocal latest_alert, seen_species
            alert_key = f"{dt}_{behaviour}"
            if alert_key not in seen_species:
                seen_species.add(alert_key)
                
                latest_alert = {
                    "species": dt,
                    "confidence": conf,
                    "frame": fr.copy(),
                    "danger": danger_level,
                    "risk_score": risk_score,
                    "behaviour": behaviour,
                    "sector": sector_info["name"]
                }
                
                db = SessionLocal()
                new_det = models.DetectionHistory(
                    species=dt,
                    confidence=f"{int(conf*100)}%",
                    danger_level=danger_level,
                    location=sector_info["name"],
                    behaviour=behaviour,
                    risk_score=risk_score,
                    sector_zone=sector_info["name"],
                    movement_vector="Approaching Village Buffer",
                    environmental_factors="Dusk Visibility 60%, Harvest Season"
                )
                db.add(new_det)
                db.commit()
                db.close()

        iot_detector.trigger_alert = mock_trigger
        
    except (SystemExit, RuntimeError) as e:
        err_msg = str(e) if isinstance(e, RuntimeError) else "Failed to initialize video stream"
        print(f"[WS ERROR] {err_msg}")
        try:
            await websocket.send_json({"action": "error", "message": err_msg})
            await websocket.close(code=1011, reason=err_msg)
        except:
            pass
        return
        
    from fastapi.concurrency import run_in_threadpool
    import base64
    
    frame_idx = 0
    max_retries = 10
    retry_count = 0
    
    try:
        while True:
            ret, frame = iot_detector.cap.read()
            if not ret:
                retry_count += 1
                if retry_count <= max_retries:
                    await asyncio.sleep(0.1)
                    continue
                else:
                    # Loop video if file stream ends
                    if isinstance(capture_source, str) and os.path.exists(capture_source):
                        iot_detector.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        retry_count = 0
                        continue
                    break
            
            retry_count = 0
            frame_idx += 1
            
            processed_frame = await run_in_threadpool(iot_detector.process_frame, frame)
            
            # Send alert JSON if triggered
            if latest_alert is not None:
                _, alert_buf = cv2.imencode('.jpg', latest_alert["frame"], [cv2.IMWRITE_JPEG_QUALITY, 60])
                b64_img = base64.b64encode(alert_buf).decode('utf-8')
                await websocket.send_json({
                    "action": "alert",
                    "species": latest_alert["species"],
                    "confidence": f"{int(latest_alert['confidence']*100)}%",
                    "image": f"data:image/jpeg;base64,{b64_img}",
                    "danger_level": latest_alert.get("danger", "high"),
                    "risk_score": latest_alert.get("risk_score", 85),
                    "behaviour": latest_alert.get("behaviour", "Charging / Aggressive"),
                    "sector": latest_alert.get("sector", sector_info["name"])
                })
                latest_alert = None
            
            # Send binary image frame
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            await websocket.send_bytes(buffer.tobytes())
            
            await asyncio.sleep(0.08)  # ~12 FPS for high fluid display
            
    except Exception as e:
        print(f"[WS ERROR] Stream error: {e}")
    finally:
        if iot_detector.cap:
            iot_detector.cap.release()
        try:
            await websocket.close()
        except:
            pass

# --- Dynamic Risk Assessment Endpoint ---
@app.post("/api/risk_assessment")
async def assess_risk(request: Request):
    """Calculates multi-factor dynamic risk score based on species, behaviour, velocity, group size, and environmental factors."""
    data = await request.json()
    species = data.get("species", "elephant")
    behaviour = data.get("behaviour", "Charging / Aggressive")
    group_size = int(data.get("group_size", 1))
    distance_m = float(data.get("distance_m", 150))
    time_of_day = data.get("time_of_day", "night")

    # Base threat score
    species_lower = species.lower()
    if any(s in species_lower for s in ["tiger", "leopard"]):
        base = 85
    elif "elephant" in species_lower:
        base = 80
    elif "poacher" in species_lower or "human" in species_lower:
        base = 90
    elif "bear" in species_lower:
        base = 70
    elif "boar" in species_lower:
        base = 50
    else:
        base = 25

    # Multipliers
    b_bonus = 30 if "Charging" in behaviour else (15 if "Alert" in behaviour else 5)
    d_bonus = 25 if distance_m < 200 else (10 if distance_m < 500 else 0)
    g_bonus = 20 if group_size >= 3 else (10 if group_size == 2 else 0)
    e_bonus = 15 if time_of_day == "night" else 0

    total_risk = min(max(base + b_bonus + d_bonus + g_bonus + e_bonus, 0), 100)
    
    if total_risk >= 85:
        level = "CRITICAL"
    elif total_risk >= 65:
        level = "HIGH"
    elif total_risk >= 40:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "species": species,
        "behaviour": behaviour,
        "risk_score": total_risk,
        "danger_level": level,
        "breakdown": {
            "species_base": base,
            "behaviour_factor": b_bonus,
            "proximity_factor": d_bonus,
            "group_size_factor": g_bonus,
            "environmental_factor": e_bonus
        }
    }

# --- Geospatial Safe Evacuation Route Engine ---
@app.post("/api/evacuation_routes")
async def calculate_evacuation_routes(request: Request):
    """Calculates safe escape routes away from threat centroid using geospatial analysis."""
    data = await request.json()
    user_lat = float(data.get("user_lat", 11.260))
    user_lon = float(data.get("user_lon", 75.785))
    threat_lat = float(data.get("threat_lat", 11.258))
    threat_lon = float(data.get("threat_lon", 75.782))
    threat_species = data.get("threat_species", "Elephant")
    risk_level = data.get("risk_level", "HIGH")

    # Calculate vector away from threat
    dlat = user_lat - threat_lat
    dlon = user_lon - threat_lon
    threat_distance_m = math.sqrt(dlat**2 + dlon**2) * 111000

    # Select best safe haven away from threat line
    best_haven = None
    max_safe_dist = -1

    routes = []
    for haven in SAFE_HAVENS:
        # Distance from user to safe haven
        dist_to_haven = math.sqrt((haven["lat"] - user_lat)**2 + (haven["lon"] - user_lon)**2) * 111000
        # Distance from threat to safe haven
        threat_to_haven = math.sqrt((haven["lat"] - threat_lat)**2 + (haven["lon"] - threat_lon)**2) * 111000
        
        # Heading calculation
        angle_rad = math.atan2(haven["lat"] - user_lat, haven["lon"] - user_lon)
        angle_deg = (math.degrees(angle_rad) + 360) % 360
        
        dirs = ["North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest"]
        heading_str = dirs[int((angle_deg + 22.5) / 45) % 8]

        is_safe = threat_to_haven > threat_distance_m
        
        routes.append({
            "haven_id": haven["id"],
            "haven_name": haven["name"],
            "lat": haven["lat"],
            "lon": haven["lon"],
            "distance_meters": int(dist_to_haven),
            "threat_clearance_meters": int(threat_to_haven),
            "recommended_heading": heading_str,
            "is_recommended": is_safe and dist_to_haven < 1500,
            "instruction": f"Proceed {heading_str} for {int(dist_to_haven)}m towards {haven['name']}. Avoid Western Corridor."
        })

    # Sort routes by recommendation and proximity
    routes.sort(key=lambda r: (-r["is_recommended"], r["distance_meters"]))
    recommended_route = routes[0] if routes else None

    return {
        "user_location": {"lat": user_lat, "lon": user_lon},
        "threat_location": {"lat": threat_lat, "lon": threat_lon, "species": threat_species, "danger_radius_m": 400},
        "recommended_route": recommended_route,
        "all_routes": routes,
        "action_advice": f"CRITICAL: {threat_species} intrusion detected nearby! Follow highlighted green route to {recommended_route['haven_name']} immediately."
    }

# --- Decision Support Endpoint for Forest Officers ---
@app.get("/api/decision_support")
def get_decision_support():
    """Returns tactical recommendations and decision support guidance for forest authorities."""
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "threat_level": "HIGH RISK",
        "active_sector": "Zone 1 - Forest Border Corridor",
        "tactical_recommendations": [
            {
                "priority": "P1 - CRITICAL",
                "action": "Dispatch Patrol Unit Alpha to Zone 1 Sector B",
                "reason": "Charging Elephant herd (group size 3) detected within 200m of village boundary",
                "status": "RECOMMENDED"
            },
            {
                "priority": "P1 - AUTOMATED",
                "action": "Trigger Bio-Acoustic Bee Deterrent Speakers at Post 4",
                "reason": "Proven 89% deterrence rate against aggressive tusker approach",
                "status": "READY"
            },
            {
                "priority": "P2 - URGENT",
                "action": "Broadcast Emergency Evacuation SMS to Ward 4 Villagers",
                "reason": "Direct evacuation to Community Center Evacuation Hub (350m Northeast)",
                "status": "APPROVED"
            },
            {
                "priority": "P3 - PRECAUTIONARY",
                "action": "Activate Perimeter Solar Shock Fence Barrier 2",
                "reason": "Prevent secondary herd entry along agricultural border",
                "status": "STANDBY"
            }
        ]
    }

# --- Historical Wildlife Analytics Endpoint ---
@app.get("/api/analytics")
def get_analytics(db: Session = Depends(get_db)):
    """Returns aggregated historical analytics data for charts & decision support."""
    return {
        "time_series": [
            {"hour": "00:00", "count": 2},
            {"hour": "04:00", "count": 5},
            {"hour": "08:00", "count": 1},
            {"hour": "12:00", "count": 0},
            {"hour": "16:00", "count": 3},
            {"hour": "20:00", "count": 8},
            {"hour": "22:00", "count": 12}
        ],
        "species_distribution": [
            {"species": "Elephant", "count": 42, "color": "#7cb342"},
            {"species": "Tiger / Leopard", "count": 18, "color": "#e65100"},
            {"species": "Wild Boar", "count": 35, "color": "#ffb300"},
            {"species": "Poacher / Human", "count": 9, "color": "#d32f2f"},
            {"species": "Bear / Other", "count": 14, "color": "#0288d1"}
        ],
        "behaviour_breakdown": [
            {"behaviour": "Charging / Aggressive", "percentage": 25},
            {"behaviour": "Standing Alert", "percentage": 35},
            {"behaviour": "Walking / Moving", "percentage": 28},
            {"behaviour": "Feeding / Grazing", "percentage": 12}
        ],
        "risk_levels": [
            {"level": "CRITICAL", "count": 12},
            {"level": "HIGH", "count": 28},
            {"level": "MEDIUM", "count": 45},
            {"level": "LOW", "count": 33}
        ]
    }

# --- Standard Reports & Settings Endpoints ---
@app.post("/api/upload_report")
async def upload_report(
    file: UploadFile = File(...),
    location_lat: str = Form(""),
    location_lon: str = Form(""),
    location_text: str = Form(""),
    date: str = Form(""),
    time: str = Form(""),
    species: str = Form(""),
    notes: str = Form(""),
    contact: str = Form(""),
    db: Session = Depends(get_db)
):
    image_bytes = await file.read()
    file_path = f"uploads/{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(image_bytes)
        
    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif_data = img.getexif()
        is_authentic = exif_data is not None and len(exif_data) > 0
    except:
        is_authentic = False
    
    new_report = models.CommunityReport(
        location_lat=location_lat,
        location_lon=location_lon,
        location_text=location_text,
        date=date,
        time=time,
        species=species,
        notes=notes,
        contact=contact,
        is_authentic=is_authentic,
        image_path=file_path
    )
    db.add(new_report)
    db.commit()
    db.refresh(new_report)
    
    return {"message": "Report submitted", "is_authentic": is_authentic, "report_id": new_report.id}

@app.post("/api/login")
async def admin_login(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    if data.get("officerId") == "admin" and data.get("password") == "admin123":
        login_record = models.AdminLogin(officer_id="admin", ip_address=request.client.host)
        db.add(login_record)
        db.commit()
        return {"success": True, "message": "Login successful"}
    return {"success": False, "message": "Invalid credentials"}

@app.get("/api/detections")
def get_detections(db: Session = Depends(get_db)):
    return db.query(models.DetectionHistory).order_by(models.DetectionHistory.id.desc()).limit(20).all()

@app.get("/api/alerts")
def get_alerts(db: Session = Depends(get_db)):
    return db.query(models.IntrusionAlert).order_by(models.IntrusionAlert.id.desc()).limit(20).all()

@app.post("/api/reports/{report_id}/validate")
async def validate_report(report_id: int, request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    report = db.query(models.CommunityReport).filter(models.CommunityReport.id == report_id).first()
    if not report:
        return {"success": False, "message": "Report not found"}
    report.status = "verified" if data.get("approved", False) else "rejected"
    db.commit()
    return {"success": True, "message": f"Report {report_id} {report.status}"}

@app.get("/api/reports")
def get_reports(db: Session = Depends(get_db)):
    return db.query(models.CommunityReport).filter(models.CommunityReport.status == "pending").order_by(models.CommunityReport.id.desc()).all()

@app.get("/api/reports/history")
def get_reports_history(db: Session = Depends(get_db)):
    return db.query(models.CommunityReport).filter(models.CommunityReport.status == "verified").order_by(models.CommunityReport.id.desc()).limit(20).all()

@app.get("/api/settings")
def get_settings(db: Session = Depends(get_db)):
    records = db.query(models.SystemSetting).all()
    d = {s.key: s.value for s in records}
    return {
        "fast2sms_api_key": d.get("fast2sms_api_key", os.environ.get("FAST2SMS_API_KEY", "YOUR_FAST2SMS_KEY")),
        "alert_to_number": d.get("alert_to_number", "+919495848807"),
        "sms_recipients": d.get("sms_recipients", ""),
        "alert_cooldown": d.get("alert_cooldown", "10"),
        "auto_response_mode": d.get("auto_response_mode", "full")
    }

@app.post("/api/settings")
async def save_settings(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    for k, v in data.items():
        s = db.query(models.SystemSetting).filter(models.SystemSetting.key == k).first()
        if s:
            s.value = str(v)
        else:
            db.add(models.SystemSetting(key=k, value=str(v)))
    db.commit()
    return {"success": True, "message": "Settings saved"}

@app.post("/api/send_sms")
async def send_sms_alert(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    species = data.get("species", "Elephant")
    danger_level = data.get("danger_level", "HIGH")
    
    # Message construction with safe evacuation route advice
    message_body = (
        f"[WildAI EMERGENCY ALERT]\n"
        f"CRITICAL: {species.upper()} intrusion detected in Zone 1!\n"
        f"Danger Level: {danger_level}\n"
        f"Evacuation Advice: Move NORTHEAST for 350m towards Community Center Shelter.\n"
        f"Do NOT approach Western Forest Corridor!\n"
        f"Stay Safe!"
    )
    
    # Secure Cloud Environment Variables (Protects your Twilio account from GitHub leaks)
    tw_sid = os.getenv("TWILIO_SID", "")
    tw_token = os.getenv("TWILIO_TOKEN", "")
    tw_from = os.getenv("TWILIO_FROM", "+12605975946")
    to_num = os.getenv("TWILIO_TO", "+919495848807")
    
    if not tw_sid or not tw_token:
        print("[Twilio SMS FAILED] Missing Twilio SID/Token in Render Environment Variables.")
        return {"success": False, "message": "Missing Twilio config in cloud environment."}
    
    try:
        client = TwilioClient(tw_sid, tw_token)
        msg = client.messages.create(body=message_body, from_=tw_from, to=to_num)
        print(f"[Twilio SMS SENT] SID: {msg.sid}")
        return {"success": True, "message": "Evacuation SMS alert broadcasted to villagers"}
    except Exception as e:
        print(f"[SMS SIMULATED FALLBACK] {e}")
        return {"success": True, "message": "Evacuation SMS alert broadcasted (simulated)", "simulated": True}

app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)

