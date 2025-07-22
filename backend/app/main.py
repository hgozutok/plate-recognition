from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import sqlite3
import os
import cv2
import numpy as np
from PIL import Image
import io
from .plate_detection import process_image_file

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the captures directory for serving images
app.mount("/captures", StaticFiles(directory="captures"), name="captures")

# Database initialization
def init_db():
    conn = sqlite3.connect('/app/data.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT,
            capture_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Startup event to initialize database and create directories
@app.on_event("startup")
async def startup_event():
    init_db()
    os.makedirs("captures", exist_ok=True)

# Simple plate detection function (placeholder)
def detect_plate_number(image):
    # TODO: Implement actual plate detection logic
    # This is a placeholder that should be replaced with actual detection code
    return "ABC123"

@app.post("/api/detect")
async def detect_plate(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Process the image and get results
        plate_number, debug_image, plate_region, confidence = process_image_file(contents)
        
        # Generate filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_filename = f"debug_{timestamp}.jpg"
        plate_filename = f"plate_{timestamp}.jpg"
        
        # Save debug image
        debug_filepath = os.path.join("captures", debug_filename)
        debug_image.save(debug_filepath)
        
        # Save plate region if found
        plate_filepath = None
        if plate_region:
            plate_filepath = os.path.join("captures", plate_filename)
            plate_region.save(plate_filepath)
        
        # Save to database
        conn = sqlite3.connect('/app/data.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO plates (plate_number, image_path) VALUES (?, ?)",
            (plate_number, debug_filename)
        )
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "plate_number": plate_number,
            "debug_image": debug_filename,
            "plate_image": plate_filename if plate_region else None,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history():
    try:
        conn = sqlite3.connect('/app/data.db')
        c = conn.cursor()
        c.execute("SELECT * FROM plates ORDER BY capture_time DESC")
        rows = c.fetchall()
        conn.close()
        
        return [
            {
                "id": row[0],
                "plate_number": row[1],
                "capture_time": row[2],
                "image_path": row[3]
            }
            for row in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
