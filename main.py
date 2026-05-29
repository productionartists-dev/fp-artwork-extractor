import os
import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware


# =====================================================
# APP SETUP
# =====================================================

app = FastAPI(title="Fresh Prints Artwork Extractor")


# =====================================================
# CONFIG
# =====================================================

PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "https://fp-artwork-extractor-production.up.railway.app"
).rstrip("/")

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# ALLOW PUBLIC ACCESS / CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/")
async def root():
    return {
        "status": "running",
        "public_base_url": PUBLIC_BASE_URL,
        "output_dir": str(OUTPUT_DIR),
        "temp_dir": str(TEMP_DIR),
    }


# =====================================================
# PUBLIC FILE ACCESS
# Anyone with the link can access:
# /output/{job_id}/{filename}
# =====================================================

@app.get("/output/{file_path:path}")
async def public_output_file(file_path: str):
    requested_path = (OUTPUT_DIR / file_path).resolve()

    # Security check: prevent access outside /output
    if not str(requested_path).startswith(str(OUTPUT_DIR.resolve())):
        return JSONResponse(
            status_code=403,
            content={"detail": "Access denied"}
        )

    if not requested_path.exists() or not requested_path.is_file():
        return JSONResponse(
            status_code=404,
            content={
                "detail": "File not found",
                "requested_file": file_path,
                "resolved_path": str(requested_path),
            }
        )

    return FileResponse(
        path=str(requested_path),
        filename=requested_path.name,
        media_type="image/png"
    )


# =====================================================
# DEBUG ROUTE: CREATE A TEST PNG
# =====================================================

@app.get("/debug/create-test-image")
async def create_test_image():
    try:
        from PIL import Image, ImageDraw
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Pillow is not installed",
                "error": str(e),
                "fix": "Add pillow to requirements.txt"
            }
        )

    job_id = "debug-test"
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    image_path = job_dir / "test.png"

    image = Image.new("RGB", (600, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 560, 260), outline="black", width=6)
    draw.text((70, 130), "Static file access works", fill="black")
    image.save(image_path)

    return {
        "status": "created",
        "file": f"output/{job_id}/test.png",
        "url": f"{PUBLIC_BASE_URL}/output/{job_id}/test.png"
    }


# =====================================================
# EXTRACT ARTWORK ENDPOINT
# Replace the placeholder section with your existing
# PDF cutout/extraction code.
# =====================================================

@app.post("/extract-artwork")
async def extract_artwork(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())

    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    uploaded_pdf_path = TEMP_DIR / f"{job_id}.pdf"

    with open(uploaded_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    artworks = []

    # =================================================
    # IMPORTANT:
    # Your actual cutout code must save PNG files into:
    #
    # job_dir
    #
    # Example expected saved file:
    # output/{job_id}/artwork_page_1_1.png
    #
    # Do not save to /tmp/output unless you also change
    # OUTPUT_DIR to that same location.
    # =================================================

    # -------------------------------------------------
    # PLACEHOLDER TEST IMAGE
    # Remove this block after adding your real cutout code.
    # -------------------------------------------------
    try:
        from PIL import Image, ImageDraw

        test_png_path = job_dir / "artwork_page_1_1.png"

        image = Image.new("RGB", (800, 500), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((50, 50, 750, 450), outline="black", width=8)
        draw.text((100, 230), "Placeholder artwork cutout", fill="black")
        image.save(test_png_path)

        artworks.append({
            "page": 1,
            "file": f"output/{job_id}/artwork_page_1_1.png",
            "url": f"{PUBLIC_BASE_URL}/output/{job_id}/artwork_page_1_1.png",
            "hash": None,
            "width": 800,
            "height": 500,
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Failed to create placeholder image",
                "error": str(e)
            }
        )

    return {
        "job_id": job_id,
        "artworks_found": len(artworks),
        "artworks": artworks
    }


# =====================================================
# LIST FILES FOR DEBUGGING
# =====================================================

@app.get("/debug/list-output")
async def list_output_files():
    files = []

    for path in OUTPUT_DIR.rglob("*"):
        if path.is_file():
            relative_path = path.relative_to(OUTPUT_DIR)
            files.append({
                "file": f"output/{relative_path.as_posix()}",
                "url": f"{PUBLIC_BASE_URL}/output/{relative_path.as_posix()}",
                "size_bytes": path.stat().st_size,
            })

    return {
        "output_dir": str(OUTPUT_DIR),
        "file_count": len(files),
        "files": files,
    }
