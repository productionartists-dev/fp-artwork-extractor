import os
import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image, ImageDraw


app = FastAPI(title="Fresh Prints Artwork Extractor")

PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "https://fp-artwork-extractor-production.up.railway.app"
).rstrip("/")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "status": "running",
        "public_base_url": PUBLIC_BASE_URL,
        "base_dir": str(BASE_DIR),
        "output_dir": str(OUTPUT_DIR),
        "temp_dir": str(TEMP_DIR),
    }


@app.get("/output/{file_path:path}")
async def public_output_file(file_path: str):
    requested_path = (OUTPUT_DIR / file_path).resolve()

    if not str(requested_path).startswith(str(OUTPUT_DIR.resolve())):
        return JSONResponse(
            status_code=403,
            content={"detail": "Access denied"},
        )

    if not requested_path.exists() or not requested_path.is_file():
        return JSONResponse(
            status_code=404,
            content={
                "detail": "File not found",
                "requested_file": file_path,
                "resolved_path": str(requested_path),
                "output_dir": str(OUTPUT_DIR),
            },
        )

    return FileResponse(
        path=str(requested_path),
        media_type="image/png",
        filename=requested_path.name,
    )


@app.get("/debug/create-test-image")
async def create_test_image():
    job_id = "debug-test"
    filename = "test.png"

    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    png_path = job_dir / filename

    image = Image.new("RGB", (800, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 50, 750, 450), outline="black", width=8)
    draw.text((100, 230), "Static file access works", fill="black")
    image.save(png_path)

    return {
        "status": "created",
        "exists": png_path.exists(),
        "local_path": str(png_path),
        "file": f"output/{job_id}/{filename}",
        "url": f"{PUBLIC_BASE_URL}/output/{job_id}/{filename}",
    }


@app.get("/debug/list-output")
async def list_output_files():
    files = []

    for path in OUTPUT_DIR.rglob("*"):
        if path.is_file():
            relative_path = path.relative_to(OUTPUT_DIR).as_posix()
            files.append({
                "file": f"output/{relative_path}",
                "url": f"{PUBLIC_BASE_URL}/output/{relative_path}",
                "local_path": str(path),
                "size_bytes": path.stat().st_size,
            })

    return {
        "output_dir": str(OUTPUT_DIR),
        "file_count": len(files),
        "files": files,
    }


@app.post("/extract-artwork")
async def extract_artwork(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())

    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    uploaded_pdf_path = TEMP_DIR / f"{job_id}.pdf"

    with open(uploaded_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    artworks = []

    # =====================================================
    # IMPORTANT:
    # Replace this placeholder block with your real cutout code.
    # Your real cutout code MUST save PNGs into job_dir.
    # Example:
    # png_path = job_dir / "artwork_page_1_1.png"
    # image.save(png_path)
    # =====================================================

    filename = "artwork_page_1_1.png"
    png_path = job_dir / filename

    image = Image.new("RGB", (800, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 50, 750, 450), outline="black", width=8)
    draw.text((100, 230), "Placeholder artwork cutout", fill="black")
    image.save(png_path)

    print("SAVED PNG:", str(png_path))
    print("PNG EXISTS:", png_path.exists())

    artworks.append({
        "page": 1,
        "file": f"output/{job_id}/{filename}",
        "url": f"{PUBLIC_BASE_URL}/output/{job_id}/{filename}",
        "local_path": str(png_path),
        "exists": png_path.exists(),
        "hash": None,
        "width": 800,
        "height": 500,
    })

    return {
        "job_id": job_id,
        "uploaded_pdf_path": str(uploaded_pdf_path),
        "output_dir": str(OUTPUT_DIR),
        "artworks_found": len(artworks),
        "artworks": artworks,
    }
