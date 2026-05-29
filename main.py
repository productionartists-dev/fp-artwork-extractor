from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid
import shutil
import os

app = FastAPI()

# =====================================================
# CONFIG
# =====================================================

BASE_URL = "https://fp-artwork-extractor-production.up.railway.app"

OUTPUT_DIR = Path("output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMP_DIR = Path("temp").resolve()
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# SERVE OUTPUT FILES
# =====================================================

app.mount(
    "/output",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="output"
)

# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/")
async def root():
    return {
        "status": "running",
        "output_directory": str(OUTPUT_DIR)
    }

# =====================================================
# TEST FILE ROUTE
# =====================================================

@app.get("/test-file")
async def test_file():

    test_folder = OUTPUT_DIR / "test"
    test_folder.mkdir(parents=True, exist_ok=True)

    test_file = test_folder / "test.txt"

    with open(test_file, "w") as f:
        f.write("hello world")

    return {
        "url": f"{BASE_URL}/output/test/test.txt"
    }

# =====================================================
# PDF ARTWORK EXTRACTION
# =====================================================

@app.post("/extract-artwork")
async def extract_artwork(
    file: UploadFile = File(...)
):

    job_id = str(uuid.uuid4())

    job_folder = OUTPUT_DIR / job_id
    job_folder.mkdir(parents=True, exist_ok=True)

    uploaded_pdf = TEMP_DIR / f"{job_id}.pdf"

    with open(uploaded_pdf, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    artworks = []

    # =================================================
    # PUT YOUR EXISTING CUTOUT LOGIC HERE
    # =================================================
    #
    # Example:
    #
    # artwork_page_1_1.png
    # artwork_page_2_1.png
    #
    # Save all PNGs into:
    #
    # job_folder
    #
    # =================================================

    example_png = job_folder / "artwork_page_1_1.png"

    with open(example_png, "wb") as f:
        f.write(b"PNG PLACEHOLDER")

    artworks.append({
        "page": 1,
        "file": f"output/{job_id}/artwork_page_1_1.png",
        "url": f"{BASE_URL}/output/{job_id}/artwork_page_1_1.png",
        "hash": None,
        "width": None,
        "height": None
    })

    return {
        "job_id": job_id,
        "artworks_found": len(artworks),
        "artworks": artworks
    }
