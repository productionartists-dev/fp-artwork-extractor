import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# --------------------------------------------------
# CREATE OUTPUT FOLDER
# --------------------------------------------------

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# SERVE OUTPUT FOLDER PUBLICLY
# --------------------------------------------------

app.mount(
    "/output",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="output"
)

# --------------------------------------------------
# TEST ROUTE
# --------------------------------------------------

@app.get("/")
def root():
    return {"status": "running"}

# --------------------------------------------------
# EXAMPLE EXTRACT ENDPOINT
# --------------------------------------------------

@app.post("/extract-artwork")
async def extract_artwork(file: UploadFile = File(...)):

    job_id = str(uuid.uuid4())

    job_folder = OUTPUT_DIR / job_id
    job_folder.mkdir(parents=True, exist_ok=True)

    #
    # YOUR PDF EXTRACTION LOGIC GOES HERE
    #
    # Example:
    #
    # artwork_page_1_1.png
    # artwork_page_2_1.png
    #

    png_filename = "artwork_page_1_1.png"
    png_path = job_folder / png_filename

    # Example file write
    with open(png_path, "wb") as f:
        f.write(b"placeholder")

    return {
        "job_id": job_id,
        "artworks_found": 1,
        "artworks": [
            {
                "page": 1,
                "file": f"output/{job_id}/{png_filename}",
                "url": f"/output/{job_id}/{png_filename}"
            }
        ]
    }
