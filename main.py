from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import fitz
import cv2
import numpy as np
from PIL import Image
import imagehash
import uuid

app = FastAPI()

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


@app.get("/")
def home():
    return {"status": "Artwork extractor is running"}


@app.post("/extract-artwork")
async def extract_artwork(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = job_dir / file.filename

    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    extracted = process_pdf(pdf_path, job_dir)

    return {
        "job_id": job_id,
        "artworks_found": len(extracted),
        "artworks": extracted
    }


def process_pdf(pdf_path: Path, job_dir: Path):
    doc = fitz.open(pdf_path)
    results = []

    for page_index in range(len(doc)):
        page = doc[page_index]

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        page_img_path = job_dir / f"page_{page_index + 1}.png"
        pix.save(page_img_path)

        crops = extract_artwork_panels(page_img_path, job_dir, page_index + 1)
        results.extend(crops)

    return results


def extract_artwork_panels(page_img_path: Path, job_dir: Path, page_number: int):
    img = cv2.imread(str(page_img_path))
    height, width = img.shape[:2]

    # The artwork panels are usually below the garment mockup.
    # Start by looking in the lower half of the page.
    lower_half = img[int(height * 0.45):height, :]

    gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)

    # Find large non-black / non-white rectangular areas.
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    results = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        area = w * h
        if area < 50000:
            continue

        # Avoid tiny color circles and text areas
        if h < 150 or w < 150:
            continue

        y_absolute = y + int(height * 0.45)

        crop = img[y_absolute:y_absolute+h, x:x+w]

        if is_blank_crop(crop):
            continue

        crop_path = job_dir / f"artwork_page_{page_number}_{i + 1}.png"
        cv2.imwrite(str(crop_path), crop)

        pil_img = Image.open(crop_path)
        hash_value = str(imagehash.phash(pil_img))

        results.append({
            "page": page_number,
            "file": str(crop_path),
            "hash": hash_value,
            "width": w,
            "height": h
        })

    return results


def is_blank_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # If very low contrast, likely blank artwork panel
    contrast = gray.std()

    return contrast < 8
