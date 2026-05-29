import os
import uuid
import shutil
from pathlib import Path

import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware


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
        "output_dir": str(OUTPUT_DIR),
    }


@app.get("/output/{file_path:path}")
async def public_output_file(file_path: str):
    requested_path = (OUTPUT_DIR / file_path).resolve()

    if not str(requested_path).startswith(str(OUTPUT_DIR.resolve())):
        return JSONResponse(status_code=403, content={"detail": "Access denied"})

    if not requested_path.exists() or not requested_path.is_file():
        return JSONResponse(
            status_code=404,
            content={
                "detail": "File not found",
                "requested_file": file_path,
                "resolved_path": str(requested_path),
            },
        )

    return FileResponse(
        path=str(requested_path),
        media_type="image/png",
        filename=requested_path.name,
    )


def render_pdf_page(page, zoom=3):
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return np.array(image)


def find_artwork_boxes(page_image):
    rgb = page_image
    h, w = rgb.shape[:2]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = np.zeros((h, w), dtype=np.uint8)

    # Detect colored filled artwork boxes, excluding black background and white shirt mockup.
    mask[(saturation > 15) & (value > 35) & (value < 245)] = 255

    # Ignore top area where product mockup usually appears.
    mask[: int(h * 0.32), :] = 0

    kernel = np.ones((19, 19), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh

        if area < w * h * 0.025:
            continue

        if bw < w * 0.15 or bh < h * 0.08:
            continue

        if y < h * 0.32:
            continue

        boxes.append((x, y, bw, bh))

    return sorted(boxes, key=lambda b: (b[1], b[0]))


def crop_inside_box(page_image, box):
    x, y, w, h = box

    pad_x = int(w * 0.025)
    pad_y = int(h * 0.025)

    x1 = max(0, x + pad_x)
    y1 = max(0, y + pad_y)
    x2 = min(page_image.shape[1], x + w - pad_x)
    y2 = min(page_image.shape[0], y + h - pad_y)

    return page_image[y1:y2, x1:x2]


def extract_artwork_from_box(box_crop):
    """
    Removes the colored box background and returns only the artwork
    as a transparent PNG crop.
    """

    rgb = box_crop
    h, w = rgb.shape[:2]

    sample_size = max(10, min(h, w) // 20)

    corners = np.vstack([
        rgb[:sample_size, :sample_size].reshape(-1, 3),
        rgb[:sample_size, -sample_size:].reshape(-1, 3),
        rgb[-sample_size:, :sample_size].reshape(-1, 3),
        rgb[-sample_size:, -sample_size:].reshape(-1, 3),
    ])

    bg_color = np.median(corners, axis=0)

    diff = np.linalg.norm(
        rgb.astype(np.int16) - bg_color.astype(np.int16),
        axis=2
    )

    # This finds pixels that differ from the box background.
    mask = (diff > 14).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    pad_x = max(8, int(w * 0.025))
    pad_y = max(8, int(h * 0.025))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    cropped_rgb = rgb[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]

    # Make background transparent.
    alpha = cropped_mask

    rgba = np.dstack([
        cropped_rgb[:, :, 0],
        cropped_rgb[:, :, 1],
        cropped_rgb[:, :, 2],
        alpha,
    ])

    return rgba


def save_transparent_png(rgba_array, path):
    image = Image.fromarray(rgba_array, mode="RGBA")
    image.save(path)


@app.post("/extract-artwork")
async def extract_artwork(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())

    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    uploaded_pdf_path = TEMP_DIR / f"{job_id}.pdf"

    with open(uploaded_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    artworks = []

    try:
        doc = fitz.open(str(uploaded_pdf_path))

        for page_index in range(len(doc)):
            page_number = page_index + 1
            page = doc[page_index]

            page_image = render_pdf_page(page, zoom=3)
            boxes = find_artwork_boxes(page_image)

            for box_index, box in enumerate(boxes, start=1):
                box_crop = crop_inside_box(page_image, box)
                artwork_rgba = extract_artwork_from_box(box_crop)

                if artwork_rgba is None:
                    continue

                filename = f"artwork_page_{page_number}_{box_index}.png"
                png_path = job_dir / filename

                save_transparent_png(artwork_rgba, png_path)

                height, width = artwork_rgba.shape[:2]

                artworks.append({
                    "page": page_number,
                    "design_location_index": len(artworks) + 1,
                    "artwork_url": f"{PUBLIC_BASE_URL}/output/{job_id}/{filename}",
                    "file": f"output/{job_id}/{filename}",
                    "local_path": str(png_path),
                    "exists": png_path.exists(),
                    "width": width,
                    "height": height,
                })

        doc.close()

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Failed to extract artwork",
                "error": str(e),
            },
        )

    return {
        "job_id": job_id,
        "artworks_found": len(artworks),
        "artworks": artworks,
    }


@app.get("/debug/list-output")
async def list_output_files():
    files = []

    for path in OUTPUT_DIR.rglob("*"):
        if path.is_file():
            relative_path = path.relative_to(OUTPUT_DIR).as_posix()
            files.append({
                "url": f"{PUBLIC_BASE_URL}/output/{relative_path}",
                "file": f"output/{relative_path}",
                "local_path": str(path),
                "size_bytes": path.stat().st_size,
            })

    return {
        "output_dir": str(OUTPUT_DIR),
        "file_count": len(files),
        "files": files,
    }
