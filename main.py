import os
import uuid
import shutil
from pathlib import Path

import fitz
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

OUTPUT_DIR = Path("/app/output")
TEMP_DIR = Path("/tmp/artwork-temp")

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
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return np.array(image)


def find_apparel_artwork_box(page_image):
    h, w = page_image.shape[:2]

    search_y1 = int(h * 0.42)
    search = page_image[search_y1:h, :]

    hsv = cv2.cvtColor(search, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    mask = np.zeros(search.shape[:2], dtype=np.uint8)
    mask[(sat > 3) & (val > 35) & (val < 255)] = 255

    white = (sat < 6) & (val > 245)
    black = val < 30
    mask[white | black] = 0

    kernel = np.ones((31, 31), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        y_global = y + search_y1
        area = bw * bh

        if area < w * h * 0.015:
            continue
        if bw < w * 0.12 or bh < h * 0.08:
            continue

        aspect = bw / max(bh, 1)
        if aspect < 0.35 or aspect > 3.8:
            continue

        fill_ratio = cv2.contourArea(contour) / max(area, 1)
        if fill_ratio < 0.55:
            continue

        candidates.append((x, y_global, bw, bh))

    candidates = sorted(candidates, key=lambda b: (b[1], b[0]))
    return candidates[0] if candidates else None


def find_large_bordered_artboard(page_image):
    """
    For banner/signage proofs:
    Finds the large artwork artboard with a rectangular border.
    """

    h, w = page_image.shape[:2]

    gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)

    # Detect dark border/linework
    _, dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Ignore the very bottom text area if present
    dark[int(h * 0.88):, :] = 0

    kernel = np.ones((5, 5), np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh

        if area < w * h * 0.10:
            continue

        if bw < w * 0.25 or bh < h * 0.25:
            continue

        aspect = bw / max(bh, 1)
        if aspect < 0.25 or aspect > 2.5:
            continue

        # Prefer artboards above print-detail text area
        if y > h * 0.45:
            continue

        candidates.append((x, y, bw, bh))

    if candidates:
        candidates = sorted(candidates, key=lambda b: b[2] * b[3], reverse=True)
        return candidates[0]

    # Fallback for page 2 banner proofs:
    # crop the main centered artwork region above the print details.
    nonwhite = np.any(page_image < 245, axis=2).astype(np.uint8) * 255
    nonwhite[int(h * 0.82):, :] = 0

    contours, _ = cv2.findContours(nonwhite, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh

        if area < w * h * 0.12:
            continue

        candidates.append((x, y, bw, bh))

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda b: b[2] * b[3], reverse=True)
    return candidates[0]


def crop_box(page_image, box, pad_ratio=0.015):
    x, y, w, h = box

    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)

    x1 = max(0, x + pad_x)
    y1 = max(0, y + pad_y)
    x2 = min(page_image.shape[1], x + w - pad_x)
    y2 = min(page_image.shape[0], y + h - pad_y)

    return page_image[y1:y2, x1:x2]


def remove_background_for_apparel_box(box_crop):
    rgb = box_crop
    h, w = rgb.shape[:2]

    sample = max(10, min(h, w) // 18)

    corners = np.vstack([
        rgb[:sample, :sample].reshape(-1, 3),
        rgb[:sample, -sample:].reshape(-1, 3),
        rgb[-sample:, :sample].reshape(-1, 3),
        rgb[-sample:, -sample:].reshape(-1, 3),
    ])

    bg_color = np.median(corners, axis=0)

    diff = np.linalg.norm(
        rgb.astype(np.int16) - bg_color.astype(np.int16),
        axis=2,
    )

    p95 = np.percentile(diff, 95)
    p98 = np.percentile(diff, 98)
    threshold = max(1.8, min(9.0, ((p95 + p98) / 2) * 0.18))

    mask = (diff > threshold).astype(np.uint8) * 255

    border_x = max(2, int(w * 0.018))
    border_y = max(2, int(h * 0.018))
    mask[:border_y, :] = 0
    mask[-border_y:, :] = 0
    mask[:, :border_x] = 0
    mask[:, -border_x:] = 0

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))

    ys, xs = np.where(mask > 0)

    if len(xs) < 10 or len(ys) < 10:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    pad_x = max(8, int((x2 - x1) * 0.10))
    pad_y = max(8, int((y2 - y1) * 0.12))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    cropped_rgb = rgb[y1:y2, x1:x2]
    cropped_alpha = mask[y1:y2, x1:x2]

    return np.dstack([
        cropped_rgb[:, :, 0],
        cropped_rgb[:, :, 1],
        cropped_rgb[:, :, 2],
        cropped_alpha,
    ])


def save_rgb_png(rgb_array, path):
    Image.fromarray(rgb_array, mode="RGB").save(path)


def save_rgba_png(rgba_array, path):
    Image.fromarray(rgba_array, mode="RGBA").save(path)


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

        # Try page 1 first, then page 2 if available.
        page_indexes = [0]
        if len(doc) > 1:
            page_indexes.append(1)

        for page_index in page_indexes:
            page = doc[page_index]
            page_number = page_index + 1
            page_image = render_pdf_page(page, zoom=3)

            apparel_box = find_apparel_artwork_box(page_image)

            if apparel_box:
                box_crop = crop_box(page_image, apparel_box)
                artwork = remove_background_for_apparel_box(box_crop)

                if artwork is not None:
                    filename = f"artwork_page_{page_number}_1.png"
                    png_path = job_dir / filename

                    save_rgba_png(artwork, png_path)

                    height, width = artwork.shape[:2]
                    url = f"{PUBLIC_BASE_URL}/output/{job_id}/{filename}"

                    artworks.append({
                        "page": page_number,
                        "design_location_index": 1,
                        "artwork_url": url,
                        "image_url": url,
                        "file": f"output/{job_id}/{filename}",
                        "local_path": str(png_path),
                        "exists": png_path.exists(),
                        "width": width,
                        "height": height,
                        "extraction_type": "apparel_box_transparent",
                    })

                    break

            artboard_box = find_large_bordered_artboard(page_image)

            if artboard_box:
                artwork = crop_box(page_image, artboard_box, pad_ratio=0.005)

                filename = f"artwork_page_{page_number}_1.png"
                png_path = job_dir / filename

                save_rgb_png(artwork, png_path)

                height, width = artwork.shape[:2]
                url = f"{PUBLIC_BASE_URL}/output/{job_id}/{filename}"

                artworks.append({
                    "page": page_number,
                    "design_location_index": 1,
                    "artwork_url": url,
                    "image_url": url,
                    "file": f"output/{job_id}/{filename}",
                    "local_path": str(png_path),
                    "exists": png_path.exists(),
                    "width": width,
                    "height": height,
                    "extraction_type": "full_artboard_crop",
                })

                break

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
            relative = path.relative_to(OUTPUT_DIR).as_posix()
            files.append({
                "file": f"output/{relative}",
                "url": f"{PUBLIC_BASE_URL}/output/{relative}",
                "local_path": str(path),
                "size_bytes": path.stat().st_size,
            })

    return {
        "output_dir": str(OUTPUT_DIR),
        "file_count": len(files),
        "files": files,
    }
