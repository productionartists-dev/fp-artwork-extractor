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
        "output_dir": str(OUTPUT_DIR),
        "public_base_url": PUBLIC_BASE_URL,
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
        str(requested_path),
        media_type="image/png",
        filename=requested_path.name,
    )


def render_page(page, zoom=3):
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return np.array(image)


def crop_nonwhite_region(page_image):
    """
    Fallback for banner/signage PDFs.
    Crops the main artwork area and avoids bottom print-details text.
    """

    h, w = page_image.shape[:2]

    # Ignore bottom details area.
    work = page_image.copy()
    work[int(h * 0.82):, :] = 255

    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)

    # Anything not close to white is content.
    mask = (gray < 245).astype(np.uint8) * 255

    # Remove top header area.
    mask[: int(h * 0.08), :] = 0

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh

        if area < w * h * 0.08:
            continue

        if bw < w * 0.20 or bh < h * 0.20:
            continue

        candidates.append((x, y, bw, bh))

    if not candidates:
        return None

    x, y, bw, bh = sorted(candidates, key=lambda b: b[2] * b[3], reverse=True)[0]

    pad = 10
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)

    return page_image[y1:y2, x1:x2]


def find_apparel_box(page_image):
    """
    Finds lower artwork box in apparel proofs.
    """

    h, w = page_image.shape[:2]

    search_y1 = int(h * 0.38)
    search = page_image[search_y1:, :]

    hsv = cv2.cvtColor(search, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    mask = np.zeros(search.shape[:2], dtype=np.uint8)

    mask[
        (sat > 3) &
        (val > 30) &
        (val < 255)
    ] = 255

    white = (sat < 6) & (val > 245)
    black = val < 25
    mask[white | black] = 0

    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        y = y + search_y1

        area = bw * bh

        if area < w * h * 0.012:
            continue

        if bw < w * 0.10 or bh < h * 0.07:
            continue

        aspect = bw / max(bh, 1)

        if aspect < 0.25 or aspect > 4.5:
            continue

        fill_ratio = cv2.contourArea(contour) / max(area, 1)

        if fill_ratio < 0.45:
            continue

        candidates.append((x, y, bw, bh))

    if not candidates:
        return None

    # Prefer lower artwork box, not garment mockup.
    candidates = sorted(candidates, key=lambda b: (b[1], -b[2] * b[3]))
    return candidates[0]


def crop_inside_box(page_image, box):
    x, y, w, h = box

    pad_x = int(w * 0.025)
    pad_y = int(h * 0.025)

    x1 = max(0, x + pad_x)
    y1 = max(0, y + pad_y)
    x2 = min(page_image.shape[1], x + w - pad_x)
    y2 = min(page_image.shape[0], y + h - pad_y)

    return page_image[y1:y2, x1:x2]


def extract_artwork_from_apparel_box(box_crop):
    """
    Removes solid colored box background and keeps only low-contrast artwork.
    """

    rgb = box_crop
    h, w = rgb.shape[:2]

    sample = max(10, min(h, w) // 18)

    corners = np.vstack([
        rgb[:sample, :sample].reshape(-1, 3),
        rgb[:sample, -sample:].reshape(-1, 3),
        rgb[-sample:, :sample].reshape(-1, 3),
        rgb[-sample:, -sample:].reshape(-1, 3),
    ])

    bg = np.median(corners, axis=0)

    diff = np.linalg.norm(
        rgb.astype(np.int16) - bg.astype(np.int16),
        axis=2,
    )

    p95 = np.percentile(diff, 95)
    p98 = np.percentile(diff, 98)

    threshold = max(1.5, min(10.0, ((p95 + p98) / 2) * 0.16))

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

    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w < w * 0.015 or crop_h < h * 0.015:
        return None

    pad_x = max(8, int(crop_w * 0.10))
    pad_y = max(8, int(crop_h * 0.12))

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


def save_image(image_array, path):
    if image_array.shape[2] == 4:
        Image.fromarray(image_array, mode="RGBA").save(path)
    else:
        Image.fromarray(image_array, mode="RGB").save(path)


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

        page_indexes = list(range(min(len(doc), 3)))

        for page_index in page_indexes:
            page = doc[page_index]
            page_number = page_index + 1

            page_image = render_page(page, zoom=3)

            # First try apparel lower box extraction.
            apparel_box = find_apparel_box(page_image)

            if apparel_box:
                box_crop = crop_inside_box(page_image, apparel_box)
                artwork = extract_artwork_from_apparel_box(box_crop)

                if artwork is not None:
                    filename = f"artwork_page_{page_number}_1.png"
                    png_path = job_dir / filename
                    save_image(artwork, png_path)

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
                        "extraction_type": "apparel_artwork_only",
                    })
                    break

            # Fallback for banner/artboard PDFs.
            artboard = crop_nonwhite_region(page_image)

            if artboard is not None:
                filename = f"artwork_page_{page_number}_1.png"
                png_path = job_dir / filename
                save_image(artboard, png_path)

                height, width = artboard.shape[:2]
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
