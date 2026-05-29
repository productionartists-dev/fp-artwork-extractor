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


def find_apparel_artwork_box(page_image):
    h, w = page_image.shape[:2]

    hsv = cv2.cvtColor(page_image, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    mask = np.zeros((h, w), dtype=np.uint8)

    # Colored/pastel regions only.
    mask[
        (sat > 3) &
        (val > 28) &
        (val < 255)
    ] = 255

    # Remove black page background and white page/mockup areas.
    white = (sat < 6) & (val > 245)
    black = val < 24
    mask[white | black] = 0

    # Critical: ignore upper mockup region.
    # Apparel artwork boxes are usually below the product mockup and print details.
    mask[: int(h * 0.50), :] = 0

    kernel = np.ones((31, 31), np.uint8)
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
        area = bw * bh
        page_area = w * h

        if area < page_area * 0.012:
            continue

        if bw < w * 0.12 or bh < h * 0.08:
            continue

        aspect = bw / max(bh, 1)

        # Artwork boxes are normally square-ish/rectangular.
        if aspect < 0.30 or aspect > 3.80:
            continue

        contour_area = cv2.contourArea(contour)
        fill_ratio = contour_area / max(area, 1)

        # Garment mockups are irregular; artwork boxes are more solid.
        if fill_ratio < 0.55:
            continue

        candidates.append((x, y, bw, bh, area))

    if not candidates:
        return None

    # Pick the largest lower solid rectangle.
    candidates = sorted(candidates, key=lambda b: b[4], reverse=True)
    x, y, bw, bh, _ = candidates[0]

    return (x, y, bw, bh)


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
    rgb = box_crop
    h, w = rgb.shape[:2]

    sample = max(12, min(h, w) // 16)

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

    # Low-contrast sensitive threshold.
    p90 = np.percentile(diff, 90)
    p98 = np.percentile(diff, 98)
    threshold = max(1.2, min(8.5, ((p90 + p98) / 2) * 0.14))

    mask = (diff > threshold).astype(np.uint8) * 255

    # Remove box edges.
    bx = max(3, int(w * 0.020))
    by = max(3, int(h * 0.020))
    mask[:by, :] = 0
    mask[-by:, :] = 0
    mask[:, :bx] = 0
    mask[:, -bx:] = 0

    # Clean noise but preserve letters.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Remove small speckles.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]

        if area < 25:
            continue

        if bw < 4 or bh < 4:
            continue

        cleaned[labels == label] = 255

    ys, xs = np.where(cleaned > 0)

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
    cropped_alpha = cleaned[y1:y2, x1:x2]

    return np.dstack([
        cropped_rgb[:, :, 0],
        cropped_rgb[:, :, 1],
        cropped_rgb[:, :, 2],
        cropped_alpha,
    ])


def crop_banner_artboard(page_image):
    h, w = page_image.shape[:2]

    work = page_image.copy()

    # Ignore top Fresh Prints header and bottom print details.
    work[: int(h * 0.08), :] = 255
    work[int(h * 0.82):, :] = 255

    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    mask = (gray < 245).astype(np.uint8) * 255

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

        candidates.append((x, y, bw, bh, area))

    if not candidates:
        return None

    x, y, bw, bh, _ = sorted(candidates, key=lambda b: b[4], reverse=True)[0]

    pad = 8
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)

    return page_image[y1:y2, x1:x2]


def save_image(image_array, path):
    if image_array.ndim == 3 and image_array.shape[2] == 4:
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

        # Apparel proof: use page 1 only to avoid duplicated artboards/pages.
        page = doc[0]
        page_image = render_page(page, zoom=3)

        artwork = None
        extraction_type = None

        box = find_apparel_artwork_box(page_image)

        if box:
            box_crop = crop_inside_box(page_image, box)
            artwork = extract_artwork_from_box(box_crop)
            extraction_type = "apparel_artwork_only"

        # Fallback: banner/signage, usually page 2 has full artboard.
        if artwork is None and len(doc) > 1:
            for page_index in range(1, min(len(doc), 3)):
                page_image = render_page(doc[page_index], zoom=3)
                artboard = crop_banner_artboard(page_image)

                if artboard is not None:
                    artwork = artboard
                    extraction_type = "full_artboard_crop"
                    break

        if artwork is not None:
            filename = "artwork_page_1_1.png"
            png_path = job_dir / filename
            save_image(artwork, png_path)

            height, width = artwork.shape[:2]
            url = f"{PUBLIC_BASE_URL}/output/{job_id}/{filename}"

            artworks.append({
                "page": 1,
                "design_location_index": 1,
                "artwork_url": url,
                "image_url": url,
                "file": f"output/{job_id}/{filename}",
                "local_path": str(png_path),
                "exists": png_path.exists(),
                "width": width,
                "height": height,
                "extraction_type": extraction_type,
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
