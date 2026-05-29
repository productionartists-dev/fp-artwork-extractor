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


def find_lower_artwork_box(page_image):
    h, w = page_image.shape[:2]

    hsv = cv2.cvtColor(page_image, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    mask = np.zeros((h, w), dtype=np.uint8)

    # Find colored/pastel solid boxes.
    mask[(sat > 3) & (val > 30) & (val < 255)] = 255

    # Remove white page and black background.
    mask[((sat < 8) & (val > 245)) | (val < 25)] = 0

    # Only search the lower half where the proof artwork box lives.
    mask[: int(h * 0.48), :] = 0

    kernel = np.ones((35, 35), np.uint8)
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

        if area < page_area * 0.025:
            continue

        if bw < w * 0.35 or bh < h * 0.15:
            continue

        aspect = bw / max(bh, 1)
        if aspect < 0.35 or aspect > 4.5:
            continue

        fill_ratio = cv2.contourArea(contour) / max(area, 1)
        if fill_ratio < 0.55:
            continue

        candidates.append((x, y, bw, bh, area))

    if not candidates:
        return None

    # Largest lower solid rectangle = artwork proof box.
    candidates.sort(key=lambda b: b[4], reverse=True)
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


def extract_logo_from_box(box_crop):
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

    # Low contrast sensitive, but not so low that it selects the whole box.
    p90 = np.percentile(diff, 90)
    p97 = np.percentile(diff, 97)
    p99 = np.percentile(diff, 99)

    threshold = max(2.0, min(12.0, (p97 + p99) * 0.12))

    if p90 > 8:
        threshold = max(threshold, p90 * 0.75)

    mask = (diff > threshold).astype(np.uint8) * 255

    # Remove box edges.
    bx = max(4, int(w * 0.025))
    by = max(4, int(h * 0.025))
    mask[:by, :] = 0
    mask[-by:, :] = 0
    mask[:, :bx] = 0
    mask[:, -bx:] = 0

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]

        # Remove tiny noise.
        if area < 30:
            continue

        # Remove giant accidental background selections.
        if area > w * h * 0.65:
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

    pad_x = max(10, int(crop_w * 0.08))
    pad_y = max(10, int(crop_h * 0.08))

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
    work[: int(h * 0.08), :] = 255
    work[int(h * 0.84):, :] = 255

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

        if area < w * h * 0.10:
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

        artwork = None
        source_page = None
        extraction_type = None

        # Apparel proof: page 2 usually has the clean large artwork box.
        apparel_page_order = []
        if len(doc) > 1:
            apparel_page_order.append(1)
        apparel_page_order.append(0)

        for page_index in apparel_page_order:
            page_image = render_page(doc[page_index], zoom=3)
            box = find_lower_artwork_box(page_image)

            if not box:
                continue

            box_crop = crop_inside_box(page_image, box)
            logo = extract_logo_from_box(box_crop)

            if logo is not None:
                artwork = logo
                source_page = page_index + 1
                extraction_type = "apparel_logo_only"
                break

        # Banner/signage fallback.
        if artwork is None:
            for page_index in range(min(len(doc), 3)):
                page_image = render_page(doc[page_index], zoom=3)
                artboard = crop_banner_artboard(page_image)

                if artboard is not None:
                    artwork = artboard
                    source_page = page_index + 1
                    extraction_type = "full_artboard_crop"
                    break

        if artwork is not None:
            filename = "artwork_page_1_1.png"
            png_path = job_dir / filename
            save_image(artwork, png_path)

            height, width = artwork.shape[:2]
            url = f"{PUBLIC_BASE_URL}/output/{job_id}/{filename}"

            artworks.append({
                "page": source_page,
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
