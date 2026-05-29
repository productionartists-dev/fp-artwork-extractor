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

    if not requested_path.exists():
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


def find_bottom_artwork_boxes(page_image):
    h, w = page_image.shape[:2]

    # Only search bottom half where artwork boxes usually are
    search_y1 = int(h * 0.38)
    search = page_image[search_y1:h, :]

    hsv = cv2.cvtColor(search, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Detect pastel/colored box backgrounds, including very light pink
    mask = np.zeros(search.shape[:2], dtype=np.uint8)
    mask[
        (sat > 3) &
        (val > 35) &
        (val < 255)
    ] = 255

    # Remove white areas and black background
    white = (sat < 6) & (val > 245)
    black = val < 30
    mask[white | black] = 0

    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        y = y + search_y1

        area = bw * bh

        # Artwork box should be medium/large, not color dots
        if area < w * h * 0.006:
            continue

        if bw < w * 0.08 or bh < h * 0.06:
            continue

        aspect = bw / max(bh, 1)

        if aspect < 0.25 or aspect > 5.0:
            continue

        boxes.append((x, y, bw, bh))

    # Prefer boxes lower on the page and remove nested boxes
    filtered = []

    for box in boxes:
        x, y, bw, bh = box
        contained = False

        for other in boxes:
            ox, oy, ow, oh = other

            if box == other:
                continue

            if (
                x >= ox and
                y >= oy and
                x + bw <= ox + ow and
                y + bh <= oy + oh
            ):
                contained = True
                break

        if not contained:
            filtered.append(box)

    return sorted(filtered, key=lambda b: (b[1], b[0]))


def crop_inside_box(page_image, box):
    x, y, w, h = box

    pad_x = int(w * 0.02)
    pad_y = int(h * 0.02)

    x1 = max(0, x + pad_x)
    y1 = max(0, y + pad_y)
    x2 = min(page_image.shape[1], x + w - pad_x)
    y2 = min(page_image.shape[0], y + h - pad_y)

    return page_image[y1:y2, x1:x2]


def extract_low_contrast_artwork(box_crop):
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

    # Difference from box background
    diff = np.linalg.norm(
        rgb.astype(np.int16) - bg.astype(np.int16),
        axis=2,
    )

    # Adaptive threshold: catches very faint pink-on-pink artwork
    p95 = np.percentile(diff, 95)
    p98 = np.percentile(diff, 98)
    threshold = max(1.8, min(9.0, (p95 + p98) * 0.10))

    mask = (diff > threshold).astype(np.uint8) * 255

    # Remove box edges by ignoring outer border
    border_x = int(w * 0.015)
    border_y = int(h * 0.015)
    mask[:border_y, :] = 0
    mask[-border_y:, :] = 0
    mask[:, :border_x] = 0
    mask[:, -border_x:] = 0

    # Clean tiny noise, preserve faint letters
    open_kernel = np.ones((2, 2), np.uint8)
    close_kernel = np.ones((4, 4), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # Keep largest meaningful connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]

        if area < 20:
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

    # Skip tiny color swatches/noise
    if crop_w < w * 0.015 or crop_h < h * 0.015:
        return None

    pad_x = max(10, int(crop_w * 0.10))
    pad_y = max(10, int(crop_h * 0.12))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    cropped_rgb = rgb[y1:y2, x1:x2]
    cropped_alpha = cleaned[y1:y2, x1:x2]

    rgba = np.dstack([
        cropped_rgb[:, :, 0],
        cropped_rgb[:, :, 1],
        cropped_rgb[:, :, 2],
        cropped_alpha,
    ])

    return rgba


def save_png(rgba, path):
    Image.fromarray(rgba, mode="RGBA").save(path)


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
            boxes = find_bottom_artwork_boxes(page_image)

            for box_index, box in enumerate(boxes, start=1):
                box_crop = crop_inside_box(page_image, box)
                artwork = extract_low_contrast_artwork(box_crop)

                if artwork is None:
                    continue

                filename = f"artwork_page_{page_number}_{box_index}.png"
                png_path = job_dir / filename

                save_png(artwork, png_path)

                height, width = artwork.shape[:2]
                url = f"{PUBLIC_BASE_URL}/output/{job_id}/{filename}"

                artworks.append({
                    "page": page_number,
                    "design_location_index": len(artworks) + 1,
                    "artwork_url": url,
                    "image_url": url,
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
