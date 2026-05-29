from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageOps, ImageFilter
import imagehash
import requests
import numpy as np
from io import BytesIO
from typing import Optional


app = FastAPI(title="Fresh Prints Image Hash Generator")


class HashRequest(BaseModel):
    image_url: str
    order_item_id: Optional[str] = None
    design_location_index: Optional[int] = None
    pdf_url: Optional[str] = None
    group_order_design_group_id: Optional[str] = None


def download_image(url: str) -> Image.Image:
    if not url or not isinstance(url, str):
        raise ValueError("Missing image_url")

    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid image_url: {url}")

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")

    if "image" not in content_type.lower():
        raise ValueError(
            f"URL did not return an image. "
            f"content_type={content_type}, "
            f"body_preview={response.text[:200]}"
        )

    return Image.open(BytesIO(response.content)).convert("RGBA")


def flatten_transparency(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)

    return image.convert("RGB")


def normalize_image(image: Image.Image) -> Image.Image:
    image = flatten_transparency(image)

    image.thumbnail((800, 800))

    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)

    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)

    return edges.resize((256, 256))


def analyze_blank(image: Image.Image):
    image = flatten_transparency(image)

    image.thumbnail((800, 800))

    gray = ImageOps.grayscale(image)
    arr = np.array(gray)

    variance = float(np.std(arr))

    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges)

    edge_density = float(np.mean(edge_arr > 20))

    is_blank = variance < 8 and edge_density < 0.01

    return is_blank, round(variance, 4), round(edge_density, 5)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "image-hash-generator"
    }


@app.post("/generate-hash")
def generate_hash(payload: HashRequest):
    try:
        image = download_image(payload.image_url)

        is_blank, variance, edge_density = analyze_blank(image.copy())
        normalized = normalize_image(image.copy())

        return {
            "order_item_id": payload.order_item_id,
            "design_location_index": payload.design_location_index,
            "pdf_url": payload.pdf_url,
            "group_order_design_group_id": payload.group_order_design_group_id,
            "image_url": payload.image_url,

            "is_blank": is_blank,
            "phash": str(imagehash.phash(normalized)),
            "dhash": str(imagehash.dhash(normalized)),
            "ahash": str(imagehash.average_hash(normalized)),
            "edge_density": edge_density,
            "variance": variance,
            "width": image.width,
            "height": image.height,
            "hash_method": "pil_grayscale_autocontrast_edges"
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": str(e),
                "image_url_received": payload.image_url,
                "order_item_id": payload.order_item_id,
                "design_location_index": payload.design_location_index,
                "pdf_url": payload.pdf_url,
                "group_order_design_group_id": payload.group_order_design_group_id,
            }
        )
