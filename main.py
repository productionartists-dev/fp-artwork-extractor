def find_artwork_boxes(page_image):
    rgb = page_image
    h, w = rgb.shape[:2]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # More flexible mask for pastel/light colored artwork boxes
    mask = np.zeros((h, w), dtype=np.uint8)

    # Colored boxes, including light pink/pastel
    color_mask = (
        (saturation > 6) &
        (value > 45) &
        (value < 252)
    )

    # Exclude obvious black background and white mockup areas
    not_black = value > 35
    not_white = ~((saturation < 8) & (value > 245))

    mask[color_mask & not_black & not_white] = 255

    # Ignore header/top brand area, but not too aggressively
    mask[: int(h * 0.25), :] = 0

    kernel = np.ones((21, 21), np.uint8)
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

        # Allow smaller boxes like shorts / left-leg designs
        if area < w * h * 0.008:
            continue

        if bw < w * 0.06 or bh < h * 0.05:
            continue

        if y < h * 0.25:
            continue

        aspect = bw / max(bh, 1)

        # Avoid very skinny random text/color swatches
        if aspect < 0.25 or aspect > 5.5:
            continue

        boxes.append((x, y, bw, bh))

    # Remove boxes contained inside bigger boxes
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
