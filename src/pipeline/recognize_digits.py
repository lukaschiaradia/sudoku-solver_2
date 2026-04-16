import os
import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import easyocr
except ImportError:
    easyocr = None

EASY_OCR_READER = None
DIGIT_TEMPLATES = {}


def crop_cell_roi(cell_image, margin_ratio: float = 0.12):
    h, w = cell_image.shape[:2]
    top = int(h * margin_ratio)
    left = int(w * margin_ratio)
    bottom = int(h * (1 - margin_ratio))
    right = int(w * (1 - margin_ratio))
    return cell_image[top:bottom, left:right]


def preprocess_cell(cell_image):
    cell = crop_cell_roi(cell_image, margin_ratio=0.12)
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned


def build_easyocr_reader():
    global EASY_OCR_READER
    if easyocr is None:
        raise ImportError('easyocr is required for EasyOCR recognition')
    if EASY_OCR_READER is None:
        EASY_OCR_READER = easyocr.Reader(['en'], gpu=False)
    return EASY_OCR_READER


def build_digit_templates(size: int = 64):
    global DIGIT_TEMPLATES
    if DIGIT_TEMPLATES:
        return DIGIT_TEMPLATES

    font = cv2.FONT_HERSHEY_SIMPLEX
    for digit in range(1, 10):
        img = np.full((size, size), 255, dtype=np.uint8)
        text = str(digit)
        scale = 2
        thickness = 4
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        x = (size - text_size[0]) // 2
        y = (size + text_size[1]) // 2
        cv2.putText(img, text, (x, y), font, scale, 0, thickness, cv2.LINE_AA)
        _, bw = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
        DIGIT_TEMPLATES[digit] = bw
    return DIGIT_TEMPLATES


def extract_digit_crop(cell_image):
    cell = crop_cell_roi(cell_image, margin_ratio=0.12)
    processed = preprocess_cell(cell)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return processed

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 25:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 8 or h < 8:
            continue
        pad = 4
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(processed.shape[1], x + w + pad)
        y2 = min(processed.shape[0], y + h + pad)
        return processed[y1:y2, x1:x2]
    return processed


def recognize_digit_easyocr(cell_image, reader=None):
    if easyocr is None:
        raise ImportError('easyocr is required for EasyOCR recognition')
    if reader is None:
        reader = build_easyocr_reader()

    crop = extract_digit_crop(cell_image)
    crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)
    results = reader.readtext(crop, detail=0, allowlist='123456789')
    for text in results:
        digit_text = ''.join([c for c in text if c.isdigit()])
        if digit_text:
            return int(digit_text[0])

    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_LINEAR)
    results = reader.readtext(gray, detail=0, allowlist='123456789')
    for text in results:
        digit_text = ''.join([c for c in text if c.isdigit()])
        if digit_text:
            return int(digit_text[0])
    return 0


def recognize_digit_tesseract(cell_image):
    if pytesseract is None:
        raise ImportError('pytesseract is required for Tesseract OCR')

    crop = extract_digit_crop(cell_image)
    crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)
    config = '--psm 10 -c tessedit_char_whitelist=123456789'
    text = pytesseract.image_to_string(crop, config=config)
    text = text.strip()
    if text.isdigit():
        return int(text)

    processed = preprocess_cell(cell_image)
    processed = cv2.resize(processed, (128, 128), interpolation=cv2.INTER_LINEAR)
    text = pytesseract.image_to_string(processed, config=config)
    text = text.strip()
    if text.isdigit():
        return int(text)

    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_LINEAR)
    text = pytesseract.image_to_string(gray, config=config)
    text = text.strip()
    return int(text) if text.isdigit() else 0


def is_cell_blank(cell_image, threshold: float = 0.04):
    cell = crop_cell_roi(cell_image, margin_ratio=0.12)
    processed = preprocess_cell(cell)
    filled_pixels = np.count_nonzero(processed)
    return filled_pixels < processed.size * threshold


def recognize_digit_template(cell_image, templates: dict | None = None):
    processed = extract_digit_crop(cell_image)
    if processed is None or processed.size == 0:
        return 0

    templates = templates or build_digit_templates()
    best_score = 0.0
    best_digit = 0
    for digit, template in templates.items():
        resized = cv2.resize(template, (processed.shape[1], processed.shape[0]), interpolation=cv2.INTER_LINEAR)
        result = cv2.matchTemplate(processed, resized, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)
        if score > best_score:
            best_score = score
            best_digit = digit
    return best_digit if best_score > 0.45 else 0


def recognize_digit(cell_image, method: str = 'best', templates: dict | None = None, reader=None):
    if method == 'easyocr':
        digit = recognize_digit_easyocr(cell_image, reader=reader)
        if digit == 0 and pytesseract is not None:
            digit = recognize_digit_tesseract(cell_image)
        if digit == 0:
            digit = recognize_digit_template(cell_image, templates)
        return digit
    if method == 'tesseract':
        digit = recognize_digit_tesseract(cell_image)
        if digit == 0 and easyocr is not None:
            digit = recognize_digit_easyocr(cell_image, reader=reader)
        if digit == 0:
            digit = recognize_digit_template(cell_image, templates)
        return digit
    if method == 'template':
        digit = recognize_digit_template(cell_image, templates)
        if digit == 0 and pytesseract is not None:
            digit = recognize_digit_tesseract(cell_image)
        return digit
    if method == 'best':
        digit = 0
        if easyocr is not None:
            digit = recognize_digit_easyocr(cell_image, reader=reader)
        if digit == 0 and pytesseract is not None:
            digit = recognize_digit_tesseract(cell_image)
        if digit == 0:
            digit = recognize_digit_template(cell_image, templates)
        return digit
    raise ValueError(f'Unsupported recognition method: {method}')


def build_templates(template_folder: str):
    templates = {}
    for digit in range(1, 10):
        path = f'{template_folder}/{digit}.png'
        template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue
        templates[digit] = cv2.threshold(template, 128, 255, cv2.THRESH_BINARY_INV)[1]
    return templates
