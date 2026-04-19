import glob
import os
import cv2
import numpy as np

try:
    import pytesseract
    from pytesseract import Output
except ImportError:
    pytesseract = None
    Output = None

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms as T
except ImportError:
    torch = None

EASY_OCR_READER = None
DIGIT_TEMPLATES = {}
CNN_MODEL = None
CNN_CLASSES = None


if torch is not None:
    class DigitCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, 3, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
            )
            self.classifier = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(128 * 8 * 8, 256), torch.nn.ReLU(), torch.nn.Dropout(0.4),
                torch.nn.Linear(256, 9),
            )

        def forward(self, x):
            return self.classifier(self.features(x))
else:
    DigitCNN = None


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

    # sudoku.com uses colored digits on white background — invert so digit = white
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)

    # combine both to handle light-colored and dark digits
    combined = cv2.bitwise_or(otsu, adaptive)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
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

    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
    ]
    for digit in range(1, 10):
        DIGIT_TEMPLATES[digit] = []
        for font in fonts:
            img = np.full((size, size), 255, dtype=np.uint8)
            text = str(digit)
            scale = 2
            thickness = 4
            text_size = cv2.getTextSize(text, font, scale, thickness)[0]
            x = (size - text_size[0]) // 2
            y = (size + text_size[1]) // 2
            cv2.putText(img, text, (x, y), font, scale, 0, thickness, cv2.LINE_AA)
            _, bw = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
            DIGIT_TEMPLATES[digit].append(bw)
    return DIGIT_TEMPLATES


def build_templates(template_folder: str):
    templates = {}
    if template_folder is None or not os.path.isdir(template_folder):
        return build_digit_templates()

    for digit in range(1, 10):
        pattern = os.path.join(template_folder, f'{digit}*.png')
        for path in glob.glob(pattern):
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                continue
            _, bw = cv2.threshold(template, 128, 255, cv2.THRESH_BINARY_INV)
            templates.setdefault(digit, []).append(bw)

    if not templates:
        return build_digit_templates()

    return templates


def extract_digit_crop(cell_image):
    processed = preprocess_cell(cell_image)
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


def recognize_digit_easyocr_candidates(cell_image, reader=None):
    if easyocr is None:
        return []
    if reader is None:
        reader = build_easyocr_reader()

    candidates = {}
    crop = extract_digit_crop(cell_image)
    crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)
    results = reader.readtext(crop, detail=1, allowlist='123456789')
    for _, text, score in results:
        digit_text = ''.join([c for c in text if c.isdigit()])
        if digit_text:
            digit = int(digit_text[0])
            candidates[digit] = max(candidates.get(digit, 0.0), float(score))

    if not candidates:
        gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_LINEAR)
        results = reader.readtext(gray, detail=1, allowlist='123456789')
        for _, text, score in results:
            digit_text = ''.join([c for c in text if c.isdigit()])
            if digit_text:
                digit = int(digit_text[0])
                candidates[digit] = max(candidates.get(digit, 0.0), float(score))

    return sorted(candidates.items(), key=lambda item: item[1], reverse=True)


def recognize_digit_tesseract_candidates(cell_image):
    if pytesseract is None or Output is None:
        return []

    def gather_candidates(image):
        config = '--psm 10 -c tessedit_char_whitelist=123456789'
        try:
            data = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)
        except (FileNotFoundError, RuntimeError, OSError, pytesseract.pytesseract.TesseractNotFoundError):
            return {}
        results = {}
        for text, conf in zip(data.get('text', []), data.get('conf', [])):
            text = text.strip()
            if text.isdigit():
                try:
                    score = float(conf)
                except (ValueError, TypeError):
                    score = 0.0
                digit = int(text)
                results[digit] = max(results.get(digit, 0.0), score)
        return results

    candidates = {}
    crop = extract_digit_crop(cell_image)
    crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)
    candidates.update(gather_candidates(crop))

    if not candidates:
        processed = preprocess_cell(cell_image)
        processed = cv2.resize(processed, (128, 128), interpolation=cv2.INTER_LINEAR)
        candidates.update(gather_candidates(processed))

    if not candidates:
        gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_LINEAR)
        candidates.update(gather_candidates(gray))

    return sorted(candidates.items(), key=lambda item: item[1], reverse=True)


def recognize_digit_template_candidates(cell_image, templates: dict | None = None):
    processed = extract_digit_crop(cell_image)
    if processed is None or processed.size == 0:
        return []

    templates = templates or build_digit_templates()
    candidates = {}
    for digit, template_list in templates.items():
        for template in template_list:
            resized = cv2.resize(template, (processed.shape[1], processed.shape[0]), interpolation=cv2.INTER_LINEAR)
            result = cv2.matchTemplate(processed, resized, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            candidates[digit] = max(candidates.get(digit, 0.0), float(score))

    return sorted(candidates.items(), key=lambda item: item[1], reverse=True)


def analyze_digit_line_orientation(cell_image):
    crop = crop_cell_roi(cell_image, margin_ratio=0.15)
    processed = preprocess_cell(crop)
    edges = cv2.Canny(processed, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=5)
    if lines is None:
        return None

    counts = {'horiz': 0, 'vert': 0, 'diag': 0}
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = x2 - x1
        dy = y2 - y1
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle < 20:
            counts['horiz'] += 1
        elif angle > 70:
            counts['vert'] += 1
        else:
            counts['diag'] += 1
    return counts


def resolve_one_vs_seven(cell_image, templates=None):
    line_counts = analyze_digit_line_orientation(cell_image)
    if line_counts is not None:
        if line_counts['horiz'] >= 1 and line_counts['diag'] >= 1:
            return 7
        if line_counts['vert'] >= 1 and line_counts['horiz'] == 0:
            return 1

    template_candidates = recognize_digit_template_candidates(cell_image, templates=templates)
    for digit, _ in template_candidates:
        if digit in (1, 7):
            return digit
    return None


def recognize_digit_candidates(cell_image, method: str = 'best', templates: dict | None = None, reader=None, cnn_weights: str = 'weights/digit_cnn.pth'):
    candidate_scores = {}

    if method in ('easyocr', 'best'):
        for digit, score in recognize_digit_easyocr_candidates(cell_image, reader=reader):
            candidate_scores[digit] = max(candidate_scores.get(digit, 0.0), score)

    if method in ('tesseract', 'best'):
        for digit, score in recognize_digit_tesseract_candidates(cell_image):
            candidate_scores[digit] = max(candidate_scores.get(digit, 0.0), score)

    if method in ('template', 'best'):
        for digit, score in recognize_digit_template_candidates(cell_image, templates=templates):
            candidate_scores[digit] = max(candidate_scores.get(digit, 0.0), score)

    if method in ('cnn', 'best'):
        for digit, score in recognize_digit_cnn_candidates(cell_image, weights_path=cnn_weights):
            # CNN scores sont des probabilités (0-1), on les boost pour peser face aux autres
            candidate_scores[digit] = max(candidate_scores.get(digit, 0.0), score * 100)

    if method == 'easyocr' and not candidate_scores:
        for digit, score in recognize_digit_tesseract_candidates(cell_image):
            candidate_scores[digit] = max(candidate_scores.get(digit, 0.0), score)
        if not candidate_scores:
            for digit, score in recognize_digit_template_candidates(cell_image, templates=templates):
                candidate_scores[digit] = max(candidate_scores.get(digit, 0.0), score)

    if method == 'tesseract' and not candidate_scores:
        for digit, score in recognize_digit_easyocr_candidates(cell_image, reader=reader):
            candidate_scores[digit] = max(candidate_scores.get(digit, 0.0), score)
        if not candidate_scores:
            for digit, score in recognize_digit_template_candidates(cell_image, templates=templates):
                candidate_scores[digit] = max(candidate_scores.get(digit, 0.0), score)

    if method == 'cnn' and not candidate_scores:
        for digit, score in recognize_digit_template_candidates(cell_image, templates=templates):
            candidate_scores[digit] = max(candidate_scores.get(digit, 0.0), score)

    if not candidate_scores:
        return []

    candidates = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)

    top_digits = [digit for digit, _ in candidates[:2]]
    if set(top_digits) == {1, 7} and len(candidates) > 1 and abs(candidates[0][1] - candidates[1][1]) < 0.2:
        resolved = resolve_one_vs_seven(cell_image, templates=templates)
        if resolved is not None and resolved in (1, 7):
            candidate_scores[resolved] = max(candidate_scores.get(resolved, 0.0), 1.0)
            candidates = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)

    return candidates


def load_cnn_model(weights_path: str = 'weights/digit_cnn.pth'):
    global CNN_MODEL, CNN_CLASSES
    if torch is None:
        return None, None
    if CNN_MODEL is not None:
        return CNN_MODEL, CNN_CLASSES

    if not os.path.isfile(weights_path):
        return None, None

    checkpoint = torch.load(weights_path, map_location='cpu')
    model = DigitCNN()
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    CNN_MODEL = model
    CNN_CLASSES = checkpoint['classes']
    return CNN_MODEL, CNN_CLASSES


def recognize_digit_cnn_candidates(cell_image, weights_path: str = 'weights/digit_cnn.pth'):
    model, classes = load_cnn_model(weights_path)
    if model is None:
        return []

    transform = T.Compose([
        T.ToPILImage(),
        T.Grayscale(),
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])

    tensor = transform(cell_image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)[0]
        probs = torch.softmax(logits, dim=0)

    candidates = []
    for i, prob in enumerate(probs):
        digit = int(classes[i])
        candidates.append((digit, float(prob)))

    return sorted(candidates, key=lambda x: x[1], reverse=True)


def recognize_digit(cell_image, method: str = 'best', templates: dict | None = None, reader=None):
    candidates = recognize_digit_candidates(cell_image, method=method, templates=templates, reader=reader)
    if not candidates:
        return 0
    return candidates[0][0]


def is_cell_blank(cell_image, threshold: float = 0.01):
    cell = crop_cell_roi(cell_image, margin_ratio=0.15)
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20:
            continue
        filled_area += area

    return filled_area < cleaned.size * threshold
