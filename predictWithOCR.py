from pathlib import Path
import argparse
import json
import re
import tempfile
from urllib.parse import urlparse

import cv2
import easyocr
import requests
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg"}


def clean_text(text: str) -> str:
    text = text.upper().strip()
    text = re.sub(r'[^A-Z0-9- ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def select_best_ocr_result(results, min_conf: float):
    best_text = ""
    best_conf = 0.0
    for item in results:
        if len(item) < 3:
            continue
        _, text, conf = item
        if conf is None:
            continue
        if conf >= min_conf and conf > best_conf:
            best_text = clean_text(text)
            best_conf = float(conf)
    return best_text, best_conf


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.equalizeHist(gray)
    return gray


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def download_source(url: str) -> Path:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower() or ".tmp"
    temp_dir = Path(tempfile.mkdtemp(prefix="ocr_flexible_"))
    file_path = temp_dir / f"downloaded_source{suffix}"

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return file_path


def build_media_output_path(source_path: Path, output_arg: str | None):
    if output_arg:
        return Path(output_arg)

    ext = source_path.suffix.lower()
    out_dir = Path("runs/predict")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{source_path.stem}_out{ext}"


def build_json_output_path(source_path: Path, output_arg: str | None):
    if output_arg:
        return Path(output_arg)

    out_dir = Path("runs/predict")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{source_path.stem}_out.json"


def detect_and_annotate_frame(frame, model, reader, conf_thres, ocr_min_conf, device, frame_index=None):
    height, width = frame.shape[:2]
    results = model.predict(source=frame, conf=conf_thres, verbose=False, device=device)
    result = results[0]
    boxes = result.boxes
    detections = []
    annotated = frame.copy()

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            score = float(box.conf[0]) if box.conf is not None else 0.0
            class_id = int(box.cls[0]) if box.cls is not None else -1
            clamped = clamp_box(*xyxy, width, height)
            if clamped is None:
                continue

            x1, y1, x2, y2 = clamped
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            processed = preprocess_crop(crop)
            ocr_results = reader.readtext(processed)
            plate_text, plate_conf = select_best_ocr_result(ocr_results, ocr_min_conf)

            item = {
                "bbox": [x1, y1, x2, y2],
                "detection_confidence": round(score, 4),
                "class_id": class_id,
                "plate_text": plate_text,
                "ocr_confidence": round(plate_conf, 4),
            }
            if frame_index is not None:
                item["frame_index"] = frame_index
            detections.append(item)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"plate {score:.2f}"
            cv2.putText(annotated, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if plate_text:
                text_y = min(y2 + 25, height - 10)
                text_label = f"{plate_text} ({plate_conf:.2f})"
                cv2.putText(annotated, text_label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return annotated, detections


def save_json(data, json_output_path: Path):
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def process_image(model, reader, source_path, media_output_path, json_output_path, output_mode, conf_thres, ocr_min_conf, device):
    frame = cv2.imread(str(source_path))
    if frame is None:
        raise RuntimeError(f"Unable to read image: {source_path}")

    annotated, detections = detect_and_annotate_frame(frame, model, reader, conf_thres, ocr_min_conf, device)
    result = {
        "source": str(source_path),
        "source_type": "image",
        "detections": detections,
    }

    if output_mode in {"media", "both"}:
        media_output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(media_output_path), annotated)
        if not ok:
            raise RuntimeError(f"Unable to save output image: {media_output_path}")
        print(f"Saved output image to: {media_output_path}")

    if output_mode in {"json", "both"}:
        save_json(result, json_output_path)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Saved JSON output to: {json_output_path}")


def process_video(model, reader, source_path, media_output_path, json_output_path, output_mode, conf_thres, ocr_min_conf, device):
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_mode in {"media", "both"}:
        media_output_path.parent.mkdir(parents=True, exist_ok=True)
        if media_output_path.suffix.lower() not in VIDEO_EXTS:
            media_output_path = media_output_path.with_suffix(source_path.suffix.lower())
        writer = cv2.VideoWriter(
            str(media_output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create output video: {media_output_path}")

    frames = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        annotated, detections = detect_and_annotate_frame(
            frame, model, reader, conf_thres, ocr_min_conf, device, frame_index=frame_index
        )

        if writer is not None:
            writer.write(annotated)

        if detections:
            frames.append({
                "frame_index": frame_index,
                "detections": detections,
            })

        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved output video to: {media_output_path}")

    result = {
        "source": str(source_path),
        "source_type": "video",
        "fps": fps,
        "total_frames_processed": frame_index,
        "frames": frames,
    }

    if output_mode in {"json", "both"}:
        save_json(result, json_output_path)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Saved JSON output to: {json_output_path}")


def run_inference(model_path, source, output_mode, output_path, json_output_path, conf_thres, ocr_langs, ocr_min_conf, device):
    source_path = download_source(source) if is_url(source) else Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    model = YOLO(model_path)
    reader = easyocr.Reader(ocr_langs, gpu=device != "cpu")

    media_output = build_media_output_path(source_path, output_path)
    json_output = build_json_output_path(source_path, json_output_path)
    ext = source_path.suffix.lower()

    if ext in IMAGE_EXTS:
        process_image(model, reader, source_path, media_output, json_output, output_mode, conf_thres, ocr_min_conf, device)
    elif ext in VIDEO_EXTS:
        process_video(model, reader, source_path, media_output, json_output, output_mode, conf_thres, ocr_min_conf, device)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def parse_args():
    parser = argparse.ArgumentParser(description="Ultralytics + EasyOCR flexible output script")
    parser.add_argument("--model", required=True, help="Path to trained YOLO model, e.g. best.pt")
    parser.add_argument("--source", required=True, help="Path or URL to input image or video")
    parser.add_argument("--output-mode", choices=["media", "json", "both"], default="media", help="Save annotated media, JSON results, or both")
    parser.add_argument("--output", default=None, help="Optional media output path")
    parser.add_argument("--json-output", default=None, help="Optional JSON output path")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--ocr-min-conf", type=float, default=0.20, help="Minimum OCR confidence to display text")
    parser.add_argument("--device", default="cpu", help="cpu, 0, 0,1 etc.")
    parser.add_argument("--ocr-langs", nargs="+", default=["en"], help="EasyOCR languages")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        model_path=args.model,
        source=args.source,
        output_mode=args.output_mode,
        output_path=args.output,
        json_output_path=args.json_output,
        conf_thres=args.conf,
        ocr_langs=args.ocr_langs,
        ocr_min_conf=args.ocr_min_conf,
        device=args.device,
    )