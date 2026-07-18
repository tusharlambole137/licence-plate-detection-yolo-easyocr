from pathlib import Path
import argparse
import re

import cv2
import easyocr
from ultralytics import YOLO


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


def run_inference(model_path, source, output_path, conf_thres, ocr_langs, ocr_min_conf, device):
    model = YOLO(model_path)
    reader = easyocr.Reader(ocr_langs, gpu=device != 'cpu')

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create output video: {output_path}")

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(source=frame, conf=conf_thres, verbose=False, device=device)
        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                score = float(box.conf[0]) if box.conf is not None else 0.0
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

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"plate {score:.2f}"
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if plate_text:
                    text_y = min(y2 + 25, height - 10)
                    text_label = f"{plate_text} ({plate_conf:.2f})"
                    cv2.putText(frame, text_label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Latest Ultralytics + EasyOCR license plate inference script")
    parser.add_argument("--model", required=True, help="Path to trained YOLO model, e.g. best.pt")
    parser.add_argument("--source", required=True, help="Path to input image or video")
    parser.add_argument("--output", default="runs/predict/output.mp4", help="Path to output annotated video")
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
        output_path=args.output,
        conf_thres=args.conf,
        ocr_langs=args.ocr_langs,
        ocr_min_conf=args.ocr_min_conf,
        device=args.device,
    )