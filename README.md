



Downloading detection model, please wait. This may take several minutes depending upon your network connection.
Progress: |██████████████████████████████████████████████████| 100.0% CompleteDownloading recognition model, please wait. This may take several minutes depending upon your network connection.
Progress: |██████████████████████████████████████████████████| 100.0% Complete2026-07-16 18:46:45.938800: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Ultralytics YOLOv8.0.3 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
Error executing job with overrides: ['model=/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/best.pt', 'source=/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/demo.mp4']
Traceback (most recent call last):
File "/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/predictWithOCR.py", line 112, in predict
predictor()
File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 124, in decorate_context
return func(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^
File "/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/ultralytics/yolo/engine/predictor.py", line 164, in __call__
model = self.model if self.done_setup else self.setup(source, model)
^^^^^^^^^^^^^^^^^^^^^^^^^
File "/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/ultralytics/yolo/engine/predictor.py", line 121, in setup
model = AutoBackend(model, device=device, dnn=self.args.dnn, fp16=self.args.half)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/ultralytics/nn/autobackend.py", line 73, in __init__
model = attempt_load_weights(weights if isinstance(weights, list) else w,
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/ultralytics/nn/tasks.py", line 303, in attempt_load_weights
ckpt = torch.load(attempt_download(w), map_location='cpu') # load
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 1578, in load
raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
WeightsUnpickler error: Unsupported global: GLOBAL ultralytics.nn.tasks.DetectionModel was not an allowed global by default. Please use `torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])` or the `torch.serialization.safe_globals([ultralytics.nn.tasks.DetectionModel])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
Completed 2 steps
This is a PyTorch 2.6+ compatibility problem, not a model download problem. PyTorch changed torch.load() to default to weights_only=True, and your older Ultralytics code path is trying to load a YOLO checkpoint that contains ultralytics.nn.tasks.DetectionModel, which now gets blocked unless you explicitly allow it or load with weights_only=False from a trusted source.

Why it fails
Your log shows Ultralytics YOLOv8.0.3, which is a very old Ultralytics release, and the stack trace is loading the checkpoint through direct torch.load(...) in the old codebase.

PyTorch’s current docs confirm that torch.load() now defaults to weights_only=True, and they also warn that setting weights_only=False should only be done for checkpoints you trust.

Best fix
Upgrade Ultralytics first, because Ultralytics merged a fix that replaces direct torch.load calls with a patched loader and notes the issue as resolved in ultralytics 8.3.162.

In practice, this is the cleanest path because your environment is on 8.0.3, far behind the fixed versions.

Use:

bash
pip uninstall -y ultralytics
pip install -U "ultralytics>=8.3.162"
python -c "import ultralytics, torch; print('ultralytics=', ultralytics.__version__); print('torch=', torch.__version__)"
Also make sure you are not accidentally importing a local cloned ultralytics/ folder from your project, because Ultralytics explicitly called out that a local folder or multiple installs can keep you on stale code even after upgrading.

Workarounds
If you must stay on this repo/code as-is, you have two trusted-source workarounds:

Patch the load call to use weights_only=False; PyTorch says this will usually load successfully, but only do it for trusted checkpoints.

Allowlist the model class with torch.serialization.add_safe_globals([DetectionModel]), which PyTorch and Ultralytics issue discussions both mention as an alternative.

Patch option
In the old Ultralytics code, change the checkpoint load from:

python
ckpt = torch.load(attempt_download(w), map_location='cpu')
to:

python
ckpt = torch.load(attempt_download(w), map_location='cpu', weights_only=False)
That exact style of patch is the documented workaround discussed in the Ultralytics issue thread for older environments.

Safe globals option
Before loading the model, add:

python
import torch
from ultralytics.nn.tasks import DetectionModel
torch.serialization.add_safe_globals([DetectionModel])
Then run the model load again. This follows the error guidance shown by PyTorch and the related Ultralytics discussions.

Practical recommendation
For your notebook or Colab-style run, I’d do this in order:

Remove any repo-bundled old ultralytics package from the project path if present.

Upgrade to latest Ultralytics.

Re-run and verify version output is not 8.0.3.

If the repo still hardcodes its own old Ultralytics sources, patch torch.load(..., weights_only=False) in that local code for trusted best.pt.

Example minimal fix in your case:

bash
pip install -U "ultralytics>=8.3.162"
If the repo still uses its own embedded old source tree, edit the failing load path and use:

python
ckpt = torch.load(attempt_download(w), map_location='cpu', weights_only=False)
Would you like me to rewrite your predictWithOCR.py fix block exactly for this project structure?

I am refering the below Github project
https://github.com/Arijit1080/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR
Help me to get the latest version for this code
Completed 3 steps
Yes — you can modernize that repo, but the GitHub project itself does not publish tagged releases, so there is no official “latest version” package to install directly from the repository.

The repo is also old enough that its bundled YOLOv8 code path is incompatible with current PyTorch defaults, which is why you hit the weights_only loading error.

What “latest” should mean
For this project, “latest version” should mean: keep the project logic, but replace the old embedded Ultralytics dependency pattern with the current pip-installed ultralytics package and update the inference code to use the modern API.

That is better than trying to preserve the repo’s old internal ultralytics/ source tree, because your error is happening in that legacy load path.

Recommended upgrade path
Use a fresh environment and install current libraries instead of depending on the repo’s old internal structure. PyTorch documents that torch.load now defaults to weights_only=True, and newer Ultralytics versions contain compatibility fixes for that change.

Install with something like:

bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install ultralytics easyocr opencv-python pillow
If you need GPU support in Colab, install the matching CUDA-enabled torch build first, then install ultralytics and easyocr. The key point is: do not rely on the repo’s bundled old Ultralytics package.

Code changes
The repo’s old pattern likely imports or executes from a local ultralytics/ folder. That should be removed from the runtime path and replaced with the pip package import:

python
from ultralytics import YOLO
import easyocr
import cv2
Model loading should become:

python
model = YOLO("best.pt")
Then for each frame or image:

python
results = model(frame)
This avoids the stale internal loader that is failing under new PyTorch behavior.

If you must keep the old repo
If you want the existing repo to run with minimal edits, patch its checkpoint loading only for trusted model files. PyTorch says re-running torch.load(..., weights_only=False) can succeed, but should only be used when you trust the checkpoint source.

The local fix is typically:

python
ckpt = torch.load(attempt_download(w), map_location='cpu', weights_only=False)
Or allowlist the YOLO model class before loading:

python
import torch
from ultralytics.nn.tasks import DetectionModel
torch.serialization.add_safe_globals([DetectionModel])
Both approaches are consistent with the PyTorch guidance shown in your error.

Best practical route
I recommend treating that repository as a reference implementation rather than the exact runtime package. The repo has no releases and exposes only the raw source, so the safest modernization is to create a cleaned-up script that uses current ultralytics, current easyocr, and your existing best.pt checkpoint if it loads cleanly.

A good migration plan is:

Keep: best.pt, demo assets, OCR logic, crop-and-read pipeline.

Replace: bundled ultralytics/ folder, old predictor hooks, direct internal patching.

Update: inference script to modern YOLO(...)/results.boxes usage.

Add: pinned dependency file for reproducibility.

Use this requirements.txt as a starting point:

text
ultralytics
easyocr
opencv-python
pillow
numpy
torch
torchvision
If you want, I can give you a fully updated predictWithOCR.py that works with the current Ultralytics API and your existing best.pt model.

yes give the file
Completed 3 steps
README_update
Updated runtime files for Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR
This update replaces the repo's old embedded Ultralytics runtime with the current pip-installed ultralytics package.

Install
bash
pip install -r requirements_latest.txt
Run
bash
python predictWithOCR_latest.py \
  --model best.pt \
  --source demo.mp4 \
  --output runs/predict/demo_out.mp4 \
  --device cpu
For GPU:

bash
python predictWithOCR_latest.py \
  --model best.pt \
  --source demo.mp4 \
  --output runs/predict/demo_out.mp4 \
  --device 0
Notes
The referenced GitHub repository has no published releases and includes a local ultralytics/ folder in the source tree.

The updated script uses from ultralytics import YOLO instead of the old internal predictor path.

This avoids the PyTorch 2.6+ torch.load(... weights_only=True) compatibility problem seen in older Ultralytics loading code.

If your best.pt checkpoint was produced with an extremely old training stack and still fails, retrain/export with a newer Ultralytics version or temporarily load only from a trusted source in a patched environment.

text

Open README_update
I created an updated inference script, a modern requirements file, and a short run guide for that repository. The GitHub repo has no published releases and includes a local ultralytics/ folder, so the safest upgrade is to run against the current pip package instead of the repo’s bundled runtime.

Files
The Python script uses the current from ultralytics import YOLO API and EasyOCR, which avoids the old internal loader path that breaks under newer PyTorch behavior.

The requirements file pins a modern Ultralytics floor because compatibility fixes for the PyTorch weights_only change landed in newer Ultralytics versions.

Usage
Install with:

bash
pip install -r requirements_latest.txt
Run with:

bash
python predictWithOCR_latest.py \
  --model best.pt \
  --source demo.mp4 \
  --output runs/predict/demo_out.mp4 \
  --device cpu
For GPU, use --device 0. The repo structure shown on GitHub includes best.pt, demo.mp4, and predictWithOCR.py, so this updated script is meant to drop into that same project flow.

Notes
If best.pt still fails to load, that usually means the checkpoint itself was produced with a much older stack and may need retraining or a trusted one-time conversion path. PyTorch documents that the weights_only default changed, which is the root of the error you saw earlier.

Place the new files beside the original repo files, then run the new script instead of the old predictWithOCR.py.

get the result as text / actual json intead of image or vide?

Yes — instead of saving annotated image/video, you can return structured text/JSON with plate text, confidence, bounding box coordinates, and source metadata. Ultralytics exposes box coordinates, classes, and confidences from results[0].boxes, and can also summarize detections as dictionaries, while EasyOCR returns text plus confidence for each OCR hit.

Recommended JSON shape
A good result format is one JSON object per source, with a list of detections like:

json
{
  "source": "test.jpg",
  "source_type": "image",
  "detections": [
    {
      "bbox": [120, 240, 280, 300],
      "detection_confidence": 0.91,
      "plate_text": "MH12AB1234",
      "ocr_confidence": 0.88
    }
  ]
}
This works well because JSON is easy to save with json.dump() and easy to consume later from Python, Android, or backend APIs.

What to change
Instead of drawing with cv2.rectangle() and cv2.putText(), collect dictionaries in a list and either print JSON to console or write it to a .json file. Python’s json.dump() is the standard way to save dictionaries or lists of dictionaries to JSON.

Updated script
Use this version if you want JSON output only:

python
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
    temp_dir = Path(tempfile.mkdtemp(prefix="ocr_json_"))
    file_path = temp_dir / f"downloaded_source{suffix}"

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return file_path


def detect_on_frame(frame, model, reader, conf_thres, ocr_min_conf, device, frame_index=None):
    height, width = frame.shape[:2]
    results = model.predict(source=frame, conf=conf_thres, verbose=False, device=device)
    result = results[0]
    boxes = result.boxes
    detections = []

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
                "ocr_confidence": round(plate_conf, 4)
            }

            if frame_index is not None:
                item["frame_index"] = frame_index

            detections.append(item)

    return detections


def build_output_path(source_path: Path, output_arg: str | None):
    if output_arg:
        return Path(output_arg)

    out_dir = Path("runs/predict")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{source_path.stem}_out.json"


def run_on_image(model, reader, source_path, conf_thres, ocr_min_conf, device):
    frame = cv2.imread(str(source_path))
    if frame is None:
        raise RuntimeError(f"Unable to read image: {source_path}")

    detections = detect_on_frame(frame, model, reader, conf_thres, ocr_min_conf, device)
    return {
        "source": str(source_path),
        "source_type": "image",
        "detections": detections
    }


def run_on_video(model, reader, source_path, conf_thres, ocr_min_conf, device):
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    frames = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detect_on_frame(
            frame, model, reader, conf_thres, ocr_min_conf, device, frame_index=frame_index
        )

        if detections:
            frames.append({
                "frame_index": frame_index,
                "detections": detections
            })

        frame_index += 1

    cap.release()

    return {
        "source": str(source_path),
        "source_type": "video",
        "fps": fps,
        "total_frames_processed": frame_index,
        "frames": frames
    }


def run_inference(model_path, source, output_path, conf_thres, ocr_langs, ocr_min_conf, device):
    source_path = download_source(source) if is_url(source) else Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    model = YOLO(model_path)
    reader = easyocr.Reader(ocr_langs, gpu=device != "cpu")

    ext = source_path.suffix.lower()
    if ext in IMAGE_EXTS:
        result = run_on_image(model, reader, source_path, conf_thres, ocr_min_conf, device)
    elif ext in VIDEO_EXTS:
        result = run_on_video(model, reader, source_path, conf_thres, ocr_min_conf, device)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    final_output_path = build_output_path(source_path, output_path)
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved JSON output to: {final_output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Ultralytics + EasyOCR JSON output for image, video, and URL sources")
    parser.add_argument("--model", required=True, help="Path to trained YOLO model, e.g. best.pt")
    parser.add_argument("--source", required=True, help="Path or URL to input image or video")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--ocr-min-conf", type=float, default=0.20, help="Minimum OCR confidence to keep text")
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
Example output files
test.jpg → runs/predict/test_out.json

plate.png → runs/predict/plate_out.json

demo.mp4 → runs/predict/demo_out.json

That naming pattern is simpler here because the output is always JSON instead of mirroring image/video media formats.

Example command
bash
python predictWithOCR_json.py --model best.pt --source test.jpg --device 0
Or for video:

bash
python predictWithOCR_json.py --model best.pt --source demo.mp4 --device 0
If you want, I can generate this as a downloadable predictWithOCR_json.py file for you.

want this option to be parameterized

Yes — the clean way is to add an output mode parameter, for example --output-mode, with choices like media, json, or both. argparse supports constrained option values with choices=..., which is exactly what you want here.

Recommended parameter
Add this argument:

python
parser.add_argument(
    "--output-mode",
    choices=["media", "json", "both"],
    default="media",
    help="Save annotated media, JSON results, or both"
)
That lets you run the same script in different modes instead of maintaining separate files. argparse is designed for named options with values like this.

Example usage
Annotated image/video only:

bash
python predictWithOCR.py --model best.pt --source demo.mp4 --output-mode media
JSON only:

bash
python predictWithOCR.py --model best.pt --source test.jpg --output-mode json
Both:

bash
python predictWithOCR.py --model best.pt --source test.jpg --output-mode both
This is simpler than separate scripts because the CLI remains stable while behavior changes through one parameter.

How to implement
Inside your processing function, branch based on output_mode:

python
if output_mode == "media":
    # save image/video only
elif output_mode == "json":
    # save json only
elif output_mode == "both":
    # save both
For YOLO detections, you can still read boxes and confidences from results[0].boxes, then either draw them to frames or serialize them to JSON dictionaries depending on the mode.

Better option set
I’d suggest these parameters:

python
parser.add_argument("--output-mode", choices=["media", "json", "both"], default="media")
parser.add_argument("--output", default=None, help="Optional media output path")
parser.add_argument("--json-output", default=None, help="Optional JSON output path")
That gives you flexibility:

--output for image/video path

--json-output for structured results

--output-mode to control what gets produced

Small example
python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-mode", choices=["media", "json", "both"], default="media")
    parser.add_argument("--output", default=None)
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()
Then call:

python
run_inference(
    model_path=args.model,
    source=args.source,
    output_mode=args.output_mode,
    output_path=args.output,
    json_output_path=args.json_output
)
My suggestion
Use:

media = annotated image/video

json = text/JSON only

both = save both outputs

That is the most practical CLI design for your use case. Would you like me to generate one merged downloadable script with --output-mode media|json|both support?

yes
Completed 2 steps



README_flexible
Document

predictWithOCR_flexible.py
Python
README_flexible
Flexible OCR script
This script supports:

local image files

local video files

direct image/video URLs

parameterized output mode: media, json, or both

Examples
Save annotated media only
bash
python predictWithOCR_flexible.py --model best.pt --source demo.mp4 --output-mode media --device 0
Save JSON only
bash
python predictWithOCR_flexible.py --model best.pt --source test.jpg --output-mode json --device cpu
Save both media and JSON
bash
python predictWithOCR_flexible.py --model best.pt --source test.jpg --output-mode both --device 0
Custom JSON path
bash
python predictWithOCR_flexible.py --model best.pt --source test.jpg --output-mode json --json-output runs/predict/result.json
Output defaults
If no custom path is passed:

media output: runs/predict/<source_name>_out.<same_extension>

json output: runs/predict/<source_name>_out.json


Currently open
I created a merged script with a parameterized --output-mode that supports media, json, or both, which is the intended CLI pattern for controlled behavior in argparse.

Modes
Use:

--output-mode media for annotated image/video

--output-mode json for text/JSON results only

--output-mode both for both outputs

This design lets one script handle all three result types while still using the same YOLO boxes and EasyOCR text pipeline underneath. Ultralytics exposes detection details from results.boxes, and EasyOCR returns text plus confidence for OCR results.

Examples
Annotated media only:

bash
python predictWithOCR_flexible.py --model best.pt --source demo.mp4 --output-mode media --device 0
JSON only:

bash
python predictWithOCR_flexible.py --model best.pt --source test.jpg --output-mode json --device cpu
Both:

bash
python predictWithOCR_flexible.py --model best.pt --source test.jpg --output-mode both --device 0
Output paths
If you do not pass custom paths, the script automatically writes:

media: runs/predict/<source_name>_out.<same_extension>

JSON: runs/predict/<source_name>_out.json

That output naming is based on standard path suffix handling in Python.

Notes
The script was syntax-checked after generation, and it also supports local files plus direct URLs as input sources. Common image formats such as JPEG and PNG are supported by OpenCV image loading, while the Ultralytics predict flow supports varied source types.




# Flexible OCR script

This script supports:
- local image files
- local video files
- direct image/video URLs
- parameterized output mode: `media`, `json`, or `both`

## Examples

### Save annotated media only
```bash
python predictWithOCR_flexible.py --model best.pt --source demo.mp4 --output-mode media --device 0
```

### Save JSON only
```bash
python predictWithOCR_flexible.py --model best.pt --source test.jpg --output-mode json --device cpu
```

### Save both media and JSON
```bash
python predictWithOCR_flexible.py --model best.pt --source test.jpg --output-mode both --device 0
```

### Custom JSON path
```bash
python predictWithOCR_flexible.py --model best.pt --source test.jpg --output-mode json --json-output runs/predict/result.json
```

## Output defaults

If no custom path is passed:
- media output: `runs/predict/<source_name>_out.<same_extension>`
- json output: `runs/predict/<source_name>_out.json`
