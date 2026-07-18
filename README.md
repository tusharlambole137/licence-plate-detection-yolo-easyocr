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