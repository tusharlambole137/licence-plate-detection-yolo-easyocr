# Flexible OCR script

This script supports:
- local image files
- local video files
- direct image/video URLs
- parameterized output mode: `media`, `json`, or `both`

## Examples

### Save annotated media only
```bash
python predictWithOCR_flexible.py --model './models/license_plate_detector.pt' --source demo.mp4 --output-mode media --device 0
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