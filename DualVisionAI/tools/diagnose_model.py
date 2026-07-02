"""
DualVision AI — YOLO26n ONNX vs PyTorch Diagnostic Tool
========================================================
Run from the DualVisionAI/ directory:

    python tools/diagnose_model.py

Requires:
  models/yolo26n.onnx         (always)
  models/yolo26n.pt           (optional — enables PT vs ONNX comparison)

Outputs:
  debug/diagnosis_report.txt  (full text report)
  debug/diag_preprocessed.png (the test frame fed to the model)
"""

import sys
import os
import json
import datetime
import textwrap

import cv2
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_PATH = os.path.join(ROOT, "models", "yolo26n.onnx")
PT_PATH   = os.path.join(ROOT, "models", "yolo26n.pt")
DEBUG_DIR = os.path.join(ROOT, "debug")
REPORT    = os.path.join(DEBUG_DIR, "diagnosis_report.txt")

os.makedirs(DEBUG_DIR, exist_ok=True)

lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

def sep(title=""):
    bar = "─" * 60
    log()
    log(bar)
    if title:
        log(f"  {title}")
        log(bar)

# ─────────────────────────────────────────────────────────────────────────────
sep("DualVision AI — YOLO26n Diagnostic")
log(f"Timestamp : {datetime.datetime.now().isoformat()}")
log(f"ONNX path : {ONNX_PATH}")
log(f"PT path   : {PT_PATH}")
log(f"  ONNX exists: {os.path.exists(ONNX_PATH)}")
log(f"  PT   exists: {os.path.exists(PT_PATH)}")

if not os.path.exists(ONNX_PATH):
    log()
    log("ERROR: ONNX model not found.  Export it first via Settings → ONNX/CPU.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 1 — Build test frame")

INPUT_SIZE = 640

# Create a synthetic test image: gradient + white rectangle (simulates a person/chair)
img_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
# Background gradient
for y in range(480):
    img_bgr[y, :, 0] = int(y / 480 * 200)
    img_bgr[y, :, 1] = 100
    img_bgr[y, :, 2] = int((480 - y) / 480 * 200)
# Two white rectangles (chair-ish shapes)
cv2.rectangle(img_bgr, (80, 200), (250, 420), (220, 220, 220), -1)
cv2.rectangle(img_bgr, (350, 200), (520, 420), (200, 200, 200), -1)
# Tall rectangle (person-ish)
cv2.rectangle(img_bgr, (290, 80), (340, 420), (180, 160, 140), -1)

cv2.imwrite(os.path.join(DEBUG_DIR, "diag_test_input.png"), img_bgr)
log(f"Test image shape : {img_bgr.shape}  dtype={img_bgr.dtype}")

# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 2 — Preprocess (same as app)")

h0, w0 = img_bgr.shape[:2]
small   = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
rgb     = small[:, :, ::-1]                                 # BGR → RGB
blob    = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0 # HWC → CHW /255
blob    = blob[np.newaxis, ...]                              # add batch dim

log(f"Original frame   : {w0}×{h0}")
log(f"Resized to       : {INPUT_SIZE}×{INPUT_SIZE}")
log(f"Blob shape       : {blob.shape}   dtype={blob.dtype}")
log(f"Blob value range : [{blob.min():.4f}, {blob.max():.4f}]")

dbg_img = (blob[0].transpose(1, 2, 0) * 255).astype(np.uint8)[:, :, ::-1]
cv2.imwrite(os.path.join(DEBUG_DIR, "diag_preprocessed.png"), dbg_img)
log(f"Saved: debug/diag_preprocessed.png")

# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 3 — ONNX model I/O info")

try:
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH,
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    log(f"Input  name  : {inp.name}")
    log(f"Input  shape : {inp.shape}")
    log(f"Input  type  : {inp.type}")
    log()
    for i, out in enumerate(sess.get_outputs()):
        log(f"Output[{i}] name  : {out.name}")
        log(f"Output[{i}] shape : {out.shape}")
        log(f"Output[{i}] type  : {out.type}")
    ort_version = ort.__version__
    log(f"\nONNX Runtime version: {ort_version}")
except Exception as e:
    log(f"ERROR loading ONNX model: {e}")
    import traceback; log(traceback.format_exc())
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 4 — ONNX raw output")

try:
    outputs = sess.run(None, {inp.name: blob})
    raw = outputs[0]
    log(f"Number of outputs : {len(outputs)}")
    for i, o in enumerate(outputs):
        log(f"  outputs[{i}]  shape={o.shape}  dtype={o.dtype}  "
            f"range=[{float(o.min()):.6f}, {float(o.max()):.6f}]")
    log()
    log(f"Primary output (raw) shape : {raw.shape}")
    pred = raw[0]
    log(f"After stripping batch      : {pred.shape}")

    # Normalise orientation
    if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
        pred_t = pred.T
    else:
        pred_t = pred
    n_anchors, n_ch = pred_t.shape
    log(f"Normalised (N_anchors, n_channels): ({n_anchors}, {n_ch})")
    log()

    # Detect format
    if n_ch == 6 and n_anchors <= 1000:
        fmt = "FORMAT A — NMS-included  [x1,y1,x2,y2,conf,cls_id]"
    elif n_ch == 85:
        fmt = "FORMAT C — YOLOv5/YOLO26  [cx,cy,w,h,obj,cls0..79]"
    elif n_ch == 84:
        fmt = "FORMAT B — YOLOv8/v11     [cx,cy,w,h,cls0..79]"
    else:
        fmt = f"UNKNOWN — n_channels={n_ch}"
    log(f"Detected output format: {fmt}")

    log()
    log("First 20 rows of normalised pred (first 10 columns):")
    log(f"  {'idx':>5}  {'cx':>8}  {'cy':>8}  {'w':>8}  {'h':>8}  "
        f"{'col4':>8}  {'col5':>8}  {'col6':>8}  {'col7':>8}  {'col8':>8}  {'col9':>8}")
    for r in range(min(20, n_anchors)):
        row = pred_t[r, :10]
        log(f"  {r:>5}  " + "  ".join(f"{v:>8.4f}" for v in row))

    # Confidence distribution
    log()
    if n_ch == 85:
        obj = pred_t[:, 4].astype(np.float32)
        cls = pred_t[:, 5:].astype(np.float32)
        best_cls_score = cls[np.arange(len(cls)), cls.argmax(axis=1)]
        conf = obj * best_cls_score
        log("Confidence = obj × max(cls_score)  [FORMAT C]")
        log(f"  objectness   range: [{obj.min():.6f}, {obj.max():.6f}]")
        log(f"  cls_score    range: [{best_cls_score.min():.6f}, {best_cls_score.max():.6f}]")
    elif n_ch == 84:
        cls = pred_t[:, 4:].astype(np.float32)
        conf = cls[np.arange(len(cls)), cls.argmax(axis=1)]
        log("Confidence = max(cls_score)  [FORMAT B]")
    elif n_ch == 6 and n_anchors <= 1000:
        conf = pred_t[:, 4].astype(np.float32)
        log("Confidence = col4  [FORMAT A]")
    else:
        conf = np.zeros(n_anchors, dtype=np.float32)

    log(f"  Final conf   range: [{conf.min():.6f}, {conf.max():.6f}]")
    top20_idx = conf.argsort()[::-1][:20]
    log()
    log("Top 20 confidence anchors:")
    log(f"  {'rank':>4}  {'anchor':>7}  {'conf':>8}  {'box_cx':>8}  {'box_cy':>8}  {'box_w':>8}  {'box_h':>8}")
    for rank, idx in enumerate(top20_idx):
        r = pred_t[idx]
        cx, cy, bw, bh = r[0], r[1], r[2], r[3]
        log(f"  {rank:>4}  {idx:>7}  {conf[idx]:>8.4f}  {cx:>8.2f}  {cy:>8.2f}  {bw:>8.2f}  {bh:>8.2f}")

except Exception as e:
    log(f"ERROR during ONNX inference: {e}")
    import traceback; log(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 5 — PyTorch comparison (if .pt available)")

pt_dets = []
if os.path.exists(PT_PATH):
    try:
        from ultralytics import YOLO
        pt_model = YOLO(PT_PATH)
        pt_results = pt_model(img_bgr, imgsz=INPUT_SIZE, verbose=False)[0]
        boxes_pt  = pt_results.boxes
        log(f"PyTorch (.pt) detections: {len(boxes_pt)}")
        log(f"  {'#':>3}  {'cls':>5}  {'name':>12}  {'conf':>7}  {'x1':>7}  {'y1':>7}  {'x2':>7}  {'y2':>7}")
        for i in range(len(boxes_pt)):
            cls  = int(boxes_pt.cls[i])
            name = pt_results.names[cls]
            cf   = float(boxes_pt.conf[i])
            x1, y1, x2, y2 = boxes_pt.xyxy[i].tolist()
            log(f"  {i:>3}  {cls:>5}  {name:>12}  {cf:>7.4f}  "
                f"{x1:>7.1f}  {y1:>7.1f}  {x2:>7.1f}  {y2:>7.1f}")
            pt_dets.append({"cls": cls, "name": name, "conf": cf,
                            "box": [x1, y1, x2, y2]})
    except ImportError:
        log("ultralytics not installed — skipping PT comparison.")
    except Exception as e:
        log(f"PT comparison failed: {e}")
        import traceback; log(traceback.format_exc())
else:
    log(f"models/yolo26n.pt not found — skipping PT comparison.")


# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 6 — ONNX decoded detections (using app postprocessing logic)")

CONF_THRESH = 0.25
IOU_THRESH  = 0.45
sx = w0 / INPUT_SIZE
sy = h0 / INPUT_SIZE

onnx_dets = []
try:
    raw   = outputs[0]
    pred2 = raw[0]
    if pred2.ndim == 2 and pred2.shape[0] < pred2.shape[1]:
        pred2 = pred2.T
    n_anchors2, n_ch2 = pred2.shape

    if n_ch2 == 85:
        obj2  = pred2[:, 4].astype(np.float32)
        cls2  = pred2[:, 5:].astype(np.float32)
        bi    = cls2.argmax(axis=1)
        bcs   = cls2[np.arange(len(cls2)), bi]
        conf2 = obj2 * bcs
        boxes_c = pred2[:, :4]
        log(f"Using FORMAT C (objectness)  conf_thresh={CONF_THRESH}")
    elif n_ch2 == 84:
        cls2  = pred2[:, 4:].astype(np.float32)
        bi    = cls2.argmax(axis=1)
        conf2 = cls2[np.arange(len(cls2)), bi]
        boxes_c = pred2[:, :4]
        log(f"Using FORMAT B (no objectness)  conf_thresh={CONF_THRESH}")
    elif n_ch2 == 6 and n_anchors2 <= 1000:
        conf2   = pred2[:, 4].astype(np.float32)
        bi      = pred2[:, 5].astype(np.int32)
        boxes_c = pred2[:, :4]
        log(f"Using FORMAT A (NMS-included)  conf_thresh={CONF_THRESH}")
    else:
        conf2   = np.zeros(n_anchors2, dtype=np.float32)
        bi      = np.zeros(n_anchors2, dtype=np.int32)
        boxes_c = pred2[:, :4]

    mask2 = conf2 >= CONF_THRESH
    bx2, cf2, cl2 = boxes_c[mask2], conf2[mask2], bi[mask2]

    raw_boxes2, raw_confs2, raw_cls2 = [], [], []
    for i in range(len(bx2)):
        cx, cy, bw, bh = float(bx2[i,0]), float(bx2[i,1]), float(bx2[i,2]), float(bx2[i,3])
        if n_ch2 == 6:
            # Format A: already xyxy
            x1, y1, x2, y2 = cx*sx, cy*sy, bw*sx, bh*sy
        else:
            x1 = (cx - bw/2) * sx
            y1 = (cy - bh/2) * sy
            x2 = (cx + bw/2) * sx
            y2 = (cy + bh/2) * sy
        raw_boxes2.append([x1, y1, x2, y2])
        raw_confs2.append(float(cf2[i]))
        raw_cls2.append(int(cl2[i]))

    if len(raw_boxes2) > 1:
        rects = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in raw_boxes2]
        idx2  = cv2.dnn.NMSBoxes(rects, raw_confs2, CONF_THRESH, IOU_THRESH)
        idx2  = [int(i) for i in idx2.flatten()] if idx2 is not None and len(idx2) else []
    else:
        idx2 = list(range(len(raw_boxes2)))

    # COCO-80 class names fallback
    COCO80 = [
        "person","bicycle","car","motorcycle","airplane","bus","train","truck",
        "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
        "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
        "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
        "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
        "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
        "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
        "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
        "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
        "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
        "hair drier","toothbrush"
    ]

    log(f"\nONNX decoded detections (conf≥{CONF_THRESH}):  {len(idx2)} after NMS")
    log(f"  {'#':>3}  {'cls':>5}  {'name':>12}  {'conf':>7}  {'x1':>7}  {'y1':>7}  {'x2':>7}  {'y2':>7}")
    for rank, i in enumerate(idx2):
        ci   = raw_cls2[i]
        name = COCO80[ci] if ci < len(COCO80) else str(ci)
        box  = raw_boxes2[i]
        log(f"  {rank:>3}  {ci:>5}  {name:>12}  {raw_confs2[i]:>7.4f}  "
            f"{box[0]:>7.1f}  {box[1]:>7.1f}  {box[2]:>7.1f}  {box[3]:>7.1f}")
        onnx_dets.append({"cls": ci, "name": name, "conf": raw_confs2[i], "box": box})

except Exception as e:
    log(f"ERROR decoding ONNX output: {e}")
    import traceback; log(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 7 — Summary JSON")

report = {
    "timestamp": datetime.datetime.now().isoformat(),
    "onnx_model": ONNX_PATH,
    "onnx_output_shape": [list(o.shape) for o in outputs],
    "n_ch": int(n_ch) if 'n_ch' in dir() else -1,
    "detected_format": fmt if 'fmt' in dir() else "unknown",
    "onnx_detections": onnx_dets,
    "pt_detections": pt_dets,
}
json_path = os.path.join(DEBUG_DIR, "diagnosis_report.json")
with open(json_path, "w") as f:
    json.dump(report, f, indent=2)
log(f"Saved JSON: {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
sep("Done")

with open(REPORT, "w") as f:
    f.write("\n".join(lines))
log(f"Full report saved: {REPORT}")
