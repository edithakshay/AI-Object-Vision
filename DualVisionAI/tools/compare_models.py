"""
DualVision AI — Model Comparison Tool  (PART 11)
================================================
Side-by-side comparison of all available YOLO26 models on the same input.

Run from the DualVisionAI/ directory:

    python tools/compare_models.py [--video path/to/clip.mp4] [--frames 100]

Outputs:
  - Console table with FPS / Inference / Detections / Stability / Memory / CPU
  - debug/comparison_report.json
  - debug/comparison_frames/  (annotated frames, one per model, for visual check)

If a model's ONNX is missing it is skipped with a clear message.
"""

import sys
import os
import json
import time
import datetime
import argparse
import shutil

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

MODEL_VARIANTS = {
    "yolo26n": {"onnx": "yolo26n.onnx", "label": "YOLO26n (Nano)",   "color": (0, 255, 80)},
    "yolo26s": {"onnx": "yolo26s.onnx", "label": "YOLO26s (Small)",  "color": (0, 180, 255)},
    "yolo26m": {"onnx": "yolo26m.onnx", "label": "YOLO26m (Medium)", "color": (255, 180, 0)},
    "yolo26l": {"onnx": "yolo26l.onnx", "label": "YOLO26l (Large)",  "color": (200, 0, 255)},
    "yolo26x": {"onnx": "yolo26x.onnx", "label": "YOLO26x (XLarge)","color": (0, 60, 255)},
}

MODELS_DIR     = os.path.join(ROOT, "models")
DEBUG_DIR      = os.path.join(ROOT, "debug")
FRAME_OUT_DIR  = os.path.join(DEBUG_DIR, "comparison_frames")
JSON_OUT       = os.path.join(DEBUG_DIR, "comparison_report.json")

os.makedirs(FRAME_OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_frames(n: int):
    frames = []
    for i in range(n):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:, :, 1] = 60
        img[:, :, 2] = int(i / n * 120)
        x = (i * 3) % 560
        cv2.rectangle(img, (x, 100), (x + 60, 300), (200, 200, 200), -1)
        cv2.rectangle(img, (200, 80 + (i % 30)), (340, 380), (160, 140, 120), -1)
        frames.append(img)
    return frames


def _preprocess(frame, sz: int):
    s   = cv2.resize(frame, (sz, sz), interpolation=cv2.INTER_LINEAR)
    rgb = s[:, :, ::-1]
    b   = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return b[np.newaxis, ...]


def _decode_boxes(outputs, conf_thresh: float, w0: int, h0: int, sz: int):
    """Return list of (x1,y1,x2,y2,conf,cls_id) for raw ONNX output."""
    sx, sy = w0 / sz, h0 / sz
    raw  = outputs[0][0]
    if raw.ndim == 2 and raw.shape[0] < raw.shape[1]:
        raw = raw.T

    results = []
    n_ch = raw.shape[1] if raw.ndim == 2 else 0

    if n_ch == 6 and raw.shape[0] <= 1000:
        # Format A
        for row in raw:
            c = float(row[4])
            if c >= conf_thresh:
                results.append((row[0]*sx, row[1]*sy, row[2]*sx, row[3]*sy,
                                c, int(row[5])))
    elif n_ch == 85:
        # Format C
        obj  = raw[:, 4].astype(np.float32)
        cls  = raw[:, 5:].astype(np.float32)
        bi   = cls.argmax(axis=1)
        bcs  = cls[np.arange(len(cls)), bi]
        conf = obj * bcs
        boxes = raw[:, :4]
        for i in range(len(conf)):
            if conf[i] >= conf_thresh:
                cx, cy, bw, bh = boxes[i]
                x1 = (cx - bw / 2) * sx;  y1 = (cy - bh / 2) * sy
                x2 = (cx + bw / 2) * sx;  y2 = (cy + bh / 2) * sy
                results.append((x1, y1, x2, y2, float(conf[i]), int(bi[i])))
    elif n_ch >= 5:
        # Format B
        cls  = raw[:, 4:].astype(np.float32)
        bi   = cls.argmax(axis=1)
        conf = cls[np.arange(len(cls)), bi]
        boxes = raw[:, :4]
        for i in range(len(conf)):
            if conf[i] >= conf_thresh:
                cx, cy, bw, bh = boxes[i]
                x1 = (cx - bw / 2) * sx;  y1 = (cy - bh / 2) * sy
                x2 = (cx + bw / 2) * sx;  y2 = (cy + bh / 2) * sy
                results.append((x1, y1, x2, y2, float(conf[i]), int(bi[i])))

    # NMS
    if len(results) > 1:
        rects = [[r[0], r[1], r[2]-r[0], r[3]-r[1]] for r in results]
        confs = [r[4] for r in results]
        idx = cv2.dnn.NMSBoxes(rects, confs, conf_thresh, 0.45)
        idx = [int(i) for i in idx.flatten()] if idx is not None and len(idx) else []
        results = [results[i] for i in idx]
    return results


def draw_comparison_frame(frame, dets: list, model_label: str,
                           fps: float, ms: float, color: tuple):
    """Draw detections + info header on a copy of frame."""
    out = frame.copy()
    for x1, y1, x2, y2, conf, cls_id in dets:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{cls_id} {conf:.2f}"
        cv2.putText(out, label, (int(x1)+2, int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    # Header bar
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (20, 20, 40), -1)
    cv2.putText(out, f"{model_label}  |  {fps:.1f} FPS  |  {ms:.0f} ms  |  {len(dets)} dets",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    return out


# ── Comparison runner ─────────────────────────────────────────────────────────

def run_comparison(frames: list, input_size: int, conf_thresh: float):
    import onnxruntime as ort

    # Discover available models
    available = {}
    for variant, meta in MODEL_VARIANTS.items():
        p = os.path.join(MODELS_DIR, meta["onnx"])
        if os.path.exists(p):
            available[variant] = (p, meta)
        else:
            print(f"  SKIP {meta['label']} — {meta['onnx']} not found")

    if not available:
        print("No ONNX models found.  Export them via Settings → ONNX/CPU first.")
        return {}

    n_frames    = len(frames)
    all_results = {}

    for variant, (onnx_path, meta) in available.items():
        print(f"\nComparing {meta['label']} …", flush=True)
        sess     = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name

        # Warm-up
        dummy = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
        for _ in range(3):
            sess.run(None, {inp_name: dummy})

        times_ms   = []
        det_counts = []
        confs_all  = []
        saved_frame_idx = min(n_frames // 2, n_frames - 1)

        for fi, frame in enumerate(frames):
            h0, w0 = frame.shape[:2]
            blob   = _preprocess(frame, input_size)

            t0  = time.perf_counter()
            out = sess.run(None, {inp_name: blob})
            ms  = (time.perf_counter() - t0) * 1000.0
            times_ms.append(ms)

            dets = _decode_boxes(out, conf_thresh, w0, h0, input_size)
            det_counts.append(len(dets))
            confs_all.extend(d[4] for d in dets)

            # Save one annotated frame for visual comparison
            if fi == saved_frame_idx:
                avg_fps  = 1000.0 / (sum(times_ms) / len(times_ms))
                annotated = draw_comparison_frame(
                    frame, dets, meta["label"],
                    avg_fps, ms, meta["color"])
                out_path = os.path.join(FRAME_OUT_DIR, f"{variant}.png")
                cv2.imwrite(out_path, annotated)

        if not times_ms:
            print(f"  WARNING: no timing data collected for {meta['label']} — skipping")
            continue
        avg_ms       = sum(times_ms) / len(times_ms)
        fps          = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        frames_w_det = sum(1 for n in det_counts if n > 0)
        stability    = frames_w_det / n_frames * 100.0 if n_frames > 0 else 0.0

        # Detection drops (frame goes 1→0→1)
        drops = sum(
            1 for i in range(1, len(det_counts) - 1)
            if det_counts[i-1] > 0 and det_counts[i] == 0 and det_counts[i+1] > 0
        )

        avg_conf = float(sum(confs_all) / len(confs_all)) if confs_all else 0.0

        cpu_pct = None
        ram_mb  = None
        if HAS_PSUTIL:
            try:
                cpu_pct = psutil.cpu_percent(interval=None)
                ram_mb  = psutil.virtual_memory().used // (1024 * 1024)
            except Exception:
                pass

        all_results[variant] = {
            "label":          meta["label"],
            "fps":            round(fps, 1),
            "avg_ms":         round(avg_ms, 1),
            "det_count_avg":  round(sum(det_counts) / n_frames, 1),
            "stability_%":    round(stability, 1),
            "drops":          drops,
            "avg_confidence": round(avg_conf, 4),
            "cpu_pct":        cpu_pct,
            "ram_mb":         ram_mb,
        }
        print(f"  → {fps:.1f} FPS  avg_det={sum(det_counts)/n_frames:.1f}  "
              f"stability={stability:.1f}%  drops={drops}")

    return all_results


def print_table(results: dict):
    if not results:
        return
    print()
    print("=" * 100)
    print(f"  {'Model':<22} {'FPS':>6} {'ms':>7} {'Dets/f':>7} "
          f"{'Stability':>10} {'Drops':>7} {'Conf':>7} {'CPU%':>6} {'RAM MB':>8}")
    print("-" * 100)

    ranked = sorted(results.values(), key=lambda r: r["fps"], reverse=True)
    for r in ranked:
        cpu = f"{r['cpu_pct']:.0f}%" if r["cpu_pct"] is not None else " —"
        ram = f"{r['ram_mb']}"       if r["ram_mb"]  is not None else "—"
        print(f"  {r['label']:<22} {r['fps']:>6.1f} {r['avg_ms']:>7.1f} "
              f"{r['det_count_avg']:>7.1f} {r['stability_%']:>9.1f}% "
              f"{r['drops']:>7} {r['avg_confidence']:>7.3f} {cpu:>6} {ram:>8}")

    print("=" * 100)
    best_fps  = ranked[0]
    best_stab = max(results.values(), key=lambda r: r["stability_%"])
    best_conf = max(results.values(), key=lambda r: r["avg_confidence"])
    print(f"\n  Best FPS        : {best_fps['label']}")
    print(f"  Best Stability  : {best_stab['label']}  ({best_stab['stability_%']:.1f}%)")
    print(f"  Best Confidence : {best_conf['label']}  ({best_conf['avg_confidence']:.4f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="DualVision AI Model Comparison")
    parser.add_argument("--video",  default=None,
                        help="Path to a video file (default: synthetic frames)")
    parser.add_argument("--frames", type=int,   default=100,
                        help="Frames to test per model (default: 100)")
    parser.add_argument("--size",   type=int,   default=640,
                        help="ONNX input size (default: 640)")
    parser.add_argument("--conf",   type=float, default=0.45,
                        help="Confidence threshold (default: 0.45)")
    args = parser.parse_args()

    try:
        import onnxruntime
    except ImportError:
        print("ERROR: onnxruntime not installed.  Run: pip install onnxruntime")
        sys.exit(1)

    # ── Load frames ───────────────────────────────────────────────────────────
    if args.video:
        if not os.path.exists(args.video):
            print(f"ERROR: Video not found: {args.video}")
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video: {args.video}")
            sys.exit(1)
        frames = []
        while len(frames) < args.frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        src = os.path.basename(args.video)
    else:
        frames = _make_frames(max(1, args.frames))
        src    = "synthetic"

    if len(frames) == 0:
        print("ERROR: no frames loaded — cannot run comparison.")
        sys.exit(1)

    print(f"\nDualVision AI — Model Comparison")
    print(f"  Frames : {len(frames)}  Source : {src}")
    print(f"  Size   : {args.size}   Conf   : {args.conf}")

    results = run_comparison(frames, args.size, args.conf)
    print_table(results)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    payload = {
        "timestamp":  datetime.datetime.now().isoformat(),
        "source":     src,
        "frames":     len(frames),
        "input_size": args.size,
        "conf":       args.conf,
        "results":    results,
    }
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  JSON saved   : {JSON_OUT}")
    print(f"  Frames saved : {FRAME_OUT_DIR}/")


if __name__ == "__main__":
    main()
