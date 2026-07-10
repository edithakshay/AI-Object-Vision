"""
DualVision AI — Multi-Model Benchmark Tool  (PART 2)
=====================================================
Run from the DualVisionAI/ directory:

    python tools/benchmark.py [--video path/to/video.mp4] [--frames 300]

Benchmarks every available YOLO26 ONNX model and generates MODEL_BENCHMARK.md.

Each model is tested independently with the same frames.
If no video is provided, synthetic frames are generated.

Output files:
  MODEL_BENCHMARK.md
  debug/benchmark_results.json
"""

import sys
import os
import json
import time
import datetime
import argparse

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── Model registry (same as ai/model_manager.py) ─────────────────────────────
MODEL_VARIANTS = {
    "yolo26n": {"onnx": "yolo26n.onnx", "label": "YOLO26n (Nano)",   "size_mb": 6},
    "yolo26s": {"onnx": "yolo26s.onnx", "label": "YOLO26s (Small)",  "size_mb": 22},
    "yolo26m": {"onnx": "yolo26m.onnx", "label": "YOLO26m (Medium)", "size_mb": 52},
    "yolo26l": {"onnx": "yolo26l.onnx", "label": "YOLO26l (Large)",  "size_mb": 87},
    "yolo26x": {"onnx": "yolo26x.onnx", "label": "YOLO26x (XLarge)","size_mb": 136},
}

MODELS_DIR  = os.path.join(ROOT, "models")
DEBUG_DIR   = os.path.join(ROOT, "debug")
REPORT_OUT  = os.path.join(ROOT, "MODEL_BENCHMARK.md")
JSON_OUT    = os.path.join(DEBUG_DIR, "benchmark_results.json")

os.makedirs(DEBUG_DIR, exist_ok=True)


def _make_synthetic_frames(n: int, w: int = 640, h: int = 480):
    """Generate n synthetic BGR frames that contain rectangle shapes."""
    frames = []
    for i in range(n):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Slow gradient background
        img[:, :, 1] = 60
        img[:, :, 2] = int(i / n * 120)
        # Moving rectangles
        x = (i * 3) % (w - 80)
        cv2.rectangle(img, (x, 100), (x + 60, 300), (200, 200, 200), -1)
        cv2.rectangle(img, (200, 50 + (i % 40)), (340, 380), (160, 140, 120), -1)
        frames.append(img)
    return frames


def _preprocess(frame, sz: int) -> np.ndarray:
    small = cv2.resize(frame, (sz, sz), interpolation=cv2.INTER_LINEAR)
    rgb   = small[:, :, ::-1]
    blob  = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return blob[np.newaxis, ...]


def _count_detections(outputs, conf_thresh: float = 0.45) -> tuple:
    """Return (n_detections, avg_confidence) for raw ONNX output."""
    try:
        raw  = outputs[0][0]
        if raw.ndim == 2 and raw.shape[0] < raw.shape[1]:
            raw = raw.T
        if raw.shape[1] == 6 and raw.shape[0] <= 1000:
            # Format A: NMS-included
            confs = raw[:, 4].astype(np.float32)
        elif raw.shape[1] == 85:
            # Format C: objectness
            obj  = raw[:, 4].astype(np.float32)
            cls  = raw[:, 5:].astype(np.float32)
            bcs  = cls[np.arange(len(cls)), cls.argmax(axis=1)]
            confs = obj * bcs
        else:
            # Format B: no objectness
            cls   = raw[:, 4:].astype(np.float32)
            confs = cls[np.arange(len(cls)), cls.argmax(axis=1)]

        above = confs[confs >= conf_thresh]
        n     = len(above)
        avg   = float(above.mean()) if n > 0 else 0.0
        return n, avg
    except Exception:
        return 0, 0.0


def benchmark_model(onnx_path: str, frames: list,
                    input_size: int = 640,
                    conf_thresh: float = 0.45,
                    warmup: int = 5) -> dict:
    """Run full benchmark for one model.  Returns result dict."""
    import onnxruntime as ort

    print(f"\n  Loading {os.path.basename(onnx_path)} …", flush=True)
    sess      = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name  = sess.get_inputs()[0].name

    # Warm-up
    dummy = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
    for _ in range(warmup):
        sess.run(None, {inp_name: dummy})

    times_ms    = []
    det_counts  = []
    confidences = []

    cpu_samples = []
    ram_samples = []

    print(f"  Running {len(frames)} frames …", flush=True)
    for fi, frame in enumerate(frames):
        blob = _preprocess(frame, input_size)

        if HAS_PSUTIL and fi % 10 == 0:
            cpu_samples.append(psutil.cpu_percent(interval=None))
            ram_samples.append(psutil.virtual_memory().used // (1024 * 1024))

        t0  = time.perf_counter()
        out = sess.run(None, {inp_name: blob})
        ms  = (time.perf_counter() - t0) * 1000.0
        times_ms.append(ms)

        n, avg_conf = _count_detections(out, conf_thresh)
        det_counts.append(n)
        if avg_conf > 0:
            confidences.append(avg_conf)

        if (fi + 1) % 50 == 0:
            print(f"    {fi+1}/{len(frames)} frames  "
                  f"avg={sum(times_ms)/len(times_ms):.1f}ms", flush=True)

    n_frames = len(times_ms)
    avg_ms   = sum(times_ms) / n_frames
    fps      = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    min_ms   = min(times_ms)
    max_ms   = max(times_ms)
    p95_ms   = float(np.percentile(times_ms, 95))

    # Detection stability: fraction of frames that had ≥1 detection
    frames_with_det  = sum(1 for n in det_counts if n > 0)
    det_stability    = frames_with_det / n_frames * 100.0 if n_frames > 0 else 0.0

    # Dropped detections: frames going from detection to no detection (1-frame drop)
    drops = 0
    for i in range(1, len(det_counts) - 1):
        if det_counts[i-1] > 0 and det_counts[i] == 0 and det_counts[i+1] > 0:
            drops += 1

    avg_conf_val = float(sum(confidences) / len(confidences)) if confidences else 0.0

    return {
        "avg_ms":        round(avg_ms, 2),
        "min_ms":        round(min_ms, 2),
        "max_ms":        round(max_ms, 2),
        "p95_ms":        round(p95_ms, 2),
        "fps":           round(fps, 2),
        "avg_cpu_pct":   round(sum(cpu_samples) / len(cpu_samples), 1)
                         if cpu_samples else None,
        "avg_ram_mb":    round(sum(ram_samples) / len(ram_samples))
                         if ram_samples else None,
        "avg_confidence":  round(avg_conf_val, 4),
        "det_stability_%": round(det_stability, 1),
        "detection_drops": drops,
        "frames_tested":   n_frames,
    }


def write_markdown_report(results: list, video_source: str):
    """Write MODEL_BENCHMARK.md."""

    # Sort by FPS (highest first)
    ranked = sorted(results, key=lambda r: r["fps"], reverse=True)

    lines = [
        "# DualVision AI — Model Benchmark Report",
        "",
        f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source    : {video_source}",
        f"Platform  : CPU-only (ONNX Runtime CPUExecutionProvider)",
        "",
        "---",
        "",
        "## Summary Table",
        "",
        "| Rank | Model | FPS ↓ | Avg ms | P95 ms | CPU% | RAM MB | Conf | Stability% | Drops |",
        "|------|-------|-------:|-------:|-------:|-----:|-------:|-----:|-----------:|------:|",
    ]

    rank_map = {}
    for rank, r in enumerate(ranked, 1):
        rank_map[r["variant"]] = rank
        cpu  = f"{r['avg_cpu_pct']:.0f}%" if r["avg_cpu_pct"] is not None else "—"
        ram  = f"{r['avg_ram_mb']}"        if r["avg_ram_mb"]  is not None else "—"
        lines.append(
            f"| {rank} | {r['label']} | **{r['fps']:.1f}** | {r['avg_ms']:.1f} "
            f"| {r['p95_ms']:.1f} | {cpu} | {ram} "
            f"| {r['avg_confidence']:.3f} | {r['det_stability_%']:.1f}% "
            f"| {r['detection_drops']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Detailed Results",
        "",
    ]

    for r in results:
        lines += [
            f"### {r['label']}",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Average FPS | **{r['fps']:.1f}** |",
            f"| Avg Inference (ms) | {r['avg_ms']:.1f} |",
            f"| Min Inference (ms) | {r['min_ms']:.1f} |",
            f"| Max Inference (ms) | {r['max_ms']:.1f} |",
            f"| P95 Inference (ms) | {r['p95_ms']:.1f} |",
            f"| CPU Usage | {r['avg_cpu_pct']:.0f}%" if r['avg_cpu_pct'] is not None
              else "| CPU Usage | — |",
            f"| RAM Usage | {r['avg_ram_mb']} MB" if r['avg_ram_mb'] is not None
              else "| RAM Usage | — |",
            f"| Avg Confidence | {r['avg_confidence']:.4f} |",
            f"| Detection Stability | {r['det_stability_%']:.1f}% |",
            f"| Detection Drops | {r['detection_drops']} |",
            f"| Frames Tested | {r['frames_tested']} |",
            "",
        ]

    # Recommendation
    best_cpu  = ranked[0]
    best_acc  = max(results, key=lambda r: r["avg_confidence"])
    best_stab = max(results, key=lambda r: r["det_stability_%"])

    lines += [
        "---",
        "",
        "## Recommendations",
        "",
        f"### Best for CPU Speed: **{best_cpu['label']}**",
        f"  - {best_cpu['fps']:.1f} FPS average",
        f"  - {best_cpu['avg_ms']:.1f} ms average inference",
        "",
        f"### Best Detection Accuracy: **{best_acc['label']}**",
        f"  - Average confidence: {best_acc['avg_confidence']:.4f}",
        "",
        f"### Best Detection Stability: **{best_stab['label']}**",
        f"  - Stability: {best_stab['det_stability_%']:.1f}%",
        f"  - Detection drops: {best_stab['detection_drops']}",
        "",
        "### Production Recommendation (Search & Rescue, CPU-only)",
        "",
        "For reliable long-duration operation where stability matters more than FPS:",
        "",
    ]

    # Score: 60% stability + 40% confidence (FPS is secondary for SAR)
    def sar_score(r):
        return r["det_stability_%"] * 0.6 + r["avg_confidence"] * 100 * 0.4

    best_sar = max(results, key=sar_score)
    lines += [
        f"**Recommended model: {best_sar['label']}**",
        "",
        f"  - Detection Stability: {best_sar['det_stability_%']:.1f}%",
        f"  - Average Confidence: {best_sar['avg_confidence']:.4f}",
        f"  - FPS: {best_sar['fps']:.1f}",
        "",
        "Pair with: Confidence Smoother ON, Ghost Frames = 3–5, Persistence = 5–10 frames.",
        "",
        "---",
        f"*Generated by DualVision AI tools/benchmark.py — {datetime.datetime.now().isoformat()}*",
    ]

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved: {REPORT_OUT}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="DualVision AI Model Benchmark")
    parser.add_argument("--video",  default=None,
                        help="Path to a video file to use for benchmarking")
    parser.add_argument("--frames", type=int, default=200,
                        help="Number of frames to benchmark per model (default 200)")
    parser.add_argument("--size",   type=int, default=640,
                        help="ONNX input size (default 640)")
    parser.add_argument("--conf",   type=float, default=0.45,
                        help="Confidence threshold (default 0.45)")
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
        cap    = cv2.VideoCapture(args.video)
        frames = []
        while len(frames) < args.frames:
            ok, frm = cap.read()
            if not ok:
                break
            frames.append(frm)
        cap.release()
        video_source = os.path.basename(args.video)
        print(f"Loaded {len(frames)} frames from {video_source}")
    else:
        frames       = _make_synthetic_frames(args.frames)
        video_source = "synthetic test frames"
        print(f"Using {len(frames)} synthetic frames (no video provided)")

    if len(frames) == 0:
        print("ERROR: no frames loaded.")
        sys.exit(1)

    # ── Benchmark each available model ────────────────────────────────────────
    all_results = []
    for variant, meta in MODEL_VARIANTS.items():
        onnx_path = os.path.join(MODELS_DIR, meta["onnx"])
        if not os.path.exists(onnx_path):
            print(f"\nSKIPPING {meta['label']} — {meta['onnx']} not found in models/")
            print(f"  Download .pt then export: Settings → ONNX/CPU → Export to ONNX")
            continue

        result = benchmark_model(
            onnx_path=onnx_path,
            frames=frames,
            input_size=args.size,
            conf_thresh=args.conf,
        )
        result["variant"] = variant
        result["label"]   = meta["label"]
        all_results.append(result)
        print(f"  → {meta['label']}: {result['fps']:.1f} FPS  "
              f"avg={result['avg_ms']:.1f}ms  "
              f"stability={result['det_stability_%']:.1f}%")

    if not all_results:
        print("\nNo ONNX models found.  Run setup.py and export models first.")
        sys.exit(1)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    payload = {
        "timestamp":    datetime.datetime.now().isoformat(),
        "video_source": video_source,
        "frames_used":  len(frames),
        "input_size":   args.size,
        "conf_thresh":  args.conf,
        "results":      all_results,
    }
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON saved: {JSON_OUT}")

    # ── Write Markdown ────────────────────────────────────────────────────────
    write_markdown_report(all_results, video_source)
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
