"""
DualVision AI — logo & icon generator.
Run once: python assets/make_icon.py
Outputs: assets/logo.png (512x512) and assets/icon.ico (multi-size)
Requires: Pillow  (pip install pillow)
"""
import math
from pathlib import Path

HERE = Path(__file__).parent


def _draw_logo(size: int = 512):
    from PIL import Image, ImageDraw

    bg      = (5,  10,  20, 255)
    blue    = (37, 99,  235, 255)   # #2563EB  — RGB camera
    green   = (34, 197, 94, 255)    # #22C55E  — Thermal camera
    white   = (255, 255, 255, 255)
    dim     = (100, 130, 180, 255)

    img  = Image.new("RGBA", (size, size), bg)
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2
    R  = int(size * 0.42)   # outer ring radius
    r  = int(size * 0.12)   # eye radius
    sep = int(size * 0.14)  # distance of each eye from centre

    # ── outer hex ring ──────────────────────────────────────────────────
    pts = []
    for i in range(6):
        a = math.radians(60 * i - 30)
        pts.append((cx + R * math.cos(a), cy + R * math.sin(a)))
    draw.polygon(pts, outline=(*blue[:3], 60), fill=(*blue[:3], 18))

    # thin arc border
    draw.ellipse([cx-R, cy-R, cx+R, cy+R], outline=(*blue[:3], 90), width=3)

    # ── scan lines (subtle) ────────────────────────────────────────────
    for y in range(cy - R + 10, cy + R, 14):
        draw.line([(cx - R + 10, y), (cx + R - 10, y)],
                  fill=(*blue[:3], 18), width=1)

    # ── left eye (RGB / blue) ──────────────────────────────────────────
    lx = cx - sep
    draw.ellipse([lx-r, cy-r, lx+r, cy+r],
                 fill=(*blue[:3], 30), outline=blue, width=3)
    draw.ellipse([lx-r//3, cy-r//3, lx+r//3, cy+r//3],
                 fill=blue)
    # cross-hair
    draw.line([(lx, cy-r-8), (lx, cy-r+8)], fill=blue, width=2)
    draw.line([(lx-r-8, cy), (lx-r+8, cy)], fill=blue, width=2)
    draw.line([(lx, cy+r-8), (lx, cy+r+8)], fill=blue, width=2)
    draw.line([(lx+r-8, cy), (lx+r+8, cy)], fill=blue, width=2)

    # ── right eye (Thermal / green) ─────────────────────────────────────
    rx = cx + sep
    draw.ellipse([rx-r, cy-r, rx+r, cy+r],
                 fill=(*green[:3], 30), outline=green, width=3)
    # thermal rings
    for frac in (0.75, 0.50, 0.25):
        rr = int(r * frac)
        draw.ellipse([rx-rr, cy-rr, rx+rr, cy+rr],
                     outline=(*green[:3], int(255 * (1 - frac + 0.25))), width=2)
    draw.ellipse([rx-r//6, cy-r//6, rx+r//6, cy+r//6],
                 fill=green)
    # cross-hair
    draw.line([(rx, cy-r-8), (rx, cy-r+8)], fill=green, width=2)
    draw.line([(rx-r-8, cy), (rx-r+8, cy)], fill=green, width=2)
    draw.line([(rx, cy+r-8), (rx, cy+r+8)], fill=green, width=2)
    draw.line([(rx+r-8, cy), (rx+r+8, cy)], fill=green, width=2)

    # ── centre bridge line ─────────────────────────────────────────────
    draw.line([(lx + r, cy), (rx - r, cy)], fill=dim, width=2)

    # ── "DV" text below eyes ──────────────────────────────────────────
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", size // 8)
    except Exception:
        font = None  # will use default bitmap font

    text = "DualVision AI"
    if font:
        bb = draw.textbbox((0, 0), text, font=font)
        tw = bb[2] - bb[0]
        tx = cx - tw // 2
        ty = cy + R - size // 9
        draw.text((tx, ty), text, font=font, fill=white)
    else:
        draw.text((cx - 50, cy + R - size // 9), text, fill=white)

    return img


def generate(out_dir: Path = HERE):
    out_dir.mkdir(parents=True, exist_ok=True)

    logo_path = out_dir / "logo.png"
    ico_path  = out_dir / "icon.ico"

    img = _draw_logo(512)
    img.save(str(logo_path))
    print(f"[OK] logo.png  → {logo_path}")

    # ICO needs multiple sizes
    sizes = [16, 24, 32, 48, 64, 128, 256]
    icons = [img.resize((s, s), resample=3) for s in sizes]
    icons[0].save(str(ico_path), format="ICO",
                  sizes=[(s, s) for s in sizes],
                  append_images=icons[1:])
    print(f"[OK] icon.ico  → {ico_path}")
    return logo_path, ico_path


if __name__ == "__main__":
    generate()
    print("Done.")
