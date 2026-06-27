"""
Generates app icons and logo for DualVision AI Detector using only Pillow.
Run once: python generate_icons.py
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math

ICONS_DIR = Path("icons")
ASSETS_DIR = Path("assets")
ICONS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

BG_COLOR = (10, 15, 30, 255)
BLUE = (37, 99, 235, 255)
ORANGE = (249, 115, 22, 255)
WHITE = (255, 255, 255, 255)
LIGHT_BLUE = (96, 165, 250, 255)


def draw_logo(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r = size // 2 - 4

    # Outer circle - dark bg
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=BG_COLOR)

    # Outer ring - blue
    ring_w = max(2, size // 20)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=BLUE, width=ring_w)

    # Inner design — two overlapping camera/lens shapes
    eye_r = r * 0.38
    offset = r * 0.22

    # RGB camera lens (blue, left)
    lx, ly = cx - offset, cy
    d.ellipse([lx - eye_r, ly - eye_r, lx + eye_r, ly + eye_r],
              fill=(20, 40, 80, 255), outline=BLUE, width=max(1, size // 40))
    inner_r = eye_r * 0.55
    d.ellipse([lx - inner_r, ly - inner_r, lx + inner_r, ly + inner_r],
              fill=BLUE)
    pupil_r = inner_r * 0.45
    d.ellipse([lx - pupil_r, ly - pupil_r, lx + pupil_r, ly + pupil_r],
              fill=(180, 210, 255, 255))

    # Thermal camera lens (orange, right)
    tx, ty = cx + offset, cy
    d.ellipse([tx - eye_r, ty - eye_r, tx + eye_r, ty + eye_r],
              fill=(40, 20, 10, 255), outline=ORANGE, width=max(1, size // 40))
    d.ellipse([tx - inner_r, ty - inner_r, tx + inner_r, ty + inner_r],
              fill=ORANGE)
    d.ellipse([tx - pupil_r, ty - pupil_r, tx + pupil_r, ty + pupil_r],
              fill=(255, 210, 150, 255))

    # AI crosshair lines
    line_color = (100, 140, 220, 180)
    lw = max(1, size // 64)
    d.line([cx - r + ring_w, cy, cx + r - ring_w, cy],
           fill=line_color, width=lw)
    d.line([cx, cy - r + ring_w, cx, cy + r - ring_w],
           fill=line_color, width=lw)

    # Corner brackets (AI bounding box style)
    blen = r * 0.25
    bw = max(2, size // 32)
    corners = [
        (cx - r * 0.6, cy - r * 0.6),
        (cx + r * 0.6, cy - r * 0.6),
        (cx - r * 0.6, cy + r * 0.6),
        (cx + r * 0.6, cy + r * 0.6),
    ]
    for bx, by in corners:
        sx = 1 if bx > cx else -1
        sy = 1 if by > cy else -1
        d.line([(bx, by), (bx + sx * blen, by)], fill=ORANGE, width=bw)
        d.line([(bx, by), (bx, by + sy * blen)], fill=ORANGE, width=bw)

    return img


def save_ico(image: Image.Image, path: Path):
    sizes = [16, 24, 32, 48, 64, 128, 256]
    icons = []
    for s in sizes:
        ico = image.resize((s, s), Image.LANCZOS)
        icons.append(ico)
    icons[0].save(path, format="ICO", sizes=[(s, s) for s in sizes],
                  append_images=icons[1:])
    print(f"  Saved: {path}")


def save_png(image: Image.Image, path: Path, size: int):
    resized = image.resize((size, size), Image.LANCZOS)
    resized.save(path, format="PNG")
    print(f"  Saved: {path}")


def save_svg(path: Path):
    svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg width="256" height="256" viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="bg" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#0F1E3D"/>
      <stop offset="100%" stop-color="#050A14"/>
    </radialGradient>
  </defs>
  <!-- Background circle -->
  <circle cx="128" cy="128" r="124" fill="url(#bg)"/>
  <circle cx="128" cy="128" r="124" fill="none" stroke="#2563EB" stroke-width="8"/>
  <!-- Crosshair lines -->
  <line x1="8" y1="128" x2="248" y2="128" stroke="#4080CC" stroke-width="1.5" opacity="0.6"/>
  <line x1="128" y1="8" x2="128" y2="248" stroke="#4080CC" stroke-width="1.5" opacity="0.6"/>
  <!-- RGB Camera lens (blue, left) -->
  <circle cx="100" cy="128" r="38" fill="#0D1F45" stroke="#2563EB" stroke-width="4"/>
  <circle cx="100" cy="128" r="21" fill="#2563EB"/>
  <circle cx="100" cy="128" r="10" fill="#93C5FD"/>
  <!-- Thermal Camera lens (orange, right) -->
  <circle cx="156" cy="128" r="38" fill="#2A1000" stroke="#F97316" stroke-width="4"/>
  <circle cx="156" cy="128" r="21" fill="#F97316"/>
  <circle cx="156" cy="128" r="10" fill="#FED7AA"/>
  <!-- Corner brackets -->
  <polyline points="68,68 68,89 89,89" fill="none" stroke="#F97316" stroke-width="5" stroke-linecap="round"/>
  <polyline points="188,68 188,89 167,89" fill="none" stroke="#F97316" stroke-width="5" stroke-linecap="round"/>
  <polyline points="68,188 68,167 89,167" fill="none" stroke="#F97316" stroke-width="5" stroke-linecap="round"/>
  <polyline points="188,188 188,167 167,167" fill="none" stroke="#F97316" stroke-width="5" stroke-linecap="round"/>
</svg>"""
    path.write_text(svg, encoding="utf-8")
    print(f"  Saved: {path}")


def main():
    print("Generating DualVision AI Detector icons and logo ...")

    logo = draw_logo(512)

    save_ico(logo, ICONS_DIR / "app.ico")
    save_png(logo, ICONS_DIR / "app_256.png", 256)
    save_png(logo, ICONS_DIR / "app_128.png", 128)
    save_png(logo, ICONS_DIR / "app_64.png", 64)
    save_png(logo, ICONS_DIR / "app_32.png", 32)
    save_png(logo, ASSETS_DIR / "logo.png", 512)
    save_svg(ASSETS_DIR / "logo.svg")

    # Splash banner (wide)
    banner = Image.new("RGBA", (800, 200), BG_COLOR)
    d = ImageDraw.Draw(banner)
    mini = draw_logo(160)
    banner.paste(mini, (20, 20), mini)
    try:
        font_title = ImageFont.truetype("arial.ttf", 36)
        font_sub = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = font_title

    d.text((200, 50), "DualVision AI Detector", fill=WHITE[:3], font=font_title)
    d.text((200, 100), "High-FPS Dual RTSP AI Object Detection", fill=(148, 163, 184), font=font_sub)
    d.text((200, 130), "v1.0.0  •  Python 3.12  •  YOLO + CustomTkinter", fill=(71, 85, 105), font=font_sub)
    banner.save(ASSETS_DIR / "splash_banner.png", format="PNG")
    print(f"  Saved: {ASSETS_DIR / 'splash_banner.png'}")

    print("\nAll icons and assets generated successfully!")


if __name__ == "__main__":
    main()
