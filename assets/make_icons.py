"""Render the Peresmeshnik icons (line bird in a headset) to PNG / .icns / .ico.

App icon = sketch variant 5 (cream, dark line, teal cup, coral boom and mic).
Tray / menu bar = variant 6 (bold mono) on a transparent background.
SVG is rendered with PyMuPDF, which has no clip-path, gradients or rect rx=w/2,
so every rounded shape is an explicit arc path.

    python assets/make_icons.py        # needs pymupdf + pillow
"""
import io
import math
from pathlib import Path

import fitz
from PIL import Image

OUT = Path(__file__).parent / 'icons'
NAVY, CREAM, CORAL, TEAL = '#22263A', '#FBF6EE', '#FF7A59', '#2EC4B6'

BODY = ('M72.6 26.6 L90 29.5 L72.6 33.4 C74 44 70 53 63 60 C57 66 50 69 43 71 L17 85 C14 86.5 12 83.5 14 82 '
        'L37 66 C39 54 43 46 50 38.4 A13 13 0 1 1 72.6 26.6 Z')
EYE, HC, HR = (65.5, 26.5), (60, 30), 13
CUP = (45.2, 23.2, 8.4, 13.6)
BOOM = 'M49.4 36.6 C51 46.5 67 47.5 75.2 39.6'
MIC = (76.8, 38.2)
FIT = 'translate(50,50) scale(0.86) translate(-51,-47)'  # inside the rounded square, ~12 % padding
FILL = 'translate(50,50) scale(1.1) translate(-52,-48)'  # tray glyph uses the whole box


def rounded(x, y, w, h, r):
    return (f'M{x + r} {y} H{x + w - r} A{r} {r} 0 0 1 {x + w} {y + r} V{y + h - r} '
            f'A{r} {r} 0 0 1 {x + w - r} {y + h} H{x + r} A{r} {r} 0 0 1 {x} {y + h - r} '
            f'V{y + r} A{r} {r} 0 0 1 {x + r} {y} Z')


def arch(gap=7, a1=200, a2=335):
    """Headband: arc through two points of the head circle, peaking `gap` above the crown."""
    cx, cy = HC
    p1 = (cx + HR * math.cos(math.radians(a1)), cy + HR * math.sin(math.radians(a1)))
    p2 = (cx + HR * math.cos(math.radians(a2)), cy + HR * math.sin(math.radians(a2)))
    half, sag = (p2[0] - p1[0]) / 2, (cy - HR - gap) - p1[1]
    r = (half ** 2 + sag ** 2) / (2 * -sag)
    large = int(-sag > half)
    return (f'M{p1[0]:.2f} {p1[1]:.2f} A{r:.2f} {r:.2f} 0 {large} 1 {p2[0]:.2f} {p2[1]:.2f} '
            f'L{p2[0] - 1.2:.2f} {p2[1] + 1.2:.2f}')


def bird(fg, body_fill, cup_fill, boom, mic, sw, tf):
    sw /= 0.86
    x, y, w, h = CUP
    s = f'<g transform="{tf}">'
    s += f'<path d="{arch()}" fill="none" stroke="{fg}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>'
    s += f'<path d="{BODY}" fill="{body_fill}" stroke="{fg}" stroke-width="{sw}" stroke-linejoin="round"/>'
    s += f'<circle cx="{EYE[0]}" cy="{EYE[1]}" r="{sw * .62}" fill="{fg}"/>'
    s += f'<path d="{BOOM}" fill="none" stroke="{boom}" stroke-width="{sw * .75}" stroke-linecap="round"/>'
    s += f'<path d="{rounded(x, y, w, h, w / 2)}" fill="{cup_fill}" stroke="{fg}" stroke-width="{sw}"/>'
    s += f'<circle cx="{MIC[0]}" cy="{MIC[1]}" r="{sw * .9}" fill="{mic}"/>'
    return s + '</g>'


def app(pad=0.0):
    """Variant 5. `pad` shrinks the tile (macOS draws icons on an 824/1024 grid)."""
    k = 1 - 2 * pad
    tile = f'<path d="{rounded(0, 0, 100, 100, 22)}" fill="{CREAM}"/>'
    tile += bird(NAVY, CREAM, TEAL, CORAL, CORAL, 2.8, FIT)
    return f'<g transform="translate({pad * 100},{pad * 100}) scale({k})">{tile}</g>'


def tray(color):
    """Variant 6: one colour on transparent; solid cup so nothing shows through it."""
    return bird(color, 'none', color, color, color, 3.8, FILL)


def png(body, size):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">{body}</svg>'
    pix = fitz.open(stream=svg.encode(), filetype='svg')[0].get_pixmap(matrix=fitz.Matrix(size / 100, size / 100), alpha=True)
    return Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGBA')


def main():
    OUT.mkdir(exist_ok=True)
    full = {n: png(app(), n) for n in (16, 24, 32, 48, 64, 128, 256, 512, 1024)}
    for n, im in full.items():
        im.save(OUT / f'app-{n}.png')
    full[256].save(OUT / 'app.ico', sizes=[(n, n) for n in (16, 24, 32, 48, 64, 128, 256)],
                   append_images=[full[n] for n in (16, 24, 32, 48, 64, 128)])
    mac = {n: png(app(pad=100 / 1024), n) for n in (16, 32, 64, 128, 256, 512, 1024)}
    mac[1024].save(OUT / 'app.icns', append_images=[mac[n] for n in (16, 32, 64, 128, 256, 512)])
    for name, color in (('tray', '#000000'), ('tray-light', '#FFFFFF')):
        ims = {n: png(tray(color), n) for n in (16, 22, 24, 32, 44, 48, 64)}
        for n, im in ims.items():
            im.save(OUT / f'{name}-{n}.png')
        ims[64].save(OUT / f'{name}.ico', sizes=[(n, n) for n in (16, 24, 32, 48, 64)],
                     append_images=[ims[n] for n in (16, 24, 32, 48)])
    (OUT / 'app.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">{app()}</svg>\n')
    (OUT / 'tray.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">{tray("#000000")}</svg>\n')
    print('icons ->', OUT)


if __name__ == '__main__':
    main()
