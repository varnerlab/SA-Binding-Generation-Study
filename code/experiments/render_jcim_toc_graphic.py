#!/usr/bin/env python3
"""Render the 3.25 x 1.75 inch JCIM table-of-contents graphic."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "paper-jcim"
    / "sections"
    / "figs"
    / "toc_graphic.png"
)

WIDTH, HEIGHT = 975, 525
INK = "#17202A"
BLUE = "#2A6FBB"
PALE_BLUE = "#E8F1FA"
ORANGE = "#E6862D"
PALE_ORANGE = "#FCEBD9"
GRAY = "#667085"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")


def font(size, bold=False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def centered(draw, box, text, text_font, fill=INK):
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=text_font)
    x = left + (right - left - (bounds[2] - bounds[0])) / 2
    y = top + (bottom - top - (bounds[3] - bounds[1])) / 2 - bounds[1]
    draw.text((x, y), text, font=text_font, fill=fill)


def arrow(draw, start, end, fill=INK, width=5):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2 - 14, y2), fill=fill, width=width)
    draw.polygon(((x2, y2), (x2 - 18, y2 - 11), (x2 - 18, y2 + 11)), fill=fill)


def sequence_row(draw, x, y, selected=False):
    colors = [BLUE, "#7B9FC5", "#BBC9D8", "#7B9FC5", BLUE, "#BBC9D8", BLUE]
    if selected:
        colors[3] = ORANGE
    for index, color in enumerate(colors):
        x0 = x + 24 * index
        draw.rounded_rectangle((x0, y, x0 + 18, y + 25), radius=2,
                               fill=color, outline="white", width=1)


def main():
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((22, 65, 279, 417), radius=22, fill=PALE_BLUE,
                           outline=BLUE, width=4)
    centered(draw, (22, 78, 279, 125), "FAMILY ALIGNMENT", font(20, bold=True))
    for row in range(6):
        chosen = row in (1, 4)
        sequence_row(draw, 62, 145 + row * 40, selected=chosen)
        if chosen:
            draw.rounded_rectangle((45, 148 + row * 40, 52, 167 + row * 40),
                                   radius=3, fill=ORANGE)
    centered(draw, (22, 367, 279, 406), "designated rows", font(18, bold=True), ORANGE)

    arrow(draw, (292, 241), (382, 241))

    draw.rounded_rectangle((386, 65, 615, 417), radius=22, fill=PALE_ORANGE,
                           outline=ORANGE, width=4)
    centered(draw, (386, 78, 615, 125), "MULTIPLICITY BIAS", font(20, bold=True))
    centered(draw, (386, 165, 615, 215), "attention + log rho", font(22))
    centered(draw, (386, 235, 615, 285), "rho:  1  ->  500", font(24))
    arrow(draw, (432, 320), (573, 320), fill=ORANGE, width=10)
    centered(draw, (386, 362, 615, 402), "training-free", font(18), GRAY)

    arrow(draw, (628, 241), (710, 241))

    draw.rounded_rectangle((714, 65, 953, 417), radius=22, fill="#F7F9FC",
                           outline=INK, width=4)
    centered(draw, (714, 78, 953, 125), "CONDITIONED LIBRARY", font(18, bold=True))
    for row in range(6):
        sequence_row(draw, 757, 145 + row * 40, selected=True)
    centered(draw, (714, 367, 953, 406), "marker enriched", font(18, bold=True), ORANGE)

    centered(
        draw,
        (20, 447, 955, 505),
        "Exact latent reweighting  |  decoded recovery depends on representation",
        font(17),
        GRAY,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, dpi=(300, 300), optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
