"""Render source rows and their statement headers; does not approve candidates."""
import argparse
from pathlib import Path
import sys

import pymupdf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finsight_ashare.common import read_json


def image(page, rect):
    pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), clip=rect, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def render(rid, candidates, out):
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    cells = []
    with pymupdf.open(ROOT / "data/raw" / (rid + ".pdf")) as doc:
        for metric in ["assets", "liabilities", "revenue", "cost", "net_profit", "operating_cashflow"]:
            row = candidates[metric][0]
            page = doc[row["page"] - 1]
            token = row["tokens"][0][0].strip("() ")
            matches = page.search_for(token)
            if not matches:
                raise ValueError(f"Cannot locate source number: {rid} {metric} {token}")
            hit = matches[0]
            header_page = doc[row["header"]["page"] - 1]
            headers = header_page.search_for(row["header"]["title"])
            hy = headers[0].y0 if headers else 30
            header_image = image(header_page, pymupdf.Rect(0, max(0, hy-15), header_page.rect.width, min(header_page.rect.height, hy+115)))
            row_image = image(page, pymupdf.Rect(0, max(0, hit.y0-20), page.rect.width, min(page.rect.height, hit.y1+28)))
            cell = Image.new("RGB", (950, 350), "white")
            draw = ImageDraw.Draw(cell)
            draw.text((8, 5), f"{rid} | {metric} | PDF p{row['page']} | {row['value']}", fill="black", font=font)
            cell.paste(header_image, (0, 35))
            cell.paste(row_image, (0, 245))
            cells.append(cell)
    board = Image.new("RGB", (1900, 1050), "#cccccc")
    for i, cell in enumerate(cells):
        board.paste(cell, ((i % 2)*950, (i//2)*350))
    path = out / (rid + ".png")
    board.save(path)
    print(path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    out = ROOT / "outputs/data_review_v0.4"
    out.mkdir(parents=True, exist_ok=True)
    for rid, rows in read_json(ROOT / "data/review_v0.4/all-candidates.json").items():
        if not args.only or rid in args.only:
            render(rid, rows, out)
