"""Render specified physical PDF pages for fact review; original remains unchanged."""
import argparse
from pathlib import Path
import pymupdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--pages", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path(".cache/pdf-review"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(args.pdf) as document:
        for number in args.pages:
            page = document[number-1]
            print(f"\nPDF physical page {number}\n{page.get_text(sort=True)}")
            page.get_pixmap(matrix=pymupdf.Matrix(1.7, 1.7)).save(args.output / f"page-{number}.png")


if __name__ == "__main__":
    main()
