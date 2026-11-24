"""Prepare image-review sheets without exposing the baseline's citations."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

RUN = Path(__file__).resolve().parent


def prepare(uid, first_sections):
    source = RUN / "baselines" / uid / "record.json"
    folder = RUN / "audit" / uid
    if not source.is_file():
        source = source.with_name("searches.json")
    if not source.is_file():
        return False
    record = json.loads(source.read_text(encoding="utf-8"))
    folder.mkdir(parents=True, exist_ok=True)
    outline = record["outline_generation"]["sections_summary"]
    data = []
    for search in sorted(record["search_calls"], key=lambda entry: entry["section_index"]):
        index = search["section_index"]
        images = search["image_results"]
        data.append({
            "section_index": index,
            "requirement": outline[index]["section_description"],
            "checklist": outline[index]["sectional_checklist"],
            "images": [
                {key: item[key] for key in ("index", "uid", "score", "img_path", "caption", "description")}
                for item in images
            ],
        })
        if index >= first_sections:
            continue
        output = folder / f"section-{index + 1}.jpg"
        if output.is_file():
            continue
        width, height = 320, 198
        sheet = Image.new("RGB", (5 * width, 2 * height), "#f3f2ef")
        draw = ImageDraw.Draw(sheet)
        for rank, item in enumerate(images[:10]):
            x, y = rank % 5 * width, rank // 5 * height
            try:
                with Image.open(item["img_path"]) as opened:
                    thumbnail = ImageOps.exif_transpose(opened).convert("RGBA")
                    thumbnail.thumbnail((300, 168))
                    sheet.paste(thumbnail, (x + 10 + (300 - thumbnail.width) // 2, y + 4), thumbnail)
            except OSError:
                draw.text((x + 10, y + 45), "Image not readable", fill="black")
            draw.text((x + 10, y + 175), f"{item['index']}   {Path(item['img_path']).name}", fill="black")
        temporary = output.with_suffix(".tmp")
        sheet.save(temporary, format="JPEG", quality=90)
        temporary.replace(output)
    output = folder / "candidates.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid")
    parser.add_argument("--sheets", type=int, default=2)
    args = parser.parse_args()
    for task in json.loads((RUN / "sample.json").read_text()):
        uid = task["uid"]
        if args.uid and uid != args.uid:
            continue
        if prepare(uid, args.sheets):
            print(f"Ready for caption audit: {uid}")


if __name__ == "__main__":
    main()
