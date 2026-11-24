"""Export a Deep-Reporter Markdown report and its cited images to PDF."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


IMAGE_CITATION = re.compile(r"!\[([^\]\n]*)\]\(citation:(img\d+)\)")
INLINE_CITATION = re.compile(r"\[citation:(txt|img)(\d+)\]")
CJK_TEXT = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def find_session(log_path, report, session_id):
    match = None
    with log_path.open(encoding="utf-8") as logfile:
        for line in logfile:
            if not line.strip():
                continue
            record = json.loads(line)
            if session_id and record.get("session_id") != session_id:
                continue
            if record.get("final_report", {}).get("content", "").strip() != report.strip():
                continue
            if match is not None:
                raise ValueError("Several log records match this report. Use --session-id.")
            match = record
    if match is None:
        raise ValueError(f"No matching report in {log_path}. Use the original Markdown and its session log.")
    return match


def resolve_image(stored_path, image_root, log_dir):
    if not stored_path:
        return None
    path = Path(stored_path).expanduser()
    if image_root is not None:
        _, marker, relative = path.as_posix().rpartition("sandbox/images/")
        if marker:
            path = image_root / relative
        elif not path.is_absolute():
            path = image_root / path
    if not path.is_absolute() and not path.is_file():
        path = log_dir / path
    return path if path.is_file() else None


def materialize_images(report, sections, work_dir, image_root, log_dir):
    pieces = []
    image_count = 0
    remaining = report

    for section in sorted(sections, key=lambda item: item["section_index"]):
        content = section["content"]
        if not content:
            continue
        prefix, found, remaining = remaining.partition(content)
        if not found or IMAGE_CITATION.search(prefix):
            raise ValueError("Section contents in the session log do not match the Markdown.")
        pieces.append(prefix)
        number = section["section_index"] + 1
        sources = {source["citation_id"]: source for source in section.get("image_sources", [])}
        copied = {}

        def insert_image(match):
            alt, citation = match.groups()
            if citation not in copied:
                stored = sources.get(citation, {}).get("img_path", "")
                image = resolve_image(stored, image_root, log_dir)
                if image is None:
                    detail = f"image file not found ({stored})" if stored else "no image path in session log"
                    raise ValueError(f"Section {number}, {citation}: {detail}. Check the log and --image-root.")
                name = f"section-{number}-{citation}{image.suffix.lower()}"
                shutil.copyfile(image, work_dir / name)
                copied[citation] = name
            caption = alt or f"S{number}-{citation[3:]}"
            return f"![{caption}]({copied[citation]})"

        def format_reference(match):
            kind, index = match.groups()
            previous = match.string[match.start() - 1] if match.start() else ""
            space = "" if not previous or previous.isspace() else " "
            if kind == "img":
                return f"{space}[Figure S{number}-{index}]"
            return f"{space}[S{number}, source {index}]"

        content = IMAGE_CITATION.sub(insert_image, content)
        content = INLINE_CITATION.sub(format_reference, content)
        pieces.append(content)
        image_count += len(copied)

    if IMAGE_CITATION.search(remaining):
        raise ValueError("The report contains image markers outside the sections in the session log.")
    pieces.append(remaining)
    return "".join(pieces), image_count


def export_pdf(report_path, log_path, output_path, session_id, image_root, cjk_font):
    if output_path.suffix.lower() != ".pdf" or output_path in (report_path, log_path):
        raise ValueError("Choose a separate output path ending in .pdf.")
    for command in ("pandoc", "xelatex"):
        if shutil.which(command) is None:
            raise ValueError(f"{command} is required. Install Pandoc and a TeX distribution with XeLaTeX.")

    report = report_path.read_text(encoding="utf-8")
    session = find_session(log_path, report, session_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".report-pdf-", dir=output_path.parent) as directory:
        work_dir = Path(directory)
        markdown, image_count = materialize_images(
            report, session["sections"], work_dir, image_root, log_path.parent
        )
        markdown_path = work_dir / "report.md"
        markdown_path.write_text(markdown, encoding="utf-8")
        header_path = work_dir / "figures.tex"
        header_path.write_text("\\usepackage{float}\n\\floatplacement{figure}{H}\n", encoding="utf-8")
        staged_pdf = work_dir / "report.pdf"

        main_font = "Times New Roman" if sys.platform == "darwin" else "DejaVu Serif"
        command = [
            "pandoc", str(markdown_path),
            "--from=markdown+tex_math_single_backslash-raw_html-raw_tex",
            "--pdf-engine=xelatex",
            "--pdf-engine-opt=-no-shell-escape",
            "--resource-path", os.pathsep.join(map(str, (work_dir, report_path.parent))),
            "--include-in-header", str(header_path),
            "-V", "papersize=a4",
            "-V", "geometry:margin=24mm",
            "-V", f"mainfont={main_font}",
            "-V", "colorlinks=true",
            "-o", str(staged_pdf),
        ]
        if cjk_font or CJK_TEXT.search(markdown):
            font = cjk_font or ("Hiragino Sans GB" if sys.platform == "darwin" else "Noto Serif CJK SC")
            command += ["-V", f"CJKmainfont={font}"]
        result = subprocess.run(command, cwd=work_dir, check=True, capture_output=True, text=True)
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        staged_pdf.replace(output_path)
    return image_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Original Markdown report.")
    parser.add_argument("--session-log", type=Path, required=True, help="Generation session JSONL.")
    parser.add_argument("--session-id", help="Session ID to select when more than one record matches.")
    parser.add_argument("--image-root", type=Path, help="Local copy of data/sandbox/images, if relocated.")
    parser.add_argument("--cjk-font", help="Installed font family to use for Chinese, Japanese or Korean.")
    parser.add_argument("-o", "--output", type=Path, help="Output PDF. Defaults to the report path with .pdf.")
    args = parser.parse_args()
    report = args.report.expanduser().resolve()
    log = args.session_log.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else report.with_suffix(".pdf")
    image_root = args.image_root.expanduser().resolve() if args.image_root else None
    try:
        images = export_pdf(report, log, output, args.session_id, image_root, args.cjk_font)
    except subprocess.CalledProcessError as error:
        parser.exit(1, f"Pandoc could not produce the PDF:\n{error.stderr}\n")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(f"Exported {output} ({images} cited image sources)")


if __name__ == "__main__":
    main()
