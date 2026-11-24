"""Calculate citation metrics and export excerpts for blinded visual review."""

import argparse
import json
from pathlib import Path
import re

from baseline import DATA, PROJECT, RUN, write_json
from citation_precision_evaluator import CitationPrecisionEvaluator
import logging


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def target_context(content, slots):
    paragraphs = re.split(r"\n\s*\n", content)
    pattern = re.compile(r"!\[\]\(citation:(" + "|".join(slots) + r")\)")
    return [
        "…\n\n" + "\n\n".join(paragraphs[max(0, index - 1):index + 2]) + "\n\n…"
        for index, paragraph in enumerate(paragraphs) if pattern.search(paragraph)
    ]


def measure(task, result, baseline, silver, evaluator, slots):
    baseline["sections"] = result["sections"]
    raw = evaluator.evaluate_article(baseline, silver)
    index = result["section_index"]
    section = next(item for item in raw["section_results"] if item["section_index"] == index)
    precision = raw["average_precisions"]
    image_total = sum(row["precisions"]["image"]["final"]["used"] for row in raw["section_results"])
    image_sections = sum(row["precisions"]["image"]["final"]["used"] > 0 for row in raw["section_results"])
    choice = result["response"]["choices"][0]
    return {
        "uid": task["uid"], "domain": task["domain"], "arm": result["arm"],
        "section_index": index,
        "target": result["used_target"],
        "any": section["precisions"]["image"]["final"]["used"] > 0,
        "image_slots": result["references"],
        "unresolved": result["unresolved"],
        "finish": choice["finish_reason"],
        "usage": result["response"]["usage"],
        "section_text": section["precisions"]["text"]["final"],
        "section_image": section["precisions"]["image"]["final"],
        "report_text": precision["text"]["final"],
        "report_image": precision["image"]["final"],
        "report_overall": (precision["text"]["final"] + precision["image"]["final"]) / 2,
        "report_image_count": image_total,
        "report_image_sections": image_sections,
        "report_sections": len(raw["section_results"]),
        "report_image_coverage": image_sections / len(raw["section_results"]),
        "contexts": target_context(choice["message"]["content"], slots),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    logging.getLogger("citation_precision_evaluator").setLevel(logging.CRITICAL)
    task_list = read(RUN / "sample.json")
    ids = {task["uid"] for task in task_list}
    with DATA.with_name("article_deconstructions_enriched.jsonl").open(encoding="utf-8") as stream:
        silver = {row["uid"]: row for row in map(json.loads, stream) if row["uid"] in ids}
    assert set(silver) == ids

    evaluator = CitationPrecisionEvaluator()
    records = []
    pending = []
    for task in task_list:
        uid = task["uid"]
        folder = RUN / "arms" / uid
        if not (folder / "inputs.json").exists():
            pending.append(uid)
            continue
        inputs = read(folder / "inputs.json")
        complete = True
        for arm in "ABCD":
            path = folder / f"{arm}.json"
            if not path.is_file():
                complete = False
                continue
            result = read(path)
            baseline = read(RUN / "baselines" / uid / "record.json")
            assert baseline["session_id"] == inputs["baseline_session"]
            assert result["request"]["messages"] == inputs["prompts"][arm]
            section_idx = inputs["edit"]["section_index"]
            for base, changed in zip(baseline["sections"], result["sections"], strict=True):
                assert base["section_index"] == changed["section_index"]
                if base["section_index"] != section_idx:
                    assert base == changed
            records.append(measure(task, result, baseline, silver[uid], evaluator, inputs["target_slots"]))
        if not complete:
            pending.append(uid)
    write_json(RUN / "metrics.json", records)
    print(f"Scored {len(records)} arms, pending: {pending}")
    if args.require_all and (pending or len(records) != len(task_list) * 4):
        raise SystemExit("Incomplete experiment")


if __name__ == "__main__":
    main()
