"""Replay a frozen section in four conditions and replace it in the shared report."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import difflib
import json
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError

from baseline import RUN, write_json

sys.path.insert(0, str(RUN.parent))
import pilot
from instruction import ADDITION, MARKER

ARMS = "ABCD"
SEED = 1701


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(task, edit):
    uid = task["uid"]
    record = read(RUN / "baselines" / uid / "record.json")
    index = edit["section_index"]
    search = next(item for item in record["search_calls"] if item["section_index"] == index)
    model = next(item for item in record["llm_calls"]
                 if item["call_type"] == "section_writing" and item["iteration_num"] == index)
    original = model["input_messages"]
    assert len(original) == 1 and original[0]["role"] == "user" and model["model_info"] == pilot.SETTINGS["model"]
    target_uids = {edit["image_uid"], *edit.get("equivalent_image_uids", [])}
    target = [image for image in search["image_results"] if image["uid"] in target_uids]
    assert {image["uid"] for image in target} == target_uids
    assert target and target[0]["index"] == edit["image_slot"], (uid, edit["image_slot"])
    raw_prompt = original[0]["content"]
    edited_prompt = raw_prompt
    for image in target:
        old = image["caption"] or image["description"]
        label = image["index"]
        block = f"[Source {label}: {old}]"
        assert edited_prompt.count(block) == 1, (uid, label, len(old))
        updated = old
        replacements = edit.get("replacements_by_uid", {}).get(image["uid"], edit.get("replacements", []))
        for before, after in replacements:
            assert updated.count(before) == 1, (uid, before)
            updated = updated.replace(before, after, 1)
        if edit.get("clarification"):
            updated += "\n\n" + edit["clarification"]
        assert updated != old
        edited_prompt = edited_prompt.replace(block, f"[Source {label}: {updated}]", 1)
    assert raw_prompt.count(MARKER) == edited_prompt.count(MARKER) == 1
    texts = {
        "A": raw_prompt, "B": edited_prompt,
        "C": raw_prompt.replace(MARKER, ADDITION + MARKER, 1),
        "D": edited_prompt.replace(MARKER, ADDITION + MARKER, 1),
    }
    prompts = {arm: [{"role": "user", "content": text}] for arm, text in texts.items()}
    return record, search, target, prompts


def complete(request):
    for attempt in range(1, 4):
        try:
            started = time.monotonic()
            response = pilot.request("/chat/completions", request)
            choice = response["choices"][0]
            if not choice["message"]["content"]:
                raise RuntimeError("Empty model response")
            return response, round(time.monotonic() - started, 3), attempt
        except (TimeoutError, URLError, RuntimeError) as error:
            print(f"Request attempt {attempt} failed: {error}", flush=True)
            if isinstance(error, HTTPError) and error.code == 400:
                raise RuntimeError(error.read().decode(errors="replace")[:1200]) from error
            if attempt == 3:
                raise
            time.sleep(4 * attempt)


def sources(content, search):
    text_ids = sorted(set(re.findall(r"\[citation:(txt\d+)\]", content)))
    image_ids = sorted(set(re.findall(r"!\[\]\(citation:(img\d+)\)", content)))
    texts = {item["index"]: item for item in search["text_results"]}
    images = {item["index"]: item for item in search["image_results"]}
    return {
        "text_citations": text_ids,
        "image_citations": image_ids,
        "text_sources": [{"citation_id": slot, **texts[slot]} for slot in text_ids if slot in texts],
        "image_sources": [{"citation_id": slot, **images[slot]} for slot in image_ids if slot in images],
        "unresolved": sorted((set(text_ids) - texts.keys()) | (set(image_ids) - images.keys())),
    }


def replay(task_order, task, edit):
    uid = task["uid"]
    record, search, target, prompts = prepare(task, edit)
    section_index = edit["section_index"]
    folder = RUN / "arms" / uid
    folder.mkdir(parents=True, exist_ok=True)
    slots = {item["index"] for item in target}
    write_json(folder / "inputs.json", {
        "task": task,
        "edit": edit,
        "baseline_session": record["session_id"],
        "settings": pilot.SETTINGS,
        "seed": SEED,
        "target_slots": sorted(slots),
        "prompts": prompts,
    })
    (folder / "caption.diff").write_text("".join(difflib.unified_diff(
        prompts["A"][0]["content"].splitlines(keepends=True),
        prompts["B"][0]["content"].splitlines(keepends=True),
        fromfile="A", tofile="B",
    )), encoding="utf-8")
    order = ("ABDC", "BCAD", "CDBA", "DACB")[task_order % 4]
    for arm in order:
        request = {**pilot.SETTINGS, "seed": SEED, "messages": prompts[arm]}
        result_path = folder / f"{arm}.json"
        if result_path.is_file():
            assert read(result_path)["request"] == request
            continue
        response, duration, attempt = complete(request)
        choice = response["choices"][0]
        content = choice["message"]["content"]
        refs = sources(content, search)
        updated = copy.deepcopy(record["sections"])
        section = next(item for item in updated if item["section_index"] == section_index)
        section.update(content=content, **refs)
        article = "\n\n".join(item["content"] for item in sorted(updated, key=lambda item: item["section_index"]))
        used_target = any(item["citation_id"] in slots for item in refs["image_sources"])
        (folder / f"{arm}.md").write_text(article, encoding="utf-8")
        (folder / f"{arm}-section.md").write_text(content, encoding="utf-8")
        write_json(result_path, {
            "task": task, "arm": arm, "section_index": section_index, "request": request,
            "response": response, "elapsed_seconds": duration, "attempt": attempt,
            "used_target": used_target, "sections": updated, "references": refs["image_citations"],
            "unresolved": refs["unresolved"],
        })
        print(f"DONE {uid} {arm} target={used_target} images={refs['image_citations']} "
              f"finish={choice['finish_reason']} elapsed={duration}s", flush=True)
    return uid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--uid", action="append")
    args = parser.parse_args()
    tasks = read(RUN / "sample.json")
    jobs = []
    for index, task in enumerate(tasks):
        uid = task["uid"]
        if args.uid and uid not in args.uid:
            continue
        path = RUN / "edits" / f"{uid}.json"
        if not path.is_file():
            print(f"Edit pending for {uid}", flush=True)
            continue
        edit = read(path)
        if edit.get("ineligible_reason"):
            print(f"No editable candidate for {uid}: {edit['ineligible_reason']}", flush=True)
            continue
        jobs.append((index, task, edit))
    if args.check:
        for _, task, edit in jobs:
            _, _, target, _ = prepare(task, edit)
            print(f"Checked {task['uid']} section {edit['section_index']+1} slots {[item['index'] for item in target]}")
        return
    if not jobs:
        raise SystemExit("No edited tasks are ready")
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(replay, *job): job[1]["uid"] for job in jobs}
        for future in as_completed(pending):
            try:
                print(f"COMPLETE {future.result()}", flush=True)
            except Exception as error:
                uid = pending[future]
                failures.append(uid)
                print(f"FAILED {uid}: {error!r}", flush=True)
    if failures:
        raise SystemExit("Unfinished tasks: " + ", ".join(failures))


if __name__ == "__main__":
    main()
