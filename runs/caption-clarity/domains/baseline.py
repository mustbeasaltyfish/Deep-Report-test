"""Sample one report per M²LongBench domain and freeze common baselines."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path, PurePosixPath
import random

RUN = Path(__file__).resolve().parent
PROJECT = RUN.parents[2]
DATA = PROJECT / "data/benchmark/article_deconstructions.jsonl"
DOMAINS = ("AI", "Finance", "Geography", "Health", "History", "Humanity", "Religion", "Science", "Social")
MANIFEST = RUN / "sample.json"


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sample(rows):
    generator = random.Random(20261122)
    grouped = {domain: [] for domain in DOMAINS}
    for row in rows:
        path = PurePosixPath(row["source_md_path"])
        assert path.parts[0] == "TestSet" and path.parts[1] in grouped, row["uid"]
        if row["uid"] != "AG001":
            grouped[path.parts[1]].append(row)
    selected = []
    for domain in DOMAINS:
        row = generator.choice(sorted(grouped[domain], key=lambda item: item["uid"]))
        selected.append({
            "domain": domain,
            "uid": row["uid"],
            "source": row["source_md_path"],
            "query": row["deconstruction_result"]["query"],
            "sections": len(row["deconstruction_result"]["detailed_outline"]),
        })
    if MANIFEST.exists():
        assert json.loads(MANIFEST.read_text()) == selected
    else:
        write_json(MANIFEST, selected)
    return selected


def generate(row):
    from deep_reporter.api import create_openai_api
    from deep_reporter.utils.session_logger import DetailedSessionLogger

    uid = row["uid"]
    folder = RUN / "baselines" / uid
    folder.mkdir(parents=True, exist_ok=True)
    saved = folder / "record.json"
    if saved.is_file():
        return uid, len(json.loads(saved.read_text())["sections"])

    api = create_openai_api({
        "primary_model": "qwen3-32b",
        "max_tokens": 4096,
        "log_dir": str(folder),
        "log_file_prefix": "sessions",
    })
    assert api.config.llm_config["qwen_base_url"] == "http://127.0.0.1:8000/v1"
    task = row["deconstruction_result"]
    outline = task["detailed_outline"]
    log_search = DetailedSessionLogger.log_search_call

    def capture_search(logger, *args, **kwargs):
        log_search(logger, *args, **kwargs)
        snapshot = {
            "outline_generation": logger.session_record["outline_generation"],
            "search_calls": logger.session_record["search_calls"],
        }
        write_json(folder / "searches.json", snapshot)

    DetailedSessionLogger.log_search_call = capture_search
    try:
        response = api.generate_article(
            overall_query=task["query"],
            overall_checklist=task["overall_checklist"],
            generation_mode="without_planner",
            outline=outline,
            text_topk=20,
            image_topk=10,
            enable_filter=False,
            retriever_url="http://127.0.0.1:5555/search",
            userid=f"deconstruct_{uid}",
        )
    finally:
        DetailedSessionLogger.log_search_call = log_search
    if not response["success"]:
        raise RuntimeError(response["error"])
    record = next(
        item for path in sorted(folder.glob("sessions_*.jsonl"))
        for item in read_jsonl(path) if item["session_id"] == response["session_id"]
    )
    expected = set(range(len(outline)))
    sections = {item["section_index"]: item for item in record["sections"]}
    searches = {item["section_index"]: item for item in record["search_calls"]}
    writes = {
        item["iteration_num"]: item for item in record["llm_calls"]
        if item["call_type"] == "section_writing"
    }
    assert set(sections) == set(searches) == set(writes) == expected, (uid, len(outline), sections.keys())
    for index in expected:
        assert searches[index]["text_results"] and searches[index]["image_results"], (uid, index)
        content = sections[index]["content"]
        assert content and not content.startswith(("Call failed:", "[Section generation failed:")), (uid, index, content[:260])
    expected_article = "\n\n".join(sections[index]["content"] for index in sorted(expected))
    assert expected_article == record["final_report"]["content"] == response["final_article"]
    (folder / f"{uid}.md").write_text(expected_article, encoding="utf-8")
    write_json(saved, record)
    return uid, len(sections)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    rows = read_jsonl(DATA)
    assert len(rows) == 247 and len({row["uid"] for row in rows}) == len(rows)
    selected = sample(rows)
    for item in selected:
        print(f"{item['domain']} {item['uid']} sections={item['sections']} · {item['query']}", flush=True)
    if args.sample_only:
        return
    os.environ.update(
        QWEN_BASE_URL="http://127.0.0.1:8000/v1",
        QWEN_API_KEY="local",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
    )
    index = {row["uid"]: row for row in rows}
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(generate, index[item["uid"]]): item["uid"] for item in selected}
        for job in as_completed(jobs):
            try:
                uid, count = job.result()
                print(f"FROZEN {uid} {count} sections", flush=True)
            except Exception as error:
                uid = jobs[job]
                print(f"FAILED {uid}: {error!r}", flush=True)
                failed.append(uid)
    if failed:
        raise SystemExit(f"Unfinished baselines: {', '.join(failed)}")
    print(f"All {len(selected)} baselines frozen", flush=True)


if __name__ == "__main__":
    main()
