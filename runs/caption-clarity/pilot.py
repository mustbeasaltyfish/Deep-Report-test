"""Replay the paired Section 1 caption intervention from PLAN.md."""

import copy
import difflib
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

RUN = Path(__file__).resolve().parent
SOURCE = RUN.parent / "qwen32-base-smoke"
OUT = RUN / "pilot"
API = "http://127.0.0.1:8000/v1"
SEEDS = range(1701, 1707)
TARGETS = {"img1", "img2"}
SETTINGS = {
    "model": "qwen3-32b",
    "max_tokens": 4096,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "chat_template_kwargs": {"enable_thinking": False},
}
EDITS = (
    (
        "The datasets show that each user's examples are individually included.",
        "The left dataset contains three triangles from User 1. The right dataset removes exactly one of those triangles and keeps every other example.",
    ),
    (
        "The resulting models are identical, indicating that the noise addition process does not affect the model outcome despite differences in the datasets.",
        'The output models are labeled "Looks the same": noisy training limits how distinguishable the outputs are when one example is removed.',
    ),
    (
        "However, the right side shows a blue box covering User 1's data, indicating aggregation or exclusion in user-level DP.",
        "The right dataset removes all three of User 1's triangles. It keeps the other users' data unchanged.",
    ),
    (
        "The resulting models are identical, suggesting that the privacy mechanism at the user level maintains consistency despite aggregated data handling.",
        'The output models are labeled "Looks the same": noisy training limits how distinguishable the outputs are when all examples from one user are removed.',
    ),
    (
        'Both scenarios result in models that "look the same," highlighting the effectiveness of DP in maintaining model performance while ensuring privacy.',
        'The label "Looks the same" illustrates limited distinguishability of the output distributions under the indicated data change.',
    ),
)
OPENER = build_opener(ProxyHandler({}))


def request(route, body=None, timeout=240):
    data = json.dumps(body).encode() if body is not None else None
    req = Request(API + route, data=data, headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=timeout) as response:
        return json.load(response)


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_prompts():
    sid = (SOURCE / "session-id.txt").read_text().strip()
    with (SOURCE / "sessions_20261122.jsonl").open(encoding="utf-8") as stream:
        session = next(record for record in map(json.loads, stream) if record["session_id"] == sid)
    assert session["user_id"] == "deconstruct_AG001"
    assert session["user_input"]["enable_filter"] is False
    writer = next(
        call for call in session["llm_calls"]
        if call["call_type"] == "section_writing" and call["iteration_num"] == 0
    )
    assert writer["model_info"] == SETTINGS["model"]
    original = writer["input_messages"]
    assert len(original) == 1 and original[0]["role"] == "user"
    candidates = next(call for call in session["search_calls"] if call["section_index"] == 0)
    target = [item for item in candidates["image_results"] if item["index"] in TARGETS]
    assert len(target) == 2 and len({image["uid"] for image in target}) == 1
    old = target[0]["caption"] or target[0]["description"]
    assert all((image["caption"] or image["description"]) == old for image in target)
    revised = old
    for before, after in EDITS:
        assert revised.count(before) == 1, before
        revised = revised.replace(before, after)
    edited = copy.deepcopy(original)
    for slot in sorted(TARGETS):
        block = f"[Source {slot}: {old}]"
        assert edited[0]["content"].count(block) == 1, slot
        edited[0]["content"] = edited[0]["content"].replace(block, f"[Source {slot}: {revised}]")
    diff = difflib.unified_diff(
        original[0]["content"].splitlines(keepends=True),
        edited[0]["content"].splitlines(keepends=True),
        fromfile="A-original-caption",
        tofile="B-clear-caption",
    )
    (OUT / "prompt.diff").write_text("".join(diff), encoding="utf-8")
    save_json(OUT / "inputs.json", {
        "original_session": sid,
        "section_index": 0,
        "target_slots": sorted(TARGETS),
        "target_uid": target[0]["uid"],
        "image_path": target[0]["img_path"],
        "old_caption": old,
        "clear_caption": revised,
        "settings": SETTINGS,
        "seeds": list(SEEDS),
        "messages": {"A": original, "B": edited},
        "prompt_hashes": {
            label: hashlib.sha256(messages[0]["content"].encode()).hexdigest()
            for label, messages in (("A", original), ("B", edited))
        },
    })
    return {"A": original, "B": edited}


def run_one(arm, seed, messages):
    body = {**SETTINGS, "seed": seed, "messages": messages}
    path = OUT / f"{seed}-{arm}.json"
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior["request"] != body:
            raise ValueError(f"Stale request at {path}")
        return prior

    for attempt in range(1, 4):
        try:
            start = time.monotonic()
            response = request("/chat/completions", body)
            duration = time.monotonic() - start
            choice = response["choices"][0]
            content = choice["message"]["content"]
            if not content:
                raise RuntimeError("Empty model response")
            break
        except (HTTPError, URLError, TimeoutError, RuntimeError) as error:
            print(f"seed={seed} arm={arm} attempt={attempt} request error: {error}", flush=True)
            if isinstance(error, HTTPError):
                print(error.read().decode(errors="replace")[:800], flush=True)
                if 400 <= error.code < 500 and error.code != 429:
                    raise
            if attempt == 3:
                raise
            time.sleep(5 * attempt)

    image_citations = re.findall(r"!\[\]\(citation:(img\d+)\)", content)
    result = {
        "arm": arm,
        "seed": seed,
        "request": body,
        "response": response,
        "elapsed_seconds": round(duration, 3),
        "attempt": attempt,
        "image_citations": image_citations,
        "uses_target": bool(TARGETS.intersection(image_citations)),
        "image_reference_tokens": re.findall(r"citation:(img\d+)", content),
        "repeated_target": sum(slot in TARGETS for slot in image_citations) > 1,
    }
    save_json(path, result)
    (OUT / f"{seed}-{arm}.md").write_text(content, encoding="utf-8")
    print(f"seed={seed} arm={arm} target={result['uses_target']} "
          f"images={image_citations} finish={choice['finish_reason']} elapsed={duration:.1f}s", flush=True)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    prompts = load_prompts()
    for _ in range(60):
        try:
            models = request("/models", timeout=3)
            assert any(model["id"] == SETTINGS["model"] for model in models["data"])
            break
        except (URLError, TimeoutError):
            time.sleep(3)
    else:
        raise RuntimeError("vLLM did not become ready")

    pairs = []
    for index, seed in enumerate(SEEDS):
        arms = ("A", "B") if index % 2 == 0 else ("B", "A")
        result = {arm: run_one(arm, seed, prompts[arm]) for arm in arms}
        pairs.append({
            "seed": seed,
            "order": list(arms),
            "target": {arm: result[arm]["uses_target"] for arm in ("A", "B")},
            "images": {arm: result[arm]["image_citations"] for arm in ("A", "B")},
            "finish": {arm: result[arm]["response"]["choices"][0]["finish_reason"] for arm in ("A", "B")},
        })
        save_json(OUT / "pairs.json", pairs)

    plus = sum(not pair["target"]["A"] and pair["target"]["B"] for pair in pairs)
    minus = sum(pair["target"]["A"] and not pair["target"]["B"] for pair in pairs)
    summary = {
        "pairs": len(pairs),
        "A": sum(pair["target"]["A"] for pair in pairs),
        "B": sum(pair["target"]["B"] for pair in pairs),
        "A_no_B_yes": plus,
        "A_yes_B_no": minus,
        "net_gain": plus - minus,
        "passes_citation_gate": plus - minus >= 3 and minus <= 1,
        "semantic_gate": "requires manual review",
    }
    save_json(OUT / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
