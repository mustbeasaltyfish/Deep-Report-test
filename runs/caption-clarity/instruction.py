"""Run caption-clarity arms C and D using the exact previously saved inputs."""

import copy
import difflib
import hashlib
import json
import time
from urllib.error import URLError

import pilot

SOURCE = pilot.RUN / "pilot"
OUT = pilot.RUN / "instruction"
MARKER = "# Content Requirements\n"
ADDITION = (
    "# Visual Evidence Decision\n"
    "Before drafting, check whether any candidate image directly helps explain, compare, "
    "or substantiate a point required by this section. If an image helps, place it next "
    "to the relevant passage and explain the specific detail the reader should notice. "
    "If none helps, write the section without an image.\n\n"
)


def main():
    OUT.mkdir(exist_ok=True)
    original = json.loads((SOURCE / "inputs.json").read_text(encoding="utf-8"))
    assert original["settings"] == pilot.SETTINGS
    assert original["seeds"] == list(pilot.SEEDS)
    assert sorted(original["target_slots"]) == sorted(pilot.TARGETS)
    prompts = {}
    differences = []
    for base, arm in (("A", "C"), ("B", "D")):
        old = original["messages"][base]
        assert len(old) == 1 and old[0]["role"] == "user"
        text = old[0]["content"]
        assert hashlib.sha256(text.encode()).hexdigest() == original["prompt_hashes"][base]
        assert text.count(MARKER) == 1 and ADDITION not in text
        new = copy.deepcopy(old)
        new[0]["content"] = text.replace(MARKER, ADDITION + MARKER, 1)
        assert new[0]["content"].replace(ADDITION, "", 1) == text
        prompts[arm] = new
        differences.extend(difflib.unified_diff(
            text.splitlines(keepends=True),
            new[0]["content"].splitlines(keepends=True),
            fromfile=base,
            tofile=arm,
        ))
    (OUT / "prompt.diff").write_text("".join(differences), encoding="utf-8")
    pilot.save_json(OUT / "inputs.json", {
        "original_session": original["original_session"],
        "section_index": original["section_index"],
        "target_slots": original["target_slots"],
        "target_uid": original["target_uid"],
        "image_path": original["image_path"],
        "settings": original["settings"],
        "seeds": original["seeds"],
        "instruction": ADDITION,
        "messages": prompts,
    })

    pilot.OUT = OUT
    for _ in range(60):
        try:
            models = pilot.request("/models", timeout=3)
            assert any(model["id"] == pilot.SETTINGS["model"] for model in models["data"])
            break
        except (URLError, TimeoutError):
            time.sleep(3)
    else:
        raise RuntimeError("vLLM did not become ready")

    pairs = []
    for index, seed in enumerate(original["seeds"]):
        order = ("C", "D") if index % 2 == 0 else ("D", "C")
        result = {arm: pilot.run_one(arm, seed, prompts[arm]) for arm in order}
        pairs.append({
            "seed": seed,
            "order": list(order),
            "target": {arm: result[arm]["uses_target"] for arm in ("C", "D")},
            "images": {arm: result[arm]["image_citations"] for arm in ("C", "D")},
            "finish": {arm: result[arm]["response"]["choices"][0]["finish_reason"] for arm in ("C", "D")},
        })
        pilot.save_json(OUT / "pairs.json", pairs)
    counts = {
        "pairs": len(pairs),
        "target_counts": {arm: sum(pair["target"][arm] for pair in pairs) for arm in ("C", "D")},
        "requires_semantic_review": True,
    }
    pilot.save_json(OUT / "summary.json", counts)
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
