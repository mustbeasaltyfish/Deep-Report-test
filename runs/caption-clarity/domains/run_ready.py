"""Run four-arm replays once the frozen baseline and caption review are both present."""

from concurrent.futures import ThreadPoolExecutor
import time

from arms import RUN, read, replay

tasks = read(RUN / "sample.json")
finished = set()
inflight = {}
errors = {}
deadline = time.monotonic() + 5400
with ThreadPoolExecutor(max_workers=3) as pool:
    while len(finished) < len(tasks) and time.monotonic() < deadline:
        for uid, future in list(inflight.items()):
            if not future.done():
                continue
            del inflight[uid]
            try:
                print(f"COMPLETE {future.result()}", flush=True)
                finished.add(uid)
                errors.pop(uid, None)
            except Exception as error:
                errors[uid] = errors.get(uid, 0) + 1
                print(f"FAILED {uid} attempt {errors[uid]}: {error!r}", flush=True)
        for index, task in enumerate(tasks):
            uid = task["uid"]
            if uid in finished or uid in inflight or errors.get(uid, 0) >= 2:
                continue
            edit_path = RUN / "edits" / f"{uid}.json"
            baseline = RUN / "baselines" / uid / "record.json"
            if edit_path.is_file() and baseline.is_file():
                edit = read(edit_path)
                if edit.get("ineligible_reason"):
                    finished.add(uid)
                    print(f"NO CANDIDATE {uid}: {edit['ineligible_reason']}", flush=True)
                elif len(inflight) < 3:
                    print(f"START {uid}", flush=True)
                    inflight[uid] = pool.submit(replay, index, task, edit)
        if len(finished) == len(tasks):
            break
        if errors and not inflight and all(uid in finished or errors.get(uid) == 2 for uid in (t["uid"] for t in tasks)):
            break
        time.sleep(12)
if len(finished) != len(tasks):
    raise SystemExit(f"Unfinished tasks: {set(t['uid'] for t in tasks) - finished}")
print("All task replays accounted for", flush=True)
