"""Turn a Deep-Reporter session JSONL into a standalone HTML trace."""

import argparse
import base64
import json
import re
from collections import defaultdict
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory


STYLES = """
:root {
  color-scheme: light;
  --bg: #f7f8f6;
  --ink: #1c2b26;
  --muted: #4e625a;
  --line: #dce4df;
  --accent: #155a4b;
  --soft: #eaf3ee;
  --white: #fff;
  font: 15px/1.6 system-ui, -apple-system, "PingFang SC", sans-serif;
}
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); background: var(--bg); }
.layout { display: grid; grid-template-columns: 190px minmax(0, 880px); gap: 36px;
  max-width: 1180px; padding: 42px 28px 76px; margin: auto; }
aside { position: sticky; top: 32px; align-self: start; }
aside strong { display: block; margin-bottom: 12px; color: var(--accent); }
nav a { display: block; padding: 7px 10px; border-radius: 6px; text-decoration: none; }
nav a:hover, nav a:focus-visible { background: var(--soft); }
a { color: var(--accent); }
main { min-width: 0; }
header, .card { border: 1px solid var(--line); border-radius: 14px; background: var(--white);
  padding: 24px 28px; }
header { border-top: 4px solid var(--accent); }
section { margin-top: 22px; scroll-margin-top: 18px; }
h1, h2, h3 { line-height: 1.28; margin: 0 0 12px; }
h1 { font-size: clamp(1.85rem, 4vw, 2.65rem); }
h2 { font-size: 1.45rem; }
h3 { font-size: 1rem; }
p { margin: 8px 0 14px; }
ul, ol { padding-left: 23px; margin: 7px 0 14px; }
.muted, .stage-time { color: var(--muted); }
.meta { margin: 0 0 14px; overflow-wrap: anywhere; font-size: .85rem; }
.flow { color: var(--accent); font-weight: 650; margin-bottom: 0; }
.stats, .columns { display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
.stats { grid-template-columns: repeat(4, minmax(0, 1fr)); margin: 17px 0; }
.stat { padding: 10px 12px; border-radius: 8px; background: var(--soft); }
.stat b { display: block; font-size: 1.24rem; }
.stat span { color: var(--muted); font-size: .84rem; }
.query { font-size: 1.16rem; border-left: 3px solid var(--accent); padding-left: 13px;
  white-space: pre-wrap; }
.tag { display: inline-block; font-size: .84rem; background: var(--soft); border-radius: 5px;
  padding: 2px 7px; vertical-align: baseline; }
.stage { border-left: 2px solid var(--line); padding: 5px 0 8px 16px; margin: 17px 0; }
.stage-title { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 5px; }
.stage-time { font-size: .84rem; font-variant-numeric: tabular-nums; }
details { border: 1px solid var(--line); border-radius: 8px; margin: 8px 0; min-width: 0; }
summary { padding: 8px 11px; cursor: pointer; color: var(--accent); font-weight: 610;
  overflow-wrap: anywhere; }
details > .inside { padding: 0 11px 10px; }
.label { color: var(--muted); font-size: .85rem; font-weight: 600; }
.raw { background: #f5f7f5; border-radius: 6px; padding: 11px; white-space: pre-wrap;
  overflow-wrap: anywhere; max-height: 440px; overflow: auto; tab-size: 2;
  font: .86rem/1.55 ui-monospace, monospace; margin: 7px 0; }
.prose { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.73; }
.prose p { margin: 0 0 12px; }
img { display: block; max-width: 100%; max-height: 440px; object-fit: contain;
  margin: 10px auto; border-radius: 5px; }
.path { overflow-wrap: anywhere; }
.warning { border-color: #e1a988; }
@media (max-width: 760px) {
  .layout { display: block; padding: 15px 12px 54px; }
  aside { position: static; margin-bottom: 12px; }
  aside strong { display: none; }
  nav { display: flex; flex-wrap: wrap; gap: 2px; }
  nav a { padding: 4px 8px; }
  header, .card { padding: 16px; }
  .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 470px) { .columns { grid-template-columns: 1fr; } }
"""

CALL_TYPES = {
    "query_generation": "生成图文查询",
    "section_writing": "撰写本节",
    "filter": "筛选证据",
    "filtering": "筛选证据",
    "planner": "规划大纲",
    "outline_generation": "生成大纲",
}
IMAGE_TYPES = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp", ".gif": "gif"}


def load_session(path, session_id):
    sidecar = path.with_name("session-id.txt")
    if session_id is None and sidecar.is_file():
        session_id = sidecar.read_text(encoding="utf-8").strip() or None
    selected = None
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no}: {error.msg}") from error
            if not isinstance(record, dict) or not record.get("session_id"):
                raise ValueError(f"{path}:{line_no}: expected a session record with session_id")
            if session_id is None or record["session_id"] == session_id:
                selected = record
    if selected is None:
        raise ValueError(f"Session {session_id!r} not found" if session_id else "The JSONL file is empty")
    return selected


def safe(value):
    return escape(str(value)) if value is not None else ""


def raw(value):
    content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return f'<pre class="raw"><code>{safe(content)}</code></pre>'


def details(label, body, open_by_default=False):
    state = " open" if open_by_default else ""
    return f'<details{state}><summary>{safe(label)}</summary><div class="inside">{body}</div></details>'


def items(values):
    if not values:
        return '<p class="muted">无记录</p>'
    return "<ol>" + "".join(f"<li>{safe(value)}</li>" for value in values) + "</ol>"


def timestamp(value):
    return str(value or "").replace("T", " ", 1)


def stage(title, body, time=""):
    display_time = timestamp(time)
    return (
        f'<div class="stage"><div class="stage-title"><h3>{safe(title)}</h3>'
        f'<time class="stage-time" title="{safe(display_time)}">{safe(display_time[11:19])}</time>'
        f'</div>{body}</div>'
    )


def group_calls(record):
    """Associate queries without iteration_num with the next logged section event."""
    grouped = defaultdict(list)
    unassigned = []
    events = []
    for call in record.get("llm_calls", []):
        index = call.get("iteration_num")
        if isinstance(index, int) and index >= 0:
            grouped[index].append(call)
            events.append((call, index, False))
        elif call.get("call_type") == "query_generation" and call.get("timestamp"):
            events.append((call, None, True))
        else:
            unassigned.append(call)

    for key in ("search_calls", "filter_calls", "sections"):
        for event in record.get(key, []):
            index = event.get("section_index", event.get("iteration_num"))
            if isinstance(index, int) and index >= 0:
                events.append((event, index, False))

    pending = None
    for event, index, is_query in sorted(events, key=lambda item: item[0].get("timestamp") or ""):
        if is_query:
            if pending is not None:
                unassigned.append(pending)
            pending = event
        elif pending is not None and index is not None:
            grouped[index].append(pending)
            pending = None
    if pending is not None:
        unassigned.append(pending)
    return grouped, unassigned


def render_call(call):
    kind = call.get("call_type", "")
    label = CALL_TYPES.get(kind, kind or "模型调用")
    status = "失败" if call.get("success") is False else "已记录"
    title = f"调用 {call.get('call_id', '—')} · {label} · {status}"
    metadata = f"{call.get('model_info', '—')} · {timestamp(call.get('timestamp'))}"
    body = f'<p class="meta muted">{safe(metadata)}</p>'
    body += details("模型原始输入", render_messages(call))
    body += details("模型原始输出", raw(call.get("output_content", "")))
    if call.get("reasoning_content"):
        body += details("日志中的推理字段", raw(call["reasoning_content"]))
    return details(title, body)


def render_messages(call):
    messages = call.get("input_messages", [])
    return "".join(
        f'<div class="label">{safe(message.get("role", "message"))}</div>{raw(message.get("content", ""))}'
        if isinstance(message, dict) else raw(message)
        for message in messages
    ) or '<p class="muted">无记录</p>'


def candidate_list(results, image):
    candidates = []
    for result in results:
        index = result.get("index", "")
        score = result.get("score")
        label = f"{index} · 检索分 {score:.3f}" if isinstance(score, (int, float)) else index
        metadata = " · ".join(
            str(result[key]) for key in ("uid", "img_path", "source") if result.get(key)
        )
        content = result.get("content")
        if image:
            content = result.get("caption") or result.get("description")
        body = f'<p class="meta muted path">{safe(metadata)}</p>{raw(content or "")}'
        if result.get("metadata"):
            body += details("元数据", raw(result["metadata"]))
        candidates.append(details(label or "候选", body))
    return "".join(candidates)


def render_search(call):
    text = call.get("text_results", [])
    images = call.get("image_results", [])
    body = (
        '<div class="columns"><div><div class="label">文本查询</div>'
        + items(call.get("text_queries") or call.get("input_queries", []))
        + '</div><div><div class="label">图片查询</div>'
        + items(call.get("image_queries", []))
        + '</div></div>'
    )
    if call.get("success") is False:
        body += '<p class="muted">这条检索记录标记为失败。</p>'
    body += details(f"文本候选 · {len(text)} 条", candidate_list(text, False))
    body += details(f"图片候选 · {len(images)} 张", candidate_list(images, True))
    return stage("检索证据", body, call.get("timestamp"))


def image_preview(source, log_dir):
    path = Path(source.get("img_path") or "")
    if not path.is_absolute():
        path = log_dir / path
    image_type = IMAGE_TYPES.get(path.suffix.lower())
    if image_type is None:
        return ""
    try:
        with path.open("rb") as stream:
            image = stream.read(8 * 1024 * 1024 + 1)
    except OSError:
        return ""
    if not image or len(image) > 8 * 1024 * 1024:
        return ""
    data = base64.b64encode(image).decode("ascii")
    return f'<img loading="lazy" alt="{safe(path.name)}" src="data:image/{image_type};base64,{data}">'


def section_title(section, index):
    match = re.search(r"^#{1,6}\s+(.+)$", section.get("content") or "", re.MULTILINE)
    return match[1] if match else f"第 {index + 1} 节"


def section_body(content, index):
    content = re.sub(r"^#{1,6}[^\S\n]+[^\n]+\n*", "", content, count=1)
    escaped = safe(content)

    def cite(match):
        identifier = match[1] or match[2]
        return f'<a class="tag" href="#s{index + 1}-refs">[{identifier}]</a>'

    escaped = re.sub(r"!\[\]\(citation:(img\d+)\)|\[citation:(txt\d+|img\d+)\]", cite, escaped)
    return '<div class="prose">' + "".join(
        f"<p>{paragraph}</p>" for paragraph in escaped.split("\n\n") if paragraph.strip()
    ) + "</div>"


def render_section(index, section, outline, calls, searches, filters, log_dir, embed_images):
    number = index + 1
    pieces = [
        f'<section class="card" id="s{number}"><p class="meta muted">第 {number} 节</p>'
        f'<h2>{safe(section_title(section, index))}</h2>'
    ]
    description = outline.get("section_description") or section.get("section_description")
    if description or outline.get("sectional_checklist"):
        pieces.append(details(
            "本节描述与检查清单",
            f'<p class="prose">{safe(description)}</p>' + items(outline.get("sectional_checklist", [])),
        ))
    query_calls = [call for call in calls if call.get("call_type") == "query_generation"]
    if query_calls:
        pieces.append(stage("生成图文查询", "".join(map(render_call, query_calls)), query_calls[-1].get("timestamp")))
    pieces.extend(map(render_search, searches))
    filter_models = [call for call in calls if call.get("call_type") in {"filter", "filtering"}]
    filter_details = []
    for call in filters:
        text = call.get("text_filtering", {})
        images = call.get("image_filtering", {})
        label = (
            f"文本 {text.get('original_count', '—')} → {text.get('filtered_count', '—')} · "
            f"图片 {images.get('original_count', '—')} → {images.get('filtered_count', '—')}"
        )
        filter_details.append(details(label, raw(call)))
    if filter_details or filter_models:
        last_filter = (filters or filter_models)[-1]
        pieces.append(stage(
            "筛选证据",
            "".join(filter_details) + "".join(map(render_call, filter_models)),
            last_filter.get("timestamp"),
        ))
    for call in calls:
        if call.get("call_type") not in {"query_generation", "section_writing", "filter", "filtering"}:
            pieces.append(stage("其他模型调用", render_call(call), call.get("timestamp")))
    writing = [call for call in calls if call.get("call_type") == "section_writing"]
    if writing:
        pieces.append(stage("撰写本节", "".join(map(render_call, writing)), writing[-1].get("timestamp")))
    if section.get("content"):
        pieces.append(details("查看生成的正文", section_body(section["content"], index)))
    else:
        pieces.append('<p class="muted">日志中没有本节的正文完成记录。</p>')
    texts = section.get("text_sources", [])
    images = section.get("image_sources", [])
    refs = []
    for source in texts + images:
        label = source.get("citation_id", "引用")
        body = f'<p class="meta muted path">{safe(source.get("uid"))}</p>'
        body += f'<p class="path">{safe(source.get("img_path") or source.get("source"))}</p>'
        if embed_images and source.get("img_path"):
            body += image_preview(source, log_dir)
        description = source.get("content_preview") or source.get("description")
        if description:
            body += raw(description)
        refs.append(details(label, body, bool(embed_images and source.get("img_path"))))
    pieces.append(
        f'<div id="s{number}-refs">' + details(
            f"正文引用 · 文本 {len(texts)} 条 · 图片 {len(images)} 张",
            "".join(refs) or '<p class="muted">无已解析引用</p>',
            bool(images and embed_images),
        ) + "</div></section>"
    )
    return "".join(pieces)


def by_section(records):
    indexed = defaultdict(list)
    for item in records:
        index = item.get("section_index", item.get("iteration_num"))
        if isinstance(index, int) and index >= 0:
            indexed[index].append(item)
    return indexed


def render_html(record, log_dir, embed_images):
    original = record.get("user_input", {})
    outline = record.get("outline_generation", {})
    final = record.get("final_report", {})
    metrics = record.get("performance_metrics", {})
    sections = {item["section_index"]: item for item in record.get("sections", [])}
    plan = {item["section_index"]: item for item in outline.get("sections_summary", [])}
    calls, unassigned = group_calls(record)
    searches = by_section(record.get("search_calls", []))
    filters = by_section(record.get("filter_calls", []))
    indices = sorted(set(sections) | set(plan) | set(calls) | set(searches) | set(filters))
    mode = original.get("generation_mode") or outline.get("generation_mode")
    plan_label = "输入大纲" if mode == "without_planner" else "规划节点生成的大纲"
    route = plan_label + " → 逐节生成查询 → 检索"
    if original.get("enable_filter"):
        route += " → 筛选"
    route += " → 写作 → 合并"
    duration = metrics.get("total_processing_time")
    duration_label = f"{duration:.2f} 秒" if isinstance(duration, (int, float)) else "—"
    stats = (
        ("完成章节", len(sections)),
        ("模型调用", len(record.get("llm_calls", []))),
        ("图文检索记录", len(record.get("search_calls", []))),
        ("会话耗时", duration_label),
    )
    stat_cards = "".join(
        f'<div class="stat"><b>{safe(value)}</b><span>{safe(label)}</span></div>' for label, value in stats
    )
    navigation = ['<a href="#input">原始输入</a>']
    navigation += [f'<a href="#s{i+1}">第 {i+1} 节</a>' for i in indices]
    navigation.append('<a href="#final">合并结果</a>')
    errors = record.get("error_log", [])
    if errors:
        navigation.append('<a href="#errors">异常记录</a>')
    heading = str(record.get("user_id") or record["session_id"]).removeprefix("deconstruct_")
    title = f"{heading} · 报告生成轨迹"
    page = [
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; '
        'style-src \'unsafe-inline\'">'
        f"<title>{safe(title)}</title><style>{STYLES}</style></head><body>",
        '<div class="layout"><aside><strong>生成轨迹</strong><nav aria-label="章节导航">'
        + "".join(navigation) + '</nav></aside><main>',
        f'<header><h1>{safe(title)}</h1><p class="meta muted">{safe(record["session_id"])}</p>'
        f'<p class="meta muted">{safe(timestamp(record.get("start_time")))}'
        f' 至 {safe(timestamp(record.get("end_time")))}</p>'
        f'<div class="stats">{stat_cards}</div><p class="flow">{safe(route)}</p></header>',
        '<section class="card" id="input"><h2>原始输入</h2>',
        f'<p class="query">{safe(original.get("overall_query") or outline.get("overall_query") or "日志中无原始问题")}</p>',
        '<h3>全篇检查清单</h3>' + items(outline.get("overall_checklist") or original.get("overall_checklist")),
    ]
    all_calls = record.get("llm_calls", [])
    if all_calls:
        first = all_calls[0]
        first_stage = CALL_TYPES.get(first.get("call_type"), "模型调用")
        page.append(details(f"第一次模型调用的完整输入 · {first_stage}", render_messages(first)))
    if plan:
        outline_entries = [
            details(f"第 {i+1} 节", f'<p class="prose">{safe(item.get("section_description"))}</p>'
                    + items(item.get("sectional_checklist", [])))
            for i, item in sorted(plan.items())
        ]
        page.append(details(f"{plan_label} · {len(plan)} 节", "".join(outline_entries), True))
    filter_enabled = original.get("enable_filter")
    filter_label = {True: "开启", False: "关闭"}.get(filter_enabled, "—")
    page.append(
        f'<p class="meta muted">模式 {safe(mode or "—")} · Filter {filter_label} · '
        f'文本 Top-K {safe(original.get("text_topk", "—"))} · 图片 Top-K {safe(original.get("image_topk", "—"))}</p>'
        + details("原始 user_input 字段", raw(original))
    )
    if unassigned:
        page.append(details("未归到具体章节的模型调用", "".join(map(render_call, unassigned))))
    page.append("</section>")
    page.extend(
        render_section(i, sections.get(i, {}), plan.get(i, {}), calls.get(i, []),
                       searches.get(i, []), filters.get(i, []), log_dir, embed_images)
        for i in indices
    )
    page.append('<section class="card" id="final"><h2>合并结果</h2>')
    if final.get("content"):
        page.append(
            f'<p class="muted">{safe(timestamp(final.get("timestamp")))} · '
            f'文本引用 {safe(final.get("total_text_citations", "—"))} · '
            f'图片引用 {safe(final.get("total_image_citations", "—"))}</p>'
            + details("查看最终 Markdown", raw(final["content"]))
        )
    else:
        page.append('<p class="muted">日志中没有合并后的报告。</p>')
    page.append("</section>")
    if errors:
        page.append(f'<section class="card warning" id="errors"><h2>异常记录</h2>{raw(errors)}</section>')
    page.append("</main></div></body></html>")
    return "".join(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="Deep-Reporter session log.")
    parser.add_argument("--session-id", help="Select a session. Defaults to the adjacent session-id.txt or the last record.")
    parser.add_argument("-o", "--output", type=Path, help="Output HTML. Defaults to the JSONL path with .html.")
    parser.add_argument("--embed-images", action="store_true", help="Embed readable local images cited in the report.")
    args = parser.parse_args()
    source = args.jsonl.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else source.with_suffix(".html")
    if output == source or output.suffix.lower() != ".html":
        parser.error("Choose a separate output path ending in .html.")
    try:
        record = load_session(source, args.session_id)
        page = render_html(record, source.parent, args.embed_images)
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".trace-", dir=output.parent) as directory:
            staged = Path(directory) / "trace.html"
            staged.write_text(page, encoding="utf-8")
            staged.replace(output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(f"Exported {output} · {record['session_id']}")


if __name__ == "__main__":
    main()
