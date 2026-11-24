"""Report paired results per task and across the nine-domain sample."""

import csv
import json
from statistics import mean

from baseline import RUN


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    tasks = read(RUN / "sample.json")
    measurements = read(RUN / "metrics.json")
    reviews = read(RUN / "judgments.json")
    lookup = {(row["uid"], row["arm"]): row for row in measurements}
    judgments = {(row["uid"], row["arm"]): row for row in reviews}
    assert len(judgments) == len(reviews)
    selected = {(row["uid"], row["arm"]) for row in measurements if row["target"]}
    assert set(judgments) == selected, (selected - judgments.keys(), judgments.keys() - selected)
    for row in measurements:
        review = judgments.get((row["uid"], row["arm"]), {})
        row["correct"] = review.get("correct", False)
        row["review"] = review.get("reason", "未引用目标图")
        assert row["finish"] in {"stop", "length"}, (row["uid"], row["arm"], row["finish"])
    uids = [task["uid"] for task in tasks if all((task["uid"], arm) in lookup for arm in "ABCD")]
    assert len(uids) == len(tasks) == 9
    assert len(lookup) == 36

    cols = [
        "domain", "uid", "arm", "section_index", "target", "any", "correct", "review", "finish", "section_text",
        "section_image", "report_text", "report_image", "report_overall", "report_image_count",
        "report_image_sections", "report_sections", "report_image_coverage",
    ]
    with (RUN / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(measurements)

    n = len(uids)
    lines = [
        "# 九领域配图对照实验结果",
        "",
        f"固定抽样九篇，全部完成四组。共 {len(measurements)} 份合成报告，每组使用该任务相同的其他章节。",
        "",
        "| 组别 | 目标节用了任意图 | 选中目标图 | 准确解释目标图 | 整篇 Selection Overall | 平均整篇图引用 | 平均有图章节比例 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in "ABCD":
        rows = [lookup[uid, arm] for uid in uids]
        values = (
            f"{sum(row['any'] for row in rows)}/{n}",
            f"{sum(row['target'] for row in rows)}/{n}",
            f"{sum(row['correct'] for row in rows)}/{n}",
            f"{mean(row['report_overall'] for row in rows):.2%}",
            f"{mean(row['report_image_count'] for row in rows):.2f}",
            f"{mean(row['report_image_coverage'] for row in rows):.2%}",
        )
        lines.append("| " + " | ".join((arm, *values)) + " |")

    lines.extend([
        "",
        "每篇下表单元格依次显示目标图状态与该篇 Selection Overall。无图表示目标节没有插图，其它图表示使用了其它候选，选中表示引用目标图但未讲出核对的细节，准确表示两项都成立。",
        "",
        "| 领域 | 任务 | 目标节 | A | B | C | D |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ])
    for task in tasks:
        uid = task["uid"]
        arms = [lookup[uid, arm] for arm in "ABCD"]
        labels = []
        for row in arms:
            status = "准确" if row["correct"] else "选中" if row["target"] else "其它图" if row["any"] else "无图"
            labels.append(f"{status} · {row['report_overall']:.2%}")
        lines.append("| " + " | ".join((task["domain"], uid, str(arms[0]["section_index"] + 1), *labels)) + " |")
    lines.extend([
        "",
        "## 配对差值",
        "",
        "正向、反向与持平按同一任务两组之差统计。",
        "",
        "| 对照 | 任意图 正／反／平 | 目标图 正／反／平 | 准确解释 正／反／平 | 整篇 Selection Overall 变化 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for left, right in (("A", "C"), ("B", "D"), ("A", "B"), ("C", "D")):
        deltas = {}
        for field in ("any", "target", "correct"):
            counts = [int(lookup[uid, right][field]) - int(lookup[uid, left][field]) for uid in uids]
            deltas[field] = f"{counts.count(1)}/{counts.count(-1)}/{counts.count(0)}"
        overall = mean(lookup[uid, right]["report_overall"] - lookup[uid, left]["report_overall"] for uid in uids)
        lines.append(f"| {right} − {left} | {deltas['any']} | {deltas['target']} | {deltas['correct']} | {overall*100:+.2f} 个百分点 |")

    lines.extend(["", "## 人工核对", ""])
    for item in reviews:
        mark = "准确" if item["correct"] else "未达准确解释标准"
        lines.append(f"- {item['uid']} {item['arm']} · {mark}。{item['reason']}")
    lines.extend([
        "",
        "## 替代图案例",
        "",
        "- 科学 SS035 的 A 引用了目标总览图 img1。B 换成了太阳内部结构图 img2。C/D 都采用了更聚焦的质子—质子链图 img8，两组正文也都解释了图中两个氦-3 结合并返回两个质子的步骤。D 把图直接放在解释段后。严格的目标图指标仍只计 img1 及其等价候选。",
        "- 地理 GM126 的 A/C 都选了展示五个争议地点的总览图 img7，B/D 的目标节无图。本次修订的是单独讲波弗特海的图，区域总览图是独立候选。",
    ])
    lines.extend([
        "",
        "## 口径",
        "",
        "- A 是原图注与原指令。B 只改图注。C 只加配图指令。D 同时改两者。",
        "- 整篇 Selection Overall 是文本与图片的银标精确率等权平均，沿用仓库的分节去重和平均规则。图片引用数按每节的唯一图片 ID 相加。",
        "- 每个领域只抽一篇，每篇只改一个目标章节，其他章节沿用共同基线。分数反映此次抽样的局部干预结果。",
        f"- 结束原因达到长度上限的输出有 {sum(row['finish'] == 'length' for row in measurements)} 份。",
        "- 论文 Table 3 的评审模型质量分不在本次计算内。",
    ])
    (RUN / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Published RESULTS.md and results.csv")


if __name__ == "__main__":
    main()
