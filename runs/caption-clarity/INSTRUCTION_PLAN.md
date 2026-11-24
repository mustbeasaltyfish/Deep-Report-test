# 配图判断指令对照 · AG001 第 1 节

## 问题和条件

上一轮只改图注，A 和 B 各 6 次均未用图。本轮检验通用写作指令是否改变这一现象，并观察原图注和修订图注在该指令下的差异。

沿用上一轮保存的输入与设置、Qwen3-32B、20 条文本、10 条图片候选，以及种子 1701 至 1706。A/B 使用上一轮冻结结果。C 从 A 的完整提示词复制，D 从 B 复制。在提示词的 Content Requirements 前添加同一段指令。两组的输出格式、图文候选和顺序保持一致。每个种子内交替排列 C/D 的运行顺序。

> Before drafting, check whether any candidate image directly helps explain, compare, or substantiate a point required by this section. If an image helps, place it next to the relevant passage and explain the specific detail the reader should notice. If none helps, write the section without an image.

## 指标

- 主指标是正确使用目标图。必须以正确 Markdown 引用 img1 或 img2。两者是同一张图。插图应紧邻样本级与用户级差分隐私的对照。与图相连的正文须准确指出单条样本和同一位用户全部样本的移除单位，且不可把两次训练输出说成完全相同。
- 次指标包括是否有合法目标图引用、其他图片引用、只有说明而没有插图、重复引用、模型正常结束、输入输出 token 和耗时。审查使用模型保存的原始正文。
- C 对 A 估计原图注下加入配图判断指令的影响。D 对 C 估计在明确要求配图判断时修订图注的增量。D 对 B 作为附加对照。
- 记录六组配对，逐个写明成功或失败的理由。主指标中的净增至少 3 对且反向翻转最多 1 对，作为本轮判别清晰提升的探索性门槛。相差较小的结果保留逐对数据。一个章节中的随机种子用于检查一致性，不代表领域样本。
- 本轮在 AutoDL 的 caption-clarity tmux 会话中执行。完整输入、输出和逐对记录位于 instruction 目录。
