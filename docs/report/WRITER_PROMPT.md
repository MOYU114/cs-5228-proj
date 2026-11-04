# 写作 agent 提示词

复制下面分隔线之间的全部内容，作为写作 agent 的提示词。每次只写一章，把 `<章节>` 换成要写的章，例如 "II. Task and Data（chapter02.tex）"。

---

你在本地项目 `/Users/moyu114/Documents/university document/NUS sem3/CS5228/Project` 中撰写 CS5228 课程项目的最终报告（预测新加坡 HDB 月租金，两阶段表格 BERT）。本次任务：撰写 **<章节>**。

## 必须使用的 skill

- 写正文前先调用 **`anthropic-skills:paper-writer`** skill，并按它的流程写作：每个事实性主张都要能追溯到下面列出的项目材料，引用必须经过核实，交付的正文不留占位标签。
- 正文完成后调用 **`anthropic-skills:paper-polish`** 做一遍润色。只修语法、流畅度和语气，不改变含义或数字，并去掉 AI 腔。
- 需要设计或绘制图（尤其是第 V 章的模型架构图）时，调用 **`anthropic-skills:figure-designer`**。

## 开始前必读（按顺序）

1. `docs/report/FRAMEWORK.md`：写作蓝本，是唯一的结构依据。严格按照其中该章的目的、要点、证据来源、图表和"禁止"条目来写。
2. `docs/report/REVIEW_CHECKLIST.md`：审核标准。交付前逐条自查。
3. `docs/Project Desc.md`：课程要求。
4. `report/PREPARATION.md`：证据索引和评估口径边界（第 4、5 节一定要读）。
5. 该章在 FRAMEWORK.md 中列出的数据来源，包括 `results/*.json`、`experiments/eda/**`、`configs/bert.json`、`src/**`。实现细节以代码为准，README 只能作为辅助。

## 写作规则

- 只编辑 `report/latex/` 下的文件。正文写在对应的 `chapter0X.tex` 中，图片复制到 `report/latex/photo/`，参考文献写在 `references.tex`（`thebibliography`，IEEE 格式）。不要修改 `src/`、`results/`、`docs/`、`proposal/` 或原始数据。
- 正文用英文，采用学术论文语气，IEEEtran 双栏格式。篇幅不超过 FRAMEWORK.md 中给该章的页数。
- **不编造。** 每个数字都要来自项目文件，并在 tex 中用 `% Source: <文件>` 注释标明出处。没运行过的实验（树模型、消融、Kaggle 分数）不能写成结果。缺少的证据用 `% TODO(evidence): ...` 注释标出，并在交付说明里列出来。
- 口径不能混用：原始标签和平均后标签、select 和 refit、35,158 行和 35,058 行、17,900 组和 17,777 组。
- 不做因果断言，不声称提出了新架构，也不声称预训练或辅助特征带来了增益（除非有对照结果）。
- 引用只用真实存在且已核实的原始论文。不确定的引用不要加，在交付说明中列出待核实项。
- 每章第一句要承接上一章；EDA 的每条发现都要以它引出的预处理或评估决定结尾。
- 写完后在 `report/latex/` 下运行 `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`，确认没有编译错误、没有未定义引用、没有明显越界。

## 交付说明（完成后回复以下内容）

1. 修改了哪些文件。
2. 该章的估计页数，以及编译是否通过。
3. 用到的全部数字及其来源文件（列成表）。
4. 新增的引用和核实方式；仍待核实的引用。
5. 剩余的 TODO(evidence) 缺口。
6. 对照 REVIEW_CHECKLIST.md 自查发现、但尚未解决的问题。

完成后停下，不要接着写下一章；我会把这一章交给审核者。

---

## 推荐的分章顺序

按 FRAMEWORK.md 第 4 节：II → III → IV → V → VI → VII → I → VIII → Abstract，最后补作者栏和附录 A。第 VI、VII 章的数字取决于团队是否补跑基线和消融（FRAMEWORK.md 第 5 节），可以先写结构，数字最后再填。
