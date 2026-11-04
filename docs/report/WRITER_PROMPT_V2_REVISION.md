# 写作 agent 修订提示词（v2：导师反馈与新增基线）

把分隔线之间的内容发给写作 agent。这是对已有草稿的**修订**任务，不是从头重写。

---

你继续负责 CS5228 最终报告（本地项目 `/Users/moyu114/Documents/university document/NUS sem3/CS5228/Project`，正文在 `report/latex/`）。我们收到了导师对进度报告的反馈，也新增了基线实验，现在需要按新的蓝本修订已有章节。

## 必须使用的 skill

- 改写或新增段落时，使用 **`anthropic-skills:paper-writer`**：每个主张都要能追溯到项目文件，引用必须核实。
- 修订完成后，使用 **`anthropic-skills:paper-polish`** 润色，不能改变含义或数字。
- 新增或修改图时，使用 **`anthropic-skills:figure-designer`**，并且把图加进 `report/latex/photo/make_figures.py`，输出矢量 PDF。

## 先读

1. `docs/feedback/progress_report_feedback.md`：导师反馈原文和逐条核对。
2. `docs/report/FRAMEWORK.md`：重点读开头的 "v2 调整"，以及第 I、IV、V、VI、VII 章中标有 v2 的条目，其中有两种口径的规定。
3. `docs/report/REVIEW_CHECKLIST.md`：重点读 A2 节。
4. `results/baselines/round_a/README.md`、`scores.csv`、`ablation.csv`、`by_month.csv`、`by_building_seen.csv`。

## 本次要做的修订

1. **chapter01**：加入 RQ1–RQ3 编号列表（模型复杂度与精度；特征组重要性；误差集中在哪里及原因），贡献列表与 RQ 一一对应。
2. **chapter04**：扩充 Table II，列出 8 个类别 ID 的名称，并交代每个核心字段的去向，尤其是 STREET、BLOCK、日期、FURNISHED、FEE。按 FRAMEWORK.md 中的表格写，但每一行都要对照 `src/preprocess/stage3_encode.py` 和 `src/bert_rent/data.py` 核实。
3. **chapter05**：删除 "no tree run is present under the same time split…"。改为介绍四个基线及其设定，然后说明 BERT 的设计动机（写成假设，不写成已证明的增益）。
4. **chapter06**：
   - 删除 "No independent baseline run is present…"。
   - 新增 Table III：五个模型对比，只用历史口径（n=35,058），列为 RMSE、MAE、bias。
   - 新增 Table IV：Ridge 和 CatBoost 的特征组消融，只用严格口径（n=35,158）。
   - 在误差分析中加入：逐月比较（浅层树在前 5 个月更好，BERT 在后 7 个月更好；Ridge 高估从 218 扩大到 532 SGD）；已见与未见楼栋（CatBoost 严格口径 475.50 对 548.26，n=151，注明样本小）。
   - 保留不可约误差 168 SGD 的分析。
5. **chapter07**：按 RQ 逐条回答；讨论"浅层 CatBoost 与 BERT 的 RMSE 几乎相同"说明了什么；局限中写明 BERT 没有严格口径结果、没有消融、没有去预训练对照、没有逐行预测。
6. **页数**：新增两张表后，如果正文超过 8 页，按 Fig. 2 → Fig. 5 → Table I 的顺序移到附录。

## 硬规则

- 两种口径不能出现在同一张表或同一个比较句里。BERT 只有历史口径的结果。
- 数字保留两位小数，并与 CSV 一致；每个新数字都要在 tex 中用 `% Source:` 注释标明出处。
- 不写显著性，不做因果断言，不说"BERT 优于 CatBoost"。
- 只编辑 `report/latex/`。

## 完成后回复

修改的文件；编译结果和正文页数；新增数字及其来源表；按 REVIEW_CHECKLIST A2 节逐条自查的结果；还没解决的问题。完成后停下，等待审核。

---
