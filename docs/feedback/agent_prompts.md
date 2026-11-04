# 给写作 agent 与审核 agent 的提示词（2026-11-04）

## 一、写作 agent

你负责 `report/latex/` 下的 CS5228 IEEE 报告（按 `docs/report/FRAMEWORK.md` 写作）。本轮有两类新变化，请据此修改正文。

**1. 图表已统一重做（不要再改图的生成方式）**
- 图 1、2、4 由 `report/latex/photo/make_figures.py` 生成矢量 PDF，统一使用 Times 字体，标签 8pt、刻度 7pt，按单栏宽 3.5 in 直接绘制。tex 中用 `\includegraphics[width=\columnwidth]{figX.pdf}`。
- 架构图（图 3）是 `report/latex/photo/model_architecture.tex`（TikZ，宽 17.7 cm），在 chapter05 中不缩放插入：`\includegraphics{model_architecture.pdf}`。
- 图 4 现在是上下两个面板：上面板为两条 RMSE 曲线（原始租金和清洗后租金），下面板为月平均误差柱状图。正文描述如有"左右两栏"之类的说法，需要对应调整。
- 图 5 仍是旧 PNG，因为 BERT 的逐行预测 `results/validation_predictions.csv` 不在仓库里。不要编写依赖这份文件的新分析。
- 新增的图（例如模型对比图、消融图）一律加进 `make_figures.py`，以保持风格一致。

**2. 导师反馈与新的 baseline 结果（main 提交 `3573e8f add:baselines`）**
先读 `docs/feedback/progress_report_feedback.md`（导师原文加逐条核对），以及 `results/baselines/round_a/README.md`、`scores.csv`、`ablation.csv`、`by_month.csv`、`by_building_seen.csv`。然后完成：
- 改写过时表述：chapter05 中 "no tree run is present under the same time split"，以及 chapter06 中 "No independent baseline run is present ... no tree or no-pretraining comparison was recorded"。改为引用新的 baseline。无预训练（no-pretraining）对比仍然不存在，要保留这一说明。
- 在第 VI 节新增模型对比表，使用旧口径（historical），35,058 行，列 RMSE/MAE/bias：全局均值 842.54/686.73/−534.45；近 6 个月 town×房型均值 497.97/374.20/−51.09；Ridge 606.82/459.95/+371.92；浅层 CatBoost 468.11/348.07/−37.46；两阶段 BERT 468.34/344.83/−6.73。如实写出：CatBoost 与 BERT 的总体 RMSE 只差 0.23，BERT 的 MAE 和偏差更小；这只是单一时间窗口的描述性比较。
- 新增特征组消融表，使用严格口径（strict），35,158 行，列 ΔRMSE（Ridge / CatBoost）：去掉街道和楼栋 ID +8.52 / +2.95；去掉 10 个设施字段 −0.08 / +0.84；去掉时间组（5 个字段）+85.64 / +232.26。注明 BERT 没有做消融，时间组的差值不能归因于单个字段。
- 两种口径不能混在同一张表里比较：严格口径没有 BERT 结果。正文中要分别说明两种口径的定义。
- 回应"说明每个核心特征的去向"：表 II 或第 IV 节要列出 8 个类别 ID（TOWN、STREET、BUILDING、SUBZONE、PLANNING_AREA、REGION、FLAT_MODEL、FLAT_TYPE_ORD）。写明 STREET 的处理：标准化后参与 (TOWN, BLOCK, STREET) join，保留为 STREET_ID，同时与 TOWN、BLOCK 合成 BUILDING_ID。原始字符串字段被 ID 替代；FURNISHED 和 FEE 因为是常量被丢弃。依据 `src/preprocess/stage3_encode.py` 和 `src/bert_rent/data.py`。
- 第一章考虑写出 2–3 个明确的研究问题，例如：复杂模型是否优于简单基线？哪些特征组最重要？误差集中在哪里？在第 VI、VII 节逐一回答。
- 误差分析补充 RMSE 之外的发现：Ridge 的月均高估从 218 扩大到 532 SGD（线性时间外推过强）；CatBoost（严格口径）在见过的楼栋上 RMSE 为 475.50，在未见楼栋上为 548.26（n=151，样本少）。
- 所有新增数字都要在注释中写上 Source 路径，与现有写法一致。不要改动已核实的旧数字。完成后用 latexmk 编译确认无报错。

## 二、审核 agent

你负责审核 `report/latex/` 下的报告内容。本轮新增了以下变化，请把它们纳入审核清单。

**1. 图表变更（核对与正文一致）**
- 图 1、2、4 已由 `report/latex/photo/make_figures.py` 重绘为矢量 PDF，数据直接读取 `experiments/eda/core/*.csv` 和 `results/validation_summary.json`。架构图由 `photo/model_architecture.tex`（TikZ）生成。
- 请核对：图中数值与正文、caption 一致（例如图 4 中 2024-04 的 RMSE 503.3、2024-06 的平均误差 +32.9、2025-03 的平均误差 −52.8）；图 4 已改为上下两面板，caption 和正文描述要与之匹配；架构图中 Stage 2 的 "MLP 768→256→1"、"2 calendar + 1 flag"、"29 field tokens + [CLS]" 与第 V 节一致；各图字号统一、无文字重叠。
- 图 5 仍是旧 PNG，原因是缺少 BERT 逐行预测。可以记为已知限制，不算错误。

**2. 新内容引入（导师反馈和 baseline）**
依据：`docs/feedback/progress_report_feedback.md`，以及 `results/baselines/round_a/`（README、scores.csv、ablation.csv、by_month.csv、by_building_seen.csv、manifest.json）。重点检查：
- 过时说法是否已删除：chapter05 和 chapter06 中"没有基线或树模型对比"的句子。"无预训练对比"仍然不存在，这一点不能被写成已完成。
- 对比表的数字是否与 `scores.csv` 逐项一致；旧口径（35,058 行）和严格口径（35,158 行）是否被混用。严格口径没有 BERT 结果，任何把 BERT 与严格口径数字直接比较的表述都要标为错误。
- 对 CatBoost 与 BERT 的结论是否过度：RMSE 只差 0.23，只能写作单一验证窗口内的描述性比较，不能说某个模型"显著更好"。
- 消融的解读：时间组删除的是 5 个字段，不能归因于单个字段；BERT 没有做消融；设施特征影响小于 1 SGD 的结论只适用于 Ridge 和 CatBoost。
- 导师要求逐项回应：是否写明每个核心特征的去向（尤其是 STREET → STREET_ID / BUILDING_ID，以及 FURNISHED、FEE 被丢弃）；8 个类别 ID 是否列出；是否有明确的研究问题并在后文回答；是否有 RMSE 之外的分析（按月偏差、见过与未见楼栋）。
- 所有新增数字是否带有 Source 注释、是否可以追溯到仓库文件。按 CRITICAL / MAJOR / MINOR 分级报告问题。
