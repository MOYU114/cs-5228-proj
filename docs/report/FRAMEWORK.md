# 最终报告写作蓝本（Report Framework）

更新：2026-11-04（v2）。本文件是写作 agent 的唯一结构依据，正文写在 `report/latex/`，本目录只做设计与审核。审核标准见 [REVIEW_CHECKLIST.md](REVIEW_CHECKLIST.md)。

## v2 调整（依据导师反馈和新增基线实验）

来源：[导师反馈原文与逐条核对](../feedback/progress_report_feedback.md)；提交 `3573e8f add:baselines` 的 `results/baselines/round_a/`（`scores.csv`、`ablation.csv`、`by_month.csv`、`by_building_seen.csv`、`README.md`、`monthly_bias.png`）。

| # | 调整 | 涉及章节 |
|---|---|---|
| 1 | 用 3 个明确的研究问题（RQ1–RQ3）组织全文，并在 VI/VII 逐一回答 | I、VI、VII、Conclusion |
| 2 | 交代每个核心字段的去向（例如 STREET 被保留为 STREET_ID，也用于构成 BUILDING_ID），默认读者不了解数据集 | IV（Table II） |
| 3 | 新增简单模型对比表：全局均值、近期分组均值、Ridge、浅层 CatBoost、BERT | V、VI（Table III） |
| 4 | 新增特征组消融表（Ridge 和 CatBoost，严格口径） | VI（Table IV） |
| 5 | 补充 RMSE 之外的发现：逐月偏差、已见与未见楼栋、Ridge 外推过强、浅层树与 BERT 几乎持平 | VI、VII |
| 6 | 删除过时表述："no tree run is present"（chapter05）和 "No independent baseline run is present"（chapter06） | V、VI |
| 7 | 图改为矢量 PDF，统一由 `report/latex/photo/make_figures.py` 和 `model_architecture.tex` 生成；新图也要加到这个脚本里 | 全部图 |

**两种评估口径不能混用（新增的硬规则）**
- *历史口径*：沿用 BERT 的先清洗、后切分流程，验证集保留 35,058 行。五个模型都有结果，**Table III 只用这个口径**。
- *严格口径*：先切分，只在训练期做清洗和拟合，验证集保留全部 35,158 行。只有四个基线有结果，BERT 没有重训。**Table IV 消融只用这个口径**。如果正文要报严格口径的基线分数，必须单独成列或单独成段，并写明 BERT 在这个口径下没有结果。
- BERT 没有做消融，也没有"去预训练"对照，没有逐行预测，所以不能按楼栋拆分 BERT 的误差。

## 0. 总体约束

| 项目 | 规定 |
|---|---|
| 入口 | `report/latex/main.tex`（IEEEtran conference，双栏），各章 `chapter0X.tex` 独立编辑 |
| 语言 | 正文英文；本目录中文 |
| 篇幅 | 正文 ≤ 8 页（含图表，不含参考文献和附录）。目标写到 7.5 页，留 0.5 页余量 |
| 必须覆盖（课程） | Motivation；EDA & Preprocessing（EDA 如何决定预处理）；Data Mining Methods（为何选、超参如何定、备选方案）；Evaluation & Interpretation（比较、何时差、局限、改进） |
| 评分重心 | 方法质量 60%（EDA→预处理有因果链、方法有理由、评估设置正确、讨论局限）、报告质量 30%、可复现 10% |
| 作者与附录 | 队名、姓名、学号；附录放分工表。未确认前保持占位，不得编造 |

### 0.1 证据规则（写作 agent 必须遵守）

1. **每个数字都要有出处。** 只能来自 `results/*.json`、`experiments/eda/**`、`configs/bert.json`、`src/` 代码逻辑。README 中没有原始文件支撑的数（如均值基线 842.5、训练时长）要么注明来源是记录说明，要么重新跑出后再用。
2. **没跑过的实验不写成结果。** 基线和基线消融已有结果（v2）。BERT 的严格口径重训、去预训练对照和 BERT 消融还没有运行，只能写在 Limitations / Future Work 里，并用注释标出缺口。BERT 的训练时长（约 4.75 小时）只出现在 README 中，引用时要注明是记录值。
3. **口径要写清。** 原始标签和平均后标签、选择轮（select）和全量重训（refit）、清洗前 35,158 行和保留的 35,058 行，都要写清楚，不能混用。
4. **不做因果断言。** EDA 的相关性和误差分组不能写成"某因素导致租金上涨"。
5. **引用必须核实。** 只引用确实读过、出处可查的原始论文（BERT、FT-Transformer/TabTransformer 类、实际评估过的基线）。

## 1. 章节结构与篇幅

保留现有 8 个章节文件，以减少改动；章节职责重新划分如下：

| 文件 | 标题 | 页数 | 对应课程要求 |
|---|---|---:|---|
| `abstract.tex` | Abstract | 0.25 | — |
| `chapter01.tex` | I. Introduction | 0.6 | Motivation |
| `chapter02.tex` | II. Task and Data | 0.5 | — |
| `chapter03.tex` | III. Exploratory Data Analysis | 1.0 | EDA |
| `chapter04.tex` | IV. Preprocessing and Feature Engineering | 1.0 | Preprocessing |
| `chapter05.tex` | V. Method: Two-Stage Tabular BERT | 1.4 | Data Mining Methods |
| `chapter06.tex` | VI. Experimental Evaluation | 1.6 | Evaluation |
| `chapter07.tex` | VII. Discussion and Limitations | 0.8 | Interpretation / Limitations |
| `chapter08.tex` | VIII. Conclusion | 0.25 | — |
| 标题、作者、整合余量 | — | 0.35 + 0.25 | — |
| `references.tex` | References | 不计 | — |
| `contribution_statement.tex` | Appendix A. Team and Contributions | 不计 | 分工 |
| （可选）附录 B | Supplementary EDA / per-month table | 不计 | 正文离开附录也要能读懂 |

全文主线（每章开头一句要能接上）：

> 租户和房东需要对相近房源做租金比较 → 数据是粗粒度房屋属性加地理信息，而且测试集在未来 → EDA 发现拼写不一、相同输入对应不同租金、时间漂移和少量异常值 → 预处理逐条回应这些发现，并加入楼栋和设施特征 → 将每条租赁记录视为字段序列，用遮蔽字段预训练加回归微调的表格 BERT 学习字段组合 → 用时间外验证评估，误差按月份和价格区间分析 → 说明哪些结论有证据、哪些还缺对照，以及不可约误差的下界。

## 2. 分章设计

每章列出：目的 / 核心信息 / 内容要点 / 证据来源 / 图表 / 禁止写法。

### Abstract（最后写）
- 4–5 句：问题与意义 → 方法（预处理 + 两阶段表格 BERT）→ 主要结果（注明是哪个验证口径）→ 主要局限。
- 数字必须与第 VI 章一致。

### I. Introduction
- **目的**：说明问题为什么重要、难在哪里、本文做了什么。
- **要点**
  1. 背景：新加坡租金上涨；租户、房东/中介、政策方三类受益者（课程说明里有）。
  2. 挑战（3 条，后文逐一回应）：(a) 测试集在训练期之后，是时间外推；(b) 只有粗粒度属性，相同特征对应不同租金，存在不可约误差；(c) 原始字段格式不一，辅助数据需要连接和对齐时间。
  3. **研究问题（v2 新增，用编号列表写出）**：
     - RQ1：在时间外验证下，从全局均值、近期分组均值、线性模型、浅层树到两阶段 BERT，模型复杂度能换来多少精度提升？
     - RQ2：哪些特征组对预测最重要？（位置 ID、设施、时间）
     - RQ3：误差集中在哪里，原因是什么？（逐月偏差、未见楼栋、价格两端、相同输入对应不同租金带来的不可约误差）
  4. 方法概述一句话加贡献列表（3 点），每一点对应一个 RQ：可追溯的清洗与特征流程；从简单到复杂的五个模型在统一口径下的对比，加上特征组消融；误差分析，并量化不可约误差。
- **禁止**：声称提出了新架构；把"Kaggle 排名"当动机；宣称预训练已被证明有效（除非第 VI 章有消融结果）。

### II. Task and Data
- **目的**：给出形式化任务和数据范围，压缩篇幅，细节放第 III 章。
- **要点**：输入字段和目标 `MONTHLY_RENT`；RMSE（SGD/月，每行等权）；train 150,000 行（2021-01–2025-03），test 50,000 行（2025-03–2026-07）；用了哪些辅助表（楼栋、MRT、学校、商场），没用哪些（COE、股价），并给一句理由。
- **表**：Table I 数据来源概览（文件 / 行数 / 时间 / 用途），控制在 6 行以内。
- **来源**：`experiments/eda/core/summary.json`、`auxiliary/summary.json`、`docs/Project Desc.md`。
- **注意**：本地数据截止 2026-07，与课程描述的 2026-06 不一致，可在脚注说明。

### III. Exploratory Data Analysis
- **目的**：EDA 不是罗列统计量，每条发现都以"→ 因此在第 IV/VI 章做了 X"结尾。
- **结构**（建议 4 条发现，每条一段）：

| 发现 | 证据 | 引出的决定 |
|---|---|---|
| F1 时间漂移：月均租金上升后趋平；测试集 60% 行距训练截止 7–16 个月 | `rent_by_month.csv`、`docs/research/PRIORITY_PLAN.md` §2 | 时间外验证（VI）；加入月份特征；局限中讨论外推（VII） |
| F2 户型/面积/位置决定主要差异 | `rent_by_flat_type.csv`、`rent_by_town.csv`、`rent_by_area_band.csv` | 保留结构字段；加入楼栋坐标和设施特征（IV） |
| F3 数据质量：拼写变体（STREET 1,146→576）、常量列（FURNISHED、FEE）、楼栋缺失值、少量极端租金 | `category_normalization_map.csv`、`anomaly_flags.csv`、`rent_extreme_examples.csv` | 规范化、删除常量列、插补、异常过滤（IV） |
| F4 相同输入对应不同租金（原始 17,900 组；清洗后 17,777 组，不能混用） | `identical_X_multiple_targets.csv` | 组内平均标签（IV）；不可约误差下界 168 SGD（VI） |

- **图**：Fig. 1 月均租金和每月样本数（`slides_eda_rent_timeline.png` 或 `eda_month_coverage_and_rent.png`）；Fig. 2 户型租金 IQR（`slides_eda_flat_type_iqr.png`）。最多 2 张，其余放附录。复制到 `report/latex/photo/`。
- **禁止**：把月均变化归因于市场（可能是样本构成变化）；写成因果关系。

### IV. Preprocessing and Feature Engineering
- **目的**：按三步流水线描述，每一步回指 EDA 中的发现。
- **小节**
  1. *Normalization and Rent Cleaning*（回应 F3、F4）：文本规范化；异常判定规则（同单元 ±2 月、≥3 个可比样本、偏离中位数超过 2 倍或不到一半且差值 ≥800 SGD，逐个迭代剔除）；删除 315 行；组内平均标签，并说明理由（平方损失下的最优解是组均值）。
  2. *Building and Facility Features*（回应 F2）：按 (town, block, street) 精确连接，全部匹配；Haversine 距离和半径内计数；MRT 按开通月份过滤（防止使用未来信息）；学校和商场是静态快照（作为局限说明）。
  3. *Encoding*：类别 ID（0 为未知）、log1p 加 z-score、月份的 sin/cos、缺失插补和缺失标记；30 个输入。
- **表**：Table II 特征组一览（组 / 列数 / 编码方式 / 回应的 EDA 发现），可替代大段文字。**v2 要求**：表中要列出 8 个类别 ID 的名称（TOWN、STREET、BUILDING、SUBZONE、PLANNING_AREA、REGION、FLAT_MODEL、FLAT_TYPE_ORD，依据 `src/bert_rent/data.py`），并用表注或一段话交代每个核心字段的去向：

| 核心字段 | 去向 |
|---|---|
| RENT_APPROVAL_DATE | 转为 MONTH_INDEX、MONTH_SIN/COS，并用于计算 AGE_YEARS、COMPLETION_AGE；原始字符串不进入模型 |
| TOWN | 规范化后作为 TOWN_ID，也参与连接键 |
| BLOCK | 只用于连接键和构成 BUILDING_ID（TOWN\|BLOCK\|STREET），本身不单独进入模型 |
| STREET | 规范化（1,146→576 种写法），作为 STREET_ID，同时参与连接键和 BUILDING_ID |
| FLAT_TYPE | 统一拼写（11→6 种），作为有序编码 FLAT_TYPE_ORD |
| FLAT_MODEL | 作为 FLAT_MODEL_ID |
| FLOOR_AREA_SQM、LEASE_COMMENCE_DATE | 作为数值特征，经 z-score 标准化 |
| FURNISHED、FEE | 常量列，删除 |
| MONTHLY_RENT | 目标：剔除异常值、组内取平均后标准化；评分时用原始值 |

写作 agent 必须对照 `src/preprocess/stage3_encode.py` 核实上表每一行，与代码不符时以代码为准。
- **必须如实写明**：清洗和标准化在时间切分之前进行（或者届时如已修复，就写修复后的流程）。
- **来源**：`src/preprocess/*.py`、README §2。

### V. Method: Two-Stage Tabular BERT
- **目的**：说明为什么选这个方法、怎么实现、超参怎么定。不需要解释 BERT 原理，但要给出引用。
- **小节**
  1. *Baselines and design rationale*（课程明确要求讨论备选方案）：**v2 改写**。先介绍四个从简单到复杂的基线及其设定（依据 `results/baselines/round_a/README.md`）：全局训练均值；最近 6 个月按 town×户型分组的均值（样本少于 10 条时依次回退到户型均值、全局均值）；Ridge（8 个 ID 做 one-hot，正则强度 10）；浅层 CatBoost（深度 4、500 轮、学习率 0.1，不用验证集选轮数）。再说明 BERT 的设计动机：用字段 token 和自注意力学习字段之间的组合；用遮蔽字段预训练从训练期特征中学习结构，不使用租金。这只是一个假设，不能写成已证明的增益。TabM、TabPFN、ModernNCA 各用一句话作为没有实现的备选方案。**删除** "no tree run is present under the same time split"。
  2. *Field tokenization and encoder*：29 个 token 加 CLS；12 层、768 维、12 个头；随机初始化。
  3. *Masked-field pretraining*：按字段组以 0.15 的概率遮蔽；相互决定的字段一起遮蔽（月份组、年龄组）；CE/MSE/BCE 三种损失。
  4. *Regression fine-tuning and training protocol*：CLS 回归头、目标标准化、分层学习率、EMA；select 轮选轮数，refit 轮按选定轮数全量重训。
- **图**：Fig. 3 模型架构图（新画：输入字段 → token → 编码器 → 两个头），这是最值得投入的一张图。
- **表**：超参表放附录，正文写一句"chosen a priori / by early stopping"。
- **禁止**：声称做过系统调参（只有一组配置加早停）。

### VI. Experimental Evaluation
- **目的**：给出评估设置、结果对比和误差分析。
- **小节**
  1. *Protocol*：时间切分（训练 2021-01–2024-03 / 验证 2024-04–2025-03）；指标；随机种子和环境；保留行与原始行的口径。
  2. *Model comparison (RQ1)*：Table III，**历史口径，n=35,058，原始标签**。列：RMSE / MAE / bias（预测值减记录值）/ 拟合时间（可选）。

| 模型 | RMSE | MAE | bias |
|---|---:|---:|---:|
| 全局均值 | 842.54 | 686.73 | −534.45 |
| 近 6 个月 town×户型均值 | 497.97 | 374.20 | −51.09 |
| Ridge | 606.82 | 459.95 | +371.92 |
| 浅层 CatBoost | 468.11 | 348.07 | −37.46 |
| 两阶段 BERT | 468.34 | 344.83 | −6.73 |

  这张表要讲的要点：(a) 一个简单的近期分组均值就把 RMSE 从 842 降到 498，大部分可解释的差异来自"同类房源的近期价格水平"；(b) 浅层 CatBoost（约 9 秒）与 86M 参数的 BERT（约 4.75 小时）RMSE 只差 0.23，模型复杂度几乎没有带来 RMSE 提升；(c) BERT 的优势在偏差上（−6.7 对 −37.5）和 MAE 上；(d) Ridge 的线性时间外推导致系统性高估。这些都是**单一验证窗口上的描述性比较**，不要写成显著性结论。平均后标签的 437.2 可以留在不可约误差那一段里说明。
  3. *Feature-group ablation (RQ2)*：Table IV，**严格口径，n=35,158**，列出完整输入的 RMSE 和三个 ΔRMSE（Ridge / CatBoost）：去掉街道和楼栋 ID +8.52 / +2.95；去掉 10 个设施字段 −0.08 / +0.84；去掉时间组（5 个字段）+85.64 / +232.26。要点：在保留坐标和 town 的条件下，设施字段几乎不起作用；时间组最关键。写明时间组一次删除了 5 个字段，不能把差值归给单个字段；BERT 没有做消融。
  4. *Irreducible error*：468.3² ≈ 437.2² + 168²，说明组内离散度给出的下界。这是本文最有说服力的分析点之一。
  5. *Error analysis (RQ3)*：
     - Fig. 4 按月误差和偏差（`photo/fig4_validation_by_month.pdf`）。BERT 的偏差从 +30 漂到 −50，说明模型跟不上趋势，这也是 refit 的理由。可以结合 `by_month.csv` 补一句：浅层树在前 5 个月 RMSE 较低，BERT 在后 7 个月较低；Ridge 的月均高估从 218 扩大到 532 SGD。如果要把多个模型画进同一张逐月图，就修改 `make_figures.py`，不要直接引用 `monthly_bias.png`。
     - Fig. 5 预测值对实际值。预测范围被压缩到 1,382–5,425，两端回归均值。
     - 已见与未见楼栋（`by_building_seen.csv`，严格口径，CatBoost）：已见楼栋 475.50，未见楼栋 548.26（n=151，样本很小，要注明）。BERT 没有逐行预测，不能做这个拆分。
  6. （可选）训练曲线放附录。
- **来源**：`results/*.json`、`results/figures/`、`results/baselines/round_a/*`。
- **删除**："No independent baseline run is present in the current tree…" 整句。

### VII. Discussion and Limitations
- **要点**：(0) **v2**：按 RQ1–RQ3 逐条给出简短回答；明确说明复杂模型在 RMSE 上没有优于浅层树，这本身就是一个关于数据的发现（导师反馈也提到了这一点）；(1) 现有证据能支持什么、不能支持什么（预训练的增益没有对照；设施特征只在基线模型上做过消融）；(2) 评估协议的局限：切分前清洗、只评估保留行、验证跨度 12 个月而测试最远 16 个月；(3) 数据局限：缺少楼层和装修信息、新楼栋（102 行 / 50 栋）、学校和商场是静态快照；(4) 计算成本（86M 参数、约 4.75 小时）与树模型相比是否值得；(5) 未来工作按优先级排列：严格时间安全的全窗口评估 → 基线和消融 → 时间感知建模。
- **禁止**：在这一章引入新结果。

### VIII. Conclusion
- 一段：任务、方法、有证据支持的最强结果、最主要的局限。不引入新数字或新主张。

### Appendix A. Team and Contributions
- 表：成员 / 学号 / 主要负责部分。待团队提供；当前的 A–D 角色表不能作为最终版本。

## 3. 图表预算（正文）

| 编号 | 内容 | 章 | 来源 | 状态 |
|---|---|---|---|---|
| Table I | 数据来源概览 | II | EDA summary | 已写 |
| Fig. 1 | 月均租金和样本数 | III | photo/fig1_rent_timeline.pdf | 已有 |
| Fig. 2 | 户型租金分布 | III | photo/fig2_flat_type_iqr.pdf | 已有 |
| Table II | 特征组、8 个 ID 与核心字段去向 | IV | src/preprocess | 需按 v2 补充 |
| Fig. 3 | 两阶段模型架构 | V | photo/model_architecture.pdf | 已有 |
| Table III | 五个模型对比（历史口径） | VI | baselines/scores.csv + validation_summary.json | 待写 |
| Table IV | 特征组消融（严格口径） | VI | baselines/ablation.csv | 待写 |
| Fig. 4 | 按月误差与偏差 | VI | photo/fig4_validation_by_month.pdf | 已有，可扩展为多模型 |
| Fig. 5 | 预测值对实际值 | VI | photo/fig5_predicted_vs_recorded | 已有 |

双栏下单栏图约 0.25–0.3 页；v2 增加两张表后，正文共 5 图 4 表，约 2.3 页。超页时按以下顺序移到附录：Fig. 2 → Fig. 5 → Table I。Table III 和 Table IV 必须留在正文。

## 4. 写作顺序与交付

1. II → III → IV（事实性最强，最先写）
2. V（对照代码写）
3. VI → VII（依赖实验口径是否会更新；若会新增基线或消融，先写框架，数字最后填）
4. I → VIII → Abstract
5. 作者、附录、页数核查

每完成一章：`cd report/latex && latexmk -pdf main.tex` 能编译通过，然后交审核（按 REVIEW_CHECKLIST.md）。

## 5. 待团队决定的事项

| 事项 | 影响 |
|---|---|
| ~~是否补跑基线~~：已完成（`3573e8f`） | Table III、Table IV |
| 是否用严格口径重训 BERT，或补"去预训练"对照 | 如果补了，就把 Table III 换成严格口径并新增一行；如果不补，就在 VII 中作为局限说明 |
| 是否对 BERT 做特征消融，或导出 BERT 的逐行验证预测 | 决定 RQ2 和 RQ3 能否覆盖 BERT |
| 队名、成员、学号、实际分工 | 作者栏和附录 A |
| 是否有 Kaggle public 分数可以报告 | 第 VI 章可增加一行（注明是 public leaderboard） |
