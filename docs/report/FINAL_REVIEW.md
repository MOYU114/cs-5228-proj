# 最终报告终审（2026-11-04）

审阅对象：`report/latex/` 当前工作区版本（main.tex + abstract + chapter01–08 + references + 附录；基于提交 `3573e8f` 加未提交的报告改动）。方法：`pre-submission-reviewer` 五维度审阅，加 `docs/report/REVIEW_CHECKLIST.md`（含 A2 节）。我在临时副本中从零编译了一次：编译通过，共 9 页，正文和参考文献在第 8 页结束（正文约 7.5 页，**符合 8 页限制**），附录在第 9 页。唯一的警告是 fixltx2e，无害。

范式判定：STEM／应用型数据挖掘课程报告。重点看 EDA→预处理的因果链、基线覆盖、消融归因和评估口径。

## 总结

- CRITICAL：1
- MAJOR：4
- MINOR：10
- 最先要修的三件事：
  1. 附录 A 的分工仍然是 "Pending confirmation"，也没有队名（C1）。
  2. 参考文献只有 3 篇，没有 Related Work，CatBoost、TabM 等提到的方法都没有引用（M1）。
  3. 同一条限定语在全文重复了大约 20 次，摘要过满（M3、M4）。第 8 页还剩约 0.4 页，可以用来放 Related Work。

**数字核对**：Table III 与 Table V、Eq. 2（√(468.3424² − 437.1584²) = 168.04）、Table IV 中的 RMSE、按月描述（CatBoost 在前 5 个月 RMSE 更低，BERT 在后 7 个月更低；Ridge 偏差从 +217.5 扩大到 +532.3）、图 4 的抽查值（503.3 / +32.9 / −52.8），以及已见楼栋 475.50、未见楼栋 548.26（n=151），全部与 CSV 和 JSON 一致。**两种评估口径没有混用**：Table III 只有历史口径，Table IV 只有严格口径，正文也分别给出了两种口径的定义。导师的各项反馈都已经回应：RQ1–RQ3 写在第 I 章并在第 VII 章回答；STREET 等核心字段的去向写在第 IV 章；有基线和消融；有 RMSE 之外的分析。过时的句子已经删除，"没有去预训练对照"的说明保留了。

## 维度 1：宏观逻辑

| # | 问题（引文） | 级别 | 修改建议 |
|---|---|---|---|
| 1 | 附录 A："CHEN CONGRUI & A0329163A & Pending confirmation"（四行都是）；正文开头写的是 "Team name remains generic because it was not provided"，作者栏只有 "GROUP 58"。课程要求报告写出队名，并在附录给出分工。 | CRITICAL | 团队提供每人实际负责的部分（EDA、预处理、BERT、基线、写作和图表），填进表格；在标题下或作者栏写上队名。 |
| 2 | 参考文献只有 [1] BERT、[2] Transformer、[3] FT-Transformer。正文提到 "shallow CatBoost"、"Ridge"、"TabM, TabPFN and ModernNCA"，但都没有引用；全文没有 Related Work，也没有提到 HDB 或住房价格预测的既有工作。课程评分写明 "Related work should be appropriately referenced"。 | MAJOR | 利用第 8 页剩余的约 0.4 页，在第 I 章末尾或第 V 章开头加一段 Related Work（4–6 句）。至少补上这些引用：CatBoost（Prokhorenkova et al., NeurIPS 2018）、TabM（ICLR 2025）、TabPFN（Hollmann et al.）、ModernNCA（ICLR 2025）；GBDT 在表格数据上仍然很强的证据（Grinsztajn et al., NeurIPS 2022），以及一篇 hedonic pricing 或住房价格预测的文献。每篇都要核实出处，链接见 `docs/research/PRIORITY_PLAN.md`。 |
| 3 | 主方法的主要结论只有历史口径的证据："no strict BERT result or no-pretraining BERT ablation exists"（摘要）。全文的核心模块（遮蔽字段预训练）没有任何归因证据，这是 skill 定义的 "attribution unverified"。 | MAJOR | 时间允许的话，至少补一个"不预训练、直接微调"的历史口径对照（只需要 stage 2，约 30 分钟），或者在严格口径下重训 BERT（约 4.75 小时）。如果不补，现在的如实披露已经足够，但这一条会扣方法质量分。 |
| 4 | RQ2 的回答只说了现象："the five-field time group matters far more than the ten facility fields"，没有解释原因。数据里有现成的解释：去掉时间组后，CatBoost 的 bias 从 −33.28 变成 −505.62，接近全局均值基线的 −528.94，说明模型失去了跟踪租金水平的能力。 | MINOR | 在 VI-C 或 VII 中加一句这个解释（数据来自 `scores.csv` 严格口径的 bias 列）。这正好回应导师"理解比 RMSE 重要"的意见。 |
| 5 | RQ1 最有价值的发现只有半句："the recent grouped mean already reduces RMSE from 842.54 to 497.97"。它说明大部分可预测的差异来自"同类房源的近期价格水平"，本身就是关于数据的洞察，但在第 VII 章和结论中没有再出现。 | MINOR | 在第 VII 章的 RQ1 回答中加一句：从近期均值到最好的模型，RMSE 只再降了约 30 SGD。 |

## 维度 2：写作细节

| # | 问题（引文） | 级别 | 修改建议 |
|---|---|---|---|
| 1 | 同一条限定语在多处重复："descriptive comparison on one historical validation window"（VI-B）、"single-window comparison does not establish a general advantage"（VII）、"descriptive single-window comparison rather than evidence of general superiority"（VIII）、"does not establish a gain"（摘要）。全文 "descriptive / does not establish / not evidence / rather than" 共出现 21 次，"therefore" 13 次。读起来很防御，也占了篇幅。 | MAJOR | 每条限定语只在定义它的地方（VI-B）完整写一次，在第 VII 章总结时再写一次；摘要和结论各只保留一个短语。 |
| 2 | 摘要一共 6 句，塞了 7 个数字和多个内部术语（"historical whole-month protocol"、"averaged cleaned labels"、"strict-protocol"），第一句也没有说明为什么这个问题重要。 | MAJOR | 按五句式重写：问题和意义 → 挑战（时间外推、相同输入对应不同租金）→ 方法 → 结果（只写 BERT 468.34、CatBoost 468.11、近期均值 497.97 三个数）→ 一个洞察（时间特征最重要，复杂度收益很小）。437.16 和 MAE 留给正文。 |
| 3 | 第 I 章第 1 段："The course task is therefore a supervised regression problem…""A useful report must describe both…" 这是讲作业的元语言，不是研究动机。 | MINOR | 改成研究式的写法，例如 "We formulate rent estimation as supervised regression in which the test period lies after the labelled period." |
| 4 | 第 II 章末尾："we retain the local file range and flag the discrepancy for submission documentation"。这是写给团队的内部备注。 | MINOR | 改为 "The local files extend to July 2026, one month beyond the course description; we use the files as supplied." 或者放进脚注。 |
| 5 | 正文里出现仓库路径："\texttt{results/baselines/round\_a}"（VI-A）、"\texttt{results/validation\_predictions.csv}"（VI-E）。 | MINOR | 正文改成文字描述；路径放进附录或者一个 Reproducibility 脚注，同时给出 GitHub 链接或代码包说明（可复现性占 10%，报告里目前没有任何代码获取方式）。 |
| 6 | V-A 的小节标题是 "Design Rationale and Alternatives"，但第一段讲的是四个基线。 | MINOR | 改名为 "Baselines and Design Rationale"。 |

## 维度 3：英语语法

整体语法很干净，没有发现冠词或主谓一致错误。

| # | 问题 | 级别 | 规则与建议 |
|---|---|---|---|
| 1 | 英式和美式拼写混用："labelled"（12 次）、"modelling"、"analyse(s)" 是英式，"unlabeled"（3 次）、"normalize(d)" 是美式。 | MINOR | 拼写一致性规则：IEEE 默认用美式。统一为 labeled、modeling、analyze。 |
| 2 | 长句偏多，例如 IV-A 第 1 段和 VI-A 第 1 段都有超过 40 个词的句子。 | MINOR | 长句拆分规则：在 "Specifically," 或分号处拆成两句。 |

## 维度 4：LaTeX 格式

| # | 问题 | 级别 | 修改建议 |
|---|---|---|---|
| 1 | Table IV 中显示的 Δ 和显示的 RMSE 对不上：624.02 − 615.49 = 8.53，但表中写 +8.52；478.79 − 475.83 = 2.96，表中写 +2.95。这是由未四舍五入的原值算出来的，读者心算会觉得有错。 | MINOR | 用显示的两位小数重新计算 Δ（+8.53、+2.96），或者在表注里加一句 "Δ computed before rounding"。 |
| 2 | 编译时出现 fixltx2e 警告，没有 overfull 或未定义引用；公式 1、2 都被引用了；label 命名规范。 | — | 通过。 |
| 3 | 页数：正文和参考文献在第 8 页结束，附录在第 9 页。 | — | 通过。加 Related Work 之后要再核对一次页数。 |

## 维度 5：图表质量

| # | 问题 | 级别 | 修改建议 |
|---|---|---|---|
| 1 | 图 5 仍然是位图 PNG，风格也和其他图不统一：无衬线字体，图内有标题 "Validation: predicted vs recorded rent"，还有色条。 | MINOR（已知限制） | 如果拿不到逐行预测，至少把图内标题裁掉；页数紧张时，图 5 是最先移到附录的候选。 |
| 2 | 图 1、2、4 是矢量图，统一 Times 字体；图 4 是上下两个面板，caption 和正文都写的是 "upper/lower panel"，一致。架构图写的是 "29 field tokens + [CLS]"、"2 calendar + 1 flag"、"MLP 768→256→1"，与第 V 节一致，也没有文字重叠。 | — | 通过。 |
| 3 | 图 4 上面板的 y 轴最低刻度是 425，但清洗后标签的曲线在 2024-10 降到约 408，刻度标签没有覆盖最低值。 | MINOR | 在 `make_figures.py` 中把 y 轴下限设为 400。 |

## 禁用词与破折号扫描

我对摘要和第 I–VIII 章做了全文扫描，没有抽样：

- 破折号：0 处。
- AI 腔词汇："superiority" 2 次（ch07 第 6 行、ch08 第 2 行），"reveal" 1 次（ch01 第 14 行，RQ3）。都不到 3 次，算 MINOR，可以分别换成 "advantage" 和 "show"。
- 非禁用词但过度使用的："therefore" 13 次，限定语 21 次，见维度 2 第 1 条。

## 最终评分（pre-submission-reviewer，1–10）

**6 / 10**

依据：1 个 CRITICAL、4 个 MAJOR。按 skill 的规定，有未解决的 CRITICAL 就不能给"可提交"。修掉 C1、M1、M3、M4 后（工作量大约半天，不需要新实验），预计可以到 **8/10**；再补一个"不预训练"对照，可以到 **9/10**。

**按课程评分标准的定性判断**（不是官方分数）：
- 方法质量（60%）：强。EDA→预处理的因果链清楚，口径披露非常诚实，基线从简单到复杂都有，消融也有。主要扣分点是主模型的结果只有历史口径，而且预训练没有归因证据。
- 报告质量（30%）：中上。数字准确，结构完整；扣分点是相关工作不足、重复的限定语和过满的摘要。
- 可复现性（10%）：报告里没有代码链接或运行说明，需要补一行。

## 提交建议

**还需要 1–2 天的工作**：先填附录分工和队名，补 Related Work 和引用，再精简限定语和摘要；MINOR 项顺手修掉即可。
