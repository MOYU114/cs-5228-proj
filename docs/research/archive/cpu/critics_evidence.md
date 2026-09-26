# HDB 租金预测：反方与验证方法证据备忘录

检索日期：2026-09-26。只做文献与官方实现核验；未训练模型、未改动数据。本文件服务于冻结的问题：RQ1 低成本 CPU 机制；RQ2 时间漂移、楼栋依赖、辅助数据泄漏；RQ3 五方案按可行性与成本排序。任务规模由主代理提供：150k 训练样本（2021-01 至 2025-03）、50k 测试样本（2025-03 至 2026-07）、楼栋重合约 99.796%；这些不是本代理重新核验的数据事实。

## 检索记录与引用口径

两轮独立视角：第一轮检索经典树模型/神经网络比较和时间/空间验证；第二轮追踪 TabReD、TabM、TabPFN 最新版本及 TabICLv2，并专门寻找会推翻“树模型永远最好”“基础模型只能处理 10k 行”“所有楼栋必须分组隔离”等断言的证据。

代表性 queries：

- `Grinsztajn Oyallon Varoquaux 2022 Why do tree-based models still outperform deep learning tabular data`
- `McElfresh 2023 2024 When Do Neural Nets Outperform Boosted Trees Tabular Data`
- `TabM ICLR 2025 parameter efficient ensembling tabular`
- `TabReD benchmark 2024 time based splits tabular deep learning`
- `Roberts 2017 cross-validation strategies data temporal spatial hierarchical structure`
- `Cerqueira Torgo Mozetic 2020 Evaluating time series forecasting models empirical study performance estimation methods`
- `TabPFN v2.5 2026 CPU 50000 rows official`
- `2026 TabICLv2 regression CPU official large data tabular foundation model`
- `site:scikit-learn.org decision trees regression extrapolate predictions constant`
- `site:otexts.com/fpp3 time series cross validation multistep forecasting horizon seasonal naive`

以下文献均先确认标题/作者，再打开摘要、正文或官方文档检查所用论断。精确硬件与规模只采用已打开的官方来源；没有以博客转述或搜索摘要单独支持性能数字。最新技术报告只作为作者报告的证据，不等同独立复现。

## A. 模型类别：支持树模型优先，但反对从旧 benchmark 推导永恒优越性

| ID / 已核验文献 | 证据与条件 | 本任务的用途 / 反证边界 |
|---|---|---|
| C1. Léo Grinsztajn, Edouard Oyallon, Gaël Varoquaux. **Why do tree-based models still outperform deep learning on typical tabular data?** NeurIPS 2022 Datasets and Benchmarks. [会议摘要](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html) | 45 个数据集、约 10k 样本的中等规模 tabular 场景中树模型表现强；研究提出无信息特征、坐标方向和不规则函数等归纳偏置问题。VERIFIED。 | 支持把成熟 GBDT 当低成本起点；不能据此断言 150k HDB 长期外推一定胜过 2026 神经网络。会议正式题名含 typical，arXiv 题名不含该词。 |
| C2. Duncan McElfresh, Sujay Khandagale, Jonathan Valverde, Vishak Prasad C, Benjamin Feuer, Chinmay Hegde, Ganesh Ramakrishnan, Micah Goldblum, Colin White. **When Do Neural Nets Outperform Boosted Trees on Tabular Data?** NeurIPS 2023 Datasets and Benchmarks；arXiv v4 于 2024 更新。[论文摘要](https://arxiv.org/abs/2305.02997) | 比较 19 算法、176 数据集；很多情况下类别差距很小，轻度 GBDT 调参的重要性超过选择 NN 或 GBDT；不规则/偏斜特征是条件变量。VERIFIED。 | 支持有限搜索预算，反对用架构名当准确率排名。该文涉及早期 TabPFN 的规模限制属于旧版本历史，不能当现行限制。 |
| C3. Yury Gorishniy, Akim Kotelnikov, Artem Babenko. **TabM: Advancing tabular deep learning with parameter-efficient ensembling.** ICLR 2025。[会议摘要](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html) | 基于 MLP 与参数高效 ensemble；作者评估中，复杂 attention/retrieval 方法的额外复杂度未得到相应收益，TabM 改善 tabular DL 的性能/效率权衡。VERIFIED。 | 排除“DL 都复杂且低效”的绝对断言。该摘要没有给出本任务普通电脑全量耗时证据，故可以列 CPU 预算后的储备候选，但不能承诺 HDB RMSE 提升。 |
| C4. Ivan Rubachev, Nikolay Kartashev, Yury Gorishniy, Artem Babenko. **TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks.** 2024 preprint / ICLR 2025。[arXiv 摘要](https://arxiv.org/abs/2406.19380)；[会议摘要](https://proceedings.iclr.cc/paper_files/paper/2025/hash/571799482291411607c54984153190b0-Abstract-Conference.html) | 8 个工业 tabular 数据集强调时间漂移与已有特征工程；时间切分会改变方法排名；简单 MLP 类与 GBDT 在其设定中较强。VERIFIED（arXiv 与会议摘要均已打开核验；OpenReview PDF 被浏览器验证阻挡，故采用官方会议页）。 | 比无时间戳 benchmark 更贴近本题；不能把随机划分得出的优胜关系直接迁移到未来租金。也支持保留简单 NN 的可能性。 |

三篇经典/近期比较的交叉结论：C1 给出传统树优先的条件性证据，C2 弱化“类别之争”的重要性，C3 和 C4 则说明简单神经模型并非天然落后，而且评价方式本身会改变结论。因此五方案应按工程成熟度、预处理复杂度、CPU 成本与漂移处理能力排序，而不是宣称文献已证明本数据上的准确率次序。

## B. 2026 更新：不能用过时的 TabPFN 上限排除基础模型

| ID / 已核验来源 | 可说的事实 | 不可推导的结论 |
|---|---|---|
| C5. Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan. **TabICLv2: A better, faster, scalable, and open tabular foundation model.** ICML 2026（arXiv v2: 2026-09-16）。[摘要](https://arxiv.org/abs/2602.11139)；[官方文档](https://tabicl.readthedocs.io/en/latest/) | v2 支持分类和回归；官方文档明确支持 CPU/GPU，建议更大数据用 GPU。论文大规模扩展证据包含 GPU 内存条件；CPU/disk offloading 是减轻 GPU 内存的机制，不能等同“全部在 CPU 上同样快”。VERIFIED。 | 不得说 TabICL 只支持分类，或基础模型全都不能在普通电脑运行；也不能把 GPU 速度写成 150k+50k 全量 CPU 速度。 |
| C6. Léo Grinsztajn et al. **TabPFN-3: Technical Report.** 2026-05。[摘要](https://arxiv.org/abs/2605.13986) | 作者报告扩展到 1M 训练行，并明确提到单 H100 上的扩展；属于新一代模型而非 Nature v2。VERIFIED。 | “超过 10k/50k 所以模型根本用不了”在现行版本下已经错误；GPU 扩展不代表普通电脑低成本成立。 |
| C7. Benjamin Jäger, Nick Erickson, Léo Grinsztajn et al. **TabPFN-3.5: Technical Report.** 2026-09-15，v2: 2026-09-22。[摘要](https://arxiv.org/abs/2609.17895) | 截止本次检索已经发布；作者专门报告 temporal/grouped splits、高基数类别等非 IID 条件的进展。VERIFIED。 | 不能声称 TFM 研究完全没有考虑时间漂移或分组数据；本题长期 HDB 预测的效果仍未实测。 |
| D1. Prior Labs 官方 [Models](https://docs.priorlabs.ai/models) 与 [TabPFN README](https://github.com/PriorLabs/TabPFN)（2026-09-26 查阅） | 官方现行表列 3/3.5 最多 1M training rows；README 支持 CPU-only 安装，CPU 样本保护阈值为 3/3.5 的 5000，旧版为 1000；可通过 `TABPFN_ALLOW_CPU_LARGE_DATASET` 覆盖，并警告大数据 CPU 可能慢。模型总规模上限与 CPU 保护阈值是两回事。VERIFIED。 | 不得把保护阈值称为不可突破硬上限；同样不能因为有 override 就声称 150k 训练上下文加 50k 测试可在普通电脑经济运行。 |

建议给主报告的短表述：**TabM、TabICLv2 和 TabPFN-3.5 值得保留为后续候选；本轮未排在五个优先方案内的理由是全量 CPU 成本与工程预算尚未证实，而不是深度学习天然无效、不能回归、不能在 CPU 运行或只能接收 10k 行。** 最新模型证据削弱了“树永远最好”，但并未直接建立本题的耗时与 RMSE 保证。

## C. 验证设计：未来已知楼栋，与新楼栋迁移是不同问题

| ID / 已核验文献 | 支持的主张 | 条件与反方 |
|---|---|---|
| V1. David R. Roberts et al. **Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.** Ecography 40(8):913–929, 2017（online 2016）。[出版方全文](https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881) | 忽略依赖结构可低估预测误差；分块方法要匹配预测目的。全文及摘要均说明分块也可能强迫外推，因而高估原本插值任务的误差。VERIFIED。 | 楼栋重合本身不自动等于违规泄漏；本题约 99.8% 已知楼栋时，不应只用严格全楼栋 holdout 来选择主模型。保留按时间的主评估，另做新楼栋/空间 holdout 作为压力测试。此应用判断是从论文原则与任务分布推导，非论文直接研究 HDB。 |
| V2. Vitor Cerqueira, Luis Torgo, Igor Mozetič. **Evaluating time series forecasting models: an empirical study on performance estimation methods.** Machine Learning 109:1997–2028, 2020。[作者托管最终 PDF](https://kt.ijs.si/igor_mozetic/papers/CerTorMoz-TimeSer-MACH-20.pdf) | 最终版本评估 174 条真实序列与 3 类合成序列；非平稳情形下，保留时间顺序的多时段 holdout 更准确；平稳情形允许不同结论。VERIFIED。 | 不说“时间数据任何交叉验证都无效”。2019 arXiv 摘要仍写 62 条序列，本文以正式 2020 版本为准。 |
| C4. TabReD（见上） | 随机切分与时间切分实际出现模型排名差异。 | 直接对应 tabular，而 V1 的原始领域为生态学、V2 为时间序列；三者共同支持本题的时序验证，但不提供 HDB 上具体误差预测。 |

补充方法文档：Rob J. Hyndman 与 George Athanasopoulos 的 *Forecasting: Principles and Practice*, 3rd ed. (2021，线上更新 2026)：[rolling-origin / 多步验证](https://otexts.com/fpp3/tscv.html)、[回归预测的 ex-ante 与 ex-post 区别](https://otexts.com/fpp3/forecasting-regression.html)、[趋势与季节分解](https://otexts.com/fpp3/forecasting-decomposition.html)。这些是作者教学来源，用来明确实施规则，不冒充新的 HDB 实证论文。

### 适合本任务的实施约束（建议，不是本轮训练结果）

1. 主验证应按整月向未来推进；除近期短窗外，至少包含近似最终远期跨度的约 12–16 个月验证窗。仅验证未来一月不能衡量远期市场漂移。每个 origin 的历史只能来自该 cutoff 之前。
2. 训练和测试都含 2025-03，所以边界月是否存在完整时间先后需要单独审查；只有月份粒度时不要把同月目标视作已知历史。最稳妥的验证定义是整月隔离。
3. `building × flat type` 历史均值、近邻价格、target encoding、残差均值等目标衍生特征全部按 fold 和时间重建。未来 validation 标签不能回流；若模拟固定起点预测未来 16 个月，也不能在中途喂入真实 validation 租金。
4. 已知楼栋与未知楼栋分别汇总误差；主目标仍为时间外推 RMSE。楼栋 holdout 测试泛化到新楼，不替代本题部署分布。重复交易/近重复行另行检查，因为这些可能有比“同楼栋”更直接的泄漏。
5. 辅助变量要标注 `available_at`，而不只有对应月份。未来实际租赁指数、后发布转售数据或完工信息用于过去 cutoff 会将 ex-ante 任务变成 ex-post。日历变量可预知；非预知宏观变量必须用 cutoff 前的数值、预测或显式场景。
6. RMSE 在原货币尺度统一比较；另按预测月份/预测距离、区域、房型、楼栋是否出现过分层，避免总体指标掩盖远期系统性偏差。验证方案与计算预算对五方案一致。

## D. 树的时间外推与季节性必须分开

[scikit-learn 官方树文档](https://scikit-learn.org/stable/modules/tree.html) 明确描述普通回归树为分段常数、不擅长外推。因而仅向普通 constant-leaf GBDT 输入递增月份编号，不能指望它在超出训练月份后自动续接上涨趋势。这是模型结构的推论；不要误说所有 GBDT 输出一定被原始训练目标最大值限制，也不要扩大到带线性叶节点或显式趋势残差化的所有树变体。

月份 one-hot / sin-cos 可以表示重复季节形状，但不会自动表达跨年租金水平上涨。Hyndman 与 Athanasopoulos 的分解章节将季节部分和去季节后的部分分别预测，可作为“平滑/收缩趋势 + 树学习房屋差异或残差”的机制依据。是否真有稳定年季节性、线性趋势是否合理，都要由训练历史与多 cutoff 验证决定；不应凭“房租每年都会涨”外推。

## 给五方案可行性排序的审查标准

- 区分算法与完整预测管线：把 CatBoost、LightGBM、XGBoost 分成三个名字，并不能覆盖时间趋势、楼栋历史与空间信息这些主要机制。
- 可行性优先意味着先比较依赖数量、预处理与泄漏控制工作量、内存/训练/推理预算，再说明潜在收益；不得给出未经实验的 RMSE 顺序。
- 普通 CPU 上的树与正则化线性/加性基线有成熟实现价值，但“普通电脑若干分钟”必须有硬件与实测，文献不能代替本机测量。
- 大量 seed、大网格、全量空间距离矩阵、重型 AutoML 多层 stacking 都会侵蚀低成本优势；小量互补模型平均可以列方案，但权重必须来自历史验证。
- 一个两阶段时间趋势/房屋残差方案可比换第三种树库更贴近本题；这是任务结构推断，不是已证实的最优方法。复杂方案只有在多时间窗一致受益后才值得增加成本。

## 已纠正的潜在误述

| 容易出现的误述 | 证据后的正确表达 |
|---|---|
| 树模型在 tabular 上永远更准 | 2022/2023 benchmark 支持成熟树起点；2025/2026 已有有力 NN/TFM 反证，具体任务仍须评估。 |
| TabPFN 只能处理 10k 或 50k 行 | 那是特定旧版本范围；现行 3/3.5 官方规模可达 1M，但全量 CPU 成本不能据此保证。 |
| TabICL 只能做分类 | v1 是分类；2026 v2 原生支持回归。 |
| 不能用 GPU，所以所有 DL 都不可能 | 多个官方实现支持 CPU；合理理由是本题规模和预算下的优先级，不是不可运行。 |
| 同楼栋出现在 train/test 必然泄漏 | 同楼历史用于未来同楼预测可以合法；目标来源、真实时间先后和重复记录才决定泄漏。 |
| 强制 group split 一定是最正确验证 | 它评估新组泛化，也可能改变主任务；按时间的同楼未来评估与新楼压力测试应分别报告。 |
| 加 year/month 就解决未来趋势 | 普通常数叶树不会自动延伸时间趋势；季节性与市场水平变化需分别处理。 |

覆盖审查：模型类别含 C1–C4 四篇独立研究；近期反证含 C3、C5–C7；结构化验证含 V1、V2、C4 三篇研究。空间依赖原则跨领域迁移已显式标注，未声称找到直接验证本数据设定的 HDB 最优方案。没有使用无法确认的引用。
