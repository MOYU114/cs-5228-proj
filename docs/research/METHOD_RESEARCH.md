# HDB 月租金预测：2025–2026 现代方法调研与五方案评估

检索截至 **2026-09-26**。当前五项主候选全部来自 **2025–2026** 年论文或技术报告。GPU 可以使用，低成本是软偏好；实际设备、显存和付费额度尚未确认。本轮只做调研与可行性评估，**没有训练、预测或本地 RMSE/耗时/显存结果**。初始版本保存在 [历史报告](archive/cpu/METHOD_RESEARCH_CPU_INITIAL.md)；其中 CPU 门槛和旧优先级已被本文件取代。

## 摘要

本项目的难点不仅是拟合住房属性，还包括训练截止 2025-03、测试延伸至 2026-07 的时间变化。EDA 显示，大部分测试行来自训练已见楼栋，类别规范化能减少表观差异，但同输入不同租金和市场水平变化仍然存在。本调研围绕现代参数模型、历史检索、表格基础模型及时间条件化开展增量检索，核验九篇研究及官方实现。五项候选为 TabM、TabICLv2、ModernNCA、TabPFN-3.5 和采用 feature-aware temporal modulation 的 TabM。结合独立 idea-evaluator 与用户最新的争分目标，当前尝试顺序为 **TabM → 时间调制 TabM → TabICLv2 → TabPFN-3.5 → ModernNCA**；这是风险调整后的实验投资顺序，不是预测精度排名。完整的 RMSE 决策与交接见 [最佳尝试清单](PRIORITY_PLAN.md)。经典树与线性模型保留为共同基线。所有候选均需在同一组向前验证窗口中检验，尤其要控制历史标签检索、辅助数据生效时间和基础模型上下文的边界。

## 1. 研究问题与 EDA 依据

[研究 brief](RESEARCH_BRIEF.md) 在定向检索前冻结了三个问题：**RQ1**，哪些现代机制适合 150,000 行、高基数楼栋与混合字段？**RQ2**，最长 16 个月预测间隔、重复交易和辅助快照会怎样改变收益假设与失效条件？**RQ3**，GPU 开放且成本只是偏好时，五项候选应如何按可行性安排、如何否证？

以下数字来自已运行的 [EDA 报告](../EDA_REPORT.md)、[核心统计](../../experiments/eda/core/summary.json) 和 [辅助审计](../../experiments/eda/auxiliary/summary.json)。方法动机与事实分栏，避免把 EDA 关联写成实验结论。

| 已观测事实 | 支持后续测试的假设 | 尚不能推断 |
| --- | --- | --- |
| train 150,000 行、51 个月；test 50,000 行，延伸到 2026-07；最大月序差 16 | 同时做近期与长跨度验证 | 近期成绩代表全部测试期 |
| 测试 49,620 行（99.24%）晚于训练最大月份；另有 380 行处于共同边界月 2025-03 | 主要评估未来月份；边界月单列 | 同一月内各交易的先后顺序 |
| 训练有 9,406 个规范化楼栋；测试 99.796% 行属于已见楼栋；新楼栋 50 个、102 行 | 楼栋表示及历史检索值得试验 | 同栋等于同单元，或记住楼栋就解决漂移 |
| 户型分布 TVD 从原始 0.2394 降至规范化后的 0.0101 | 先统一文本，再比较模型 | 目标条件分布已经稳定 |
| 月均租金从 2021-01 的 2,048.90 到 2025-03 的 3,162.71 SGD/月，后期增速趋缓 | 检验随时间改变的属性映射与预测偏差 | 未来上涨、平台或季节性必然延续 |
| 面积与租金 Pearson 相关约 0.329，楼龄代理约 −0.178 | 学习非线性属性关系与交互 | 因果效应或单调约束成立 |
| 相同规范化 X、不同 y 涉及 17,900 组、39,540 行；没有 unit ID | 保留交易、正则化、检查残差 | 这些都是脏记录，或已求得不可约误差 |
| HDB 三字段规范化键 100% 匹配、无一对多；设施表缺历史有效期 | 辅助属性可作独立消融 | 完整连接代表时间安全或预测有效 |

## 2. 检索范围和证据等级

本轮在已完成调研上补查现代方法，使用三个并行视角，每个均包含广查与定向核验：神经表征/检索、表格基础模型及资源反例、时间漂移机制。关键词与元数据核验分别见 [检索模型底稿](evidence/modern_retrieval_evidence.md)、[基础模型底稿](evidence/modern_foundation_evidence.md)、[时间方法底稿](evidence/modern_temporal_evidence.md)。另一个代理按 idea-evaluator 的五维框架独立评估，没有让文献作者的平均排名直接决定本题次序。

正文采用 **9 篇研究文献**，年份为 2024–2026；主候选的正式发表/当前版本均为 2025–2026。每个相关分支至少有三项相邻证据，而非单篇宣传式介绍。会议论文、预印本/技术报告、官方软件能力及本地事实分别标识。TabPFN-3.5 的新报告不冒充已有独立复现；官方大显存设备结果不转写为普通电脑承诺。网页与仓库核验发生在上述日期，后续实验必须锁定包版本、权重和 commit。

这是有范围的证据调查，不声称穷尽所有模型。旧树模型、统计方法和 GPU 中间版的证据保留在 [材料索引](README.md)，供基线与历史追踪使用；它们不再占主候选名额。

## 3. 分类：预测时依赖什么

按预测所依赖的已学习对象，一级分类为任务内训练的参数模型、显式历史标签检索、预训练模型的上下文预测。时间条件化是第一类的可控扩展，GPU 是运行后端，均不另造算法类别。

| 机制 | 预测时依赖 | 五项中的成员 | 本项目最需要审查的边界 |
| --- | --- | --- | --- |
| 任务内参数学习 | 用本地历史数据学到的权重 | N1 TabM；N5 时间调制 TabM | 时间外推、类别表示、数值稳定性 |
| 任务内距离学习与检索 | 学到的表示 + 带标签历史候选库 | N3 ModernNCA | 候选截止时间、查询成本、历史目标范围 |
| 预训练模型的上下文学习 | 固定预训练权重 + 本折训练上下文 | N2 TabICLv2；N4 TabPFN-3.5 | 上下文容量、推理开销、权重版本与数据边界 |

N4 可另做任务内微调，但必须与冻结权重的上下文版本分开报告。N1/N5 共享主干是有意的机制对照；N5 并非第六种独立基础架构。所有路线都仍须解决日历向前评估，没有哪一种因名称带“foundation”或“temporal”就自动具备可靠远期预测。

## 4. 现代参数模型与检索：表示能力不等于时间适应

TabM [1] 通过共享参数的 MLP 集成学习表格表示；ModernNCA [2] 则学习距离，再对候选标签加权。TabR [3] 同样检索历史邻居，但将邻居特征/标签表示送入参数化预测模块。因此 TabM 省去推理时的带标签候选扫描，ModernNCA 提供更直接的历史相似性归纳，TabR 说明“检索模型”内部也存在不同输出机制。[TabM 正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html)、[ModernNCA 正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ecd294194d5155432aab533f7ce0f6fe-Abstract-Conference.html)、[TabR 正式论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html)。

这些差异与高楼栋覆盖率有关，但三篇都不能证明本项目未来租金误差谁更低。ModernNCA 论文还指出未显式处理邻居间潜在时间关系；随机拆分上的平均表现不直接回答长跨度问题。[ModernNCA 全文附录 B.5](https://proceedings.iclr.cc/paper_files/paper/2025/file/ecd294194d5155432aab533f7ce0f6fe-Paper-Conference.pdf)。

| 已核验的实现区别 | 实验含义 |
| --- | --- |
| TabM 官方 `cat_cardinalities` 默认 one-hot，不是楼栋 embedding；训练平均各成员损失，推理平均预测 | 类别压缩必须另标适配；不能先平均预测再计算训练损失冒充原版 |
| ModernNCA 使用 `softmax(-欧氏距离/T)` 对候选 y 求和，无独立回归头 | 数值正常时预测是候选 y 的凸组合；标准化再反变换也不突破该范围 |
| ModernNCA 训练排除当前行自身标签，但其他索引的同 X 行仍可作邻居 | 自身排除不等于去重，也不能据此删除不同交易 |
| TabR 有 top-k 检索与参数预测头 | ModernNCA 的凸组合边界不能移植到所有检索方法 |

来源：[TabM 官方包](https://github.com/yandex-research/tabm)、[ModernNCA 模型](https://github.com/LAMDA-Tabular/TALENT/blob/main/TALENT/model/models/modernNCA.py)、[训练/预测路径](https://github.com/LAMDA-Tabular/TALENT/blob/main/TALENT/model/methods/modernNCA.py)、[TabR 实现](https://github.com/yandex-research/tabular-dl-tabr/blob/main/bin/tabr.py)。这里的范围约束是源码推导，不是本地误差结论；测试租金未知，不能声称它已因此失效。

## 5. 2026 表格基础模型：大表支持已经变化，但成本转移到上下文

TabICLv2 [4]、TabPFN-3 [5] 和 TabPFN-3.5 [6] 已把讨论扩展到回归及更大规模表格。不能沿用早期版本的 10k/50k 行限制，将 150k 行一概排除。与 TabM 的本地梯度训练相比，这类方法首先利用预训练权重读取训练样本及标签形成上下文；免任务内梯度训练并不表示上下文编码和预测免费。[TabICLv2](https://arxiv.org/abs/2602.11139)、[TabPFN-3](https://arxiv.org/abs/2605.13986)、[TabPFN-3.5](https://arxiv.org/abs/2609.17895)。

TabPFN-3.5 的论文对分组、时间和大表设置也保留了竞争方法占优的结果，不能把总体宣传结论解释成这种部署条件下普遍最好。TabICLv2 的大规模测试使用高端设备、混合精度与内存策略，TabPFN-3.5 的规模分析也有专门设备条件。它们支持“值得纳入”而不是“普通单卡已保证能跑”；当前主问题是本折约 85k–133k 历史上下文、最终 150k 上下文和 50k 查询的端到端资源。

| 比较项 | TabICLv2 | TabPFN-3.5 |
| --- | --- | --- |
| 主候选模式 | 固定预训练模型的回归上下文预测 | 先冻结模型做上下文回归，微调另作后续消融 |
| 可用性证据 | 2026 论文与公开回归接口 | 2026-09 技术报告、官方本地接口及微调文档 |
| 需计入成本 | 预处理、上下文表示/缓存、各集成成员、查询 | 权重加载、上下文及缓存、各估计器预测；微调另计 |
| 不能继承的结论 | 最大规模的论文设备结果 ≠ 团队设备容量 | 早期行数上限不能直接套用；当前限制按版本、后端与实际内存核验 |

实际 API 和设备参数以 [TabICL 文档](https://tabicl.readthedocs.io/en/latest/) 与 [TabPFN 官方仓库](https://github.com/PriorLabs/TabPFN) 为准。两者均有回归微调入口，首轮仍采用冻结权重。TabPFN 当前权重获取流程要求登录并接受对应模型许可；无界面服务器可通过 `TABPFN_TOKEN` 认证。这与上传 HDB 数据进行云推理是不同动作，本轮未下载权重或上传数据。两条路线均计划优先本地运行，不从头预训练基础模型。

缓存方式必须显式登记：TabICL 默认 `kv_cache=False`，外部分批可能重复上下文编码；`repr` 与完整 KV 缓存也不是相同代价。TabPFN 默认 `fit_preprocessors` 只缓存预处理，`fit_with_cache` 才构建相应神经缓存。不能只测 `fit` 就宣称方法便宜，完整验证/测试查询的时间和内存都须计入。[缓存与分块审计](evidence/modern_foundation_evidence.md)。实施前还应记录预训练数据来源和权重版本，核实不包含本题未来真实租金；本轮没有对下载后的权重做数据污染审计，不能宣称该风险已被实证排除。

## 6. 2025 时间漂移方法：把时间表示和验证协议分开

TabReD [7] 表明实际表格的时间切分会影响方法比较；Cai 与 Ye 的 ICML 2025 研究 [8] 进一步区分训练滞后、内部验证偏差及时间表示；其 NeurIPS 2025 工作 [9] 则让特征变换显式依赖时间。三者分别说明评估问题、输入/训练协议和表示适应机制，不能简化成“加月份就解决漂移”。[TabReD](https://proceedings.iclr.cc/paper_files/paper/2025/hash/571799482291411607c54984153190b0-Abstract-Conference.html)、[ICML 论文](https://proceedings.mlr.press/v267/cai25j.html)、[NeurIPS 论文](https://papers.nips.cc/paper_files/paper/2025/hash/ddc20f403d0d296c87940393838d5971-Abstract-Conference.html)。

N5 采用最后一篇的 feature-aware temporal modulation：时间表示产生逐特征变换参数，改变尺度、平移和形状。相比只拼接时间 embedding，它让同一属性的映射随时间变化；相比从未来样本估计分布，它可以在推理中以该行已知日历为条件。但在 51 个训练月份之后延伸 16 个月是否稳定，仍完全需要本地检验。[NeurIPS 全文式 (2)](https://proceedings.neurips.cc/paper_files/paper/2025/file/ddc20f403d0d296c87940393838d5971-Paper-Conference.pdf)。

官方仓库提供 `tabm`、`tabm_temporal`、`tabm_modulated`，因此这一组合有直接实现依据。必须用同一版主干做三臂对照；不能把当前独立 TabM 包与论文仓库的不同版本混用后，将全部差异归给调制模块。[官方仓库](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation)、[TabM 调制实现](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation/blob/main/model/models/tabm_modulated.py)。

ICML 文献关于随机内部划分的发现不构成本项目随机拆分外层测试的理由。首轮固定共同的完整月份外层窗口，内部早停与最终重新拟合分开；若后续研究作者的内部协议，作为独立消融，保持外层截止日不变。时间标准化同样必须在实际训练段拟合，不能依赖未来日期分布。

源码核验发现，官方 `indices` 路径的时间均值/标准差取原始 `M["train"]`，不一定等于内部重切分后的训练段；因此每折需要覆盖为实际训练月份的统计并固定复用。另一条 `period/period_t` helper 按各分片分别归一化，但 N5 的 `indices` 路径不使用它。这里是时间特征统计的协议问题，不是发现读取未来租金标签。[固定 commit 的统计来源](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation/blob/8feabbd580d59757dd108622ea93b9bdb6d38fa1/model/lib/data.py#L570-L609)。

## 7. 五项候选卡与可行性排序

N 编号用于追踪方案身份，**不是排名**。以下结合独立 [idea-evaluator 评估](IDEA_EVALUATION.md) 更新为以未来 RMSE 为目标的尝试顺序；此前工程顺序将 N2 放在 N5 前，现将直接针对时间问题的 N5 提前。具体评价与晋级规则以 [最佳尝试清单](PRIORITY_PLAN.md) 为准，每项仍为待验证假设。软件入口是后续实施建议，不是已安装依赖或可直接运行的本地训练代码。

| 优先级 | 候选及年份 | 与 EDA 的联系 | 计算和工程判断，非实测 |
| --- | --- | --- | --- |
| 1 | **N1 TabM，ICLR 2025** | 非线性物业关系、高基数楼栋；提供现代可控主干 | CPU 可研究，GPU 更值得校准；直接监督训练，工程相对清楚 |
| 2 | **N5 TabM + 时间特征调制，NeurIPS 2025** | 直接检验月份变化是否需要改变属性映射 | 复用 N1，但要对齐主干与三臂消融；远期数值风险 |
| 3 | **N2 TabICLv2，ICML 2026** | 检验预训练表格先验能否改善同输入未来回归 | 首轮不做梯度微调；GPU 上下文/推理资源需实测 |
| 4 | **N4 TabPFN-3.5，2026-09 技术报告** | 独立基础模型路线，检查时间/分组分布泛化 | 新版权重、环境及大上下文校准；微调为可选增量 |
| 5 | **N3 ModernNCA，ICLR 2025** | 已见楼栋占绝大多数，学习历史相似性 | 全量候选编码/距离计算使适配工作较大；排后不表示准确率差 |

### N1：TabM 参数共享集成

从 [官方 tabm](https://github.com/yandex-research/tabm) 的回归训练语义起步。首版保持官方类别表示、训练折数值预处理、原尺度平方误差；目标可按训练均值/标准差线性标准化，评分反变换至 SGD/月。高基数 one-hot 尽量按批生成，先核算首层与激活成本；自定义楼栋 embedding 作为明确的适配变体。

**待测假设：** 共享集成在本题提供优于普通 MLP、或与树不同的误差结构。**最小否证实验：** 相同输入、时间窗与开发机会下比较小 MLP、CatBoost/LightGBM 和 N1；再分别去掉楼栋、改变数值表示/集成成员数。若只提高训练拟合、远期偏差不降，不称其改善时间泛化。月份编号是预测特征，不是外推保证。

### N2：TabICLv2 预训练上下文回归

使用当前官方 `TabICLRegressor` 和对应 v2 checkpoint，首轮固定权重。每个外层窗口仅以训练段 X/y 建立上下文，单独记录预处理、上下文缓存和完整验证查询的成本。若全量不适配设备，可预先登记训练段分层/近期上下文选择，再与同样样本子集的基线比较；不能靠删掉困难验证行来适配，也不能把子集实验写成全量结果。

**待测假设：** 预训练先验在有限调参下产生有竞争力且互补的误差。**否证：** 比较原尺度 RMSE、按未来跨度偏差以及已见/未见楼栋结果；全历史与近期上下文只在开发窗口选择。缓存必须每折重建，类别 ID 不能未经接口支持就当连续数值。先证明模型版本、回归接口与本地资源可用，再考虑更大集成，不直接复现整个预训练过程。

### N5：TabM + feature-aware temporal modulation

以 [官方时间调制仓库](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation) 为起点，将三臂统一为普通日历输入、仅时间 embedding、特征调制。保持类别规则、主干版本、优化器和搜索机会一致；记录相较 N1 独立包的任何实现差异。时间 t 使用真实日历，不能按交易行号伪造连续时间。官方 Fourier 周期以秒为单位，因此需统一约定月份锚点、时区并转换为秒时间戳，或同步将全部周期改为月份单位；不能直接传 month index 而保留秒周期。趋势标准化不会自动修复这一单位错配。[时间表示源码](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation/blob/8feabbd580d59757dd108622ea93b9bdb6d38fa1/model/lib/temporal_embeddings.py#L95-L126)。

**待测假设：** 时间条件化的属性映射能降低较远月份的系统性偏差。**否证：** N5 对相同版 N1 的整体与逐跨度 RMSE、均值残差若未改善，或收益只来自换内部划分，就不支持模块有效。增加未来日期扰动和数值有限性诊断；不根据图形平滑度决定胜负。训练只有 51 个时间点，控制模块容量，单独消融周期及谐波数量；月粒度记录不能证明日/周信号有效，月份锚点也不能制造真实交易日期。检查面积/户型/城镇分层，防止把房源构成变化误解为价格机制变化。

### N4：TabPFN-3.5，上下文预测优先，微调可选

使用明确指定的 3.5 checkpoint 与官方回归接口；首轮固定权重，随后才在共同训练段中评估官方 `FinetunedTabPFNRegressor` 路径。权重、预处理和集成设置完整登记，不能让默认版本更新悄悄改变方法。3.5 微调从适配的 3.5 模型出发，不把旧版微调权重视为等价。查询期间保持模型/上下文固定，不随未来真实租金更新。

**待测假设：** 最新基础模型对结构化表格的先验可转移到时间外推任务，微调可能改善本地适配。**否证：** 先与 N2/N1 比较冻结版，再只改是否微调；分别报告训练与上下文/查询成本。若论文的总体排名无法在长窗口复现，接受本题差异。更高资源消耗允许存在，但必须有可复核的误差或实验信息收益，不能以“更新”代替选择理由。[官方模型文档](https://docs.priorlabs.ai/models)。

### N3：ModernNCA 学习距离的历史检索

按 [TALENT 官方实现](https://github.com/LAMDA-Tabular/TALENT) 核验类别、dtype、候选采样与纯推理接口。默认 `tabr_ohe` 常规流程将未知类别替换为训练众数，再稠密 one-hot；本题应明确适配训练过的 OTHER/UNK，避免把新楼当作最常见楼。该变化要记录，并与相同编码对照比较。

**待测假设：** 已见楼栋及相似物业的历史交易提供参数模型未充分利用的信息。**否证：** 同窗对 N1/树检查 RMSE、邻居月份组成和远期偏差；全历史与近期候选池作受控对照。固定 cutoff 的候选库只含该时点已知标签；训练排除自身，未来查询不补入真实验证标签。加趋势校正或改回归头是另一个变体，不得归为原版表现。

资源上，150,000 候选 × 50,000 查询意味着 **75 亿个距离配对**，这是形状算术而非耗时实测。当前方法每查询 batch 重新编码全候选；训练 SNS 抽样不降低原版推理候选数。减小查询 batch 仅降低峰值距离矩阵，不能消除全量稠密输入和总计算。候选缓存/分块是待开发优化，近似检索或候选子采样则改变估计器，应分开标注。详细源码与复杂度见 [检索审计](evidence/modern_retrieval_evidence.md)。

## 8. 共用清洗、辅助信息和验证协议

### 输入边界

所有方法从相同派生数据版本开始：NFKC、首尾/连续空格规范化、大小写统一、户型别名合并；以规范化城镇/街道/楼栋组成键。去掉已知常量 FURNISHED、FEE；ID 不作为可迁移属性。无 unit ID 的同 X 交易保留，租金尾部不因不常见而自动删除。楼龄是记录年份减租约起始年的代理，负值异常单列并采用明确的缺失/标记规则，不擅自修正原始值。

词表、频数阈值、缩放、分箱、目标标准化和任何编码均在每个训练折内拟合。稀有/未知楼栋回退须在训练中得到支持，不能给一个从未训练的随机 embedding 就称稳健处理。共同的是原始信息和评估协议，允许各模型采用其合理编码；额外机制归因需同编码对照。

第一轮只用核心字段。HDB 静态字段可独立加入，但 100% 连接率不证明 point-in-time 安全；NIL、−1 和建设年份异常按 EDA 规则标记。MRT 规划状态与学校/商场快照缺历史生效时间，先核实再纳入。COE/股票表包含晚于训练截止的实际数据，不能按目标月份直接连接；固定起点预测未来 16 个月时，简单 lag-1 仍可能读取截止日之后的真实观测。仅允许截止时已知信息或单独生成并验证的预测值，当前不生成此类预测。

### 外层时间窗口

下面样本数由 [validation_windows.py](../../experiments/research/validation_windows.py) 从已有 EDA 月份计数实际运行获得，无目标读取或拟合。

| 窗口 | 训练月份 | 训练行数 | 验证月份 | 验证行数 | 用途 |
| --- | --- | ---: | --- | ---: | --- |
| V1 | 2021-01–2024-03 | 114,842 | 2024-04–2024-09 | 17,960 | 近期开发 |
| VL | 2021-01–2023-05 | 84,902 | 2023-06–2024-09 | 47,900 | 16 个月长跨度压力检验 |
| V2 | 2021-01–2024-09 | 132,802 | 2024-10–2025-03 | 17,198 | 开发决策固定后的近期确认 |

V1 与 VL 的验证月份重叠，不能作为独立重复实验合并计算置信区间或重复加权。V2 的数据已经用于 EDA，所以只能称保留给算法确认的窗口，**不是完全未见的盲测试**；若此前已用它反复选模型，必须降低其确认性表述。2025-03 的 train/test 同月无法确定交易先后，历史验证采用整月隔离。

每折内部另留最后若干完整月份做早停/超参数选择，选择完成后用整个外层训练段重新拟合，未来验证期间不更新真实目标。基础模型上下文及检索库也服从相同 cutoff。外层 V1/VL 可用于开发比较，V2 确认前冻结配置；不得根据 V2 或测试特征不断调整日历周期与候选池。

主指标为原尺度 **RMSE（SGD/月）**，按同一窗口全体行计算；另报逐月/预测跨度 RMSE、平均残差、城镇/户型/租金尾部、已见/未见及稀疏楼栋的样本数。月份平均 RMSE 与总体 RMSE 分开命名。最终比赛测试没有目标，只做预测输入审计，不能产生测试 RMSE。

### 公平性、资源和依赖

保留 Ridge 诊断基线以及 CatBoost/LightGBM 强基线，它们不占五项现代候选。相同切分、相同允许信息、公开各自调参机会；无法做到等搜索空间时，同时报告默认配置和受控开发后的结果。不同方法的试验次数、失败/OOM、预处理、拟合、上下文构建、全部查询和峰值 RAM/VRAM都计入成本。少量种子先验证稳定性，扩展次数由团队预先决定，不自动照抄论文大搜索。

GPU 不是硬约束或效果指标。现有机器是否支持 CUDA/MPS、可用显存和是否租用云卡仍待团队决定；macOS 工作目录不能推出已有 NVIDIA GPU。基础模型预训练成本不由本项目重做，但本地权重加载与上下文推理必须计时；微调另列。没有依据承诺某卡容量、训练分钟数或付费金额。

后续实现分别需要 PyTorch 与 `tabm`、`tabicl`、`tabpfn`/其微调依赖、TALENT 和 temporal-modulation 仓库环境；这些应分环境锁定，记录 Python、包版本、commit、checkpoint 校验及设备。本轮未安装模型依赖。EDA 复现依赖仍见 [README_EDA.md](../../notebooks/README_EDA.md)。窗口计数仅需 Python 3 标准库：

```bash
python3 experiments/research/validation_windows.py
```

## 9. 判断、未解决问题与 B/C 交接

**RQ1：** 现代候选覆盖参数集成、历史检索、两条预训练路线及显式时间条件化，且均有公开实现依据；不是把旧方法换名。**RQ2：** EDA 支持高基数表示、相似交易和时间适应的研究动机，但尚未证明条件漂移、未来季节性或任何方法增益。**RQ3：** 当前可行性顺序来自独立评估，成本不再是准入门槛；效果与资源结果均待共同实验。

团队仍需确定 GPU/内存或租用预算、B/C 每周投入与库熟练度、进度报告前愿意完成的范围、基础模型权重获取，以及辅助快照可用边界。[PROGRESS_PLAN.md](../PROGRESS_PLAN.md) 已给出 2026-10-04 进度报告建议节点；最终报告具体截止日和团队可支配工时不能由此推断。新论文提供方法参考，不改变“未来信息只能在当时可得”的要求。

给 B/C 的交接清单：

- 共用 A 的规范化规则、楼栋键、保留交易/异常标记和按训练折拟合的编码；不要改原 CSV。
- 先建立核心字段、强基线与完整月份验证，再按 N1 → N5 → N2 → N4 → N3 分配现代路线；有成熟环境可登记调整顺序。
- B 可负责 N1/N5 同主干三臂对照；C 可负责 N2/N4 上下文路线。N3 在候选存储、纯推理和计算预算审计后排入，具体分工由团队确认。
- 比较楼栋输入、时间表示与辅助数据时一次改变一个信息组；N5 内部划分变化、N4 微调和 N3 候选池变化单独登记。
- 所有“更准确、更稳、更快、更省”的主张必须经共同 V1/VL/V2、原尺度 RMSE 和端到端资源记录检验。记录失败与负结果，不将可行性评分当实验结果。

本轮在调研与审阅材料处停止，没有训练模型、制作 proposal 幻灯片、提交 Kaggle 或提交/推送 Git。

## 参考文献

[1] Yury Gorishniy, Akim Kotelnikov, Artem Babenko. **TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling.** ICLR 2025。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html)。

[2] Han-Jia Ye, Huai-Hong Yin, De-Chuan Zhan, Wei-Lun Chao. **Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later.** ICLR 2025，方法名 ModernNCA。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ecd294194d5155432aab533f7ce0f6fe-Abstract-Conference.html)。2024 预印本题名不同，以正式版本为准。

[3] Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev, Daniil Shlenskii, Akim Kotelnikov, Artem Babenko. **TabR: Tabular Deep Learning Meets Nearest Neighbors.** ICLR 2024。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html)。

[4] Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan. **TabICLv2: A better, faster, scalable, and open tabular foundation model.** ICML 2026；本轮读取 arXiv v2（2026-09-16）。[版本页](https://arxiv.org/abs/2602.11139)。

[5] Léo Grinsztajn et al. **TabPFN-3: Technical Report.** 2026 技术报告，首发 2026-05-13，本轮读取 v2（2026-05-28）。[版本页](https://arxiv.org/abs/2605.13986)。

[6] Benjamin Jäger, Nick Erickson, Léo Grinsztajn et al. **TabPFN-3.5: Technical Report.** 2026 技术报告，首发 2026-09-15，本轮读取 v2（2026-09-22）。[版本页](https://arxiv.org/abs/2609.17895)。

[7] Ivan Rubachev, Nikolay Kartashev, Yury Gorishniy, Artem Babenko. **TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks.** ICLR 2025。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/571799482291411607c54984153190b0-Abstract-Conference.html)。

[8] Haorun Cai, Han-Jia Ye. **Understanding the Limits of Deep Tabular Methods with Temporal Shift.** ICML 2025，PMLR 267:6366–6386。[正式论文](https://proceedings.mlr.press/v267/cai25j.html)。

[9] Haorun Cai, Han-Jia Ye. **Feature-aware Modulation for Learning from Temporal Tabular Data.** NeurIPS 2025。[正式论文](https://papers.nips.cc/paper_files/paper/2025/hash/ddc20f403d0d296c87940393838d5971-Abstract-Conference.html)。
