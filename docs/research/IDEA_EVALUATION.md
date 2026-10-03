# 五个现代候选的独立 idea-evaluator 评估

评估日期：2026-09-26。依据 [研究 brief](RESEARCH_BRIEF.md)、[EDA](../EDA_REPORT.md)、[进度计划](../PROGRESS_PLAN.md)、[神经实现证据](archive/gpu/gpu_neural_evidence.md)、[基础模型证据](evidence/modern_foundation_evidence.md)、[检索证据](evidence/modern_retrieval_evidence.md)、[时间调制证据](evidence/modern_temporal_evidence.md) 及官方论文/代码。本文替代旧候选评估；[CPU 初评](archive/cpu/IDEA_EVALUATION_CPU_INITIAL.md) 仅供历史追溯。

**结论：五项均为 Accept with Revisions；当前尝试优先级 N1 > N5 > N2 > N4 > N3。** 原可行性顺序为 N1 > N2 > N5 > N4 > N3；按用户最新要求强调未来 RMSE，独立复核将 N5 提前。当前是条件性的实验投资顺序，不是预测精度排名，指标与晋级规则见 [最佳尝试清单](PRIORITY_PLAN.md)。GPU 已获允许，低成本是软偏好，旧 15/75 分钟准入门槛失效。实际 GPU、显存、地点和软件环境未知；本轮没有安装、训练、下载权重、生成预测、付费调用或提交，所有本项目 RMSE、耗时与峰值显存均未测量。

## 1. 第一印象与五项定义

**定位：New Setting，即现代方法在 HDB 固定起点租金预测中的应用研究；范式为 STEM 的比较实验与消融。** 不宣称新算法或全球首次问题。一句话故事：在高楼栋复用与最长 16 个月外推中，比较参数集成、预训练上下文学习、历史检索和时间条件化，辨认现代机制的真实收益及失败边界。

| ID | 主候选 | 本项目的具体角色 |
| --- | --- | --- |
| N1 | TabM，ICLR 2025 | 从头监督训练的参数共享 MLP 集成，作为现代可训练主干 |
| N2 | TabICLv2，ICML 2026 | 先用预训练回归 checkpoint 做 ICL，必要时将微调列为独立消融 |
| N3 | ModernNCA，ICLR 2025 | 学习表示/距离后，对训练候选标签作邻域加权预测 |
| N4 | TabPFN-3.5，2026-09 技术报告 | 本地 Base checkpoint 的 ICL 强对照；微调可选，不能混用服务变体成绩 |
| N5 | TabM + Feature-aware Modulation for Learning from Temporal Tabular Data，NeurIPS 2025 | 官方 `tabm_modulated` 的时间条件化输入调制，与相同版本 TabM 成对比较 |

Ridge、CatBoost、LightGBM 是共同基线，不占五项；小 MLP 和简单 KNN 仅用于机制诊断。N1 与 N5 共用家族有意用于隔离时间机制，不当作两次独立的“神经模型获胜”证据。发布日期满足范围要求，不能代替任务适配性。

## 2. 致命缺陷审计与最接近工作

按 F1「相对最接近工作增加什么」及 F6「什么实验能证实主张」检测。当前没有已成立的 CRITICAL；以下是应在最终论证前修复的两个缺口。EDA 不是模型比较结果，**没有候选已被数据反驳**。

| 缺口 | 严重性 | 具体修订 |
| --- | --- | --- |
| F1：模型/模块均为他人已发表或已公开的贡献，把采用新年份模型写成本团队方法创新会失实 | MINOR，课程应用定位可修复；不足以支持新方法论文定位 | 明确下表对象/设置差异；贡献写成受控比较、无未来信息的验证与机制失败分析 |
| F6：通用 benchmark、历史拟合和短窗成绩不能证实本题 16 个月泛化；N5 的收益也可能来自底座版本/训练协议变化 | MAJOR，需实质实验，不能靠文案修补 | 共用 V1/VL/V2；N5 同版三臂控制；N2/N4 冻结 checkpoint/上下文；N3 固定截止候选库 |

两轮增量检索覆盖“tabular foundation model regression scalability”“deep nearest neighbor tabular”“temporal tabular feature modulation”，再核读 primary paper 与作者代码。以下五个最接近工作中四项有正式会议发表，一项是新近技术报告；检索不等于穷尽，也不把“未检到本份数据”当原创性证明。

| 最接近工作、作者、年份 | 采用的既有机制 | 本项目真正不同的轴 |
| --- | --- | --- |
| [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](https://arxiv.org/abs/2410.24210)，Yury Gorishniy、Akim Kotelnikov、Artem Babenko，ICLR 2025 | 参数共享、多预测 MLP；N1 直接复用 | 对象/设置：HDB 月租金、高基数楼栋、固定起点长窗；机制本身相同 |
| [TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model](https://arxiv.org/abs/2602.11139)，Jingang Qu、David Holzmüller、Gaël Varoquaux、Marine Le Morvan，ICML 2026；v2 于 2026-09-16 更新 | 多样合成任务预训练、可扩展注意力与 ICL；N2 使用发布权重 | 对象/设置不同；不重做作者预训练，不宣称本团队发明先验或架构 |
| [Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later](https://arxiv.org/abs/2407.03257)，Han-Jia Ye、Huai-Hong Yin、De-Chuan Zhan、Wei-Lun Chao，ICLR 2025 | ModernNCA 的深表示与随机邻域采样 | 对象/设置：重复楼栋与未来租金；沿用检索机制，不能把旧 sklearn NCA 当完整等价实现 |
| [TabPFN-3.5: Technical Report](https://arxiv.org/abs/2609.17895)，Benjamin Jäger、Nick Erickson、Léo Grinsztajn 等，2026-09；v2 为 2026-09-22 | 预训练表格模型与非 IID 设置的评估 | 对象/设置不同；属于作者技术报告，不能写成已独立复现或同行评审的 HDB 结论 |
| [Feature-aware Modulation for Learning from Temporal Tabular Data](https://arxiv.org/abs/2512.03678)，Hao-Run Cai、Han-Jia Ye，NeurIPS 2025 | 时间表示驱动逐特征平移、缩放与形状变换；已有 TabM 变体 | 对象/设置：51 个历史月份到 16 个月未来；组合已有官方实现，不是本团队首次把该模块接入 TabM |

这些论文支持候选值得检验，不能把跨数据集平均名次填入本项目成绩。N3 的主要原实验采用随机切分；N5 已研究时间漂移，仍不等于已覆盖本题跨度。N4 报告 §2.2 明确 tuned/ensembled MLP 在 grouped、temporal、large 切片仍领先，≤100k 子集结果不能直接套到 150k；本地 Base 也不能借用 Thinking 成绩。N2 对较早 RealTabPFN-2.5 的作者比较不能推出胜过随后发布的 3.5。[TabPFN-3.5 全文](https://arxiv.org/html/2609.17895v2)

## 3. 生命周期与能力匹配

| 方面 | 已知输入 | 评估 |
| --- | --- | --- |
| 类别 | 四人课程团队，EDA 完成，目标是比较现代方法 | Application research；不是从头预训练基础模型或开发新理论 |
| 生命周期 | 2025–2026 方法/实现仍在更新 | 技能框架的 3–6 个月是应用研究参考；不是本项目到期日或必须投入时长，团队须结合领域节奏判断 |
| 里程碑 | 进度报告建议 2026-10-04；最终报告截止未知 | 距本轮约 8 天，不承诺此前完成五项全套实验；可报告已完成结果和待验证设计 |
| 周工时 | 未提供 | 不虚构团队总工时，无法计算可靠交付日期 |
| 技能与偏好 | Python/PyTorch、统计、GPU 系统经验和个人能力等级未提供 | N1 训练工程、N2/N4 上下文资源、N3 检索工程、N5 时间建模各需负责人；理论/应用偏好未知 |
| 硬件 | GPU 被允许，型号/显存/环境/地点未定 | “允许”不等于“已具备某块卡”；也不等于确认无法获得设备 |
| 综合 | 数据可用，能力和资源不完整 | **Yellow，条件性匹配**；无不可修复能力错配证据，先确认实际执行条件 |

## 4. 五维评分：从 5 起算

每项维度从 5 起，括号为移动量。全部为**机制型、尚未经本项目数据确认**的潜力判断；没有依据则留在 5。comparator 与验证实验逐项注明，不能把不同 comparator 的分数相加或当速度/精度排行榜。高分不替代第 6 节的否证实验。

| 候选 / 维度 | 分数 | 机制、比较对象与验证 |
| --- | --- | --- |
| N1 Higher | 7（+2） | 对同输入 CatBoost/LightGBM，数值表示与共享 MLP 集成提供不同连续交互假设；E1 检验增益并隔离表示与集成 |
| N1 Faster | 8（+3） | 对同规模独立多 MLP 集成，共享参数与批量多头减少重复工作；不是快于单 MLP/树的承诺，E1 检验相当误差下端到端时间 |
| N1 Stronger | 6（+1） | 对单 MLP，多预测平均有减少个体波动的机制；E1 检查月份/必要重复的稳定性，不等同抗长期漂移 |
| N1 Cheaper | 7（+2） | 对独立多 MLP，减少权重副本与独立训练维护；激活/搜索仍有成本，E1 检验同效果总成本 |
| N1 Broader | 5（0） | 本设计无跨地区/任务验证；原方法通用性不是本团队新贡献 |
| N2 Higher | 8（+3） | 对树/N1，多样任务预训练先验结合本表标签上下文提供不同函数推断机制；E2 在完整时间窗检验，不用论文平均排名替代 |
| N2 Faster | 6（+1） | 对从头训练 N1，冻结权重 ICL 可省本地梯度迭代；上下文前向可能抵消收益，E2 必须计 fit、缓存和全部 predict |
| N2 Stronger | 5（0） | 多样先验不直接保证未来市场鲁棒，尚无本题优势依据；E2 检查 VL、尾部与未知楼栋 |
| N2 Cheaper | 6（+1） | 对从头训练/搜索 N1，复用 checkpoint 减少任务专用优化；不减少本题标签需求，E2 计内存/offload与实际总投入 |
| N2 Broader | 5（0） | ICL 的既有跨任务能力属于原工作，本题没有新跨域实验 |
| N3 Higher | 7（+2） | 对简单 KNN 与 N1，学习距离可找到更相关历史合同；99.796% 测试行楼栋已见是动机，不是收益证据；E3 检验检索增量 |
| N3 Faster | 4（−1） | 对 N1 参数预测，官方每查询批扫描全训练候选且重算表示；训练采样不消除推理成本，E3 校准全量流程 |
| N3 Stronger | 4（−1） | 对无此硬边界的 N1，原版回归输出受候选标签凸包约束，对超出历史范围的目标有限制；E3 检查远期水平和尾部。N1 也未保证正确外推 |
| N3 Cheaper | 4（−1） | 对 N1，额外保留候选库、稠密类别输入及检索工程；E3 计训练和查询成本，不以超旧额度判无效 |
| N3 Broader | 5（0） | 仅当前租金任务，无跨域增益依据 |
| N4 Higher | 8（+3） | 对树/N1/N2，扩展的预训练表示及高基数/非 IID 适配提供竞争假设；E4 用同信息本地 Base 检验，不混入 Thinking 成绩 |
| N4 Faster | 5（0） | 相对同为 ICL 的 N2，未给出本任务净提速依据；E4 测完整上下文构建与 50k 查询 |
| N4 Stronger | 6（+1） | 对仅在本表从头拟合的 N1，预训练表示及报告中的 temporal/grouped 证据支持有限鲁棒性假设；E4 验证本题 16 个月边界 |
| N4 Cheaper | 5（0） | 相对 N2，没有降低资源或接入总成本的依据；本地权重可用不等于零计算成本 |
| N4 Broader | 5（0） | 采用原模型通用能力，不构成本项目跨域创新 |
| N5 Higher | 7（+2） | 对同版 N1/仅时间 embedding，逐特征条件变换可拟合随时间改变的属性关系；E5 隔离模块与主干/协议差异 |
| N5 Faster | 4（−1） | 对同版静态 TabM 增加时间编码与调制模块，没有提速机制；E5 记录真实增量，不能套用相对完整超网络的效率宣传 |
| N5 Stronger | 8（+3） | 对同版静态/时间 embedding TabM，显式时间条件化直接针对映射漂移；E5 是决定性验证，51 个月拟合与远期延伸仍有风险 |
| N5 Cheaper | 5（0） | 相对静态 TabM 没有总方案降本依据；模块轻量不代表额外消融/维护无成本 |
| N5 Broader | 5（0） | 只在本题研究时间适应，没有跨领域统一实验 |

重点是 **Higher** 与 **Stronger**；N1 的 **Faster/Cheaper** 只相对独立多 MLP 集成。N2/N4 的强潜力和 N5 的时间机制值得实测，均不是“必胜”判断；N3 的已知结构限制不是数据已反驳，不能据此触发 Reject and Pivot。

## 5. 四项范式变化探针

| 问题 | 判断 | 理由 |
| --- | --- | --- |
| First Principles：是否挑战隐含假设？ | Partial | N5 对固定属性映射、N2/N4 对每任务从头训练提出替代；这些改变来自原方法，本项目检验其适用边界 |
| Elephant in the Room：是否解决社区回避的重大问题？ | No | 时间漂移是已被研究的问题，本项目未证明社区回避或自己已解决 |
| Technology Cycle：近期能力是否使方案现在才可行？ | Yes | 2025–2026 公开回归 checkpoint、规模化 ICL 与时间调制实现扩大可检验范围；GPU 允许本身不是新发明 |
| Hamming's Rule：成功是否改变领域优先级？ | No | 单份 HDB 租金比较有应用价值，但不足以重排整个领域议程 |

Yes=2、Partial=1、No=0，总计 3/8：增量研究，有技术更新带来的探索空间，尚不足以称范式突破。只有一个 Yes，不满足两项肯定的突破潜力判断。

## 6. 可行性与实验约束

| 风险 | 当前水平 | 缓解与边界 |
| --- | --- | --- |
| 计算 | N1 中；N5 中高；N2/N4 全量上下文条件性高；N3 全库检索高 | 在真实设备校准，分别记录输入、缓存、激活、RAM/显存与端到端时间；不给显存或分钟数保证 |
| 数据 | 核心低；历史设施/宏观可得性高 | 150k 标签和 9,406 楼栋已可读；设施快照不等于历史可用；固定起点后真实经济值禁入 |
| 工程 | N1 成熟入口；N2/N4 有 API 但资源/权重需核验；N3/N5 适配较多 | 固定 commit/checkpoint、纯推理接口、类别回退和时间预处理；方法改动与等价优化分开登记 |
| 时间 | 10/4 前全套实验存在风险，最终截止与周工时未知 | 先交共同协议、基线和已完成模型结果；未完成候选仍可进入后续计划 |
| 成本 | 软偏好 | 误差、信息增量、工程与真实计算成本共同决定追加投入；昂贵、CPU 不便或超旧额度不构成自动淘汰 |

### 6.1 各方案必须落实的实现边界

**N1。** 官方 TabM 的 `cat_cardinalities` 默认 one-hot，`num_embeddings` 是数值表示；不要声称默认学习紧凑楼栋 embedding。按批编码避免全表稠密化；训练平均各 head loss，推理平均预测。预留但从未训练的 UNK 输入/向量不能当可靠回退，可用训练中的稀有类共享桶或受控 mask，并保留城镇/属性输入；阈值、词表与缩放在 inner-train 拟合，inner-validation 选择后才重拟合外层训练段。[官方实现](https://github.com/yandex-research/tabm)

**N2。** 首轮是预训练回归权重的 ICL，`fit` 储存/处理有标签上下文，主要计算可在 `predict`，缓存会重新分配二者工作量；作者预训练成本不计成本项目训练时间。代码支持缓存、offload 和 `FinetunedTabICLRegressor`，微调另有梯度优化与早停义务。百万行论文实验的 H100/offload 条件不证明 150k 可装入未知消费卡，缩小 ensemble batch 也不消除完整上下文成本。[官方 API/流程](https://github.com/soda-inria/tabicl)、[论文附录 H](https://arxiv.org/html/2602.11139v2)

**N3。** 当前官方实现是欧氏距离经 softmax 后乘候选 y，回归没有独立预测头；非负权重和为一，反标准化后仍在候选标签范围内。训练排除来源行自身，其他索引的相同 X 不自动排除；无 unit ID，不能据此删重。默认 `tabr_ohe` 将未知类别映射到训练众数，不是原生楼栋 embedding/可靠 UNK，应显式适配受训练的共享桶。预测每查询批仍用全部训练候选，未内置候选缓存/分块/ANN；150k×50k 是 75 亿配对的形状算术，不是耗时预测。[官方模型](https://github.com/LAMDA-Tabular/TALENT/blob/main/TALENT/model/models/modernNCA.py)、[方法层](https://github.com/LAMDA-Tabular/TALENT/blob/main/TALENT/model/methods/modernNCA.py)

N3 如需等价分块，必须保留全局 softmax 分母，不能各块 softmax 后平均；缓存是待适配工程，不冒称现成能力。候选抽样、top-k/ANN、加趋势或换预测头都改变估计器，另列消融。外层验证候选库只能来自截止前训练，固定起点期间不补入验证标签；同一 cutoff 的监督训练使用完整训练段邻居并排除自身，不自动等于泄漏，也不能冒称逐行历史在线回放。

**N4。** 官方模型表的 TabPFN-3.5 Base 可本地运行，最多 1M 训练行，行列上限需联合考虑；不能用 v2 的 10k 或旧版 50k 限制排除本题。支持微调，首轮先冻结 Base 权重做 ICL；微调单独用内层验证，不等同从头预训练。权重访问需要登录并接受许可，无界面服务器可用 token 认证；这不意味着本地租金必须上传，本轮未下载或传输数据。Base/Fast/Plus/Thinking 的访问、资源与成绩分开记录。[模型表](https://docs.priorlabs.ai/models)、[微调文档](https://docs.priorlabs.ai/capabilities/fine-tuning)

**N5。** 官方仓库已有 `tabm`、`tabm_temporal`、`tabm_modulated`；先采用已核验的输入调制路径，类别仍 one-hot，宽度会进入调制成本。普通独立 TabM 包与该仓库主干可能不同：同版静态、仅时间 embedding、调制三臂必须对齐；当前最新版 N1 可另报管道结果，不能混成模块消融。所有时间尺度与预处理只从相应训练段拟合，未来推理仅接入可预知日历 t 和合法属性，不能依赖未来分布统计。[官方仓库](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation)

源码审计的 [commit 8feabbd](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation/tree/8feabbd580d59757dd108622ea93b9bdb6d38fa1) 中，N5 强制使用 `indices`，其时间均值/标准差来自原始 `M['train']`；重新划分内部验证后可能仍包含被移入内部验证的行，须覆盖为当前训练折统计。`period/period_t` 的分片归一化不属于 N5 实际路径。这是特征统计与验证协议问题，不能称为未来目标泄漏。

N5 的 150k 行不提供 150k 个独立时间点，只有 51 个历史月份；月份变化不证明条件属性关系必然漂移。日历输入可知、Fourier 可延伸都不保证未来 16 个月正确；静态普通模型接收月序也不保证。作者的大规模调参、种子数与内部 `holdout_foremost_sample` 不能直接替代本项目统一协议。

独立代码复查还要求统一时间单位：官方 Fourier 周期以秒定义，不能输入月序却沿用秒周期。月份锚点与时区需明确，或同步修改周期单位；在 E5 中消融周期与谐波容量，避免月粒度数据被人为锚点引入无依据的日/周信号。[时间表示源码](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation/blob/8feabbd580d59757dd108622ea93b9bdb6d38fa1/model/lib/temporal_embeddings.py#L95-L126)

### 6.2 共同窗口与指标

行数来自既有 [validation_windows.csv](../../experiments/research/validation_windows.csv)，本轮未重跑 EDA。

| 窗口 | 训练范围 / 行数 | 验证范围 / 行数 | 角色 |
| --- | --- | --- | --- |
| V1 | 2021-01–2024-03 / 114,842 | 2024-04–2024-09 / 17,960 | 近期开发 |
| VL | 2021-01–2023-05 / 84,902 | 2023-06–2024-09 / 47,900 | 固定起点 16 个月压力 |
| V2 | 2021-01–2024-09 / 132,802 | 2024-10–2025-03 / 17,198 | 有限选择后算法确认 |

每个外层训练段内部留完整较后月份做选择/早停；词表、目标标准化、数值分箱、时间尺度、候选库、checkpoint 微调和调制均只用该折合法历史。外层不能选 epoch 或回退阈值，预测中途不回流真值。V1/VL 验证重叠，不能当独立重复或合并增权；VL 更早市场阶段不能保证复制真实未来。EDA 已观察 V2，故其不是分析者盲测；若用于调参则降为开发窗。

2025-03 两集重叠且月内先后未知，边界月另报，显式历史目标统计只用严格更早月份。共同输出是完整验证分布的 SGD RMSE（总 SSE/行数再开方），辅以 MAE、月份/horizon 均值偏差、城镇/户型、尾部、已见/未见楼栋的样本量和误差；全测试只有 102 行未见楼栋不能外推为每个历史折同样少。统一输入组与信息量，分别记录单配置和搜索累计成本。

### 6.3 决定性实验及否证条件

| 实验 | 最小受控比较 | 会削弱或否证什么 |
| --- | --- | --- |
| E1 / N1 | 同输入树基线、同数值/类别表示单 MLP 对 TabM；若主张集成效率，再用受控独立多 MLP 作 comparator | 仅数值编码改善不能归给共享集成；未同时满足误差与成本条件则撤回效率主张 |
| E2 / N2 | 固定回归 checkpoint/ensemble/上下文，对 N1 和树；缓存/精度只作明确控制，微调另列 | 小样本成功不证明全量可行；全量资源超实际可得范围时调整规模/路线并披露，不宣称模型无效；无稳定误差或信息增量则停止扩张 |
| E3 / N3 | 同输入 N1、简单 KNN；核验自身排除，记录邻居月份/楼栋、预测范围；全历史/近期候选控制需报告覆盖损失 | 如果只复用旧价格形成长窗偏差、检索无额外收益或实际资源不可接受，则拒绝相应收益主张或改有标记的变体 |
| E4 / N4 | 固定本地 Base 对 N2/N1/树，同截止与合法字段；ICL 和微调成绩分开 | 不借 Thinking/其他版本成绩证明 Base；若官方上限内仍无法在实际资源完成，报告工程边界；时间/高基数收益须本地出现 |
| E5 / N5 | 同版静态 TabM、仅时间 embedding、modulated；相同输入/训练协议，V1/VL 后有限 V2 确认 | 仅更强主干、不同验证或更大搜索获得收益不支持调制；只短窗改善而远期偏差加重则撤回长期适应主张 |

不设虚构最小提升百分点，不用最佳 seed 冒充稳定性；只有近似并列、波动或新问题才加必要重复。上表是未来实验决策，没有把未知写成已失败；若干净实验确实反驳核心机制，保留负结果并改变主张。

## 7. Reviewer verdict 与行动顺序

| 顺位 | 候选 | Verdict | 开始前最关键的修订 |
| ---: | --- | --- | --- |
| 1 | N1 TabM | **Accept with Revisions**，值得推进，待 E1 | 正确多头训练、默认 one-hot 与折内表示、真实设备校准 |
| 2 | N5 时间调制 TabM | **Accept with Revisions**，值得推进，待 E5 | 复用同版主干，三臂消融与严格时间尺度，优先验证长窗偏差 |
| 3 | N2 TabICLv2 | **Accept with Revisions**，值得推进，待 E2 | 锁定回归权重与全量上下文路径，计完整 fit/predict/offload 成本 |
| 4 | N4 TabPFN-3.5 | **Accept with Revisions**，值得推进，待 E4 | 完成权重访问与设备核验，区分 Base/服务变体及 ICL/微调 |
| 5 | N3 ModernNCA | **Accept with Revisions**，值得推进，待 E3 | 类别回退、候选库/自身排除、纯推理接口与全量检索资源校准 |

N1 的训练规模可通过 batch 调节且能支撑 N5；N2 的公开回归 API 提供与从头训练不同的参照；N5 直击时间问题，可在 N1 跑通后提前并行；N4 很新且有竞争力，但权重/全量资源与具体变体需核验；N3 的全库检索和官方稠密路径需要最多专门工程。该排序会随真实设备与团队熟练度改变，不能解释为较后模型精度较差。

前三项行动：

1. 冻结时间协议、信息可得边界、统一特征组和四人负责人；确认周工时、实际 GPU 与最终截止，先建立共同基线记录。
2. 下一实验阶段优先 N1 与同版 N5 的受控校准，N2 可由另一负责人并行准备；完整记录真实误差、时间、内存和工程投入，取消旧时间准入门槛。
3. 再按资源与新增信息推进 N4、N3；用 E1–E5 区分编码、预训练、检索和时间条件化贡献，保留失败分析，不把新年份本身写成有效性证据。

未授予 Strong Accept 的原因是能力/资源仍为 Yellow、收益尚待对应实验且缺完整生命周期匹配，并非禁止 GPU 或要求最低成本。
