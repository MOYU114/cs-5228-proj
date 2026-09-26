# GPU 表格神经网络：增量证据与实施边界

检索日期：2026-09-26。依据 [RESEARCH_BRIEF.md](../../RESEARCH_BRIEF.md)、[EDA_REPORT.md](../../../EDA_REPORT.md) 与既有 [critics_evidence.md](../cpu/critics_evidence.md)。沿用最新约束：GPU 可以使用，低成本仅为软偏好。本轮只核验文献和官方代码；未安装依赖、训练、生成预测或改动原始数据。

## 摘要与问题

本轮检查三个问题：150k 回归能否合理采用 TabM / FT-Transformer；高基数楼栋与数值字段如何实际接入；GPU 能解决哪些计算问题、不能解决哪些未来预测问题。证据支持把 **TabM 与 FT-Transformer 均纳入主五项，TabM 优先级更高**。小型 MLP / ResNet 加数值嵌入适合作机制消融与运行诊断，不必再独占主候选名额。这个排序是实验信息量与实施判断，不是 HDB 准确率排名。

本地 EDA 的约束是：训练 150,000 行、10 个原始输入字段，其中两个常量；规范化三字段楼栋键有 9,406 类。测试 49,898/50,000 行属于训练已见楼栋，另有 50 栋、102 行未见楼栋。训练截止 2025-03，测试到 2026-07，最大间隔 16 个月，且两集都包含 2025-03。没有 unit ID；相同全部 X 仍可能对应不同租金，不能把楼栋 embedding 当作单元身份识别，也不能据此去重。

## 检索与核验方法

两轮有界检索：第一轮定位 TabM、Revisiting Deep Learning、Numerical Embeddings 三篇原论文及作者仓库；第二轮聚焦 GPU 硬件、类别编码、训练范例与时间漂移反例。代表 queries 为 `TabM parameter efficient ensembling official github`、`Revisiting Deep Learning Models for Tabular Data FT Transformer official`、`On Embeddings Numerical Features official`、`TabM GPU training time`、`TabReD temporal splits`。随后只打开命中的论文、官方 README 与源码核对细节。另一个代理独立复核数值分箱和未知类别边界。

四篇核心论文均完成标题、作者、版本/会议和所用论断核验，评级 VERIFIED。引用的是作者实验，非独立 HDB 复现；前三篇主要来自同一研究团队，不能算三次独立复现。没有以 Kaggle 名次、第三方博客或搜索摘要证明本题性能。

## 1. 模型与编码机制的证据

| 已核验 primary paper | 原研究支持什么 | 对本题的条件性含义 |
|---|---|---|
| P1. Yury Gorishniy, Akim Kotelnikov, Artem Babenko. **TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling.** ICLR 2025。[论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf)、[arXiv](https://arxiv.org/abs/2410.24210) | 参数共享的 MLP 集成，在作者多数据集评估中取得较好的性能/效率权衡；评估包含 TabReD 时间切分。 | 比“神经网络一定复杂昂贵”更合适的反例；支持正式试验，不直接证明 HDB 最佳。 |
| P2. Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko. **Revisiting Deep Learning Models for Tabular Data.** NeurIPS 2021；arXiv v5 更新于 2023。[论文](https://proceedings.neurips.cc/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf)、[元数据](https://arxiv.org/abs/2106.11959) | ResNet 是强基线；FT-Transformer 用 feature tokens 和 attention 建模，在其 benchmark 中较强，但通常比 MLP / ResNet 耗费更多训练资源。 | 面积、户型、楼龄和位置的交互值得检验；attention 不是收益保证，更不是时间序列建模的同义词。 |
| P3. Yury Gorishniy, Ivan Rubachev, Artem Babenko. **On Embeddings for Numerical Features in Tabular Deep Learning.** NeurIPS 2022。[论文](https://proceedings.neurips.cc/paper_files/paper/2022/file/9e9f0ffc3d836836ca96cbf8fe14b105-Paper-Conference.pdf)、[元数据](https://arxiv.org/abs/2203.05556) | 分段线性与周期数值嵌入可改善多个 backbone；数值编码后的 MLP 类模型可接近 attention 模型。 | 编码与 backbone 应分开比较，不能把编码收益全部记作 Transformer 的功劳；周期嵌入也不证明租金存在稳定年度季节性。 |
| P4. Ivan Rubachev, Nikolay Kartashev, Yury Gorishniy, Artem Babenko. **TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks.** 2024 preprint / ICLR 2025。[论文页](https://arxiv.org/abs/2406.19380)、[会议页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/571799482291411607c54984153190b0-Abstract-Conference.html) | 八个工业数据集反映时间漂移与特征工程；时间切分会改变模型排序，简单 MLP 类和 GBDT 表现较强。 | 本题需要长跨度向前验证；P2 的较强平均成绩与 P4 的结论不矛盾，因为数据与验证条件不同。 |

P1 支持先试 TabM，P2 支持保留不同交互结构的 FT-Transformer，P3 要求认真处理数值输入，P4 限制所有随机切分成绩的外推范围。三者不能合并成“NN 已全面击败树”的结论。

## 2. 官方实现路径与高基数楼栋

以下是查阅日的官方推荐入口，而不是安装或运行记录；正式实验应固定 release/commit 和 PyTorch/CUDA 版本。

| 方法 | 已核验入口 | 本题必须处理的差别 |
|---|---|---|
| TabM | [官方仓库](https://github.com/yandex-research/tabm)、[example.ipynb](https://github.com/yandex-research/tabm/blob/main/example.ipynb)、[当前源码](https://github.com/yandex-research/tabm/blob/main/tabm.py)。实际应用推荐根目录 `tabm` package；论文复现材料在 `paper/`。 | `TabM.make` 的 `cat_cardinalities` **默认 one-hot，不是类别 embedding**。`num_embeddings` 指数值嵌入。高基数不能未经处理就声称“官方自动学习紧凑楼栋 embedding”。 |
| FT-Transformer / ResNet | [作者仓库](https://github.com/yandex-research/rtdl-revisiting-models)、[package README](https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/README.md)、[完整范例](https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/example.ipynb)。官方推荐 `rtdl_revisiting_models`，而非直接复用旧论文环境。 | `FTTransformer` 分别接收 `x_cont, x_cat`，原生每类别字段一张 embedding 表。ResNet 接收拼接后的连续表示；类别编码要自行准备。 |
| 数值嵌入 | [rtdl-num-embeddings](https://github.com/yandex-research/rtdl-num-embeddings)、[package README](https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/README.md)、[源码](https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py) | `compute_bins` 与 `PiecewiseLinearEmbeddings` 可用于 TabM 或 MLP；改 FT-Transformer 数值 tokenizer 需要显式适配，不能把 PLE 误称原版默认。 |

**楼栋编码建议（本项目设计推断）**：词表只在当前训练折拟合，使用规范化 `TOWN+BLOCK+STREET`，不把 `BLOCK` 当连续整数。FT-Transformer 用预留 UNK 的楼栋表，同时保留城镇、街道/位置与房屋属性供共享信息；9,406 类仍只是**一个楼栋 token**，不是 9,406 个 attention tokens。TabM 先保留官方 one-hot 作透明起点；若其开销不合适，再通过官方自定义输入/backbone 接口接入紧凑楼栋 embedding，并明确标成适配变体，与同输入 one-hot 作消融。

两个官方 API 都要求类别索引位于合法范围，没有自动理解未见楼栋的机制。仅新建一个从未收到梯度的 UNK 向量会留下随机回退。建议在训练折内把罕见楼栋映射到共同桶，或训练时以受控概率屏蔽楼栋到 UNK，使回退有训练数据；同时保留更粗层级输入。阈值/屏蔽率需由历史验证选择，并比较不使用楼栋 ID 的对照。楼栋 dropout 不能凭经验直接声称有效。实现依据见 [FT 类别模块](https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/rtdl_revisiting_models.py) 与 [TabM 类别模块](https://github.com/yandex-research/tabm/blob/main/tabm.py)。

## 3. 数值编码与时间外推

面积、租约年份/楼龄、月序及合法加入的地理字段可采用训练折标准化；常量移出数值分箱。目标可用训练折均值/标准差标准化，最后反变换到 SGD 评 RMSE。首轮无需同时搜索全部编码：TabM 比较线性/无嵌入与分位数 PLE；FT-Transformer 先用原生线性 tokenizer；小型 MLP + 同一 PLE 用于判断收益来自编码还是 backbone。

当前 `compute_bins` 默认使用训练 X 的分位数，不使用 y；指定 `tree_kwargs` 才进入监督分箱并要求 y 与任务类型。当前 PLE 推荐 `version="B", activation=False`，TabM 源码亦要求 B 版本；没有 `extrapolate=True` 这个构造参数。源码在两端允许编码继续线性延伸，不能误述成所有越界值都裁剪在训练范围。以上为 [官方数值实现](https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py) 与 [TabM 范例](https://github.com/yandex-research/tabm/blob/main/example.ipynb) 的核验结果。

这些编码行为不等于租金外推正确。更具体地说，模型能接收超过训练范围的月序，只说明函数有输出；输出斜率、波动与市场结构是否合理仍未知。把年月整体当离散类别会把未来月份送进未知桶；周期表示能表达重复形状，却不会自动续接市场水平上涨。建议直接网络与显式趋势分解使用相同时间窗口比较；若使用趋势残差目标，趋势拟合与残差构造也必须限于每折可用历史。

FT-Transformer 的 attention 连接同一行的字段，原版不会读取跨月份交易序列；没有 unit ID 时更不能假装拥有逐户轨迹。按楼栋聚合的时间序列是另一个建模任务，需重新处理样本构成与稀疏月份。GPU 也不能让预测起点之后的真实宏观数据变成合法输入。

## 4. GPU 批训练、资源与公平比较

150k 是小批量优化的样本总量，并非必须同时驻留 GPU 的上下文长度；可以逐批搬运数据，50k 预测也可分批。TabM 输出形状为 `B×k×1`；训练应平均各 head 的 MSE，**不能先平均预测再计算训练损失**；推理才平均 head。官方范例提供 CUDA/CPU、AdamW、早停和可选 AMP / compile 路径；其示例将数据张量放到设备上不是本题必须照抄的内存策略。[TabM 官方训练范例](https://github.com/yandex-research/tabm/blob/main/example.ipynb)

资源只作透明估算，不承诺显存或分钟数。以下取“9,406 已知楼栋 + 1 个 UNK”作满训练词表示例：

- 只算楼栋 embedding 权重，维度 16 是 `9,407×16×4≈0.57 MiB`，维度 192 是约 `6.89 MiB`（FP32）；不包括梯度、优化器、其他层和激活，不能拿此数当峰值显存。
- TabM 若先物化全训练楼栋 one-hot，单 FP32 矩阵约 `150,000×9,407×4≈5.26 GiB`；按 batch=256 生成仅该矩阵约 `9.19 MiB`，但后续 `B×k×D` 运算仍会扩大激活。应保留整数类别，按批编码，避免先物化全表稠密 one-hot。
- FT-Transformer 的 attention 尺度取决于字段 token 数、宽度和层数；增加成百上千个派生字段会改变成本，类别基数本身主要扩大 embedding 表。TabM 的 k、宽度与批大小则直接影响并行激活。以上均为结构推断，非测量。

P1 的效率图多数训练使用 **NVIDIA A100**，另有 2080 Ti 实验；CPU 单样本吞吐与 GPU 批吞吐是不同测量。论文明确将效率比较限定为所选配置和硬件，不能换算为本题的消费级 GPU 耗时。[P1 §4.3、附录 B.3/D.1](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf)

后续若获准训练，应先记录设备、可用显存、软件版本，再做一个受控配置的校准，报告峰值 allocated/reserved、每 epoch 时间、最佳 epoch 与总墙钟时间。8/12/16 GB 只能列作**待验证设备档**；从小 batch 起，逐级增加，OOM 时降低 batch/宽度/k。AMP 是否支持、是否稳定、compile 的预热开销都要测；不能把 CUDA 示例视为已验证 Apple MPS 路径。需要缩小配置时同时记录模型改变，不能只报“更快”。

公平性采用统一输入组、原尺度 RMSE、整月向前验证、近似 12–16 个月长窗，以及每折已见/未见楼栋分层；2025-03 的月内先后不能猜测。早停在训练段内部的较后月份完成，外层评估窗口用于比较固定管线。主模型报告单次配置与实际搜索总成本，可用相同试验次数或预先说明的预算；低成本软偏好不再转换成旧 15/75 分钟准入门槛。

## 5. 给五主候选的建议与否证条件

| 神经候选 | 建议位置 | 保留理由与可否证条件 |
|---|---|---|
| TabM + 折内数值处理 | **主五项，神经方法优先** | 共享 MLP 集成提供与树不同的机制，近期文献与官方实现可复查。若多个时间窗口都不优于同输入树/小 MLP，且没有互补误差价值，应停止扩展搜索。 |
| FT-Transformer | **主五项，排在 TabM 之后作为结构对照** | 字段少、高基数类别可用原生 embedding，GPU 放开后值得实证检验；不能因通用效率排名就判定本题不可行。若同输入 TabM 稳定优于它、远期误差没有改善，则降为储备。 |
| MLP / ResNet + PLE 或 periodic embedding | **诊断与消融，不另占主五项** | 与前两项共享大量预处理，适合检查编码收益、训练管线与复杂度价值；没有证据要求同时开展完整三类神经架构大搜索。 |

因此，与成熟树及时间机制合并时，建议主五项为 **CatBoost、LightGBM、TabM、显式时间趋势 + 固定物业残差树、FT-Transformer**。Ridge 保留最小诊断基线，EBM 保留解释性储备；这里仅对神经候选作证据判断，不替代树模型和趋势方案的独立审查。

RQ1：150k 与少字段使两种神经网络具有合理试验价值，高基数需要不同编码策略。RQ2：GPU 降低批训练成本，不解决市场外推、未知楼栋学习不足或不可观测单元差异。RQ3：TabM 和 FT-Transformer 应进入主五项，小型 MLP / ResNet 是辅助对照；排名只能在共同时间协议下由本地结果确认。

尚未解决的边界：未检索到直接覆盖这份 HDB 数据、9,406 楼栋词表和未来 16 个月的模型比较；设备尚未提供，峰值显存与耗时均无本地实测。论文平均排名和示例 notebook 训练结果均不填补这些空白。
