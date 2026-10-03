# ModernNCA：现代检索表格回归的证据与适配边界

检索日期：2026-09-26。范围为增量文献与官方源码核验；未安装、训练、生成预测或改动 CSV。依照最新要求，主候选聚焦现代方法，经典树、线性模型与 FT-Transformer 只作 baseline/储备。本文件仅决定 ModernNCA 的候选价值，不代替 TabICLv2、TabPFN-3.5 或 temporal modulation 的独立审查。

## 摘要与研究问题

问题是：ModernNCA 是否提供区别于 TabM 的有价值机制；150k 历史记录与 50k 查询下检索是否可实施；同楼栋重合与 16 个月预测间隔会放大哪些收益或失效条件。证据支持 **ModernNCA 进入新的五个主候选，作为学习距离的历史检索模型**，但不能将其包装成开箱即用的低显存方案或自动时间外推模型。TabR 保留为检索结构的相邻参照，不另占五个名额。

本地事实来自 [EDA_REPORT.md](../../EDA_REPORT.md)：训练 150k 行，规范化楼栋 9,406 类，测试 50k 行中 99.796% 的行所属楼栋在训练出现；测试 50 栋、102 行未见楼栋。无 unit ID，相同全部 X、不同 y 的训练组有 17,900 个。训练到 2025-03，测试到 2026-07；两集均有 2025-03，不能假定月内先后。

## 检索与引用核验

使用 deep-research 流程的两轮有界增量检索。首轮查正式题名、会议及作者代码；次轮查 `ModernNCA candidate_y sample_rate`、`ModernNCA unknown_value onehot`、`TabR memory_efficient candidate`，并打开论文全文及命中的源码。另一个代理独立复核 TALENT 的候选池、编码和推理路径。三篇 primary works 均核实题名、作者、年份与所用结论，评级 VERIFIED；不采用第三方分数或搜索摘要证明本题 RMSE。

| 已核验研究 | 机制和证据 | 对本题的边界 |
|---|---|---|
| Han-Jia Ye, Huai-Hong Yin, De-Chuan Zhan, Wei-Lun Chao. **Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later.** ICLR 2025。[会议页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ecd294194d5155432aab533f7ce0f6fe-Abstract-Conference.html)、[全文](https://proceedings.iclr.cc/paper_files/paper/2025/file/ecd294194d5155432aab533f7ce0f6fe-Paper-Conference.pdf)、[arXiv](https://arxiv.org/abs/2407.03257) | ModernNCA 把 NCA 改成可学习非线性表示，加数值编码和 stochastic neighborhood sampling（SNS）。作者报告 300 个分类/回归数据集结果；论文附录 A 的主实验采用随机切分，多数资源实验使用 Tesla V100。 | 是 2025 年现代方法，并非直接调用旧 sklearn NCA。论文 B.5 自述未显式处理邻居的潜在时间关系；平均排名不能替代 HDB 向前验证。 |
| Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev, Daniil Shlenskii, Akim Kotelnikov, Artem Babenko. **TabR: Tabular Deep Learning Meets Nearest Neighbors.** ICLR 2024；arXiv 题名含 “in 2023”。[会议页](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html)、[全文](https://proceedings.iclr.cc/paper_files/paper/2024/file/4ef594af0d9a519db8fb292452c461fa-Paper-Conference.pdf)、[官方代码](https://github.com/yandex-research/tabular-dl-tabr) | 神经网络检索 top-k 上下文，把邻居特征和标签表示交给预测模块；作者在其 benchmark 中报告有竞争力的表现。 | 与 ModernNCA 同属检索路线，但不是同一预测规则；TabR 有参数化预测头，不能套用 ModernNCA 的原标签凸组合边界。 |
| Yury Gorishniy, Akim Kotelnikov, Artem Babenko. **TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling.** ICLR 2025。[全文](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf)、[官方实现](https://github.com/yandex-research/tabm) | 参数共享的 MLP 集成，不需要推理时扫描带标签候选库；其评估包含时间切分数据，作者报告较好的性能/效率权衡。 | 提供关键反方：新的检索模型未必优于现代参数模型。ModernNCA 应证明检索信息增益，不能仅以年份较新作为入选理由。 |

ModernNCA 比 TabR 简化了检索后的预测机制；TabM 则去掉推理候选库。这三个机制的比较比再堆叠若干 Transformer 名称更有实验区分度，但没有任何一篇证明本地租金任务上谁更准。

## 1. 官方实现究竟做了什么

正式会议页直接链接 [LAMDA-Tabular/TALENT](https://github.com/LAMDA-Tabular/TALENT)，因此官方来源已确认。以下核验针对查阅日 `main`，正式实验须固定 commit，不能假定论文公式、仓库默认值与调优配置完全相同。

模型在 [models/modernNCA.py](https://github.com/LAMDA-Tabular/TALENT/blob/main/TALENT/model/models/modernNCA.py) 里先产生表示，计算 `torch.cdist(..., p=2)`，再 `softmax(-distance / T)`，最后乘候选 y。**当前实现用欧氏距离而非平方距离；不做 top-k 截断；回归没有独立预测头。** 可写成

\[
\hat y(x)=\sum_{j\in R}w_j(x)y_j,\qquad
w_j(x)\ge0,\quad\sum_j w_j(x)=1.
\]

由这条代码路径可直接推导：数值正常时，原版回归预测位于候选标签最小值与最大值之间；训练标签标准化后再反变换仍保留边界。这不证明本题必然失败——未来租金可能仍在历史全局范围内——但它无法仅靠改变邻居权重突破这个范围，也容易受邻近历史价格水平约束。加趋势校正或改预测头属于新变体，应另标，不把其效果算作原版 ModernNCA。

**自己标签排除已核验**：[methods/modernNCA.py](https://github.com/LAMDA-Tabular/TALENT/blob/main/TALENT/model/methods/modernNCA.py) 先把当前 batch 的索引从候选池移出；模型按 `sample_rate` 从剩余候选无放回抽样，再拼回 batch，最后把对角距离设为正无穷。这样自己不能贡献标签，batch 中其他行仍可作邻居。类构造参数默认 `sample_rate=0.8`，实际实验配置可覆盖，不能误报为固定超参数。

源码的验证/测试按查询 batch 运行，每批候选仍是全部训练数据，并重新计算候选表示；未发现内置候选分块、表示缓存或近似索引。SNS 降低训练时的候选量，**不降低原版推理候选量**。[官方训练与预测路径](https://github.com/LAMDA-Tabular/TALENT/blob/main/TALENT/model/methods/modernNCA.py)

对照来看，[TabR 官方模型](https://github.com/yandex-research/tabular-dl-tabr/blob/main/bin/tabr.py) 用 FAISS `IndexFlatL2/GpuIndexFlatL2` 取邻居，提供 `memory_efficient` 与 `candidate_encoding_batch_size`，也显式排除当前训练样本。它们不能作为 ModernNCA 已有同等选项的证据；FAISS Flat 也不能误称近似索引。

## 2. 9,406 楼栋与未知类别

TALENT 的 `ModernNCAMethod` 要求 `cat_policy="tabr_ohe"`。公共 [data.py](https://github.com/LAMDA-Tabular/TALENT/blob/main/TALENT/model/lib/data.py) 在训练上拟合 ordinal 与 one-hot 编码，常规带验证集流程把验证未知值映射到训练列众数，测试复用该映射，再得到稠密 one-hot。因此不能写“原版 ModernNCA 自动提供楼栋 embedding / UNK”，也不能把此路径描述为“未知默认全零”。数据和模型还受 `use_float` 控制，不能默认所有张量都是 FP32。

本项目不宜把一栋新楼直接伪装成交易最多的某栋楼。建议在派生输入中明确建立训练过的 OTHER/UNK 策略：仅按训练折频数合并极少样本楼栋，或在训练输入上屏蔽少量楼栋 ID 到共同桶；保留城镇、房型、面积、街道/位置等共享特征。仅预留空桶、没有任何训练行到达它，不会获得可靠回退。回退策略与阈值仍须验证，属于对官方管线的明确适配。

全楼栋 one-hot 会显著增加数据存储。可优先审查按批生成 one-hot；若采用紧凑类别嵌入，则标明改变了输入参数化。也可研究把第一层对 one-hot 的线性运算实现成对应权重查表以保持数学等价，但这是待开发优化，不是当前 TALENT 已实现的能力。没有楼栋 ID 的对照仍需保留，以判断高覆盖率是否只是诱导记忆。

## 3. 150k × 50k 的资源账

令训练候选数 N=150,000、测试查询 Q=50,000、表示宽度 d、查询 batch 大小 B。全库精确距离计算量约为 `O(QNd)`，对应 **75 亿个查询—候选配对**；按批只把距离矩阵峰值降到 `O(BN)`，不会减少总配对数。

以下只是形状算术，不是峰值显存实测：一次物化全体 Q×N 距离，FP32 需约 27.94 GiB，FP64 为其两倍；B=256 时，一个距离矩阵约 146.5 MiB（FP32）。还必须加上 softmax 临时量、候选表示、输入、网络激活和运行时开销。只算 150k×9,406 的稠密楼栋 one-hot，FP32 约 5.26 GiB；官方预处理包含 float64 路径，且会把数据移到所选 device，不能拿单个距离矩阵证明整管线“8 GB 可跑”。

SNS 使用候选比例 r 时，训练某批约需计算 `B×(r(N−B)+B)` 个距离；固定 r 的整 epoch 距离量仍约为 `O(rN²d)`。此外当前方法层在模型抽样前已构造排除 batch 后的候选输入，降低 r 不必然按同比降低所有内存峰值。这些都是从源码推出的规模判断，未测耗时。

**GPU 建议**：保留 ModernNCA 主候选，但给它单独的资源校准步骤；记录 dtype、N、d、B、r、峰值 allocated/reserved、训练及 50k 推理墙钟时间。8/12/16 GB 只能作为待验证设备档，不承诺任一档能直接承载当前默认实现。先减查询 batch 能缓解距离矩阵压力；不能解决全量稠密输入驻留。

如需要优化，冻结模型后可考虑缓存候选表示、分块生成候选输入，并以数值稳定的全局 softmax 累积加权和。不能把“每块独立 softmax 后平均预测”当作等价算法。此优化不改变全库距离总量；候选子采样、top-k 或 ANN 才改变检索量，但也改变估计器，必须单独标记并对照。原版方法层的 `predict` 还计算带测试 y 的 benchmark 指标；本题无测试目标，应接出纯推理接口，不能制造假 y 当成有效评估。

## 4. 时间截止、重复 X 与同楼栋利弊

候选库应记为 `R_cutoff`，只能包含该预测起点已知的训练标签。主验证每折重建预处理、网络和候选库；候选不含验证/测试标签。固定起点预测未来 16 个月时，不能月月把真实验证标签补入候选库。2025-03 只有月份粒度，验证采用整月隔离，不能把同月某行租金视作另一行已知历史。

需要区分两层时间规则：在一个 cutoff 上训练监督模型时，使用截止前完整训练段内部的邻居标签、且排除当前样本，不自动构成泄漏；若另行声称每个训练 query 模拟“当时在线预测”，则该 query 的候选也必须更早。这两种协议不能混用。完整未来评估是共同外层时间窗口，不能为检索方法单独使用随机验证来获得较好名次。

楼栋重合 99.796% 是 ModernNCA 的机会：它可能检索同栋相近户型的历史成交，减少仅靠全局函数拟合的偏差。但是“同栋”不代表同单元或同合同；相同 X 的不同 y 体现不可观测差异。自己标签的对角屏蔽仅排除同一来源行，**不会排除其他索引的相同 X**。这些记录不能自动删除或视为错误，也不应把检索到相同 X 宣称为可复制答案。

纯加权平均可能稳定噪声，也可能把尾部与新市场水平拉回旧邻居。邻居年龄、所属月份、城镇、楼栋与户型组成应进入诊断；全历史、近期候选池可作受控对照，但后者会损失楼栋覆盖。不要直接强制只检索同栋：新楼、稀疏楼栋与户型迁移需要更粗层级相似度。主指标仍是原尺度 RMSE，并报告远期月份、稀疏楼栋和未知楼栋的样本数及误差。

## 5. 主五项建议与退出条件

**建议纳入 ModernNCA**，因为它同时满足 2025 年正式研究、公开官方代码、原生回归、与参数集成/基础模型不同的历史检索机制。建议新的五项组合保留 ModernNCA、TabM、TabICLv2、TabPFN-3.5、TabM + feature-aware temporal modulation；后三者的版本、资源与组合可行性应以各自核验为准。本文件不把它们当作已共同实测的最终榜单。

TabR 作为原理和实现参照留储备；经典 KNN、Ridge、树模型只作最小比较基线。ModernNCA 的主要否证条件是：在统一多个向前窗口里，同输入 TabM 稳定占优，且检索误差没有互补价值；或长跨度系统性偏差经邻居诊断显示只是在复用旧价格；或完成等价分块等必要工程后，实际资源仍超出团队接受范围。不能仅因它有检索成本就在调研阶段淘汰，也不能只因它较新就保留到底。

RQ1：ModernNCA 提供值得保留的现代检索机制。RQ2：150k 全库可作为工程目标，但原版的稠密编码与重复候选计算使 GPU 可行性需要校准。RQ3：高楼栋重合支持历史匹配假设，缺失 unit ID 与时间漂移则限制该假设；自己标签排除和固定 cutoff 的候选库是必要条件，不能替代长跨度实证。
