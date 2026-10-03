# 树模型与低成本实现：独立证据笔记

检索日期：2026-09-26。范围：RQ1 成熟表格模型机制；RQ2 时间、重复楼栋与辅助数据泄漏；RQ3 在暂定 4–8 CPU 核、16 GB RAM、无付费 GPU 下的实施优先级。本笔记只使用父任务给定的数据画像，没有读取或修改数据，也没有训练模型。所有参数和资源预算均为拟议方案，非实测结果。

## 检索轨迹

第一轮先检索完整论文标题和作者，再打开会议主页、摘要和全文：

- `"CatBoost: unbiased boosting with categorical features" Prokhorenkova 2018`
- `"LightGBM: A Highly Efficient Gradient Boosting Decision Tree" Ke 2017`
- `"XGBoost: A Scalable Tree Boosting System" Chen Guestrin 2016`

第二轮围绕实现风险缩窄至官方来源：

- `site.catboost.ai/docs has_time boosting_type CPU Plain Ordered thread_count used_ram_limit early_stopping_rounds`
- `site.lightgbm.readthedocs.io Features leaf-wise categorical high cardinality Parameters Tuning min_data_in_leaf cat_smooth histogram_pool_size num_threads`
- `site.xgboost.readthedocs.io tree_method hist CPU categorical max_cat_threshold early_stopping_rounds max_bin`
- 因 CatBoost common 页面多次抓取失败，补搜 `site.github.com/catboost/catboost "CPU" "Plain" "boosting_type" "Default value"`，打开官方仓库中的文档源文件和 RELEASE.md 完成复核。

搜索结果中的第三方百科、博客、论坛均未作为技术主张证据。论文是 3 篇，官方软件文档另计。以下 URL 均实际请求打开；XGBoost 的正式 DOI 返回 403，但作者预印本摘要、全文及其中 KDD 2016 会议脚注可读，DOI 亦由 arXiv 的 Related DOI 交叉确认。每篇论文均完成“标题/作者/年份/venue 存在性”和“摘要/全文主张匹配”两步核验。

## 已核验论文

| ID | 已核验元数据与原始来源 | 支持位置与准确主张 | 本项目使用边界 |
|---|---|---|---|
| P1 | Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, Andrey Gulin. **CatBoost: unbiased boosting with categorical features**. NeurIPS 2018, Advances in Neural Information Processing Systems 31. [会议主页](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html)，[全文](https://proceedings.neurips.cc/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf)。结论 VERIFIED。 | PDF 第 3–4 页，§3.2：把自身标签纳入类别均值会造成 target leakage；ordered TS 在随机排列中只用先前样本。第 4 页 §4 区分 ordered boosting 处理的 prediction shift。 | 排列构造的 artificial time 不是日历时间。理论背景采用 i.i.d. 样本（第 2 页）；不能据此证明跨月份预测无泄漏，也不能承诺 HDB 租金收益。 |
| P2 | Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. **LightGBM: A Highly Efficient Gradient Boosting Decision Tree**. NIPS 2017, Advances in Neural Information Processing Systems 30. [会议主页](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)，[全文](https://proceedings.neurips.cc/paper/6907-a-highly-efficient-gradient-boosting-decision-tree.pdf)。结论 VERIFIED。 | PDF 第 1–2 页，摘要、§1 和 §2.1：GOSS 保留大梯度样本并采样小梯度样本；EFB 捆绑近乎互斥特征；直方图分箱减少分裂搜索成本。第 7 页 Table 1 载明实验数据。 | 原实验包括百万级稀疏分类/排序数据；本项目仅 150k 训练行且原始列数不高，不能把论文加速倍数或 EFB 优势照搬。初版可使用常规 histogram GBDT，无须为复现论文而强制 GOSS。 |
| P3 | Tianqi Chen, Carlos Guestrin. **XGBoost: A Scalable Tree Boosting System**. KDD 2016, Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785–794, DOI 10.1145/2939672.2939785. [作者预印本主页](https://arxiv.org/abs/1603.02754)，[全文](https://arxiv.org/pdf/1603.02754)，[正式 DOI](https://doi.org/10.1145/2939672.2939785)。结论 VERIFIED。 | PDF 第 1 页摘要和会议脚注核对标题、作者、venue；第 2 页 §2.1 式(2) 对叶数与叶权重惩罚；第 5–6 页 §4 讨论列块、并行和 cache-aware 系统优化。 | 2016 论文不能用来证明当前原生类别处理 API；后者另由当前官方文档核验。本文的大规模系统成果不能作为本项目运行时间或排名预测。 |

## 当前官方实现证据

文档版本说明：本次 `/stable/` 页面显示 LightGBM 4.7.0、XGBoost 3.4.2；它们不是本地已安装版本的确认。实施时应锁定实际包版本并显式设置关键参数。搜索也返回 XGBoost 3.5.0-dev 的变化，因此没有依赖 `latest` 的默认值推断当前行为。

| 来源 | 已打开的段落/锚点 | 可以使用的主张 |
|---|---|---|
| [CatBoost 官方 common 文档源](https://github.com/catboost/catboost/blob/master/catboost/docs/en/references/training-parameters/common.md) | `boosting_type`，`has_time`；网页提取行 1359–1393、892–911 | CPU 默认 boosting scheme 是 Plain。Ordered 往往成本更高。`has_time` 保留输入顺序、关闭类别转换与树结构阶段的随机排列；提供 Timestamp 时使用该顺序。ordered categorical statistics 与 `boosting_type=Ordered` 不应混为一谈。 |
| [CatBoost RELEASE.md](https://github.com/catboost/catboost/blob/master/RELEASE.md) | Release 0.18，提取行 1156–1158 | CPU 回归/二分类切换至默认 MVS 和 Plain，覆盖大小数据集。此处用于解决官方 speed-up 页对“小数据默认 Ordered”的简化表述与 CPU 专门规则的差异。 |
| [CatBoost Parameter tuning](https://catboost.ai/docs/en/concepts/parameter-tuning) | Internal dataset order；Tree growing policy；提取行 334–398 | 默认 SymmetricTree；`min_data_in_leaf` 仅适用于 Depthwise/Lossguide，不能把 LightGBM 的叶样本控制机械照搬到默认 CatBoost。 |
| [CatBoost Speeding up training](https://catboost.ai/docs/en/concepts/speed-up-training) | Boosting type；Number of categorical features to combine | Plain、减少轮数、限制类别组合可减轻训练成本；这些设置可能影响预测质量。 |
| [CatBoost CTR settings](https://catboost.ai/docs/en/references/training-parameters/ctr) | `max_ctr_complexity`、`counter_calc_method`、`ctr_leaf_count_limit` | 组合阶数可限制；`SkipTest` 不用验证集对象计算 Counter CTR；限制类别叶数可省内存，但会损失类别信息。 |
| [CatBoost Performance settings](https://catboost.ai/docs/en/references/training-parameters/performance) | `thread_count`、`used_ram_limit`；提取行 59–129 | CPU 可设线程数；`used_ram_limit` 仅约束 CTR 计算，而且不保证完全遵守，绝不是整个训练进程的硬 RAM 上限。 |
| [CatBoost Overfitting detection](https://catboost.ai/docs/en/references/training-parameters/overfitting-detection) | `early_stopping_rounds` | 在最佳验证指标后等待指定轮数停止；控制的是轮数，不是墙钟时间。 |
| [LightGBM Parameters Tuning](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html) | Tune Parameters for the Leaf-wise Tree；Use Early Stopping；Decrease max_bin；Decrease max_cat_threshold | leaf-wise 容易长出深且稀疏的分支，应联合限制叶数、深度、叶内样本。分箱、类别分裂候选上限、早停可减少计算；增大最小叶样本也可能欠拟合。 |
| [LightGBM Advanced Topics](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html) | Categorical Feature Support，提取行 13–24 | 类别需一致整数映射或 pandas category；用 `min_data_per_group` 和 `cat_smooth` 对付类别少样本/高基数风险。官方还提出高基数类别转数值/embedding 的替代方向；这不证明任意楼栋编号具有真实顺序。 |
| [LightGBM Parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html) | `force_row_wise`、`histogram_pool_size`、`num_threads` | `force_row_wise=true` 会使 Dataset 对象内存翻倍；内存紧张可选 col-wise。histogram pool 限制的是历史直方图缓存，非整个进程。 |
| [XGBoost Tree Methods](https://xgboost.readthedocs.io/en/stable/treemethod.html) | Approximated Solutions；Implications，提取行 12–16 | `hist` 在训练前做一次全局 sketch；官方建议对恒定 Hessian 的 `reg:squarederror` 使用 hist。 |
| [XGBoost Categorical Data](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html) | Training with scikit-learn Interface；Auto-recoding | 原生类别需 `category` dtype 和 `enable_categorical=True`，支持 hist/approx；Python 3.1 起支持类别自动 recoding。仍需锁定训练/预测类型和类别映射。 |
| [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html) | `device`、`nthread`、`max_bin`、`max_cat_to_onehot`、`max_cat_threshold` | 可显式使用 CPU；分箱数控制计算与分裂粒度；类别分裂候选上限控制过拟合。应显式给类别 one-hot 阈值，避免随版本默认变化。 |

## 面向该项目的推理与候选 pipeline

以下判断来自“给定数据画像 + 已核验机制”的综合推理，而非已有 HDB 实验结论。建议在最终五方案中按机制组织，不必用三个相近 boosting 库占据三个名额；XGBoost 可作为 LightGBM 的替代实现或有限对照。

### CatBoost：优先尝试的原生类别基线

适配原因是 town、building、flat type/model 等异质类别以及面积/位置交互。楼栋 9,406 类不宜默认展开成密集 one-hot；CatBoost 的类别处理能降低手写目标编码的工程负担。起点是原始金额目标、`loss_function=RMSE`、CPU、4 个线程、depth 6、`boosting_type=Plain`、`max_ctr_complexity=1`、最多 1,500 轮、早停 100 轮。第二个小配置只增加 depth 至 8 或组合阶数至 2，避免同时改变大量因素。参数均为建议搜索起点，不是文献给出的最优值。

如果报告强调 ordered boosting，必须另设显式 `boosting_type=Ordered` 的比较，不能把 CPU 默认运行写成 Ordered。严格日历历史可以先排序并研究 `has_time=True`，但同月样本无真实先后次序时，它不能自动构造“只用之前月份”的统计。针对显式历史租金特征，按月份批量生成严格滞后统计更容易审计。

### LightGBM：低成本 histogram + 类别正则化对照

保留连续面积、年月索引、静态地理量；低基数和楼栋列显式设为 categorical。拟议首轮：31/63 叶、`max_depth=8`、`min_data_in_leaf=100` 或 300、`max_bin=127`、`cat_smooth=20` 或 100、4 线程、最多 1,500 轮、早停 100 轮。不要把这些候选展开成完整笛卡尔积：先两组固定配置，再依据验证结果有针对性调整。若楼栋原生类别出现过拟合，可比较只使用训练折生成的平滑/历史编码；不要直接把任意楼栋整数解释为连续量。

相较 CatBoost 的 ordered TS，LightGBM 的类别分裂依赖另一种机制，可能提供互补错误；是否值得简单均值融合需要相同时间验证上的预测证据。当前不能承诺融合收益。

### XGBoost hist：备选实现，不宜一开始扩张搜索

用 `device=cpu`、`tree_method=hist`、`objective=reg:squarederror`、depth 4–6、`max_bin=128`、固定少量 L2/min_child_weight 候选、4 线程；类别开启原生支持并显式 `max_cat_to_onehot=4`。它提供另一种正则化/生长策略作为有限对照，但不是必要第三轮大规模调参。2016 原论文机制与当前原生 categorical API 分开归因。

### 所有常规树模型的长期预测限制

根据叶子给常数预测的模型定义可以推导：若其他特征固定，月份数值大于所有已学阈值后，继续增大月份将沿同样的树路径，因此仅加 `month_index` 不会自动延伸未来上涨趋势。这是结构推理，不是 HDB 实测。最长 16 个月 horizon 下，应把“显式趋势/季节项 + 树拟合残差”作为与纯树机制不同的候选。month-of-year 能表示循环季节，不能替代长期趋势；面积/位置等其他特征改变时，预测当然仍会变化。

## 时间与实体泄漏边界

1. **外部时间验证是首要边界。** 每个 fold 的训练月份必须早于验证月份；编码、平滑先验、类别计数、特征筛选与调参均在该 fold 内拟合。CatBoost 的 ordered TS 解决自身标签污染的一类问题，不替代时间切分，也不会替外部 join 自动检查时间可得性。
2. **月粒度数据不能制造月内先后。** 严格历史特征使用 `month < current_month`；同月行整批计算后再更新统计。验证期所有月份的标签都不可用于更新滚动特征，除非任务真的允许逐月获得已揭晓标签，且该条件在评估中明确模拟。测试起始月与训练末月均为 2025-03，需要把边界月作为单独层报告，不能自行假设它们有可辨识日级顺序。
3. **重复楼栋与泄漏不是同义词。** 已知楼栋在未来再出现是合法预测情境。给定 99.796% 测试行楼栋已见，整楼栋 GroupKFold 会刻意制造大量冷启动并偏离主要测试分布；应以未来月份为主验证，同时报告“已见/未见楼栋”分层，并把楼栋隔离做补充压力测试。真正重复记录或近乎重复事件需另审，不能仅靠模型处理。
4. **HDB 辅助表 100% 匹配不等于时间无泄漏。** 地理位置可能较稳定，但楼栋设施、状态、数量等若来自当前快照，可能带有历史时点尚不可得信息。记录数据版本与提取时间，区分静态属性和动态属性，并保留不加辅助表的基线。没有日期证明时不能声称严格 point-in-time safe。
5. **早停也是模型选择。** 可以用靠后的训练窗口作为内层早停集，保留最外层未来窗口用于评估；至少另留一个未参与反复调参的最终时间窗口，防止把早停/调参分数当作完全独立成绩。

以上是方法设计建议；本笔记的三篇算法论文不能独立证明最优验证方案，需与父任务的时间评估文献证据合并。

## 内存和计算预算：估算与限额分开

以下仅是数组容量算术，既不是 peak RSS 预测，也不是训练实测：

| 假设表示 | 算术 | 大小 |
|---|---|---|
| 150k 训练行 × 40 数值/编码列，float32 | 150,000 × 40 × 4 | 24 MB，约 22.9 MiB |
| 训练+测试 200k 行 × 40 列，float32 | 200,000 × 40 × 4 | 32 MB，约 30.5 MiB |
| 仅楼栋密集 one-hot，200k × 9,406，float32 | 200,000 × 9,406 × 4 | 7.525 GB，约 7.01 GiB；尚未含其他列和任何复制 |
| 同一楼栋列的 CSR one-hot，每行 1 非零，float32 值/int32 索引 | 200,000 × (4+4) + 200,001 × 4 | 约 2.4 MB；类别字典等另计 |

因此应避免“高基数 one-hot 必然不可行”的过度概括：稀疏 one-hot 用于线性模型可以很省内存；密集展开才会逼近 16 GB 的实际可用空间。树的直方图、梯度、类别 CTR、内部数据副本和 Python 对象会使真实内存明显高于表中的原始矩阵。

资源控制建议：一次只运行一个模型训练过程；4–8 核总预算内控制线程，禁止“多个 trial × 每个 8 线程”的过度并行。初次批准训练后可先安排一个小规模计时检查，再给每个模型族同样的墙钟预算。例如单配置 5–10 分钟/每族 30 分钟是可选择的项目限额，不是预计完成时间。必须在日志记录时间、peak RSS、实际轮数；早停只缩短无收益的迭代，不构成硬时间限额。

对 CatBoost，优先限制 depth 和类别组合，再考虑 CTR 内存选项；对 LightGBM 可用 `histogram_pool_size=512` MB 作为缓存预算并在内存紧张时用 col-wise，但这两者都不是进程总内存保障。保留至少数 GB 系统与运行环境余量，具体停止阈值应以实际可用 RAM 决定。

## 给主报告的优先级建议

树模型内部：CatBoost 原生类别基线优先；LightGBM 少量配置为速度/机制对照；XGBoost hist 作为可替换备选。任务整体更值得保留一个独立名额给长期趋势处理，另一个给严格历史楼栋/区域统计，而非把五个名额填成相似 boosting 库。最终优先级应以相同月份 backtest 的 RMSE、horizon 退化、RSS、运行时间和实现审计成本共同决定，当前没有任何本地验证结果可以宣称“最佳模型”。

自审：未迁移 benchmark 改善幅度；未将 ordered permutation 写成日历防泄漏；未把内存参数写成硬限制；未把 GroupKFold 误设为唯一正确划分；未声称静态辅助表天然无泄漏；未将运行预算写成实测耗时。
