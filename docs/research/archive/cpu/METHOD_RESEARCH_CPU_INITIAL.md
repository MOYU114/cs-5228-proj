> 历史 CPU 优先版本：旧编号、预算和候选顺序已失效。当前入口为 [最佳尝试清单](../../PRIORITY_PLAN.md)；本文件仅用于追踪。

# HDB 月租金预测的低成本方案调研：属性拟合与长期时间外推

检索截至 **2026-09-26**。角色 A；本轮增加文献调研、五条候选管道及可行性评估。候选尚未训练，也没有本项目 RMSE、耗时或峰值内存结果。用户已确认普通电脑、CPU 优先、无需 GPU；具体核心数、内存、团队技能和每周工时尚未确认。

## 摘要

本项目有 150,000 条训练租金记录，训练截止 2025-03，而测试延伸至 2026-07。现有 EDA 显示，大部分测试记录来自历史已见楼栋，类别写法规范化能够消除相当部分表观分布差异，但这些现象不能消除未来市场水平的不确定性。本调研结合本地审计，从统计结构、类别树模型、近期反例、住房应用和时间验证四个视角检索并核验文献。证据支持将静态物业差异的拟合能力、未来时间的外推规则和数据可得性分别审查：高基数类别处理不能代替时间验证，低训练成本也不等于低验证风险。由此形成五条 CPU 候选管道，并通过独立 idea-evaluator 评估执行顺序。报告给出共用清洗规则、有限预算、历史长短窗口和失败后的处理方式；可行性优先级只是安排实验的建议，预测效果仍须统一时间验证。

## 1. 从 EDA 界定需要解决的问题

与单纯比较回归算法不同，本项目同时涉及高基数地址、未观测房屋差异和最长 16 个月的未来时间间隔。先前 [EDA 报告](../../../EDA_REPORT.md) 保持原样；本文件响应后续明确提出的“五方案调研”要求，列出待测试方案，不将它们写成已有效的方法选型结论。

检索前冻结的三个问题见 [研究 brief](../../RESEARCH_BRIEF.md)：

- **RQ1：** 哪些成熟、适合普通 CPU 的机制，能够处理本项目的混合字段和高基数楼栋？
- **RQ2：** 面对时间漂移、重复楼栋和缺少历史生效日期的辅助数据，各方案何时可能受益、何时可能失效？
- **RQ3：** 小团队应按什么可行性顺序测试五条管道，如何用有限计算否证假设？

下表将已运行的 EDA 与方法推断分开。所有数据数字来自 [核心输出](../../../../experiments/eda/core/summary.json) 或 [辅助输出](../../../../experiments/eda/auxiliary/summary.json)，不是文献数据。

| 已观测事实 | 支持提出的假设 | 仍不能推出什么 |
| --- | --- | --- |
| train 150,000 行、51 个月；test 50,000 行、17 个日历月份；最远月序差 16 | 验证同时覆盖近期与较长预测间隔 | 单个近期月份的好成绩能代表全部测试期 |
| 49,620 条测试行，即 99.24%，晚于训练最大月份；2025-03 仍有 380 条测试行 | 主任务是向未来预测，边界月另行解释 | 所有 test 都严格晚于所有 train 交易 |
| 训练楼栋 9,406 个；测试 99.796% 的行来自训练已见楼栋；新楼栋 50 个、102 行 | 正则化楼栋效应或类别处理值得测试 | 楼栋记忆已经解决未来租金变化 |
| 户型原始分布 TVD 为 0.2394，规范化后为 0.0101 | 确定性文本规范化必须先于模型比较 | 规范化已经消除了时间或条件目标漂移 |
| 月均租金从 2021-01 的 2,048.90 到 2025-03 的 3,162.71 SGD/月，后期变化趋缓 | 比较固定近期水平、受控趋势与非线性属性拟合 | 未来继续上涨，或平台必然持续 |
| 面积与租金 Pearson 相关 0.329；楼龄代理为 −0.178；分组差异并不严格线性 | 少量样条和交互项有测试动机 | 这些关联是因果效应或单调约束成立 |
| 相同规范化输入、不同租金涉及 17,900 组和 39,540 行；没有 unit ID | 避免自动删重，采用收缩并检查残差 | 已经测得测试集不可约误差下界 |
| HDB 规范化连接 100% 匹配且无一对多；其他设施表缺少历史有效期 | 静态辅助属性可以列为受控消融 | 匹配率高代表时间上安全或预测上有益 |

以下先比较证据与机制，再给出五条具体管道、评估顺序和验证设计。

## 2. 检索与证据范围

调研使用网页检索、出版方/作者页面和官方软件文档；三个独立子代理分别负责树模型、统计结构、反方与近期进展，主代理负责住房应用、可解释实现和验证。各视角都做了宽检索及第二轮定向核验。四份可复查记录为 [树模型](tree_evidence.md)、[统计结构](structure_evidence.md)、[反方证据](critics_evidence.md)、[应用与验证](application_validation_evidence.md)。

关键词覆盖 `HDB rental hedonic forecasting`、`CatBoost categorical CPU`、`LightGBM high cardinality`、`penalized spline sparse mixed effects`、`damped trend`、`temporal tabular benchmark`、`EBM interactions` 及论文完整标题。反向检索特意检查“树永远优于神经网络”“基础模型只能处理小数据”“同楼栋重合就是泄漏”等断言。

最终正文引用 **20 篇研究文献**，发表或预印本年份为 1985–2026；另引用一本作者公开教材及实现文档。纳入标准是题名、作者、年份/版本可核验，且所用主张可以在已打开的摘要、正文或官方文档中确认。仅能读摘要的文献只承担摘要级论断；会议预览稿和新近技术报告明确保留其性质。博客、排行榜和无法核验的细节没有作为性能证据。这不是穷尽检索或文献计量研究。

检索中部分出版页受访问限制，替换为作者全文或正式摘要；具体失败与替代链接保存在证据底稿。没有把 2019 预印本与 2020 正式论文不同版本的实验数量混用，也没有把旧版 TabPFN 的容量限制写成现行限制。所有本地效果问题均保持未验证状态。

## 3. 以预测职责划分机制

一级划分依据是“是否显式预测截止日之后的市场水平”；不含该模块的属性模型，再按函数结构分成受限加性与一般树交互。这样可以区分完整管道，又允许一个管道复用其他分支的模块。

| 分支 | 属性关系 | 未来市场水平规则 | 对应候选 | 主要证据 |
| --- | --- | --- | --- | --- |
| 受限加性属性模型 | 样条/分箱主效应，必要时少量交互 | 显式低阶时间项，或边界常量延伸 | P1、P4 | Wood、Bates、Lou、Nori |
| 一般类别树交互 | 多变量树分裂与类别处理 | 单靠常数叶树不会延伸月份趋势 | P2、P3 | Prokhorenkova、Ke、Chen |
| 显式时间分解管道 | 先调整属性，再拟合物业残差 | 独立的固定水平/阻尼趋势预测 | P5 | Meese/Wallace、Gardner/McKenzie、Wang/Tu |

P4 的底层也使用树，但最终函数限制为加性项，因此归入第一分支。P5 使用 P2 作为残差模块，跨越两个分支；完整管道的区别在于它承担了额外的时间外推职责。不是所有模型实现都落入此有限候选集，近期神经网络作为反证与储备在第 7 节讨论。

没有在检索到的工作中找到“与本项目相同 CSV、相同截止日、相同 CPU 预算”的直接比较。这是当前证据的空白，不能解释成相应模型无效，也不能用其他数据集分数填补。

## 4. 加性结构：用低成本换取可检查的假设

Wood 等的惩罚样条计算研究 [1] 说明加性模型可以扩展到大数据，Bates 等的稀疏混合效应实现 [2] 则说明分组信息可以通过收缩共享。两者分别处理函数形状和组间差异；都不证明 9,406 个楼栋加入任意软件后必然便宜。这里采用稀疏样条基与 Ridge，是受这些机制启发的工程方案，不等同完整 GAM 的自动平滑估计，也不等同估计随机效应方差的混合模型。[Wood 等，2017](https://www.research.ed.ac.uk/en/publications/generalized-additive-models-for-gigadata-modelling-the-uk-black-s/)、[Bates 等，2015](https://www.jstatsoft.org/article/view/v067i01)。

Lou 等的 GA2M [3] 允许在主效应之外加入有限成对交互，Nori 等的 InterpretML [4] 提供 EBM 实现；它们比严格加性模型更灵活，但仍比一般树集成限制更强。因此 P1 与 P4 可以检验“少量可解释非线性是否已经足够”，而不是事先断言可解释性一定牺牲精度或一定提升泛化。[GA2M 原文](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/06/kdd13.pdf)、[InterpretML](https://arxiv.org/abs/1909.09223)。

| 机制比较 | 可处理的问题 | 留下的限制 |
| --- | --- | --- |
| 样条基 + L2 惩罚 | 面积/楼龄的平滑变化，控制自由度 | 结点、尺度和惩罚需要折内确定 |
| 稀疏楼栋效应/分组收缩 | 已见楼栋的稳定差异，小样本向总体收缩 | 无法恢复缺失的 unit 信息，也不预测市场拐点 |
| GA2M / EBM 主效应与少量交互 | 属性关系可分项查看，容许有限交互 | 高基数项仍可能稀疏，分箱时间项不自动外推 |

该比较支持的是三类约束各自解决不同问题，而非它们的误差排序。可视化的效应图依然是预测关联；面积、户型、城镇、楼栋和年代高度相关时，不能把某个分项当作独立因果效应。

## 5. 类别树模型：降低预处理负担，但保留时间边界

CatBoost [5] 通过有序类别统计处理自身目标污染问题，LightGBM [6] 强调直方图及高效分裂，XGBoost [7] 强调正则化和系统实现。相较前者的类别机制，后两者更需要显式检查类别映射与复杂度；三篇论文都没有给出本项目长期租金预测的准确率或普通电脑耗时保证。[CatBoost](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html)、[LightGBM](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)、[XGBoost](https://arxiv.org/abs/1603.02754)。

CatBoost 的随机排列是算法内部的人工顺序，不能替代日历切分；ordered target statistics 与 `boosting_type=Ordered` 也不是同一设置。CPU 方案明确用 `Plain`，不能在报告中写成已经运行 ordered boosting。LightGBM 则需要限制叶数、叶内样本及类别平滑，防止高基数楼栋形成脆弱分支。两者都必须在每个训练折内学习类别相关信息。[CatBoost 官方配置源](https://github.com/catboost/catboost/blob/master/catboost/docs/en/references/training-parameters/common.md)、[LightGBM 调参指南](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html)。

| 方法 | 本项目值得测试的机制 | 主要实现风险 | 本轮定位 |
| --- | --- | --- | --- |
| CatBoost | 减少手写类别编码；学习物业交互 | CTR 组合开销；把内部顺序误当真实时间 | P2 |
| LightGBM | 直方图计算；受正则化的类别分裂 | 深而稀疏的叶；训练/预测类别编码错位 | P3 |
| XGBoost | 正则化树与成熟 CPU 实现 | 再增加一套库与调参工作 | 替换储备，不占第六个实验名额 |

表中差异足以支持保留两种类别处理路线；目前没有证据要求用三种相近树库占满候选。对于普通常数叶树，固定其他输入、让月份编号超过所有已学阈值后，继续增加月份会沿相同路径。因此年月特征可以拟合历史阶段，却不会自动接续未来上涨；树的总输出并不因此被训练租金最大值严格界定。[回归树的边界行为](https://scikit-learn.org/stable/modules/tree.html)。

## 6. 时间分解：针对真正的远期风险，也增加假设

Meese 与 Wallace [8] 的动态 hedonic 研究容许住房属性关系随时间和地区改变；Gardner 与 McKenzie [9] 的阻尼趋势则约束长期线性延伸。前者支持区分属性与市场构成，后者提供一种受控外推规则，但住房价格指数和一般序列的证据都不能证明本项目租金将遵循该趋势。[动态 hedonic](https://onlinelibrary.wiley.com/doi/abs/10.1111/1540-6229.00555)、[阻尼趋势原文](https://www.bauer.uh.edu/egardner/3301H%20Operations%20Management/ESG%20Publications/1985%20Forecasting%20trends.pdf)。

Wang 与 Tu 的 HDB 租金指数会议稿 [10] 将住房属性与状态变化结合，应用对象比一般表格 benchmark 更接近本项目；但其历史平滑使用某时点前后信息，研究目标也偏向回溯指数重建。与只用截止日前信息的趋势预测相比，它提供的是分解动机，并非可直接照搬的前向验证成绩。截止日以内重建历史指数可以合法，使用截止日以后的验证目标平滑过去则不合法。[HDB 租金指数会议稿](https://www.aeaweb.org/conference/2020/preliminary/paper/T587D4ZR)。

| 时间机制 | 能表达什么 | 关键边界 |
| --- | --- | --- |
| 动态属性/时间效应分解 | 共同市场水平与房源构成的区别 | 效应估计受属性模型错设影响 |
| 固定近期水平 / 阻尼趋势 | 截止月以后的显式预测轨迹 | 平台、斜率和拐点均需验证 |
| 双向历史平滑 | 回溯历史状态 | 不能读取预测起点以后的目标 |

P5 因此采用“属性调整的月度效应 + 受控外推 + 固定残差树”，而不是直接外推未经调整的月均租金。训练只有 51 个月，不能由此假设每个楼栋都有足够的连续月度观测；逐楼栋拟合数千个独立时间序列模型既增加成本，也缺少信息共享。稳定年度季节性同样尚未得到 EDA 证实。

## 7. 交叉证据：算法家族不是效果结论

Grinsztajn 等 [11] 支持传统树作为表格任务起点，McElfresh 等 [12] 则表明适度调参与任务差异会削弱单纯的架构排名。TabM [13] 提供参数高效神经集成的反例，TabReD [14] 进一步显示时间切分会改变方法排名。因此成熟 CPU 树的执行优先级有依据，“树永远更准”没有依据。[树与深度学习比较](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html)、[更广的模型比较](https://arxiv.org/abs/2305.02997)、[TabM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html)、[TabReD](https://proceedings.iclr.cc/paper_files/paper/2025/hash/571799482291411607c54984153190b0-Abstract-Conference.html)。

截至检索日，TabICLv2 [15] 已支持回归，TabPFN-3 与 3.5 [16–17] 已扩展训练规模，后者还报告时间/分组划分结果。其官方实现支持 CPU，但大表能力或 GPU 实验不能证明本项目全量 CPU 成本可接受。这些作者技术报告和最新方法构成必须保留的反证；本轮未纳入五项优先候选，理由是当前资源约束下缺乏可确认的全量成本与实施收益，而不是不能回归、不能用 CPU 或只能处理旧版小样本上限。[TabICLv2](https://arxiv.org/abs/2602.11139)、[TabPFN-3](https://arxiv.org/abs/2605.13986)、[TabPFN-3.5](https://arxiv.org/abs/2609.17895)、[TabICL 官方文档](https://tabicl.readthedocs.io/en/latest/)、[TabPFN 官方实现](https://github.com/PriorLabs/TabPFN)。

验证也需要条件判断：Bergmeir 等 [18] 给出某些自回归、误差不相关情形下 K-fold 成立的条件，而 Cerqueira 等 [19] 在非平稳情形更支持保序留出；Roberts 等 [20] 又提醒分组阻断可能改变要评估的外推问题。本项目没有验证前者的成立条件，且部署主要是已见楼栋的未来租金，所以主评估采用完整月份向前验证，未见楼栋作为分层或压力测试。[Bergmeir 等](https://robjhyndman.com/publications/cv-time-series/)、[Cerqueira 等正式全文](https://kt.ijs.si/igor_mozetic/papers/CerTorMoz-TimeSer-MACH-20.pdf)、[Roberts 等](https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881)。

由这些证据产生的共同判断是：模型能力、预算和预测信息边界必须同时成立。时间外推规则正确也可能拟合不足；属性模型很强也可能在未来整体偏低；有大量已见楼栋也不能使用验证时期的楼栋租金更新模型。

## 8. 五条完整候选管道与独立评估

以下参数均是**后续实验的预算起点**，不是已调好的最优参数。五条管道首先使用相同的规范化核心字段和原值租金平方损失；辅助数据、历史统计、近期加权和对数目标作为后续共用消融，避免把不同特征包造成的收益误归因于模型。

### P1：稀疏样条 Hedonic + Ridge

形式为 `预测租金 = 截距 + 面积样条 + 楼龄样条 + 类别效应 + 楼栋效应 + 显式时间项`。使用 `OneHotEncoder(handle_unknown="ignore")` 的稀疏输出、少量 `SplineTransformer` 基函数及 `Ridge(solver="lsqr")`；月份编号采用单独的线性项，面积和楼龄样条在边界外保持端点值。所有词表、缩放和样条结点只在当前训练子折拟合。[SplineTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html)、[Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)。

第一配置可固定每个数值样条 5 个结点、`alpha=10`；若值得继续，仅比较预先列明的 `alpha∈{1,10,100}`，不展开大网格。新楼栋回退到已知属性效应。Ridge 楼栋系数有收缩作用，但不能声称已经估计混合模型的楼栋方差。

适配 EDA 的理由是高楼栋复用和面积/楼龄的非线性迹象。主要反例是复杂交互、全期线性时间斜率在后期平台附近失真。低成本否证实验是与训练折内的 town×户型均值控制比较，再依次移除楼栋项或将线性时间项固定到训练终点；若增加复杂度没有稳定收益，保留更小的诊断基线。

### P2：CPU CatBoost 类别回归

直接输入规范化的 town、building key、flat type/model 等类别，以及面积、楼龄代理、月份编号和可预知日历特征；不手写全量目标编码。起点为 `task_type=CPU, loss_function=RMSE, boosting_type=Plain, depth=6, max_ctr_complexity=1, iterations=1500, thread_count≤4`，学习率预先固定，内层完整月份验证可设早停 100 轮。类别 Counter 的计算明确排除验证对象，例如 `counter_calc_method=SkipTest`。[CPU 与 CTR 控制](https://catboost.ai/docs/en/references/training-parameters/ctr)、[计算预算参数](https://catboost.ai/docs/en/references/training-parameters/performance)。

它降低手动编码的工程负担，适合测试类别与物业交互；是否能优于 P1 仍未知。不要把 `has_time=True` 当成月内真实顺序，也不要将有序统计误写成已经完成日历防泄漏。否证点是同预算下相对 P1 的多窗口 RMSE、稀有楼栋表现和远期月份偏差；若只改善近期而远期系统性偏移，后者交给 P5 检验。

### P3：CPU LightGBM 直方图 + 类别正则化

沿用 P2 的共用输入，类别列使用训练折确定的一致映射，预测新类别设为显式未知状态；不能将任意楼栋整数当作有真实次序的数值。首个配置可用 `objective=regression, num_leaves=31, max_depth=8, min_data_in_leaf=100, max_bin=127, cat_smooth=20, num_threads≤4`，最多 1,500 轮，内层早停 100 轮。第二配置若有预算，只同时按预登记方案加强叶样本和类别平滑，不做参数笛卡尔积。[类别支持与平滑](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html)。

它与 CatBoost 的类别机制不同，可能给出更适合预算的属性拟合方式，也可能在高基数楼栋上过拟合。比较时固定特征组、时间窗和资源上限，记录真实时间/RSS；不能将原论文的加速倍数搬到本数据。否证点是资源受限时是否仍提供稳定的误差或效率收益；若完全被 P2 支配，就停止扩张这条路线。

### P4：预算化 EBM 主效应与少量交互

先拟合 `ExplainableBoostingRegressor` 主效应：`objective=rmse, interactions=0, outer_bags=1, validation_size=0, max_bins=128, max_rounds=1000, n_jobs=2`。这些限制减少自动搜索和重复拟合，但可能影响精度。只在外部内层时间窗比较少量固定配置；每次 `fit` 只接收该内层训练行，避免默认随机内部验证。后续交互限于预先指定的面积×户型等少数项；不自动展开 building×month。[EBM 参数与接口](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html)、[预算相关超参数](https://interpret.ml/docs/hyperparameters.html)。

它可以检查主效应和有限交互是否足够解释 EDA 的分组差异，模型图有助于发现稀疏类别异常。分箱时间项仍不能自动外推；袋数减少也不保证精度保持。否证实验是相对 P1 比较额外非线性的收益、相对 P2 比较受限交互的代价，以及固定主效应配置后再加指定交互的消融。效应图用于诊断，不用于因果结论。

### P5：属性调整的月度趋势 + 固定 CatBoost 残差树

本管道由本项目设计推导，未被某篇论文直接验证。对每个训练折执行：

1. 用稀疏 hedonic 属性项和受正则化的月份指示项联合估计历史月份效应 `δ_t`，固定基准/中心化约束以消除截距歧义。此阶段仅用面积、租约起始年、town/type/model/building 等静态物业描述，不用按记录日期计算的 `AGE_YEARS` 或其他可重构绝对时间的物业特征。不能直接把原始月均值当成已控制构成的指数。
2. 在该折月度效应序列上，用无季节、加性阻尼 Holt 预测未来 `δ`，例如 `Holt(..., exponential=False, damped_trend=True, initialization_method="estimated")`；固定最后水平作为必须比较的简单控制。不读取截止日后的租金，也不在验证过程中用真实月份结果更新。[Holt 官方接口](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.Holt.html)。
3. 用训练标签 `y−δ_t` 拟合物业残差树，复用已冻结的 P2 参数及上述静态物业输入；去除绝对月份、按行日期变化的年龄和其他时间重构变量。预测未来月份时保持该物业描述不变。
4. 输出 `未来月份效应预测 + 物业残差预测`。每个内层和外层截止日都重新拟合整个分解，最终比较原值租金 RMSE。

在训练折内构造残差标签不自动构成泄漏；把含自身标签的聚合量作为同一行输入，或用验证真值更新指数，才跨越信息边界。若以后将指数/残差统计改成输入特征，必须另做按时间的折外生成。趋势预测和未来解释变量的可得性应按前瞻预测处理。[前瞻与回溯回归预测](https://otexts.com/fpp3/forecasting-regression.html)。

楼龄等于批准年份减租约起始年，结合楼栋项后会与时期效应混淆；仅中心化月份项不能解决它。静态参数化让 `δ_t` 表示给定这组物业描述后的共同月份差异，仍不是因果市场指数；也可能牺牲真实的老化效应。因此核心否证需增加一个 P2 消融控制：固定 P2 参数，让直接树和 P5 使用同一静态物业组，直接树另外接收月份编号，P5 由独立模块处理月份。比较**直接树 / 固定水平残差管道 / 阻尼趋势残差管道**；原来含动态年龄的 P2 再作完整管道参照。该控制不是第六项方案，原 P2 与 P5 的全部差异也不能只归给分解。

P5 最直接回应 16 个月间隔和树的时间边界，但月效应受属性模型错设、稀疏时期和市场拐点影响；单一共同趋势也可能忽略城镇差异。检查后段月份误差与偏差，以及静态/动态年龄输入的敏感性；若收益仅来自更强底座或额外调参，不能归功于时间分解。

### 可行性优先级

独立评估已实际按 [idea-evaluator skill](</Users/moyu114/.codex/skills/idea-evaluator/SKILL.md>) 执行，详见 [当时完整评估](IDEA_EVALUATION_CPU_INITIAL.md)。评估定位为成熟方法在特定时间外推场景的课程应用，不宣称算法创新；五项都需要验证设计或实施范围的修订，结论均为 **Accept with Revisions**。五维评分是有解释的机制判断，不是测得的准确率或资源分数，也不加总成伪精确排名。

| 可行性顺序 | 管道 | 相对计算负担：待实测 | 排在此处的理由 | 首要失败风险 |
| --- | --- | --- | --- | --- |
| 1 | P1 稀疏样条 + Ridge | 预期低 | 稀疏表示、依赖少、易诊断，先建立共同基线 | 加性不足或时间线性外推失真 |
| 2 | P2 CatBoost CPU | 预期低至中 | 类别接口成熟，手动编码负担小 | 未来水平偏移、类别计算开销 |
| 3 | P3 LightGBM CPU | 预期低至中 | 提供另一种成熟的低成本树机制 | 类别/叶复杂度控制需要额外检查 |
| 4 | P4 预算化 EBM | 预期低至中，受配置影响 | 可检查主效应与少量交互，实施需控制默认预算 | 分箱外推与高基数稀疏项 |
| 5 | P5 月度趋势 + 残差树 | 比固定 P2 多一个估计阶段 | 更贴近远期风险，但实现与归因审计最多 | 指数错设、趋势假设失效、阶段间泄漏 |

P5 排后反映工程依赖，不代表研究价值或 RMSE 潜力最低。P2 与 P3 的真实速度顺序尤其需要实测。若团队已熟悉某个库，可以降低其实施成本，但不能因此改变验证标准。

## 9. 共用数据规则、验证和 CPU 预算

### 9.1 清洗与辅助信息

五条管道必须共享下列确定性规则；复用 [EDA 规范化代码](../../../../notebooks/eda_core.py)，保留原始文件与审计标记。

- NFKC、首尾去空白、小写、连续空白压缩；`FLAT_TYPE` 的 `N room/N-room` 统一。楼号保留字母后缀，楼栋键用规范化 `TOWN + STREET + BLOCK`，不能只用 BLOCK。
- 排除常量 `FURNISHED` 与 `FEE`；月份严格解析，楼龄代理按该行批准年份减租约起始年。负楼龄的一条记录先保留异常标记，处理方式等待字段口径确认，不擅自改成年龄零。
- 不因 IQR 尾部、重复楼栋或相同输入不同目标自动删行。不从全训练集沿用早期验证的分位数、词表、填补量或类别统计。
- 稀有类别的频数与回退规则在训练折确定；缺失类别统一显式标记。所有候选先预测原始 SGD/月；对数目标若后续实验，仍需在原值尺度评价并检查回变换偏差。
- 目标派生的楼栋/区域历史统计只能用该行之前的完整月份；同月整批处理后再更新。验证期坚持固定预测起点，不逐月喂入真实租金。边界 2025-03 缺少月内顺序，历史特征保守截止 2025-02，并将该月与后续月份分开解释。

严格滞后规则针对显式历史输入。CatBoost 的内部类别统计属于训练算法，可以在当前训练折内部使用训练标签；不得访问外部验证标签，也不能将内部统计解释为每行当时可得的逐月历史特征。

辅助数据的“可连接”和“在预测时可得”必须分开：

| 表 | 当前可用信息 | 后续实验条件 |
| --- | --- | --- |
| HDB 楼栋表 | 规范化三键全量唯一匹配；位置和楼栋描述 | `NIL` 邮编、`−1` 年份/楼层按字段哨兵处理；竣工年与租约年冲突保留标记。地理位置虽较稳定，仍需记录快照来源；动态设施/数量不得默认历史已知 |
| MRT | 坐标、线路/站名、当前 open/planned 状态 | 没有启用日期时不能还原每个批准月的可达性；同名多线路去重不能误删换乘语义。需核实历史开通表才进入严格时间特征 |
| 学校 | 坐标、类型/层级 | 当前存在不代表历史存在；缺少迁址/开校日期，先保留无该表的对照 |
| 商场 | 坐标、名称 | 同样缺少历史生效信息，距离只作有来源假设的静态消融 |
| COE | 按月份/轮次/类别的经济指标 | 表延伸至 2026-08；不能使用截止日后已实现数值。即便滞后一个月，固定起点预测远期时仍会泄漏 |
| 股票 | 日度市场数据 | 表到 2026-07；需审计缺失和 OHLC 逻辑，并核实发布时间。未来实际价格不能作为已知解释变量 |

首轮不依赖这些辅助表也能完成五条管道。后续按“仅核心 → 审核后的静态地理 → 严格历史统计”逐组增加，每次只改变一组。不可确认历史状态的字段暂缓纳入严格时间实验；COE/股票的预测期真实值不进入任何方案。外部网页只用于方法调研，没有下载未来租金或经济序列补标签。

### 9.2 拟议统一时间验证

以下是交团队审阅的方案，尚未训练，也未替 B/C 冻结最终 split。行数已由 [标准库脚本](../../../../experiments/research/validation_windows.py) 从 EDA 月份计数重新计算，输出见 [validation_windows.csv](../../../../experiments/research/validation_windows.csv)。

| 用途 | 训练月份（闭区间） | 训练行 | 验证月份（闭区间） | 验证行 | 验证月数 |
| --- | --- | ---: | --- | ---: | ---: |
| V1 近期开发 | 2021-01–2024-03 | 114,842 | 2024-04–2024-09 | 17,960 | 6 |
| VL 长间隔开发 | 2021-01–2023-05 | 84,902 | 2023-06–2024-09 | 47,900 | 16 |
| V2 方案冻结后的确认 | 2021-01–2024-09 | 132,802 | 2024-10–2025-03 | 17,198 | 6 |

V1 与 VL 的验证月份重叠，是互补诊断，不是独立重复实验，不能把两窗重复行合并成一个“总 RMSE”。VL 较早的市场阶段并不等价于真实 2025-03 起点；它检验长间隔脆弱性，不能消除未来变化的不确定性。V2 与开发验证月份分开，但其分布已经在 EDA 中看过，因此只能称为对算法选择保留的确认窗，不能宣称分析者从未接触过。如果 B/C 已用它调参，就应降格为开发结果。

每个开发窗口内部再用训练段末尾的完整月份做参数/轮数选择；选定后用该窗口全部训练段重拟合，外部验证只计算结果。CatBoost/LightGBM 早停只用内部未来段；EBM 用固定轮数的少量配置做外部内层比较；P5 的分解和趋势选择全部嵌在同一边界内。V2 在冻结方案后一次性比较，看到结果后若继续修改，必须公开其已变成开发反馈。

统一主指标为原值 `RMSE = sqrt(Σ(y−ŷ)² / N)`，单位 SGD/月；一个窗口内先汇总 SSE 和样本数，再开方，不能平均各月 RMSE 代替。另报逐月/预测间距的 RMSE 与平均残差、户型、城镇、训练截止前已见/未见楼栋及各组样本量。未见楼栋样本少，不能把小样本波动解释为稳定优势。边界同月问题单列，不人为制造行级时间顺序。多步评估依据见 [作者教材的时间序列验证](https://otexts.com/fpp3/tscv.html)。

### 9.3 预算是上限，运行成本尚未测量

首阶段先做 P1/P2，各一个配置，在 V1 检查可运行性和资源；每条管道的训练墙钟累计上限 **15 分钟**，两项上限共 **30 分钟**。随后按优先级扩展到五项完整筛查时，串行训练累计上限为 **75 分钟**。每项限额包含内层拟合、重拟合、P5 所有阶段和降配重试，不按单次 `fit` 重置；不含环境安装、数据准备和人工，端到端成本另行记录。此处是可调整的项目预算，不是“15 分钟一定能跑完”的预测；预算内可以先缩减轮数/复杂度，超限则记录未完成并暂停该管道，不把超时当作方法无效。

同时只运行一个训练进程，线程数最多为实际可用核心数与 4 的较小值，EBM 起点为 2。内存停止线建议不超过实机物理内存的 60%，由 B/C 按系统余量调整；4–8 核和 16 GB 仅为先前规划参考，不是用户设备实测配置。CatBoost 的 `used_ram_limit`、LightGBM 的 histogram cache 都不是整个进程的硬 RAM 限制，需要进程级计时与 RSS 监测。

仅训练楼栋 dense float64 one-hot 主矩阵的容量算术为 `150000×9406×8 = 11,287,200,000 bytes`，约 11.29 GB，尚未含副本；这不是峰值 RSS 测量。P1 必须维持稀疏表示，不允许意外转 dense。树模型用受限类别/直方图，距离特征以后按楼栋去重计算，避免全交易两两距离矩阵。

首轮之后只为仍有价值的管道追加有限配置和 VL，新增阶段另行登记预算；V2 留到冻结后。75 分钟仅是 V1 筛查训练限额，不是完成全部验证与机制消融的总成本保证。日志至少记录 split、特征组、清洗规则版本、随机种子、包版本、线程、轮数、墙钟时间、峰值 RSS、原值 RMSE 和分层误差。不得用同一模型的多 seed、大网格或自动 stacking 消耗默认预算。

实现依赖计划为：P1 使用 NumPy/SciPy/pandas/scikit-learn；P2 添加 catboost；P3 添加 lightgbm；P4 添加 interpret-core；P5 复用 P1/P2 并添加 statsmodels 的趋势实现。它们是未来训练环境的依赖清单，本轮没有安装或运行这些模型；实际实施时锁定验证可用的版本，不能把在线文档版本冒充本机安装版本。

## 10. 未解决问题与 B/C 交接

当前最关键的不确定性并非再增加算法名称，而是市场变化、数据可得时间和团队执行预算：

- 未来市场若与训练末段平台不同，固定水平和阻尼趋势都可能偏误。P2/P5 的同底座成对比较和 VL 逐月残差用于检验该风险，但不能保证覆盖所有结构变化。
- 相同输入不同目标缺少 unit/交易标识；是否存在真正错误重复、租金舍入或房屋内部差异仍待数据提供方解释。没有证据支持自动删除或宣称误差下界。
- HDB/设施表的快照时点与历史状态尚不完整；需要明确课程允许的外部数据范围，以及能否获得可核验的历史生效日期。
- 课程文字终点与实际 CSV 终点不同，2025-03 月内顺序不可恢复；应由团队确认评估口径。训练期内已有数据足够进行提议实验，无需访问测试目标。
- 具体 CPU/RAM、B/C 的库熟练度和剩余工时未知；可行性顺序可据这些现实条件调整。正式预算与 split 应在实验前共同登记。

**给 B/C 的简短交接清单：**

1. 共用 `eda_core.normalized_copy` 的确定性规则；字符串楼号与三字段楼栋键保持一致，保留全部交易与异常标记；所有学习型预处理折内拟合。
2. 先确认 V1/VL/V2 及边界月口径；不将楼栋互斥切分替代未来时间验证，不将重叠窗口当独立证据。
3. B 可先负责 P1/P2，C 可先负责 P3/P4；P5 复用冻结后的 P2，由双方共同检查分解的时间边界。这是建议分工，未替团队分派承诺。
4. 通过原值 RMSE、逐月偏差与实测成本检验：楼栋收缩是否有用、额外交互是否值得、固定水平和阻尼外推哪个更稳、辅助信息是否有独立增益。所有假设均未被本轮验证。
5. 不使用测试标签、未来已实现宏观值或未核实的历史设施状态；不因初次窗口小幅改善立即扩大搜索。结果不支持额外机制时，保留较简单管道。

## 11. 对研究问题的回答

**RQ1：** 稀疏加性回归、成熟类别树与受限 EBM 提供可在 CPU 上实施的机制基础；显式趋势加残差树提供不同的时间建模职责。五条候选在类别处理、交互复杂度、外推规则上有可审查差别，而不是五个同义算法名称。

**RQ2：** 高楼栋覆盖支持利用过去的地址信息，却不能保证未来市场稳定；文本清洗能够修正表观类别漂移，却不能证明目标分布稳定；辅助表高匹配率不能证明历史可得。完整月份、固定起点、无未来真值更新的多间隔验证，是区分这些假设的必要设计。

**RQ3：** 按目前已知资源，建议 P1 → P2 → P3 → P4 → P5 的可行性顺序，优先建立小而可诊断的共同基线，再增加复杂度。独立评估均为有条件推进；RMSE、耗时和最终保留方案只能由后续实验确定。

本调研贡献的是将本地 EDA、方法机制和预测信息边界连成同一实验设计，给出可否证的五条管道及预算，而非宣布已经选出最佳模型。本轮到调研评估为止，等待团队审阅。

## 参考文献

[1] Simon N. Wood et al., “Generalized additive models for gigadata: modelling the UK black smoke network daily data,” Journal of the American Statistical Association, 2017.

[2] Douglas Bates et al., “Fitting Linear Mixed-Effects Models Using lme4,” Journal of Statistical Software, 2015.

[3] Yin Lou et al., “Accurate Intelligible Models with Pairwise Interactions,” KDD, 2013.

[4] Harsha Nori et al., “InterpretML: A Unified Framework for Machine Learning Interpretability,” arXiv:1909.09223, 2019.

[5] Liudmila Prokhorenkova et al., “CatBoost: unbiased boosting with categorical features,” NeurIPS, 2018.

[6] Guolin Ke et al., “LightGBM: A Highly Efficient Gradient Boosting Decision Tree,” NIPS, 2017.

[7] Tianqi Chen and Carlos Guestrin, “XGBoost: A Scalable Tree Boosting System,” KDD, 2016.

[8] Richard Meese and Nancy Wallace, “Nonparametric Estimation of Dynamic Hedonic Price Models and the Construction of Residential Housing Price Indices,” Real Estate Economics, 1991.

[9] Everette S. Gardner, Jr. and Ed. McKenzie, “Forecasting Trends in Time Series,” Management Science, 1985.

[10] Yaopei Wang and Yong Tu, “Neighborhood housing rent index construction and spatial discontinuity: a machine learning approach,” AREUEA/AEA conference preview, 2020.

[11] Léo Grinsztajn, Edouard Oyallon, and Gaël Varoquaux, “Why do tree-based models still outperform deep learning on typical tabular data?,” NeurIPS Datasets and Benchmarks, 2022.

[12] Duncan McElfresh et al., “When Do Neural Nets Outperform Boosted Trees on Tabular Data?,” NeurIPS Datasets and Benchmarks, 2023.

[13] Yury Gorishniy, Akim Kotelnikov, and Artem Babenko, “TabM: Advancing tabular deep learning with parameter-efficient ensembling,” ICLR, 2025.

[14] Ivan Rubachev et al., “TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks,” ICLR, 2025.

[15] Jingang Qu et al., “TabICLv2: A better, faster, scalable, and open tabular foundation model,” ICML, 2026; arXiv:2602.11139 v2, 2026-09-16.

[16] Léo Grinsztajn et al., “TabPFN-3: Technical Report,” arXiv:2605.13986, 2026.

[17] Benjamin Jäger et al., “TabPFN-3.5: Technical Report,” arXiv:2609.17895 v2, 2026-09-22.

[18] Christoph Bergmeir, Rob J. Hyndman, and Bonsoo Koo, “A note on the validity of cross-validation for evaluating autoregressive time series prediction,” Computational Statistics & Data Analysis, 2018.

[19] Vítor Cerqueira, Luís Torgo, and Igor Mozetič, “Evaluating time series forecasting models: an empirical study on performance estimation methods,” Machine Learning, 2020.

[20] David R. Roberts et al., “Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure,” Ecography, 2017.

补充教材：Rob J. Hyndman and George Athanasopoulos, “Forecasting: Principles and Practice,” 3rd edition, 2021；相关章节链接已置于正文，访问于 2026-09-26。实现文档的具体核验位置见四份证据底稿。
