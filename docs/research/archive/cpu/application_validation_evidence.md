# 应用、验证与可解释实现：检索和核验记录

访问日期 2026-09-26；工具 web search/open，primary sources 为准。

## 搜索轨迹

第一轮：`Singapore HDB rental price prediction hedonic machine learning temporal validation research paper`；`Roberts 2017 cross validation strategies data temporal spatial hierarchical phylogenetic structure`；`Cerqueira Torgo Mozetic 2020 Evaluating time series forecasting models empirical study performance estimation methods`。

第二轮：完整标题检索 Cerqueira 与 Bergmeir 的论文；`HDB rental hedonic research Singapore housing`；`Neighborhood housing rent index construction and spatial discontinuity Wang Tu 2020`；完整标题检索 Lou 2013、Nori 2019；官方 InterpretML 参数和 fit/bags 接口核验。

通用搜索返回的 Reddit/产品页面不作为效果证据。Regensburg 的旧论文入口被导航页替代，改为期刊原页；PMC 一个入口遇到验证码，未用该页作为事实来源。既没有隐藏这些检索限制，也没有因此从记忆补写结论。

## 已核验资料

| 工作 | 一手核验链接与位置 | 可使用的结论 | 边界 |
| --- | --- | --- | --- |
| Vítor Cerqueira, Luís Torgo, Igor Mozetič, “Evaluating time series forecasting models: An empirical study on performance estimation methods”, Machine Learning 109, 2020；preprint 2019 | [作者提交的 arXiv 页面](https://arxiv.org/abs/1905.11744)，Abstract 与 Journal reference | 研究区分平稳和真实非平稳情形，后者保留时间顺序的样本外评估更可靠 | 不是本课程数据的验证结果；仅据摘要概括，不引用具体性能比率 |
| Christoph Bergmeir, Rob J. Hyndman, Bonsoo Koo, “A note on the validity of cross-validation for evaluating autoregressive time series prediction”, Computational Statistics & Data Analysis 120, 2018 | [作者页面](https://robjhyndman.com/publications/cv-time-series/)，Abstract、Cite | 纯自回归且误差不相关等条件下，普通 K-fold 可以成立 | 反驳“所有时间数据永远不能 CV”的绝对说法；当前房屋横截面+趋势任务未验证其条件 |
| David R. Roberts et al., “Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure”, Ecography 40, 2017；online first 2016 | [期刊原页](https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881)，Abstract | 结构相关数据忽略依赖会低估误差；阻断方式也可能意外改变外推任务 | 并不推出本项目必须全楼栋互斥；test 大多是历史已见楼栋，主测试应对齐未来月份，未见楼栋单列 |
| Yaopei Wang, Yong Tu, “Neighborhood housing rent index construction and spatial discontinuity: a machine learning approach”, AREUEA/AEA 2020 conference preview | [会议 PDF](https://www.aeaweb.org/conference/2020/preliminary/paper/T587D4ZR)，p.1 Abstract、pp.8–9 Data/Design；[会议议程](https://www.aeaweb.org/conference/2020/preliminary/1979) | 直接研究 Singapore HDB open-rental，结合 hedonic 与 state-space，说明房屋属性和共同时间状态的联合建模有领域依据 | 会议稿而非核实的期刊最终版；原文明确 smoother 利用时点前后信息。不能将回溯指数重建成绩当作16个月前向预测证据，不照搬其将 state-space 称 reinforcement learning 的分类 |
| Yin Lou, Rich Caruana, Johannes Gehrke, Giles Hooker, “Accurate Intelligible Models with Pairwise Interactions”, KDD 2013 | [Microsoft Research 原文](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/06/kdd13.pdf)，第一页作者/Abstract；[第一作者项目](https://www.cs.cornell.edu/~yinlou/projects/gam/) 的 Reference | 在加性主效应中加入少量成对交互并保持可视化，是 EBM 候选的机制依据 | 论文数据集的结果不等于本项目效果；没有证明一般因果可解释性 |
| Harsha Nori, Samuel Jenkins, Paul Koch, Rich Caruana, “InterpretML: A Unified Framework for Machine Learning Interpretability”, arXiv:1909.09223, 2019 | [arXiv](https://arxiv.org/abs/1909.09223)，Title/Authors/Abstract | 开源 Python 框架，包含 EBM 实现；用于确定可落地工具而非证明效果 | preprint；当前 API 应另查官方文档 |

上述六项：存在性与所采用主张均 VERIFIED。仅使用上列核验范围，不引用未读的完整实验数字。

## 官方实现资料

- [ExplainableBoostingRegressor](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html)：已核验 objective='rmse'、n_jobs、max_bins、outer_bags、interactions、estimate_mem；fit 的 bags 可按样本指定 +1 训练 / -1 验证。后续应以外层训练截止内最后完整月份作内部 early stopping；不能把外层验证当内部 bag 后仍称无偏最终分数。
- [EBM Hyperparameters](https://interpret.ml/docs/hyperparameters.html)：较少 outer_bags 可减少计算；这只是速度控制机制，不意味着降低参数后精度不变。预算草案须显式限制自动交互和 boosting rounds。
- [SplineTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html)：支持 sparse_output 与 explicit extrapolation。房屋属性样条可用端点常量延伸；月份若要外推必须独立明确设计，不能默认样条天然会预测趋势。

## 归纳与反例

Bergmeir 给出了 K-fold 成立的条件，而 Cerqueira 在非平稳情况下更支持保序留出；Roberts 又提醒分组阻断应匹配真实部署目标。由此推导当前主评估应是按完整月份向前，并将新楼栋作为额外子组，不能用随机 CV 或全楼栋互斥 CV 单独决定排名。

Lou 的交互加性机制与 Nori 的开源实现支持可解释的 CPU 候选，但 EBM 并没有解决超出训练月份的时间趋势；Wang/Tu 的时间状态思路提供互补启发，其双向平滑也展示了不得照搬的泄漏风险。这些是机制和设计推论，尚无本项目 RMSE 或运行成本测量。
