# 低成本结构化统计方法证据：HDB 租金预测

检索与核验日期：2026-09-26。此文件是主调研的独立检索视角，不扩展最终恰好五项候选的数量。没有训练模型，没有得到本项目验证 RMSE 或运行时间。

## 研究问题与约束

- RQ1：哪些成熟、低成本的统计机制适配 15 万行 HDB 租金表格回归？
- RQ2：如何处理最长 16 个月的时间漂移与楼栋层次？
- RQ3：普通 CPU 下的成本、实现风险与适用边界是什么？

项目事实来自主代理的本地数据审计：训练期 2021-01 至 2025-03，共 51 个月；测试到 2026-07；训练 150,000 行，测试 50,000 行；9,406 个训练楼栋，测试 99.796% 的行来自已见楼栋；50 个未见楼栋共 102 行。训练首末月份均值约 2048.9 → 3162.7，末段呈平台；没有单元 ID。相同输入但租金不同涉及 17,900 组、39,540 行。**4–8 CPU 核、16 GB 内存仅为规划预算，不是已测用户设备配置。**

## 两轮检索日志

| 轮次 | 查询 | 处理 |
|---|---|---|
| 第一轮：宽检索 | `generalized additive models large data sets discretization Wood Li Shaddick Augustin 2017 2018` | 初始结果混杂，第二轮用确切题名补查 |
| 第一轮：宽检索 | `fitting linear mixed effects models lme4 Bates Maechler Bolker Walker 2015` | 找到 JSS 正式论文及作者 arXiv 全文，纳入 S2 |
| 第一轮：宽检索 | `Gardner McKenzie 1985 forecasting trends time series damped trend` | 找到 INFORMS 正式页及作者大学主页 PDF，纳入 S3 |
| 第二轮：核验 | `"Generalized additive models for gigadata" Wood Li Shaddick Augustin` | 作者所在大学研究库完整元数据和摘要，纳入 S1 |
| 第二轮：核验 | `"Nonparametric Estimation of Dynamic Hedonic Price Models" "Meese" "Wallace"` | Wiley 正式页与 Berkeley 机构库 working paper，纳入 S4；仅用正式页摘要可确认结论 |
| 第二轮：背景检查 | `"Hedonic Prices and Implicit Markets" "Rosen" 1974` | 出版社确认元数据，但未核读足够理论正文，因此不作为方法性能证据 |
| 实现核验 | 官方 `mgcv` 的 `random.effects`/`bam`；`lme4` 的 `predict.merMod`；scikit-learn 的 `SplineTransformer`/`Ridge` | 检查稀疏性、外推默认值、新类别回退和求解器；软件文档不替代论文实证 |

S1–S4 均经题名、作者、年份、刊物确认，并打开 primary source 对照所用论断。S2 的 JSS 猜测 PDF 路径返回错误，改用作者 arXiv 全文，正式发表信息仍以 JSS 为准。S4 机构库 PDF 能打开但文本抽取为空，因此没有声称从其正文读取结果。

## 已核验论文与证据强度

| ID | 精确文献信息 | 已核验结论 | 局限 |
|---|---|---|---|
| S1 | Simon N. Wood, Zheyuan Li, Gavin Shaddick, Nicole H. Augustin. **Generalized additive models for gigadata: modelling the UK black smoke network daily data.** 2017, *Journal of the American Statistical Association*, 112(519):1199–1210. DOI: 10.1080/01621459.2016.1195744. | **VERIFIED**。[作者机构页](https://www.research.ed.ac.uk/en/publications/generalized-additive-models-for-gigadata-modelling-the-uk-black-s/) 摘要确认：惩罚样条 GAM 利用新的迭代、并行和边际离散化降低大数据计算成本；动机为英国空气污染监测。2016 是提前在线年份，正式卷期为 2017。 | 不能将算法可扩展性推广到任意 GAM 软件或任意高维随机效应；未证明 HDB RMSE 优于树。 |
| S2 | Douglas Bates, Martin Mächler, Ben Bolker, Steve Walker. **Fitting Linear Mixed-Effects Models Using lme4.** 2015, *Journal of Statistical Software*, 67(1):1–48. DOI: 10.18637/jss.v067.i01. | **VERIFIED**。[正式页](https://www.jstatsoft.org/article/view/v067i01)；[作者全文](https://arxiv.org/html/1406.5823) §2.2、§2.3、§3.7。随机效应实现收缩，分组矩阵具有稀疏结构，算法利用该结构拟合 ML/REML。 | 这是统计计算方法，不是 HDB benchmark；楼栋效应不会恢复未观测 unit 差异，也不自动预测未来市场。 |
| S3 | Everette S. Gardner, Jr., Ed. McKenzie. **Forecasting Trends in Time Series.** 1985, *Management Science*, 31(10):1237–1246. DOI: 10.1287/mnsc.31.10.1237. | **VERIFIED**。[出版社](https://pubsonline.informs.org/doi/10.1287/mnsc.31.10.1237)；[作者大学 PDF](https://www.bauer.uh.edu/egardner/3301H%20Operations%20Management/ESG%20Publications/1985%20Forecasting%20trends.pdf) §3、Exhibits 2–4。在 1,001 条序列比较中，阻尼趋势对较长预测期更有利，短期差异较小；所读图表使用 APE。 | 非本项目 RMSE 实验；不保证未来租金趋平或任意市场拐点可预测。 |
| S4 | Richard Meese, Nancy Wallace. **Nonparametric Estimation of Dynamic Hedonic Price Models and the Construction of Residential Housing Price Indices.** 1991, *Real Estate Economics*, 19(3):308–332. DOI: 10.1111/1540-6229.00555. | **VERIFIED，限摘要层面**。[出版社摘要](https://onlinelibrary.wiley.com/doi/abs/10.1111/1540-6229.00555)。Alameda/San Francisco 房产数据比较参数 hedonic 与 locally weighted regression；后者容许曲率及时间、地区间函数差异，并用于地区价格指数。 | 是房价指数与局部回归研究，不是新加坡租金、GAM、GBDT 或未来 16 个月 forecasting benchmark。 |

S4 支持住房属性关系不必强制全局线性，而 S1 提供控制平滑复杂度的计算路线，两者是应用动机与通用计算的互补证据。S2 处理楼栋间信息共享，S3 处理共同趋势的未来延续，不能互相替代。结合这些模块的最终管道应明确是**由成熟机制推导的本项目设计**，不能写成某篇论文已证明完整组合有效。

## 可执行设计及支持/反对证据

### 一条可解释候选：稀疏样条 hedonic + 收缩楼栋项

可在最终五项中占一项；定义成完整预测管道，不能仅称“特征工程”。例子为：

`predicted_rent = intercept + spline(area) + spline(age) + categorical_effects + building_effect + market_time_effect`。

类别包括房型、model、town；面积、屋龄用少量 B-spline 基函数；楼栋效应受正则化；时间项显式规定训练期形状和截止日期后的外推规则。主损失优先原始租金平方损失以对应 RMSE。对数租金可作为有限消融，但还原后的条件均值不等于直接指数化的对数预测，不能默认对数损失更好。

轻量 Python 路线是 `SplineTransformer` + 稀疏类别编码 + `Ridge(solver="lsqr")`。这是惩罚样条基展开的近似加性管道，**不应宣称与自动选择各平滑参数的完整 `mgcv` GAM 完全相同**。如果已有 R 环境并希望显式估计楼栋方差，可改用低维固定样条基 + `lme4::lmer(... + (1|building_id))`。它是同一候选的实现替代路线，不额外拆出候选凑数。

在已扣除属性和市场项的简化 Gaussian 楼栋残差模型中，收缩估计形式为 `u_b = n_b/(n_b+lambda) * mean_residual_b`；`lambda` 对应方差比或惩罚强度。这只是简化模型的推导，不是任意全模型的精确闭式解。小样本楼栋向总体属性预测收缩；新楼栋取 `u_b=0`，仍保留 town/房型等属性信息。99.796% 测试行见过楼栋说明此信息值得使用，不意味着未来租金不漂移。

**支持**：楼栋小样本有明确收缩机制；可画面积/屋龄效应；稀疏运算适合 CPU；Pearson 相关不能排除非线性价值。**反对**：默认加性不能覆盖复杂的面积×房型×地点交互；楼栋截距不能解释同楼栋不同 unit；时间外推仍需额外选择。仅在证据支持时增加少量交互，如房型×面积样条，避免楼栋×月份的密集交互。

### 时间趋势 + 残差：可接到树或上述管道的工程推导

`rent = market_level(t) + property_residual(x)` 是本报告推导，不是已经验证的 HDB 算法。51 点月度序列计算很便宜，但**月均租金变化可能混有房型和地区交易构成变化**。建议控制物业属性后估计月份效应，或至少检查相对稳定 town×房型构成的趋势；不能把月均上涨直接解释为市场因果变化。

截止月后比较最近水平、局部线性和阻尼趋势三种简单规则。阻尼预测写成 `level_T + (phi + ... + phi^h) * slope_T`，其中 `0 < phi < 1`。S3 支持将阻尼作为稳健候选；末段平台和 16 个月跨度使全期无阻尼线性外推有建模风险。**公式能够外推不等于未来将延续同一趋势。**

每个时间折都重新估计月份项、趋势和残差模型。如果月份项仅用于训练折内部构造残差标签 `y-m_t`，使用训练标签并非自动泄漏；但若把包含本行标签的聚合量当成本行输入特征，或依据同样本拟合分数选择二阶段组合，需要严格的 out-of-fold/时间生成。验证期真实租金不能进入 trend，不能用于更新楼栋效应。

不推荐先为 9,406 个楼栋各拟合独立 ARIMA/ETS：51 个月的全局跨度不表示每楼栋有 51 个充分观测点；稀疏短序列、数千次拟合和缺乏信息共享增加风险。共同趋势 + 稀疏楼栋截距更节约，只有验证支持时再增加 town 层趋势。

## CPU 成本与实现边界

以下是结构性预算判断，**不是实测**。

| 模块 | 成本来源 | 约束 |
|---|---|---|
| 稀疏样条 + Ridge | 约一万类别/基函数列仍可用稀疏迭代；成本随非零元及迭代次数变化 | 保持 CSR/CSC，少量结点与正则候选，禁止无必要转 dense |
| 楼栋 random intercept | 每行只激活对应楼栋，分组结构稀疏 | 使用实际利用稀疏性的实现，先只做 intercept |
| 共同月级趋势 | 51 个历史月，拟合成本低 | 主风险是趋势假设和构成混淆 |
| 复杂交互/逐楼栋时间模型 | 列数、优化次数和依赖增加 | 仅根据干净验证中的稳定残差增加复杂度 |

仅楼栋 dense float64 主矩阵就约为 `150000 × 9406 × 8 = 11.29 GB`，尚未含副本或求解器；一个每行只有一个非零元的楼栋稀疏矩阵则只有 150,000 个非零项。因此不能用“15 万行不多”证明 dense one-hot 无成本问题。

官方实现核验：

- [`SplineTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html) 支持稀疏输出，外推有 `constant`、`linear` 等选项，默认 `constant`。必须显式记录时间边界行为。
- [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) 的 `lsqr` 支持稀疏输入及截距。基函数的尺度会影响惩罚含义，数值与类别效应须审慎处理。
- [`mgcv` random effects 文档](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/random.effects.html) 明确普通 `gam` 对大量随机效应不利用其稀疏性，可能慢；[large-data `bam`](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/bam.html) 有离散化与并行机制，但不能推出任意高维模型都便宜。
- [`lme4` prediction 文档](https://lme4.github.io/lme4/reference/predict.merMod.html) 中 `allow.new.levels=TRUE` 可让未见分组使用 population-level 值；固定效应的新类别仍需另行处理。

## 泄漏、验证和解释边界

1. **完整月份的时间切分**：需覆盖接近 16 个月的长跨度，同时保留末段短跨度。可做训练截止 2023-11、验证 2023-12 至 2025-03 的长窗口；因其市场阶段不等于真实预测起点，还需较近短窗口。随机划分或只验证一个月不能回答部署问题。
2. **已见楼栋重叠不是自动泄漏**：真实测试主要为已见楼栋，保留历史合理；验证日期后的楼栋租金绝不可用。另报新楼栋结果，不能以纯 building-group split 取代主要时间切分。
3. **目标派生量只用训练折**：楼栋均值、town×房型均值、月份指数、平滑参数、趋势窗口均如此。最终拟合可用训练标签截止 2025-03，不能用此后的真实目标更新。
4. **同 X 不同 y 不能随意删重**：没有 unit ID，可能是不同合同；确定性模型对相同输入只能输出相同值，但这不能直接给出测试不可约误差。若为平方损失压缩完全同 X 行，平均标签必须配原组计数权重；树的叶子计数/采样语义仍可能改变，不作为通用优化。
5. **时间与屋龄混淆**：如屋龄由记录日期减建成年份构造，不能把同一时钟的效应都解释为独立因果效应；应按每条记录日期一致计算，保持预测关联解释。
6. **原值 RMSE**：汇总用总 SSE 与总行数计算 pooled RMSE，并看月份/horizon、房型及 seen/unseen building。平均各折 RMSE 与 pooled RMSE 不相同。

## 对主报告的建议

- RQ1：低维样条 + 稀疏正则化楼栋项是成熟、可解释的 CPU 候选；作用是稳健比较对象和结构诊断，精度排名留待训练。
- RQ2：共同时间走势与物业差异应区分；楼栋用信息共享截距，未来时间用受控水平/局部/阻尼规则；融合是本项目设计，须时间验证。
- RQ3：实现成本上，稀疏 Ridge 基展开较易落地；显式 mixed effects 增加依赖和统计复杂度；复杂非线性可能更适合树，但本文献不能预判本项目精度排序。不要承诺未实测分钟数或 RMSE 改善值。

课程项目可以通过机制与数据结构匹配、无泄漏比较和消融体现价值，无须把成熟方法重命名为新算法。证据不支持“更复杂一定更好”，也不支持“末段平台必然持续”。
