# GPU 可行性与反方证据：2026-09-26 增量审计

本底稿采用 deep-research 的两轮核验流程，回答：GPU 改变哪些候选的准入与排序；150k 训练、50k 测试、9,406 楼栋的资源证据到哪里为止；GPU 是否改变最长 16 个月时间外推的验证义务。规模沿用项目 brief/EDA，未重跑。本轮只检索、评估；没有训练、预测、安装模型、启动 GPU、租用资源或提交 Git。**CPU 不再是硬门槛，低成本只作软偏好；用户的 GPU 型号、显存与环境仍未知。**

## 1. 核验过程与资料边界

第一轮检索 `CatBoost GPU supported platforms`、`LightGBM CUDA OpenCL macOS`、`TabPFN 3.5 memory training rows`、`TabICLv2 GPU regression`；第二轮定向核验 `border_count / boosting_type / deterministic / gpu_use_dp`、TabICLv2 Appendix H、TabPFN-3.5 §2.2/2.6 与本地推理 API。来源限定官方文档、作者仓库和论文全文；下列条目均已打开核验，文档能力与作者 benchmark 分开表述。CatBoost Python 安装页、common 参数页多次超时，改用官方发布平台表、FAQ 与量化参数页；不以未打开的搜索摘要证明具体结论。

## 2. 树模型 GPU：可用后端与复现并非同一问题

| 路线 | 已核实支持 | 本项目含义 |
|---|---|---|
| CatBoost CUDA | 官方发布表注明：1.2.10 发布包支持 NVIDIA compute capability ≥3.5，驱动 ≥450.80.02；Linux/Windows 有 CUDA，macOS 无。发布二进制静态链接必要 CUDA 库，源码构建另论。[G1](https://catboost.ai/docs/en/installation/cli-installation-binaries) | `task_type='GPU'` 需要实际兼容设备与构建；现有 `/Users/...` 路径既不证明有 NVIDIA GPU，也不排除团队另有服务器。 |
| LightGBM OpenCL | 当前 stable 文档为 4.7.0：`device_type='gpu'` 在 Windows/Linux 配合 OpenCL；只把直方图计算放在 GPU，其余训练在 CPU。macOS 明确不支持。[G2](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html#build-gpu-version) | 不能把“装有 LightGBM”当作已启用 GPU；旧 GPU tutorial 的 macOS 编译提示与当前安装指南冲突，应采用当前指南。 |
| LightGBM CUDA / ROCm | CUDA 为独立实现：Linux、NVIDIA compute capability ≥6.0；Windows/macOS 不支持。当前文档另列 Linux AMD ROCm 构建，复用 `device_type='cuda'`。[G2](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html#build-cuda-version) | 报告必须同时记录构建与后端，不能仅记录字符串 `cuda`；不将某后端速度外推到另一后端。 |

CatBoost CPU/GPU 默认量化边界数分别为 254/128；一般回归在 CPU 默认 Plain，在 GPU 上超过 50k 行默认 Plain、较小训练折可默认 Ordered。bootstrap 与类别模型正则化也有差异，GPU 不支持 `ctr_leaf_count_limit`。因此切换设备不保证是同一个实验配置，应显式冻结支持的共同参数，再把后端差异单独登记。GPU 浮点求和次序非确定，即使相同 seed 也不能许诺逐位复现。[G3：FAQ](https://catboost.ai/docs/en/concepts/faq#what-are-the-differences-between-training-on-cpu-and-gpu)、[G4：量化](https://catboost.ai/docs/en/references/training-parameters/quantization)、[G5：GPU](https://catboost.ai/docs/en/features/training-on-gpu)

LightGBM 的 `deterministic=true` 仅作用于 CPU，且文档建议同时指定 `force_col_wise` 或 `force_row_wise`；跨版本、编译器、系统仍可能不同。OpenCL 默认单精度累加，可用 `gpu_use_dp=true` 换取双精度；该开关只适用于 OpenCL，CUDA 当前仅支持双精度。减小 `max_bin`（例如 63）有速度动机，却改变离散化，不能把精度或速度变化全归因于 GPU。[G6：参数](https://lightgbm.readthedocs.io/en/stable/Parameters.html)

150k 行并不构成“GPU 必快”的证据。特征数、类别组合、树深、直方图、传输与初始化都会影响收益；9,406 类楼栋尤其要求审查类别交互开销。合理的后续记录是：软件/构建版本、OS、GPU/显存、驱动与运行时、全部有效参数、seed、精度、训练/预测/端到端墙钟、CPU RAM/GPU 峰值及失败原因。首轮单 seed 筛查，只有近似并列或不稳定时追加少量重复；不靠大规模 seed 搜索挑最低误差。这是实验建议，非实测预算。

## 3. 基础模型：150k 不越现行规模上限，但硬件保证仍缺失

| 模型及证据 | 能证明什么 | 对 150k + 50k 仍不能证明什么 |
|---|---|---|
| TabPFN-3.5 Base 官方模型表 | 本地 `tabpfn` 可用，最多 1M training rows、20k columns；行列上限不能同时任意取满。Thinking 为不同服务配置，不能混用其上限/成绩。[G7](https://docs.priorlabs.ai/models) | 150k 行没有因版本上限而被排除；没有本任务显存、速度保证。 |
| TabPFN README | GPU 建议是通用说明；3/3.5 的 CPU 5,000 行保护阈值可覆盖。缓存推理可将大测试集分块，当前默认每批最多 32,768 测试行。[G8](https://github.com/PriorLabs/TabPFN/blob/main/README.md) | “8GB 可用/部分大数据需16GB”不等于本题 150k 上下文可装进16GB；分块测试也不消除训练上下文与缓存的内存。 |
| Jäger 等，TabPFN-3.5 技术报告 v2，2026-09-22 | §2.6 扩展测量用单 RTX PRO 6000 Blackwell 96GB；测单 estimator forward 再乘默认 estimator 数（Base 8、Fast 4）。§2.2 承认 tuned/ensembled MLP 在 grouped、temporal、large 切片仍最高，≤100k 子集才进一步缩小差距。[G9](https://arxiv.org/html/2609.17895v2) | 不可把 overall 领先当作 150k HDB 已胜，也不可把已缓存单行推理速度当全量 fit+50k predict 时间。 |
| Qu 等，TabICLv2 v2，2026-09-16 | Appendix H 的 1M×500 表（80% train / 20% test），在 H100+FlashAttention-3+AMP 下：CPU offload 115秒、约250GB RAM；disk offload 450秒、CPU<24GB/GPU<50GB、250GB磁盘。[G10](https://arxiv.org/html/2602.11139v2) | 这是特定全表前向条件，不是本题训练时间或最低配置；不能按行数线性缩放到消费卡。 |
| TabICLv2 官方 API | 原生回归；当前 device 支持 CUDA/XPU/MPS/CPU。ensemble batch、AMP、KV/representation cache、CPU/disk offload 均影响速度和内存；FA3 加速只适用于相应 CUDA 环境。[G11](https://tabicl.readthedocs.io/en/latest/api.html) | MPS 支持不能推出本机可承受全量，也不能把 H100 测量套给 Apple GPU。降低 ensemble batch 或 offload 都有权衡。 |

资源边界应写成**全量运行值得验证**，而不是“只能抽样”或“任意 GPU 都能全量”。后续应先锁定 feature count、checkpoint、ensemble 数、缓存与精度，在实际设备做小规模可运行性检查，再测试全量上下文及全部 50k 预测的峰值；抽样成功不算全量通过。本轮没有这类实测。

TFM 中的“训练行”通常是预测时输入的有标签上下文，**不等于本项目在这些行上重新训练基础模型权重**。普通 TabICL `fit` 储存/预处理上下文，ICL 的主要计算在 `predict`，启用缓存又会把部分计算移到 `fit`；benchmark 的 train/fit 时间常包含这些前向成本。用户只承担所选 checkpoint 的适配/推理，作者预训练成本不能记成课程项目的训练成本。真正 gradient fine-tuning 是另一条路线，例如 `FinetunedTabICLRegressor`，需单独预算、内层时间验证与权重记录。[G12：机制](https://tabicl.readthedocs.io/en/latest/)、[G13：微调](https://tabicl.readthedocs.io/en/stable/tutorials/finetune_regressor.html)

## 4. 对五条主方案的独立建议

**建议将 Ridge 与 EBM 调整为共同基线/诊断控制，主候选容纳两条 GPU 神经训练路线；但两条不应被解释成必须保留两个神经模型。** 稀疏 Ridge 仍检验加性结构和时间项，EBM 仍提供非线性效应诊断，二者并未因 GPU 可用而失去价值，只是不必各占有限的五个主方案名额。以下是研究/实施优先建议，不是 RMSE 排名：

| 优先 | 主方案 | 新约束下的理由与否证点 |
|---|---|---|
| 1 | CatBoost，CPU/GPU 后端按环境选择 | 成熟类别处理起点；先建立共同时间验证与物业拟合参照，GPU 仅作实现选项。 |
| 2 | LightGBM，CPU/OpenCL/CUDA/ROCm 按环境选择 | 比较另一类类别正则化与树生长机制；若被 CatBoost 在误差与资源上共同支配则停止扩张。 |
| 3 | 属性调整月度趋势 + 固定 CatBoost 残差树 | 针对16个月远期水平；在冻结底座后比较固定水平/阻尼趋势，保持同特征控制。工程有依赖，但研究优先级不应仅因多一步估计而排末。 |
| 4 | TabM 单卡监督训练 | 参数高效 MLP ensemble 提供与树不同的假设；TabM 2025 的跨数据集证据支持把它从“CPU预算后储备”提升为主候选。[G14](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html) |
| 5 | FT-Transformer 单卡监督训练，有条件保留 | 检验 feature attention 是否补充 TabM；2021 工作明确没有通用赢家，官方实现承认其慢于 MLP。TabM 2025 又提供 attention 复杂度未获对应收益的反证，因此该项应晚于 TabM，不因有GPU就优先。[G15](https://arxiv.org/abs/2106.11959)、[G16](https://github.com/yandex-research/rtdl-revisiting-models) |

如果目标更重“不同学习范式”而非两种从头监督训练，完成实际资源核验后，可用 **TabICLv2 或 TabPFN-3.5 的 ICL** 替换第五项。二者不应由于旧版行数限制或“只能CPU”被排除；当前未优先替换的理由是全量上下文资源与对比条件未确认，而非断言昂贵或效果差。TabM/FT-Transformer 的训练 loop、类别映射、早停与归一化也要在同一 cutoff 内完成，不能把原论文配置/耗时直接搬来。

## 5. 排序改变与不变的验证责任

成本成为软偏好后，排序优先考虑任务适配与实验信息量，再以工程和计算成本区分近似方案；取消原15/75分钟准入门槛，记录资源消耗而不把超时判为方法无效。CatBoost 与 LightGBM 后端测试属于各自方案内部控制，不应重复占名额；各路线可在共同协议下并行推进，时间分解仍依赖冻结底座。

**GPU 加速不解决分布漂移。** 普通树换后端不增加外推机制；神经网络可以产生训练时间域外输出，却没有保证正确延续租金市场。TabPFN-3.5 已有 temporal 证据，也未直接覆盖本题长期跨度。五方案必须共用整月、多个历史 cutoff、固定起点的近期与约12–16个月验证，不注入中途真实标签；报告原货币 RMSE、逐预测距离偏差、已见/新楼栋分层。目标编码、楼栋均价、归一化与辅助数据可得性均在折内审计。

综上：GPU 扩大可检验方法集，而非预先指定赢家；TFM 的规模证据支持全量尝试，但实际设备通过情况未定；保留统计诊断、提高时间分解优先级并加入 TabM/条件性第二神经路线，是成本软化后比单纯扩大树模型搜索更有信息量的修订。所有本项目 RMSE、耗时、显存与最终模型均未验证。
