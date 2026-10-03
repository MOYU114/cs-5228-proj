# 现代基础模型：回归接口、上下文与资源边界

核验日期：2026-09-26。增量底稿，沿用 `RESEARCH_BRIEF.md` 的 150,000 行训练、50,000 行测试、9,406 栋楼、最长 16 个月预测间隔。GPU 可用作实验路线，低成本仅为软偏好；用户实际 GPU、显存及软件环境未知。本次只读论文、文档与源码，未安装、下载权重、训练、预测或使用付费资源。

## 1. 三篇工作的证据不能混为三次独立验证

| 工作与版本 | 发表状态与机制 | 对本项目的有效结论及边界 |
|---|---|---|
| Qu、Holzmüller、Varoquaux、Le Morvan，*TabICLv2: A better, faster, scalable, and open tabular foundation model*；首稿 2026-02-11，v2 2026-09-16 | ICML 2026 会议论文；本次读作者 arXiv v2 修订全文。合成先验、可扩展 attention 与预训练改进，增加回归能力。[论文记录](https://arxiv.org/abs/2602.11139)、[会议清单](https://icml.cc/Downloads/2026) | 作者在 TabArena/TALENT 报告优于 RealTabPFN-2.5；这不能推出优于随后发布的 3.5，更不是 HDB 验证。百万行实验支持研究 full-context 路线的可行性，但具体设备仍需测量。[全文](https://arxiv.org/html/2602.11139v2) |
| Grinsztajn 等，*TabPFN-3: Technical Report*；首稿 2026-05-13，v2 2026-05-28 | 作者技术报告 / preprint；本次未核实同行评审发表。压缩行表示、缩小 KV cache、分块预测。[论文记录](https://arxiv.org/abs/2605.13986) | 是 3.5 的架构与资源对照，不必另占主候选名额。百万训练行与快速重复查询是作者条件化实验结论；本地 Base、API Plus、Thinking 和专门时间序列 checkpoint 不可混用成绩。[全文](https://arxiv.org/html/2605.13986v2) |
| Jäger、Erickson、Grinsztajn 等，*TabPFN-3.5: Technical Report*；首稿 2026-09-15，v2 2026-09-22 | 作者技术报告 / preprint；本次未核实同行评审发表。分类/回归共用 checkpoint，扩大模型但保留较紧凑缓存，并提供 Fast 变体。[论文记录](https://arxiv.org/abs/2609.17895) | 作者报告整体及非 i.i.d. 能力进步，但正文 §2.2 中调优并集成的 MLP 在 grouped、temporal、large 切片仍领先；≤100k 子集结果不能直接套到 150k。GPU 加速没有解决未来租金漂移。[全文](https://arxiv.org/html/2609.17895v2) |

以上是三篇工作、两个主要作者体系；两篇 PriorLabs 报告不是两次独立复现。TabICLv2 的同行评审身份也不能替代本地复现。本次未找到三者在相同 HDB 时间切分、相同信息预算下的独立比较；因此不写 RMSE 名次或预期提升百分比。

## 2. 当前官方接口、版本与访问条件

| 核验项 | TabICLv2 | TabPFN-3.5 |
|---|---|---|
| 本地回归 API | `from tabicl import TabICLRegressor`；`fit(X_train, y_train)` 后 `predict(X_test)`；默认 checkpoint `tabicl-regressor-v2-20260212.ckpt`。[官方仓库](https://github.com/soda-inria/tabicl) | `TabPFNRegressor.create_default_for_version(ModelVersion.V3_5)`；3 对照用 `V3`，Fast 用 `V3_5_FAST`。应显式选版本，避免默认值随包更新。[版本文档](https://docs.priorlabs.ai/models/selecting-model-version) |
| 权重日期与许可 | 官方 Hugging Face 模型卡列 BSD-3-Clause；公开回归权重。模型卡没有展示事前审批门槛，但本次未实际下载。[模型卡](https://huggingface.co/jingang/TabICL) | `tabpfn-v3.5-20260909.safetensors` 是共用分类/回归权重；Fast 单列。权重使用 `tabpfn-3-5-license-v1.0`，与代码 Apache-2.0 分开；官方模型页标非商业使用许可。[模型卡](https://huggingface.co/Prior-Labs/tabpfn_3_5)、[模型说明](https://docs.priorlabs.ai/models)、[代码许可](https://github.com/PriorLabs/TabPFN/blob/main/LICENSE) |
| 获取与离线运行 | 首次使用可能自动下载 checkpoint；以后可显式指定本地路径。[官方仓库](https://github.com/soda-inria/tabicl) | 当前官方流程要求 PriorLabs 登录并接受对应模型许可；无界面服务器可用 `TABPFN_TOKEN`。已下载权重可本地缓存运行；首次仅指定缓存目录不保证离线。下载认证不等于云推理或上传 HDB 数据。[访问文档](https://docs.priorlabs.ai/models/accessing-model-weights) |

对于 RMSE，两者回归接口均有默认均值输出；不能悄悄换成中位数或分位数再称同一目标。TabICL `batch_size` 是同时处理的 ensemble views 数量，不是测试行数；需要另行控制 50k 测试行分块。[TabICL API](https://tabicl.readthedocs.io/en/latest/api.html)、[TabPFN 回归源码](https://raw.githubusercontent.com/PriorLabs/TabPFN/main/src/tabpfn/regressor.py)

## 3. 150k 上下文与 50k 分批查询的工程区别

| 执行方式 | 是否反复处理训练上下文 | 主要代价 |
|---|---|---|
| TabICL 默认 `kv_cache=False` | 每次 `predict` 都把训练与测试视图送入模型；外部分批会重复处理 150k 上下文 | 较少持久缓存，较多重复计算 |
| TabICL `kv_cache=True` / `"kv"` | `fit` 建立列与 ICL 的 KV；后续查询复用 | 更大的持久缓存，但适合多次测试查询 |
| TabICL `kv_cache="repr"` | 复用列 KV 与行表示，仍重跑 ICL transformer | 文档所述约 24 倍压缩只指 ICL 缓存组成部分，不能写成整体 RAM/VRAM 缩小 24 倍 |
| TabPFN 默认 `fit_preprocessors` | 缓存预处理，不等于缓存 transformer KV；重复查询仍有上下文神经计算 | 不能把 `fit` 完成理解为之后查询已无 150k 上下文成本 |
| TabPFN `fit_with_cache` | `fit` 完成预处理及 KV 缓存，后续测试块复用 | 较高缓存内存；可减少重复编码 |

上述分支由 API 与当前源码交叉确认：[TabICL API](https://tabicl.readthedocs.io/en/latest/api.html)、[TabICL predict 实现](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_sklearn/regressor.py)、[TabPFN fit modes](https://raw.githubusercontent.com/PriorLabs/TabPFN/main/src/tabpfn/regressor.py)。不能只看 `fit` / `predict` 命名判断计算量。

TabICL 的 `offload_mode="cpu"/"disk"` 主要作用于列嵌入中间输出，不意味着所有持久 KV 都可自动逐层搬到 CPU；当前基础类把加载的缓存移到所选计算设备。保存缓存时搬到 CPU 是持久化行为，不能等同运行时卸载。[配置与缓存源码](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_sklearn/base.py)

TabPFN 当前回归类提供 `keep_cache_on_device=False`，可将 KV 保留于 CPU，用传输时间换 GPU 空间；`memory_saving_mode` 的内部计算分块不能替代输入及缓存的完整内存账。当前 README 的 cached 模式另有 `TABPFN_MAX_BATCHED_TEST_ROWS`，默认 32,768，50k 测试会拆块；官方称给定缓存后测试行条件独立，但浮点结果可能略有差别。该默认必须按最终锁定版本复核，不能把 `main` 文档等同所有已发布包。[回归实现](https://raw.githubusercontent.com/PriorLabs/TabPFN/main/src/tabpfn/regressor.py)、[分块说明](https://github.com/PriorLabs/TabPFN/blob/main/README.md?plain=1)

**资源证据的尺度：**TabICLv2 Appendix H 的百万总行、500 列、80/20 切分实验使用 H100、FlashAttention-3 与混合精度，CPU/disk offload 存在明显 RAM、磁盘和时间互换。TabPFN-3 §2.4 使用 H100 80GiB 展示单 estimator 缓存；3.5 §2.6 使用 RTX PRO 6000 Blackwell 96GB，图中的计时按单 estimator 前向再乘默认 ensemble 数。三者均不是本项目 150k+50k 的端到端消费卡实测，不能按行数比例承诺时长或最低显存。[TabICLv2 全文](https://arxiv.org/html/2602.11139v2)、[TabPFN-3 全文](https://arxiv.org/html/2605.13986v2)、[TabPFN-3.5 全文](https://arxiv.org/html/2609.17895v2)

## 4. 推理、微调、从头训练须分开报告

默认 `TabICLRegressor` / `TabPFNRegressor` 使用作者预训练参数；HDB 的带标签记录是推理上下文。`fit` 可能学预处理或建缓存，但不等于从 HDB 梯度训练基础模型。作者从合成任务预训练模型的计算成本，也不能记作团队每个 fold 的成本。

两者现已核实**回归微调入口**：TabICL 有 `FinetunedTabICLRegressor` 与显式 `X_val/y_val`；TabPFN 有 `FinetunedTabPFNRegressor`，当前文档专门说明升级到 3.5 必须从 3.5 checkpoint 重新微调，不能靠更新软件升级已有权重。TabPFN 源码还显式传递 `model_version`，因此这不是从 2.5 能力外推。微调才会更新预训练网络参数，仍不等于从头预训练。[TabICL 微调教程](https://tabicl.readthedocs.io/en/stable/tutorials/finetune_regressor.html)、[TabPFN 微调文档](https://docs.priorlabs.ai/capabilities/fine-tuning)、[微调版本源码](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/finetuning/finetuned_regressor.py)

主实验先比较默认预训练推理，微调作为独立消融。TabPFN 文档提醒渐变时间漂移下微调可能不稳定，其默认随机留出不能直接沿用；两者都应显式给时间验证集。TabPFN 微调接口与普通接口的文本/日期自动展开并不相同，预处理需另行锁定。[微调边界](https://docs.priorlabs.ai/capabilities/fine-tuning)

## 5. 对现代五主项的独立建议

**两条基础模型路线都入主候选。**不能继续按旧 CPU 时间预算淘汰 TabICLv2 / TabPFN-3.5。TabM、ModernNCA、TabICLv2、TabPFN-3.5、TabM 加 feature-aware temporal modulation 的组合覆盖参数学习、邻域检索、预训练上下文和显式时间适配；Ridge、EBM、经典树可退到必要的诊断/强基线层，不占这五个名额。此处是实验信息量判断，不是五者精度排名。

建议的实施次序：先建立可复用 TabM 与统一时间切分；TabICLv2、TabPFN-3.5 可并行做版本、缓存和资源试验；ModernNCA 的参考库与查询代价由其独立底稿约束；时间调制在固定 TabM 后配对加入，才能归因增益。仅就本次核验的两条基础模型而言，TabICL 公开许可和无登录示例使接入较直接，TabPFN 当前 CPU 缓存与 cached 测试分块接口也有工程价值，不应凭发表更新或品牌预设胜负。[TabM 会议论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html)

后续获准实验时，在实际单卡记录版本/commit、checkpoint hash、设备、精度、列数、ensemble 数、上下文大小、冷启动、建缓存、完整 50k 查询时间、峰值 RAM/VRAM/磁盘；先验证缓存/分批数值一致性，再扩到 full 150k。单 estimator 的资源探测不能冒充默认 ensemble 成绩。所有 fold 只允许使用预测起点以前的标签，16 个月长间隔期间不能偷加中途真实租金；单独报告远期月份与未见楼栋误差。

若设备不足，先调缓存、查询块和 ensemble；上下文检索/子采样可作为另一个明确命名的变体，但它改变可见标签及信息预算，必须与 full-context 分列，不得称为无损实现优化。无需从 macOS 路径推断 NVIDIA 可用，也不能因路径在 macOS 就排除所有 GPU 后端。

## 检索复核记录

第一轮从论文版本页、全文、作者仓库和模型卡核对身份、回归能力、规模与许可；第二轮定向核对官方 API 的 `kv_cache`、`fit_with_cache`、`keep_cache_on_device`、`FinetunedTabPFNRegressor` 与 `model_version`，追到预测分支及缓存搬运源码，并反查论文硬件与 temporal 切片。仅采用以上 primary sources。ICML 单篇活动页访问失败，会议身份用官方 2026 清单和论文记录交叉核对；文档 `main/latest` 与固定发行版可能不同，实施前须锁定。未用独立复现、GPU 实测或本地 RMSE 填补目前证据缺口。
