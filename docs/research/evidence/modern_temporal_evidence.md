# 现代时间表格方法：证据与适配审计

访问日期：2026-09-26。只检索论文和官方代码，未训练、下载权重或改动数据。问题在增量检索前冻结于 [RESEARCH_BRIEF.md](../RESEARCH_BRIEF.md)：15 万行、51 个训练月份、最长 16 个月未来间隔下，现代方法能否提供可检验的时间适应机制？

## 两轮检索和证据强度

首轮沿 `temporal tabular learning 2025 temporal shift`、`TabReD temporal benchmarks` 和 TabM 相关文献查找；第二轮按完整题名、`Feature-aware Modulation temporal tabular code`、`tabm_modulated` 查出版版本、公式和官方实现。阅读出版方摘要、NeurIPS 全文及官方仓库，不用搜索摘要承担具体性能结论。

| 核验文献 | 已确认内容 | 证据边界 |
| --- | --- | --- |
| Ivan Rubachev, Nikolay Kartashev, Yury Gorishniy, Artem Babenko. **TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks.** ICLR 2025。[正式页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/571799482291411607c54984153190b0-Abstract-Conference.html)、[预印本](https://arxiv.org/abs/2406.19380) | 时间切分与实际数据工程会影响 benchmark 结论，不能从随机划分排名直接推出未来表现。 | 不是本地 HDB 的方法胜负证据。 |
| Haorun Cai, Han-Jia Ye. **Understanding the Limits of Deep Tabular Methods with Temporal Shift.** ICML 2025，PMLR 267:6366–6386。[正式页](https://proceedings.mlr.press/v267/cai25j.html) | 研究训练滞后、内部验证偏差，提出 Fourier 时间 embedding 与调整后的训练/验证协议。 | 本轮以出版方摘要核实上述层级。PDF 链接读取失败，不引用其具体提升量；摘要中随机划分的讨论不能解释为应随机拆分本项目外层未来评估。 |
| Haorun Cai, Han-Jia Ye. **Feature-aware Modulation for Learning from Temporal Tabular Data.** NeurIPS 2025。[正式页](https://papers.nips.cc/paper_files/paper/2025/hash/ddc20f403d0d296c87940393838d5971-Abstract-Conference.html)、[全文](https://proceedings.neurips.cc/paper_files/paper/2025/file/ddc20f403d0d296c87940393838d5971-Paper-Conference.pdf)、[官方代码](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation) | 时间 embedding 驱动逐特征平移、缩放及形状变换；官方提供 TabM 的普通、temporal、modulated 三条实现。 | 实际 HDB 远期表现未验证，论文其他表格任务的效果不能当作本地结果。 |

这三篇依次提供评估问题、时间输入与训练协议、特征映射适应三个层次。N5 选择最后一篇的 TabM 版本，是为了检验时间相关属性映射；不是把旧趋势方法重新命名为新模型。

## 机制及源码核验

NeurIPS 全文第 5–6 页式 (2) 以时间表示 ψ(t) 产生 γ、β、λ，然后对特征做 Yeo–Johnson 形状变换和缩放/平移。该公式可作用于输入、中间或输出表示；本项目首个方案严格采用官方 TabM 输入调节路径，不额外宣称所有插入位置已核验。

[model/models/tabm_modulated.py](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation/blob/main/model/models/tabm_modulated.py) 的构造器接收 t_mean、t_std，建立 FullTemporalEmbeddings；数值输入经过 TemporalModulation，类别先 one-hot 再调节，最后进入共享 TabM 主干。forward 接收 x_num、x_cat、idx，未要求未来目标。类目宽度仍随类别数增长，因此 9,406 楼栋不仅影响普通 TabM，也影响调节模块。

[仓库 README](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation) 公开 `tabm`、`tabm_temporal`、`tabm_modulated`，说明 N5 是可实现的论文变体。README 的大规模调参与 `holdout_foremost_sample` 是作者实验设定，不应未经审查直接作为本项目外层验证。当前仓库 main 会更新；后续须锁定 commit、依赖和训练配置。

另一代理进一步核验了 commit `8feabbd580d59757dd108622ea93b9bdb6d38fa1`。`indices` 的时间均值/标准差取原始 `M['train']`，而非重切分后的 `M_trainval['train']`；`holdout_last` 情况一致，其他重划分可能把内部验证行纳入统计。因此每折应重新拟合并覆盖 `info.M_mean/M_std`。推理只使用保存的统计，不按当前 batch 重拟合。[统计来源](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation/blob/8feabbd580d59757dd108622ea93b9bdb6d38fa1/model/lib/data.py#L606-L609)、[统计传递](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation/blob/8feabbd580d59757dd108622ea93b9bdb6d38fa1/model/methods/base_temporal.py#L75-L86)、[时间 embedding](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation/blob/8feabbd580d59757dd108622ea93b9bdb6d38fa1/model/lib/temporal_embeddings.py#L95-L126)。

`period/period_t` helper 在 train、val、test 各分片分别计算日期 mean/std，确实使用各自时间分布；但 TabM temporal 强制 `indices`，N5 不走该 helper。共同协议应禁用或改写分片归一化。此处发现的是特征统计边界，不是未来目标读取。[分片代码](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation/blob/8feabbd580d59757dd108622ea93b9bdb6d38fa1/model/lib/data.py#L570-L602)、[路径约束](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation/blob/8feabbd580d59757dd108622ea93b9bdb6d38fa1/model/methods/tabm_temporal.py#L12-L21)。

独立复查还确认：FullTemporalEmbeddings 的 Fourier 年/月/周/日初始周期使用秒单位，趋势部分的标准化不改变 Fourier 输入单位。直接输入月份编号、保留秒周期会错配。实施时统一月份锚点/时区后用秒时间戳，或同步修改全部周期单位；记录改动。默认每周期 128 个谐波也不是本题已合理的容量，应以 51 个月级时间点检验周期/谐波消融，不能由人为月份锚点制造日/周信号。[永久源码](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation/blob/8feabbd580d59757dd108622ea93b9bdb6d38fa1/model/lib/temporal_embeddings.py#L95-L126)。

## 针对 EDA 的假设与反方

- **支持动机：** 训练月均租金变化明显，房型/面积/楼龄与价格存在关联。让映射随月份改变，是值得验证的假设。
- **不能混淆：** 月均变化不证明面积溢价或楼龄折价发生条件漂移，也可能来自房源构成变化。N5 要改善统一未来 RMSE 或分层偏差，不能只展示训练期动态图。
- **最大风险：** 15 万行只有 51 个独立月份；复杂时间函数可能拟合短期波动。预测延伸 16 个月，日历 t 可预知，但函数正确延伸未得到保证。Fourier 表示也不证明稳定季节性成立。
- **必要对照：** 同一版 TabM、同样的类别表示和数值预处理，比较普通日历输入、仅时间 embedding、feature-aware modulation。训练/验证协议改变需单独消融，不能混入模块收益。
- **不读取未来统计：** 预处理、时间尺度、类别词表、归一化参数只在训练段拟合；推理仅使用该行可预知的月份与允许特征。源码若依赖全体时间戳或未来分布统计，适配为训练段拟合并记录差异。不能用完整测试月份分布选择周期或参数。
- **实现漂移：** 官方 modulation 仓库内置的 TabM 与当前独立 `tabm` 包可能不同。N1、N5 的机制消融须对齐同一主干；可另报当前最新版 TabM 完整管道，但不能把库版本收益归给 modulation。

## 研究结论

N5 的研究价值在于以现代公开方法直接检验时间条件化是否有用；它应排在可复现的静态主干之后实施。没有任何已读证据证明本题必须使用这一模块。是否推进，由共同长短时间窗口、按预测跨度的偏差、数值稳定性和真实工程成本决定。
