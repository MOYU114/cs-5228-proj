# 方案研究：统一入口

更新：2026-09-26。**先看 [PRIORITY_PLAN.md](PRIORITY_PLAN.md)**：当前尝试顺序为 **TabM → 时间调制 TabM → TabICLv2 → TabPFN-3.5 → ModernNCA**，以未来月份的原尺度 RMSE 为目标；全部尚未训练，顺序不是精度排行榜。

## 当前决策与解释

| 文件 | 阅读目的 |
| --- | --- |
| [PRIORITY_PLAN.md](PRIORITY_PLAN.md) | 最佳尝试清单、数据匹配理由、指标/晋级规则、融合设计、B/C 交接 |
| [METHOD_RESEARCH.md](METHOD_RESEARCH.md) | 九篇研究的综合比较、五项现代方法、接口与时间风险 |
| [IDEA_EVALUATION.md](IDEA_EVALUATION.md) | 独立五维审查、失败条件、能力匹配；最新争分优先级已同步 |
| [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md) | 研究问题、CPU/GPU 偏好和范围演变 |
| [EDA_REPORT.md](../EDA_REPORT.md) | 所有本地数据结论的来源，保留阶段一规定路径 |

五个主候选均为 2025–2026 方法。GPU 允许使用，低成本只是建议；经典模型仍作强基线。任何“更准、更稳”的结论都待统一时间验证，不从发表年份或五维分数推断 Kaggle 成绩。

## 证据与历史材料

当前三份证据集中在 `evidence/`：

- [现代表征与检索](evidence/modern_retrieval_evidence.md)：TabM、ModernNCA、TabR；候选库、自身排除与复杂度。
- [2026 基础模型](evidence/modern_foundation_evidence.md)：TabICLv2、TabPFN-3/3.5；回归、缓存、微调和硬件边界。
- [时间漂移机制](evidence/modern_temporal_evidence.md)：TabReD、时间 embedding、feature-aware modulation；时间统计与周期单位。

旧 CPU 方案和 GPU 中间修订全部进入 [archive/](archive/README.md)。它们保存探索过程，**不代表当前选择、资源门槛或排名**。13 个文件的迁移映射见 [file_moves.json](../../experiments/research/file_moves.json)，未删除证据材料。

## 复现与目录职责

```text
docs/research/             当前决策与解释
docs/research/evidence/    可追溯文献/源码证据
docs/research/archive/     旧范围、旧候选与历史底稿
notebooks/                EDA 代码、依赖和运行说明
experiments/eda/           已运行的 EDA 表格和清单
experiments/research/      日期计数脚本/结果、迁移记录和哈希清单
proposal/figures/          原 EDA 图表，保持用户要求的路径
```

从项目根目录执行：

```bash
python3 experiments/research/validation_windows.py
python3 experiments/research/priority_profile.py
```

二者只用 Python 3 标准库和已有月份计数，不读取目标列或拟合模型。输出、验证口径与文件说明见 [experiments/research/README.md](../../experiments/research/README.md)；完整 EDA 命令与固定依赖见 [notebooks/README_EDA.md](../../notebooks/README_EDA.md)。

文献复查按 evidence 中的 query、正式论文和官方代码链接核验。后续模型环境须锁定实际版本、commit 与 checkpoint；本轮未安装模型依赖、下载权重或生成预测。

原始 CSV、EDA 代码与受保护文档一致性记录在 [research_manifest.json](../../experiments/research/research_manifest.json)。未编辑 proposal/main.md、proposal/slides.md 或最终报告，未提交/推送 Git。当前停在团队审阅阶段。
