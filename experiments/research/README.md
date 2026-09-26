# 调研复现与派生计数

这里存放调研所需的确定性计算与审计记录，**不是模型训练结果**。决策文档集中于 [docs/research/](../../docs/research/README.md)，首读 [PRIORITY_PLAN.md](../../docs/research/PRIORITY_PLAN.md)。

| 文件 | 用途 |
| --- | --- |
| [validation_windows.py](validation_windows.py) / [CSV](validation_windows.csv) | 从 EDA 月份计数核算 V1、VL、V2 样本数 |
| [priority_profile.py](priority_profile.py) / [JSON](priority_profile.json) | 以训练截止为 h=0，核算测试月份跨度分布 |
| [file_moves.json](file_moves.json) | 本次 13 个文档迁移的来源、去向及迁移前哈希 |
| [research_manifest.json](research_manifest.json) | 调研阶段的文档/脚本哈希、候选身份/顺序及文件保护检查快照 |

该清单保留调研完成时的状态；之后经授权更新的英文 proposal、slides 和入口说明由 [proposal/build_manifest.json](../../proposal/build_manifest.json) 记录。历史清单中的 proposal 哈希与未修改检查不代表后续文档的当前状态。

项目根目录运行：

```bash
python3 experiments/research/validation_windows.py
python3 experiments/research/priority_profile.py
```

依赖：Python 3 标准库（csv、pathlib、json、hashlib）；没有模型库、网络请求或权重下载。输入仅为 [monthly_counts.csv](../eda/core/monthly_counts.csv)，不读取 train/test 目标。重跑只覆盖上述 CSV/JSON；若源 EDA 更新，应重新生成并审阅计数、清单与文档，不能默认为模型协议已同步。

预期范围：V1 训练 114,842 / 验证 17,960；VL 84,902 / 47,900；V2 132,802 / 17,198。测试 h=0 为 380 行，h=1–6 为 19,386 行，h=7–16 为 30,234 行。所有数字均来自实际脚本结果。

V1/VL 的验证标签重叠，不能汇成独立重复或 pooled RMSE；V2 已被 EDA 观察，不能称完全盲测。跨度比例仅用于描述风险，没有用于训练样本加权或计算新的竞赛分数。

此处无预测或 RMSE 文件，因为尚未训练。完整 EDA 复现入口保持为 [notebooks/run_eda.py](../../notebooks/run_eda.py)，依赖说明在 [README_EDA.md](../../notebooks/README_EDA.md)。
