# CS5228 HDB Rental Prediction

本项目预测新加坡 HDB 整套住房的月租金。课程要求、数据说明和交付要求见 [`docs/Project Desc.md`](docs/Project%20Desc.md)。任务与验证原则见 [`docs/PROJECT_TASK_GUIDE.md`](docs/PROJECT_TASK_GUIDE.md)；四人分工和交接流程见 [`docs/TEAM_WORKFLOW.md`](docs/TEAM_WORKFLOW.md)；进度报告前的推进计划见 [`docs/PROGRESS_PLAN.md`](docs/PROGRESS_PLAN.md)。

已完成第一阶段 EDA、六份辅助数据审计、现代方法调研与五候选评估。当前建议按 **TabM → 时间调制 TabM → TabICLv2 → TabPFN-3.5 → ModernNCA** 尝试，以原尺度 RMSE 和共同时间验证判断结果。这是待验证清单，尚未训练模型、产生预测或提交 Kaggle。GPU 可以使用，成本仅为软偏好。

## 从这里开始

1. **决定先尝试什么：** [最佳尝试清单](docs/research/PRIORITY_PLAN.md)，包括排序、指标、晋级规则、融合设计和 B/C 交接。
2. **查看数据依据：** [EDA 报告](docs/EDA_REPORT.md)，包括实际统计、图表、清洗规则及辅助数据风险。
3. **追溯调研与评估：** [调研目录](docs/research/README.md)；当前文档、证据和历史归档已分开。
4. **审阅英文稿与 slides：** [proposal 入口](proposal/README.md)，包含英文正文、14 页 slides 工作稿、[逐页规划](proposal/SLIDE_PLAN.md)、可编辑源文件和构建说明。
5. **重新运行：** [EDA 复现说明](notebooks/README_EDA.md)；[调研计数与验证说明](experiments/research/README.md)。

英文 proposal 的问题描述与 EDA 已完成；slides 按完整项目规划为 14 页，第 1–6 页已成稿，第 7–14 页为待补充提纲，等待团队审阅。接下来由团队决定首轮实验范围，B/C 共用时间划分、输入信息与 RMSE；2026 年 10 月 4 日进度报告节点见项目计划。尚无模型结果，最终报告未修改。

## 工作区

```text
data/                     # 课程原始 train/test 和 auxiliary/ 辅助数据
configs/                  # 验证切分与候选方案配置；方案确定后再添加
docs/                     # 课程/团队文档与 EDA_REPORT.md
  research/               # PRIORITY_PLAN、完整调研、idea-evaluator、研究范围
    evidence/             # 三份现代方法的文献与源码证据
    archive/cpu/          # 早期 CPU 候选及探索记录，非现行方案
    archive/gpu/          # GPU 中间修订，非现行方案
experiments/              # 派生结果与后续实验；当前没有模型实验
  eda/                    # EDA core/auxiliary 表格与运行清单
  research/               # 时间窗/跨度计数、复现脚本、迁移记录与清单
notebooks/                # EDA 入口、模块与依赖；保持原复现路径
src/cs5228_rent/          # 后续可复用的数据处理、特征与评估代码
tests/                    # 后续针对数据切分、连接和提交格式的必要检查
proposal/                 # 英文 main.md、slides.md / slides.pdf 和构建清单
  experiments/            # 供 proposal 使用的实验结果
  figures/                # 供 proposal 使用的图表
submissions/              # Kaggle 预测文件
final report/             # 最终报告；以 latex/ 中的 IEEE 模板为写作入口
```

原始数据文件位于 `data/`，课程说明位于 `docs/`。课程说明是任务依据；工作指南中的模型和时间划分是团队建议，可根据实验结果调整。
