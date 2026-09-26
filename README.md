# CS5228 HDB Rental Prediction

本项目预测新加坡 HDB 整套住房的月租金。课程要求、数据说明和交付要求见 [`docs/Project Desc.md`](docs/Project%20Desc.md)。任务与验证原则见 [`docs/PROJECT_TASK_GUIDE.md`](docs/PROJECT_TASK_GUIDE.md)；四人分工和交接流程见 [`docs/TEAM_WORKFLOW.md`](docs/TEAM_WORKFLOW.md)；进度报告前的推进计划见 [`docs/PROGRESS_PLAN.md`](docs/PROGRESS_PLAN.md)。

目前只有课程数据、空白报告模板和工作文档；尚无分析、模型或实验结果。近期目标是先完成数据审计、时间验证和可复现基线，为 2026 年 10 月 4 日的进度报告提供真实结果。团队尚未确定候选方案数量；`experiments/` 和 `configs/` 按实验编号扩展。

## 工作区

```text
data/                     # 课程原始 train/test 和 auxiliary/ 辅助数据
configs/                  # 验证切分与候选方案配置；方案确定后再添加
docs/                     # 课程说明、任务、分工、推进计划和实验记录模板
experiments/              # 各方案的实验脚本；按方案或实验编号命名
notebooks/                # EDA 与探索分析
src/cs5228_rent/          # 后续可复用的数据处理、特征与评估代码
tests/                    # 后续针对数据切分、连接和提交格式的必要检查
proposal/                 # main.md 工作稿和 slides.md 幻灯片模板
  experiments/            # 供 proposal 使用的实验结果
  figures/                # 供 proposal 使用的图表
submissions/              # Kaggle 预测文件
final report/             # 最终报告；以 latex/ 中的 IEEE 模板为写作入口
```

原始数据文件位于 `data/`，课程说明位于 `docs/`。课程说明是任务依据；工作指南中的模型和时间划分是团队建议，可根据实验结果调整。
