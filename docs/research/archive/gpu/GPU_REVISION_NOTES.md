# GPU 开放后的增量核验记录

> 历史中间记录：本文件的 R1–R5 已被 2025–2026 现代候选 N1–N5 取代。当前审阅版见 [主报告](../../METHOD_RESEARCH.md) 和 [现代评估](../../IDEA_EVALUATION.md)；以下保留当时检索过程，不代表最新候选清单。

日期：2026-09-26。最新用户允许 GPU，低成本为软偏好。本记录由主代理负责，配合两个独立 GPU 检索代理及独立 idea-evaluator；只调研，不训练或启动计算资源。

## 两轮检索

第一轮 query：`TabM FT Transformer tabular deep learning numerical embeddings official paper regression temporal datasets`。从正式会议结果定位 TabM、FT-Transformer、数值 embedding 及较新的 temporal modulation 工作；第三方博客、排行榜和其他任务的改善率不作本地效果证据。

第二轮 query：`site:github.com/yandex-research/tabm categorical one hot encoding numerical embeddings`、`site:github.com/yandex-research/rtdl-revisiting-models FTTransformer regression example`、`site:proceedings.neurips.cc 2021 Revisiting Deep Learning Models for Tabular Data Gorishniy`。打开官方代码/包文档，逐项核验输入、输出和训练接口，并打开两篇正式 PDF 与后端文档。

## 可核验主张

| 来源 | 核验位置 | 使用范围 |
| --- | --- | --- |
| [TabM 官方仓库](https://github.com/yandex-research/tabm) | README 的 Creating TabM、Custom inputs、Training、Inference；源码链接 | 默认类别 one-hot；自定义 embedding 是适配；训练平均各成员损失、推理平均预测。不能称默认 learned building embedding |
| [FT-Transformer 2021 正式页](https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html) | 题名、作者、Abstract | Gorishniy、Rubachev、Khrulkov、Babenko；NeurIPS 2021。比较 ResNet/Transformer 与树，不存在普适优胜者 |
| [FT-Transformer 官方包](https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/README.md) | FT-Transformer、Practical notes、参数组示例 | 输入区分连续/类别；字段 token 注意力，原生类别表示；优化器参数组需保持。其示例配置不作为 HDB 最优配置 |
| [On Embeddings for Numerical Features in Tabular Deep Learning](https://proceedings.neurips.cc/paper_files/paper/2022/file/9e9f0ffc3d836836ca96cbf8fe14b105-Paper-Conference.pdf) | p.1 作者/摘要，§3 | Gorishniy、Rubachev、Babenko；NeurIPS 2022。分段线性/周期表示可用于不同主干，收益不能只归于 attention |
| [Feature-aware Modulation for Learning from Temporal Tabular Data](https://proceedings.neurips.cc/paper_files/paper/2025/file/ddc20f403d0d296c87940393838d5971-Paper-Conference.pdf) | p.1 题名/作者/会议脚注、Abstract，§2 | Hao-Run Cai、Han-Jia Ye；NeurIPS 2025。研究以时间上下文调节特征表示；只用于说明已有针对 temporal shift 的机制，不引用未复核的实验数值或保证 HDB 外推 |
| [CatBoost GPU](https://catboost.ai/docs/en/features/training-on-gpu) | Training on GPU、Parameters | task_type/devices、专属参数边界、GPU 非确定性；不承诺加速 |
| [LightGBM 安装](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html) | GPU 后端章节 | 当前实现/构建需按后端核对；完整平台审计见 gpu_feasibility_evidence.md |

这些出处支持机制和接口，不提供本项目的 GPU 显存、耗时、价格或 RMSE。当前主五项改为 R1 CatBoost、R2 LightGBM、R3 TabM、R4 趋势残差、R5 FT-Transformer；Ridge 为诊断控制，EBM 为解释储备。

原 CPU 版主报告和评估已分别保存为 `docs/research/archive/cpu/METHOD_RESEARCH_CPU_INITIAL.md` 与 `docs/research/archive/cpu/IDEA_EVALUATION_CPU_INITIAL.md`，供追踪假设变化。历史 P 编号与当前 R 编号不混用；旧分钟预算已撤销为准入条件，未变成 GPU 预算。
