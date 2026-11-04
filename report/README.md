# CS5228 Final Report

采用课程提供的 IEEE LaTeX 样式，并沿用用户指定 CS5322 模板的组织方式：`main.tex` 管理标题、作者、公共宏包与章节顺序，各章独立编辑。CS5322 的人员、银行业务正文、图片和实验日志不属于本项目，未复制。

当前是**可编译的写作框架**，尚未撰写章节正文。任务、代码进度、结果口径及待补证据见 [PREPARATION.md](PREPARATION.md)。下一步从第一章开始，逐章写作、核实数据和编译检查。

## 文件与篇幅

以下篇幅是内部规划，课程要求是正文最多 **8 页，含图表，不含参考文献及附录**；没有发现单独封面不计页的规定。标题、作者和摘要约预留 0.5 页，以下章节合计约 7 页，另留 0.5 页用于整合。

| 文件 | 内容 | 预计正文页数 |
|---|---|---:|
| `latex/main.tex` | 唯一主文档；标题、作者、公共宏包、章节导入 | — |
| `latex/abstract.tex` | 摘要；最后撰写，目前环境注释 | 包含在开头 0.5 页内 |
| `latex/chapter01.tex` | Introduction：动机、任务挑战、研究问题与工作概览 | 0.6 |
| `latex/chapter02.tex` | Task and Dataset：预测设定、指标、数据范围 | 0.5 |
| `latex/chapter03.tex` | Exploratory Data Analysis：时间变化、属性差异、数据问题 | 0.9 |
| `latex/chapter04.tex` | Data Preprocessing and Feature Engineering | 0.9 |
| `latex/chapter05.tex` | Data Mining Methods：字段编码、BERT、预训练与回归 | 1.5 |
| `latex/chapter06.tex` | Experimental Evaluation：协议、对照、结果、误差解释 | 1.5 |
| `latex/chapter07.tex` | Discussion and Limitations | 0.8 |
| `latex/chapter08.tex` | Conclusion | 0.3 |
| `latex/references.tex` | 经核实的参考文献；目前仅规划注释 | 不计正文 |
| `latex/contribution_statement.tex` | 附录：成员与实际工作分工 | 不计正文 |
| `latex/photo/` | 后续按章加入审核过的图片 | — |

原始 `latex/cs5228-template.tex`、其 PDF 和 `IEEEtran.cls` 均保留。**正式编辑与编译入口为 `latex/main.tex`**，无需向原始模板回填。

## 编辑约定

- 报告正文使用英文；本目录的准备说明使用中文。
- 每章已有 `Source`、`Evidence`、`TODO` 注释。扩写对应章即可，保留必要的证据来源。
- 后续按章节推进；某章依赖尚未完成的实验时，记录缺口，不能把计划写成完成结果。
- 分清原始数据、清洗后数据、原始标签、平均后标签，以及 selection/refit。
- 引用方法原论文前核实出处；当前不保留 IEEE 模板里的无关示例参考文献。
- 图表应标明样本、月份、单位和评估口径。当前不自动把 README 全部图搬入正文。

## 队员信息

检查了当前 `proposal/main.md`、`proposal/slides.md`、本地修改后的 `proposal/slides.pdf` 和 `proposal` 分支。没有发现姓名、学号或队名；Markdown 明确标注待确认，`docs/TEAM_WORKFLOW.md` 只有四个暂定角色 A/B/C/D。

因此作者栏目前明确标记待确认，附录保留原 proposal 的四角色计划，**并未认定为实际贡献**。补充本组队名、姓名、学号、实际分工后，应同时更新 `main.tex` 和 `contribution_statement.tex`。

## 编译与预览

从本目录运行：

```sh
cd latex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

预览文件为 `latex/main.pdf`。在 Overleaf 上传整个 `latex/` 目录，设 `main.tex` 为主文档，使用 pdfLaTeX。章节源码相互依赖，不能单独编译一个 `chapterXX.tex`。

正文完成后核对正文页数；目前的框架页数不能用于判断最终页数。附录中的暂定角色表也不能直接作为最终分工提交。
