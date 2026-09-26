# CS5228 Proposal 工作稿

[`main.md`](main.md) 是进度报告的 Markdown 工作稿。团队先在这里整理已核实的数据、方法、实验结果和后续计划；团队名称、四位成员姓名及学号留空待填。候选方案数量尚未确定。

[`slides.md`](slides.md) 是可编辑的 **Pandoc Beamer 幻灯片模板**。它当前有标题页加 12 页内容页，最后两页用于下一步计划。版式采用 16:9 的 Boadilla 主题；段落、图表和顺序可以直接在 Markdown 中调整。工作稿完成并核对后，再将结论压缩到对应幻灯片。

目录用途：`experiments/` 放可追溯的实验摘要，`figures/` 放报告使用的图表。需要插图时使用相对路径 `figures/文件名`。实验记录应先更新，再把有证据的结论写入 `main.md`。

**导出命令**：在 `proposal/` 目录运行 `pandoc slides.md --slide-level=1 -t beamer --pdf-engine=xelatex -o slides.pdf`。课程说明要求约 10–15 页的幻灯片 PDF，最后 1–2 页列下一步；普通长文档式 PDF 不符合这一形式。课程要求见 [`docs/Project Desc.md`](../docs/Project%20Desc.md)。
