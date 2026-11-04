# 候选编辑接口与独立验收接线记录

原记录日期：2026-11-03；文档归集日期：2026-11-04。来源为 `codex/overnight-4b-takeover` 的 `artifacts/eval_driven_20261103/local4b-overnight/state.json`。本次归集读取的是以 `d5c0901705db9e753fcdfe16de0f378a47a47ded` 为 HEAD 的工作区状态快照，该文件含尚未进入此提交的更新；原文哈希见[来源清单](../reports/sources.json)。不能把该提交单独当作全文状态的来源。

本文整理现有计划、源码提交和结果记录的关系，没有重新运行模型、独立验收或分析原始隐藏输入。此文进入文档分支，不表示这些产品实现已经进入远端主线。原运行和测试文件仍保存在来源分支或其登记的本地证据目录；未发布的源码提交和本地原件不提供推测的 GitHub 下载链接。

## 三个工作单元

| 单元 | 目的 | 来源记录中的状态 |
| --- | --- | --- |
| S1：已观察源码的行范围编辑 | 从模型实际见过的完整代码行派生精确旧文本，减少抄写负担 | `823b5449dced7cb8eb809a7e5606aaf14adfbab8`；实现及受控验证已记录 |
| S2：精简候选编辑接口 | 新请求只公开 `source_view_ref + line_range + after_text`，其他确定性字段由宿主派生 | 实现提交 `dd1f981fae26ec1b78fbf4247488b9776c67ec7b`；工作记录状态为 `completed_supported_negative` |
| 验收接线 | 正常入口调用适用独立验收，绑定最终候选、实际交付和运行结果 | `implementation_verified_real_comparison_owned_by_interface_chat`；真实模型对照由接口页拥有 |

S2 是候选接口改进单元；[开发者试用版 B](../specs/developer-preview-b.md) 中的 B4 是安装后的完整使用流程。它们不是同一份计划的同级任务。

## 已选接口及保留的约束

S2 中模型选择已经读取的代码位置，并提交新内容。宿主从保存请求中的源码视图派生路径、父版本、精确旧文本、换行、绑定及来源证明。历史 S1／精确文本请求仍按保存时的 schema 解释；未知 schema 拒绝准入。

原记录将 Task Intent、目标源码、模型身份、权限、公开检查、独立验收器、交付与资源政策列为共同条件；接口是所选主要变化。上述是原实验合同摘要，不在此新建或扩大实验授权。
为什么要有它：接口表达变简洁与实际任务更容易成功需要分别验证，不能用减少字段替代正确交付证据。

## 工程验证与实验记录的关系

来源记录报告 S2 实现有 171 项聚焦回归、19 项接口检查通过，并关闭独立审查发现。验收接线记录另报告 255 项最终组合检查通过。它们的选集、版本和运行不同，本摘要不把数量相加为统一覆盖或新的成绩。

原 S2 对照在运行根目录与独立验收准备的关系上存在缺口；来源记录又登记了 S3 `S3-prebound-independent-acceptance-pair`，先绑定共同验收环境再执行比较。S2、S3 和准备失败分别保留，不能混成一个配对实验。

S3 的文档终态为 `completed_supported_negative_with_incomplete_independent_verdict`。记录表示两臂都未形成正确交付；基线存在候选，但独立构建失败，未产生完整通过／失败 verdict；处理臂无候选。本文只呈现来源已经登记的终态与限制，不把它扩展为本地 4B 的一般能力结论，也不报告新的成功率。

## 找到原始材料

| 目的 | 来源位置／身份 |
| --- | --- |
| 查共同合同和两页归属 | 来源提交中的 `artifacts/eval_driven_20261103/local4b-overnight/state.json`，字段 `interface_followup`、`acceptance_followup` |
| 查候选接口代码 | 实现提交 `dd1f981fae26ec1b78fbf4247488b9776c67ec7b` 的 `src/mod_agent/task_agent_v2_tools.py` 及相应测试 |
| 查验收接线实现演进 | 记录列出 `28db6ab`、`945c6b3`、`95cbfa7`、`e960a4e`、`f241f34`；文档收口提交 `d5c0901` |
| 查最终接线汇总 | 来源提交中的 `artifacts/eval_driven_20261103/local4b-overnight/s2-acceptance-followup-result.json` |
| 查 S3 基线原运行 | Run ID `dsh_582c486c6a65495d90e8fafe83dc11c8`；由原 state 中 `acceptance_capable_comparison.runs.baseline` 定位注册配置 |
| 查 S3 处理臂原运行 | Run ID `dsh_448edf857cfc4a43ac93330d624bfede`；由同字段下 `treatment` 定位注册配置 |

需要对运行做新的归因时，回到注册执行配置和共享只读证据入口，并说明其缺失或 partial 范围；不能只拿本文摘要重新判分。
为什么要有它：文档负责帮助检索，实际结论仍需由对应版本的原件支持，私有裁判内容也不进入普通文档索引。

## 历史 HTML 分析

- [2026-11-01 架构调查](../reports/architecture-review-20261101-1438.html)
- [2026-11-02 架构调查](../reports/architecture-review-20261102-173834.html)
- [2026-11-03 候选形成结构调查](../reports/architecture-review-20261103-125657.html)
- [2026-11-03 验收与结果结构复核](../reports/architecture-review-20261103-150000.html)

这些页面是安全静态副本：保留历史分析正文、源码定位和图示原文，移除了脚本／CDN以及本机绝对路径。页面中的“下一步”、旧检查数量和状态属于原记录日期，不应直接变成新任务。它们解释为何出现后续实现；当前任务和合入情况从[统一文档管理入口](../document-management.md)查找。
