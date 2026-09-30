# GitHub发布说明

Research v1.0源码交付已整理。先读[交付说明](DELIVERY.md)；旧协议目录是历史证据，不需要全部执行。公开定位为受限年度财报研究Agent，默认基础模型可交互，新LoRA作为完成的实验保留。

## 生成可上传源码包

在干净的Git工作区运行（输出放到项目外）：

```bash
python scripts/build_source_release.py --output ../_release_work/finsight-ashare-agent-research-v1.0.zip
```

脚本只读取HEAD提交，附逐文件清单和提交号，拒绝打包数据/权重/环境/日志，校验冻结源码。已有同名包不会覆盖。验证已有包：

```bash
python scripts/build_source_release.py --verify ../_release_work/finsight-ashare-agent-research-v1.0.zip
```

源码ZIP不含Git历史；也可把ZIP作为GitHub Release附件。不要把ZIP误当成服务器升级补丁。源代码中pyproject的0.2.0指历史Python原型，Research v1.0是此次项目交付标签，不混用两者版本号。

2026-09-30：系统与本轮LoRA实验已经阶段收尾，首页和结果记录已更新。公开仓库为[QBSYGXX/finsight-ashare-agent](https://github.com/QBSYGXX/finsight-ashare-agent)，远程主分支为`main`。
源码公开不等于全新机器可以一键运行：模型快照、财报/事实、检索索引、离线环境及LoRA权重不随Git分发。

## 建议公开的内容

- 当前入口agent_v015、独立实验lora_v0161、推理环境vllm_v013、固定上游源码和许可。
- 根README、项目讲解、阶段报告、推理实验、本轮LoRA核验脚本与脱敏汇总。
- 模型/数据准备路径和来源说明，明确复现前置条件、开发诊断局限及已知退化。

现有data、models、outputs、runs、环境和ZIP由忽略规则排除。不要强制添加权重、凭据、原始财报或完整日志。
lora_v0161/data中的训练文本在离线发布包内，源码仓库只包含生成代码和指纹；新克隆不能跳过数据准备直接训练。
结果包不含LoRA权重；如需公开适配器，应从服务器单独取回并核验，附带基础模型版本与许可。

## 推送前

在本机项目根目录检查git status、当前分支和待提交差异。只暂存本次相关路径，不使用无差别git add。
当前工作分支为codex/initial-implementation，不需要为发布修改或覆盖历史。

本机origin使用`https://github.com/QBSYGXX/finsight-ashare-agent.git`。如果origin已存在，用set-url更新而不是重复add，不要覆盖其他项目的远程。
地址必须是纯文本URL，不要粘贴Markdown方括号。完成官方认证后，才按用户明确要求推送。
不要把令牌放入代码、命令或截图。本次发布由用户明确授权；使用Git凭据管理器认证。由于本机Git HTTPS连接超时，首次通过GitHub官方API发布源码快照，完整开发历史保留在本机，模型与环境仍留在本地/服务器。

远程main与本地开发分支的提交历史不同，不能直接强制推送覆盖。后续更新应先读取远程状态，通过正常提交或API更新，并校验源文件一致性。

## 版本说明

推荐演示入口是v0.15.5基础模型；新协议LoRA已经完成配对评测，但未自动切换交互入口。
历史目录和包内README保留当时状态，最新结论以根README和2026-09-30核验报告为准。
保留FinSight出处及GPL-3.0；模型与财报的许可证和来源要求独立适用。

发布可以定位为限定范围的研究项目，已知失败和独立测试未执行须明确列出，不以所有诊断题满分为条件。
