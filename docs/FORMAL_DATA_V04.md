# 正式数据 v0.4：上传与试跑

本轮完成本地财报审计和确定性工具回放，尚未运行正式数据训练或效果评测。

## 数据与证据

- 12家公司、2023至2025年度、36份官方年报，9048个PDF物理页。
- 216个核验事实：每家公司每年6项，营业收入、营业成本、归母净利润、经营现金流净额、资产、负债。
- 核验方式：AI逐项检查216条抽取行和报表表头，12份代表性报告的图像抽样核对。不是人工审计；原文、表头、单位声明、比较列、来源URL和PDF哈希均保留。
- 每个事实采用该年度年报本年列。不是最新重述后的统一历史序列。
- 美的、中兴、比亚迪使用千元；宁德时代2023年使用万元，2024/2025年使用千元。其余本次入库字段使用元。
- 美的成本原表以括号负数展示，单独转换成正的费用金额；利润和现金流的负号保留。
- 同比和增减额须通过上年比较列核对。当前年报上年数与旧年报数不一致时拒绝跨报告计算，包括重述差异和舍入差异。宁德时代单位变化后的小额舍入差异也被保守排除。
- 49项候选计算被排除；明细位于 `data/review_v0.4/omissions.json`。

本地工具回放通过1243道候选题：训练819、验证212、测试212。从中按公司和题型分层选训练800、验证100、测试100。训练800是分词前数量；服务器按实际 `qwen3_5_nothink` 模板排除超过4096 token的轨迹，最终数量写在冻结清单中。不会用重复题补足。

这些是根据真实财报生成的确定性工具轨迹，覆盖数值查询、换算、同比、比率及缺少母公司口径时的拒答。不同单位的问法共享底层事实，不能把800道题宣传成800项独立金融研究任务，也不是人工标注或大模型教师生成的开放式研究数据。公司隔离可以检查公司迁移，仍然存在模板相似性；后续应另设自然提问评测。

## 上传

将以下两个文件上传到 `/calc/users/geyuboao/workspace/FinSight`：

- `finsight-formal-data-v0.4.zip`
- `finsight-formal-data-v0.4.zip.sha256`

包内包含源代码更新、36份PDF、抽取页面、事实和审计记录。不含模型、环境和历史训练产物。不要把这个私有服务器数据包直接上传GitHub。

先确认本项目没有训练、评测、prepare等运行任务，再执行：

```bash
cd /calc/users/geyuboao/workspace/FinSight &&
sha256sum -c finsight-formal-data-v0.4.zip.sha256 &&
unzip -o finsight-formal-data-v0.4.zip apply_v04.py &&
.runtime/envs/finsight/bin/python apply_v04.py finsight-formal-data-v0.4.zip
```

安装器先逐文件校验，再将被替换的旧文件和旧pilot数据复制到 `outputs/patch_backups/pre-v0.4-时间/`，最后更新活动代码和数据。旧训练、导出包、适配器和模型不受影响。若已经存在正式数据冻结且本次安装会修改源代码/数据/配置，安装器会停止，防止破坏既有实验。旧pilot冻结仍留作历史记录，在新代码和新事实下不能直接重跑。

## 冻结正式数据

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bash server/run.sh prepare --stage audit &&
bash server/run.sh prepare --stage dataset
```

全部离线运行，不要运行默认 `prepare` 或 `--stage all`，它们包括联网获取。日志由 `server/run.sh` 自动保留在 `runs/`。成功应看到审计216个事实、冻结清单的 `usable_training`、`max_selected_tokens`、`rejected`，以及最终 `EXIT=0`。

固定产物在 `data/frozen/`：训练轨迹、验证100题、测试100题、分词长度、拒绝记录和来源摘要。当前机器上不能假定所有800条都通过分词。

## 正式数据20步试跑

只有上述两条命令都成功后执行：

```bash
bash server/run.sh train --probe --background
```

这次不带 `--pilot`。复制输出中的 `tail -f .../run.log` 命令查看进展，退出查看用 Ctrl+C，不会终止后台训练。四卡有效批量32，20个优化步；输出 `outputs/training/probe-时间/`。必须等 `EXIT=0` 和 `outputs/training/probe.json` 的 `status=completed`，不能只看启动PID。

20步试跑通过后，同一份源码、数据和配置才允许运行完整训练：

```bash
bash server/run.sh train --background
```

完整训练从基础模型重新开始，使用最多2轮BF16 LoRA；不接着小样本pilot适配器继续训练。完成后再做验证集上的direct/base/tuned三组对照。测试集保留至模型方案确定，避免根据测试成绩反复修改。

## 回传

当前先回传数据冻结输出与正式数据试跑日志。若全部成功，可执行 `bash server/run.sh export` 并下载新结果ZIP，再由本机核对训练收据和适配器。

本轮还修复了训练来源记录中的配置可变引用问题。v0.3.3历史产物按原样保留，其解释见 `SERVER_EXPORT_20260924.md`。本次本地测试和回放证明数据与工具链可执行，不证明微调后模型回答准确。早期真实模型结构化10/10仍有自由文本金额及年份错误，详见 `BASELINE_SMOKE_20260924.md`。
