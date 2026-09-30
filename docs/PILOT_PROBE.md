# 10 条核验轨迹的 20 步试训

用途：在完整财报数据审计期间，先验证四卡 BF16 LoRA 的分布式训练、显存、反向传播和适配器保存加载。
本次会重复使用 10 条已核验茅台案例，不是正式训练或模型效果评测。不得以该数据上的损失下降宣称泛化改善。
使用已有服务器环境、基础模型和种子事实；不下载新依赖，不需要本机 SSH。

## 安装与执行

要求已安装 v0.3 离线环境和 v0.3.2 Agent 补丁，且 10 题 smoke 已完成。
上传本次 ZIP 和 SHA256 文件到 `/calc/users/geyuboao/workspace/FinSight`。
确认四张 GPU 没有被其他任务占满后，在服务器终端执行：

```bash
cd /calc/users/geyuboao/workspace/FinSight &&
sha256sum -c finsight-pilot-probe-v0.3.3.zip.sha256 &&
unzip -o finsight-pilot-probe-v0.3.3.zip &&
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 &&
bash server/run.sh prepare --stage dataset --pilot &&
bash server/run.sh train --probe --pilot --background
```

末尾会打印 PID、LOG、STATUS 以及一条实际的 `tail -f '.../run.log'` 命令。
复制那条 tail 命令观察日志，Ctrl+C 只退出日志查看，不会停止后台训练。
可使用 `bash server/run.sh status` 查询运行状态。后台启动返回 0 不代表训练完成。

日志应先出现 `PILOT_TOKEN_CHECK=PASS`，随后启动 4 个训练进程并输出逐步损失。
配置为 LoRA rank 16、BF16、上下文 4096、学习率 1e-4、有效 batch 32、20 个 optimizer step。
不用小样本轮数衡量学习效果：20 步会多次重复这 10 条案例。
已有 pilot 快照若与当前源码或事实不一致，冻结校验会拒绝覆盖；保留原文件，回传报错后再安排新版本。

## 完成标准

- 对应后台运行的 `status.json` 为 completed 且 exit_code 为 0。
- `outputs/training/pilot-probe.json` 的运行路径和时间与本次任务一致，global_step 为 20，pilot 为 true。
- 对应 `outputs/training/pilot-probe-时间/` 包含适配器权重、配置、损失记录、token_lengths.json 和 reload_check.json。
- 适配器重新加载产生非空输出。这个检查只证明保存加载与生成可用，不证明该输出正确。

遇到报错，回传这次 run.log 尾部以及 status.json。不要直接运行正式 train。
小样本收据不会写入正式的 `probe.json` 或 `production.json`；全量训练仍要求正式数据快照通过独立试训。
本地回归测试覆盖模式隔离、超长拒绝、步数不足及非有限损失；真实四卡训练必须在服务器验证。

## 2026-09-24 服务器完成记录

用户回传的训练日志末尾确认任务 `20260924-215230-train-b9152f` 于北京时间 22:00:19 完成，最终 `EXIT=0 STATUS=completed`。
训练器完成 20/20 个 optimizer step，训练器计时 239.1 秒（不包含整个任务的预检查、加载和重新加载）。
训练汇总 train_loss 为 0.1074，完成收据 last_loss 为 0.009896892122924328；两者分别为汇总损失与末次记录，不应混用。
日志中的 epoch=20 来自小样本与 max_steps=20 的组合，不是正式数据训练20轮的计划。
没有运行验证集，因此缺少 eval_loss 和 eval_accuracy 的绘图警告符合本次用途。

适配器目录：`outputs/training/pilot-probe-20260924-215342`。
完成收据显示 `probe=true`、`pilot=true`、`global_step=20`，适配器保存后重新加载检查通过。
收据记录的 SHA256：

- adapter_config.json：`ac16063d6ef1ff74e8b2b4980fdbc2936ea380bbf6038de199f079c7082f9d9e`
- adapter_model.safetensors：`9c99796f1327f447f254589e367a7685e8a16b4536fa62cb905b79e9b21036ce`

本地已取回 `finsight-results-20260924-222818.zip`，适配器逐文件 SHA256 与收据一致；详见[结果包核验](SERVER_EXPORT_20260924.md)。配置说明存在已定位的共享字典记录缺陷，能够从包内原配置重现收据哈希，原始产物保持不变。
该结果证明试训和保存加载链路完成，未证明小样本适配器的回答质量或泛化收益。

## 正式数据当前状态

截至 2026-09-24，本机已下载并逐文件 SHA256 校验 12 家公司、2023-2025 年的 36 份官方年报，提取 9048 页。
隆基 2024 年采用 2025-05-07 发布的修订版，选择记录在本机 data/manifest.json。
审计表保留 12 条种子核验事实，并有 210 条待审候选，其中部分与既有比较列事实重叠。
不能把这些候选直接批准或打包成正式训练标签。需核对单位（包括千元）、会计年度、合并口径和重述比较列。
正式数据没有冻结，约 800 条训练轨迹及两个 100 题集合尚未生成。
本补丁不附带新下载的 PDF 或待审数据，也不会覆盖服务器的已有财报和日志。
