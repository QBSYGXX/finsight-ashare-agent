# LoRA v0.16.1：TrainingArguments兼容修复

日期：2026-09-29。

用户返回作业`20260929-134546-1952035`的探针日志：9B加载完成，LoRA可训练参数43,278,336，
随后在`TrainingArguments(..., overwrite_output_dir=False)`报TypeError，probe与总任务退出码均为1。
没有进入`trainer.train()`，本次日志不能证明任何优化器步骤已执行。fast path回退提示不是该异常原因。
原失败导出由用户终端报告，尚未下载独立核验；本记录依据其所贴堆栈。

## 改动

- 新增独立`lora_v0161/`，从已提交并核验清单的v0.16派生；原代码、发布包、服务器旧作业保留。
- 移除5.8中不存在的`overwrite_output_dir`，已有输出拒绝覆盖逻辑保留并有回归测试。
- 将弃用但仍兼容的`warmup_ratio=0.05`改成5.8等价的`warmup_steps=0.05`。
- 用实际构造函数签名检查全部参数，不通过静默丢弃参数来兼容接口；构造参数后检查热身步数。
- 准备阶段的小模型检查之前执行真实TrainingArguments接口检查；probe/full也在加载9B之前构造参数。
- 新目录、新实验身份、新RUN；重新准备与探针，不复用旧准备凭证或伪造续训。

690条轨迹、2,273个样本、配置文件逐字节与v0.16一致。更新后的包清单继承相同Agent依赖与基础模型
分片指纹；发布脚本重核原源码、数据及轻量依赖，服务器仍按原逻辑重新哈希模型。

## 验证

读取已哈希核验的Transformers5.8.0官方wheel中的training_args.py：全部传入字段均存在；
直接执行其get_warmup_steps方法验证20步热身1步、143步热身8步。此检查不等于在Windows运行完整Trainer。
新增测试覆盖1/2/4卡与probe/full六组参数、未知接口拒绝、旧输出保护及数据/配方不变。

运行原Agent67项、复制的LoRA16项和新增9项测试，92项通过，退出码0。
发布前检查Python AST、Git差异、原包依赖、脚本LF；发布时检查ZIP成员、CRC和全部文件SHA256。
本机未运行9B反向传播或DDP，服务器probe、正式训练、重载、合并及评测仍待验收。

## 操作

上传`finsight-lora-v0.16.1.zip`和同名`.sha256`后，在服务器项目目录执行：

```bash
sha256sum -c finsight-lora-v0.16.1.zip.sha256 &&
unzip -n finsight-lora-v0.16.1.zip &&
FINSIGHT_GPUS=1 bash lora_v0161/run.sh all --background
```

输出新的RUN、PID和PROGRESS_LOG，按打印的tail命令监控。不要使用旧RUN_ID，也不要删除旧作业。
先应看到`TRAINING_ARGUMENTS=PASS`；随后`TRAIN_PROGRESS`才表示开始训练步，probe退出码0且重载通过
才进入正式训练。错误时流程停止并导出，禁止根据模型成功加载推断训练完成。
