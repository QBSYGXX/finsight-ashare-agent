# 服务器结果包回收核验

核验对象：`finsight-results-20260924-222818.zip`，163,479,982 字节。
SHA256：`ac09fdabb5b025a985f886cdce519b425ee1f3991609a9305255d5994665d542`，与服务器导出输出一致。

使用 `scripts/verify_server_export.py` 完成核验，退出码 0：

- 63 个 ZIP 成员 CRC 检查通过；清单中 61 个文件逐一 SHA256 匹配。
- 适配器配置和权重的文件哈希与训练完成收据一致。
- pilot 数据、配置、10 条轨迹和长度记录相互匹配；最长样本 2609 tokens，低于 4096。
- trainer_state、completion、pilot-probe 和运行状态确认 20 步完成、损失有限、最终退出码 0。
- 重新加载记录含非空模型输出；本机未重新进行 GPU 加载。
- 四卡都有活跃采样，设备总显存占用采样最大值分别为 29659、29677、29675、29661 MiB，约每卡 29 GiB。这是约30秒间隔采样，不是精确的进程显存峰值。

本地独立归档目录：`outputs/server_imports/finsight-results-20260924-222818/`。
机器可读核验报告：该目录中的 `LOCAL_VERIFICATION.json`。
解压没有覆盖当前本机 data、models 或 outputs/training。

## 基线复核

对包内 `outputs/smoke/20260924-210204-llm/cases.json` 使用匹配的任务标签重新评分，结果与原记录一致。
结构化正确 10/10，工具成功 26/28（92.86%）；每题平均 41.53 秒，含工具和检索耗时，不含初始主模型加载。
正文仍有已记录的单位、年报年份和术语问题，见 [基线人工检查](BASELINE_SMOKE_20260924.md)。
这些是未微调模型在训练公司开发样例上的结果，不是微调后或固定测试集成绩。

## 配置记录缺陷

v0.3.3 的 server.train 在计算 experiment_digest 后，给同一个 training_config 字典添加运行路径和试训覆盖项。
provenance.json 因引用同一字典而保存了覆盖后的版本，直接哈希与收据不符。
本次核验确认该字段与包内实际 train.yaml 完全一致；再用同包 configs/lora.yaml 还原计算哈希前的配置，可精确重现收据中的 experiment_digest。
报告将此标为 `legacy_v0.3.3_mutated_config_reconstructed_from_archived_base_config`，不伪装成原始记录直接通过。

本地 server.py 已用 deepcopy 保留计算哈希时的配置，并增加回归测试；将在后续代码包一并同步服务器。
未改写本次原始 ZIP、完成收据或适配器，也不需要为这项记录修复重新跑已完成的试训。
源码变更会触发既有数据冻结保护；正式数据应在后续代码定版后新建快照，不修改旧 manifest 来绕过检查。

## 未完成内容

正式数据的会计口径、单位及重述比较列审计尚未完成，正式训练集未冻结，正式训练和三组对照评测尚未运行。
结果包只包含日志摘要，不含完整运行日志、PDF或服务器源码快照。归档和适配器均不应混入源码 Git 提交。
