# Research v1.0发布验证

2026-09-30在Windows上，从Git暂存树导出新的纯源码目录执行检查；该目录不含原财报、模型、环境、LoRA训练文本或历史输出。使用本机Python 3.13.5和pytest，不调用GPU或真实模型。

| 检查 | 结果 |
|---|---|
| `python -m pytest agent_v015 -q` | 67 passed，EXIT=0 |
| `python -m pytest lora_v0161 -q -k "not corpus_frozen_and_train_only and not training_data_and_recipe_unchanged"` | 17 passed，6 skipped，2 deselected，EXIT=0 |
| 新增构建/案例脚本 | Python语法检查通过；案例脚本校验固定返回包SHA256后实际运行 |

本机默认pytest临时目录曾触发WinError 5；实际成功命令额外使用`--basetemp <新建的专用测试目录>`。没有修改生产程序解决这个本机权限问题。完整检查日志仅保存在本机`_release_work/source-check-20260930-002238/`，不随源码公开。

6项跳过的测试依赖固定Transformers wheel来检查真实接口；源码包不包含wheel。2项排除测试检查冻结训练文本及旧/新配方一致性；文本不随源码分发。此处84项通过是工程回归检查，不是84道真实模型问答。

真实模型的训练与82次配对评测已经在服务器完成，验收详见[结果核验报告](experiments/lora-v0161-result-audit-20260930.md)，本次不再重复训练或评测。

发布脚本从干净HEAD构建ZIP，逐项验证成员集合、SHA256、CRC与冻结Agent/LoRA源码。LoRA清单中3份训练数据保留外部依赖，不被误作源码打包。包内`SOURCE_MANIFEST.json`记录最终提交号与所有分发文件哈希，包外`.sha256`校验整个ZIP。构建命令若校验失败会以非零退出，不应分发失败产物。

静态案例来自固定结果包的M01/base、R02/base和R02/tuned，保留原记录哈希；数字与引用直接取自已核验结果，不重新生成答案。公开材料同时保留改进与退化，不启用未使用的独立测试集来补分。
