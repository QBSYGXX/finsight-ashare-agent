# 从哪看起

先看[项目讲解](PROJECT_WALKTHROUGH.md)，再对照[真实运行案例](../examples/research_v1/README.md)。当前版阅读顺序如下，不必从历史补丁开始：

| 顺序 | 文件 | 要弄懂的问题 |
|---|---|---|
| 1 | [agent_v015/cli.py](../agent_v015/cli.py) | 如何接收问题、复用模型、保存答案与轨迹？ |
| 2 | [agent.py](../agent_v015/agent.py)、[action_contract.py](../agent_v015/action_contract.py) | plan、工具观察、下一动作和finish怎样形成循环？ |
| 3 | [retrieval.py](../agent_v015/retrieval.py)、[excerpts.py](../agent_v015/excerpts.py) | BM25与向量怎样融合？引用如何恢复到原句及页码？ |
| 4 | [finance.py](../agent_v015/finance.py)、[registry.json](../agent_v015/registry.json) | 公式、单位、报表口径及同比比较列由谁检查？ |
| 5 | [lora_v0161/corpus.py](../lora_v0161/corpus.py)、[tokenization.py](../lora_v0161/tokenization.py) | 教师轨迹怎样执行？为什么只监督当前动作与EOS？ |
| 6 | [train.py](../lora_v0161/train.py)、[merge.py](../lora_v0161/merge.py) | LoRA训练哪些层？为什么合并时要保留MTP张量？ |
| 7 | [evaluate.py](../lora_v0161/evaluate.py)、[结果报告](experiments/lora-v0161-result-audit-20260930.md) | 两个模型如何公平对比？哪些失败被总分掩盖？ |
| 8 | [vllm_v013/adapter.py](../vllm_v013/adapter.py)、[测速报告](experiments/vllm-v013-inference-audit-20260927.md) | token/s、模型调用次数和端到端耗时有何区别？ |

`src/finsight_ashare/`与根目录安装后的`finsight`命令属于早期原型，不是本版推荐Agent入口。旧800条轨迹与新2,273个动作样本也不是同一次实验。新协议只监督当前JSON动作及EOS；历史动作与工具观察不参与loss。

没有GPU的电脑也可以在项目根目录执行：

```bash
python -m pip install "pytest>=8,<10"
python -m pytest agent_v015 -q
```

测试使用合成文档和脚本动作，检查计算、引用和流程约束，不测试真实模型准确率。完整说明见[阶段交付](DELIVERY.md)。

自查练习：跟踪“中兴通讯2024年毛利率”的工具结果，找到收入和成本的来源；解释为什么缺少净利润时不能用归母净利润替代计算销售净利率。把字段口径、计算与模型选择分开，也适用于基金研究和办公数据Agent。
