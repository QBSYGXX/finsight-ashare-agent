# 来源与修改边界

## FinSight

- Repository: https://github.com/RUC-NLPIR/FinSight
- Commit: `8fc5bce28ac201c713219a47d30367c6a84342f4`
- License: GPL-3.0，完整文本在根目录 `LICENSE` 和 `vendor/FinSight/LICENSE`。
- `vendor/manifest.json` 记录下载档案哈希及100个原样保留文件的逐文件SHA256。演示视频、图片、原始示例财报、GitHub工作流、环境变量示例未打入本项目。

`agent.py` 和 `protocol.py` 适配自上游 `src/agents/base_agent.py` 的动作解析和 action-observation 迭代设计：保留 `<execute>` / `<final_result>` 以及 assistant动作、user工具观察的会话表示。实现经过缩减重写，新增严格标签校验、JSON答案、事实引用校验、工具白名单、失败记录及明确的步数上限。

上游源码作为可比对来源随仓库保留，运行时使用 `src/finsight_ashare` 中的适配实现。没有直接调用上游完整 DataCollector/DataAnalyzer/ReportGenerator 编排，也不导入其 unrestricted code executor 或 pickle/dill checkpoint。AST 只解析字面量工具参数，不执行任意生成代码。这里没有声称复现 FinSight 论文的完整系统或成绩。

新增财报数据管线、财务工具、Qwen本地推理、训练轨迹构建、公司划分、评测及服务器运维代码均在本项目中明确给出。对上游内容的修改/适配仍受 GPL-3.0 约束。

## 运行依赖

- Qwen/Qwen3.5-9B: https://huggingface.co/Qwen/Qwen3.5-9B ，Apache-2.0。下载时锁定实际 revision。
- BAAI/bge-m3: https://huggingface.co/BAAI/bge-m3 ，以该版本模型卡/许可为准，下载时锁定 revision。
- LlamaFactory: https://github.com/hiyouga/LlamaFactory ，Apache-2.0。候选固定提交 `97b32d3133b501432141a82949d5c7bc4d94f23a`，setup在独立环境安装。
- Transformers、PyTorch、PEFT、Sentence Transformers、PyMuPDF、PyYAML：使用各依赖原有许可；第三方依赖未随代码ZIP分发。PyMuPDF的AGPL/商业许可需在发布部署组合时另行遵守。

## 财报样例

`examples/maotai_2024/seed.json` 是对公开披露年报的少量结构化事实与短证据引用，包含原始链接和文件校验值。原始财报不按项目GPL再许可，不随Git发布。种子数据经过本次 Codex PDF图像和文本交叉核对，非人类审计签字；用于起步示范，研究使用前可按原页再次复核。
