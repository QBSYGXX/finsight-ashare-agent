# Qwen3.5-9B 单请求推理优化 GitHub 调研

核查日期：2026-09-27。任务是提高 token/s，保留思考能力和当前财务确定性工具。
本轮只做资料/本地快照核验，没有安装推理引擎或启动服务器实验。

## 推荐路线

1. 首选 vLLM，单 A800、BF16、文本模式、16K上下文，先不开推测解码。
   检查 Qwen3.5 GDN 路径、CUDA Graph 实际启用情况，再与单卡 Transformers 比较。
2. 通过正确性检查后启用原生 MTP，从 num_speculative_tokens=1 起步，比较关闭/开启。
   MTP 由主模型验证预测 token，适合低并发、关注单请求延迟的场景。
3. SGLang 是备选；llama.cpp CUDA 是另一种部署路线，需要 GGUF 转换与独立验证。
   暂不把量化、多卡并行和推理引擎迁移同时引入，以便分离效果。

## 直接证据

- vLLM 仓库： https://github.com/vllm-project/vllm
- 9B专项配方： https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.5-9B.yaml
  读取到 date_updated=2026-09-14，min_vllm_version=0.17.0；明确描述 BF16单卡、
  reasoning parser=qwen3、原生MTP、文本模式 language-model-only。
  配方中的已验证硬件列表不是本机A800验收记录，不能据此报A800测速结果。
- CUDA Graph设计： https://github.com/vllm-project/vllm/blob/v0.30.0/docs/design/cuda_graphs.md
  提供FULL_DECODE_ONLY、FULL_AND_PIECEWISE等模式，并按后端能力调整。
  实际是否捕获/回放图要查看启动与运行记录，不能仅凭默认配置断言已启用。
- MTP文档： https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/speculative_decoding/mtp.md
  原生MTP无需另找独立草稿模型，建议从推测深度1开始。
- 推测解码说明： https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/speculative_decoding/README.md
  官方定位包括低至中等QPS、访存受限工作负载；标准推测采样旨在保留目标分布。
  浮点计算和调度可能改变输出，同一seed不保证与Transformers逐token相同。
- 本机 models/Qwen3.5-9B/config.json 的 text_config.mtp_num_hidden_layers=1；
  model.safetensors.index.json列有mtp.fc.weight、mtp.layers.0.*等权重键。
  这是快照元数据/权重索引核验，不等于已在服务器成功加载MTP。

## 环境适配

从用户已返回的v0.9.1结果包doctor.json复核：Python3.11.16、glibc2.28、
Torch2.8.0+cu126、Transformers5.8.0、Triton3.4.0、驱动535.86.10、A80080GB。
这是最近返回的诊断快照，并非本轮远程实时检查。

vLLM v0.30.0发布页（2026-09-22）：
https://github.com/vllm-project/vllm/releases/tag/v0.30.0
发布说明默认CUDA13.0；发布资产也有
vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl。
glibc标签与服务器2.28一致，但不能因此宣称整套Torch/FlashInfer/Triton依赖兼容。
同版本安装文档仍有默认CUDA12.9描述；应以具体所选资产及完整依赖为准，避免混用说明。

旧驱动兼容文档：
https://github.com/vllm-project/vllm/blob/v0.30.0/docs/getting_started/installation/gpu.cuda.inc.md
官方镜像提供可选CUDA compatibility libraries，涉及指定数据中心/专业GPU、
驱动和内核条件。不能简单断言535完全不能运行，也不能认为现有cu126能跑就保证
新引擎能跑。服务器Docker/Apptainer权限、本机离线构建条件尚未核验。

部署时应独立环境并锁定完整依赖；优先评估CUDA12变体或受支持的容器兼容路径。
若必须源码构建，需要额外确认服务器nvcc/工具链。此次不提供未经验证的pip安装命令。

## 其他候选及限制

SGLang： https://github.com/sgl-project/sglang
当前最新v0.5.20发布说明明确退休CUDA12发布线，v0.5.19为最后cu12x版本：
https://github.com/sgl-project/sglang/releases/tag/v0.5.20
Qwen3.5-9B历史上有GDN乱码/重复报告：
https://github.com/sgl-project/sglang/issues/22087
该issue已关闭，但关闭不构成本环境的通过记录，也不能表述为当前版本必然仍有同一bug。

llama.cpp： https://github.com/ggml-org/llama.cpp
Qwen3.5 dense/MoE支持PR已合并（2026-02-10）：
https://github.com/ggml-org/llama.cpp/pull/19468
需要GGUF转换、CUDA构建与模型输出验证；并非只能做4bit量化。BF16/F16表示可先用于
隔离引擎变化，后续才考虑量化收益与准确率代价。本轮没有获取本机A800实测速度。

## FLA分歧的额外线索

https://github.com/vllm-project/vllm/issues/51779
讨论Qwen3.5 GDN BF16/FP32中间运算差异。维护者在评论中表示已与模型方确认，
问题来自Transformers fallback，并建议使用flash-linear-attention。
因此不能简单把“与当前PyTorch回退不一致”解释为加速内核一定错误。
相关PR https://github.com/vllm-project/vllm/pull/51797 已关闭且merged=false，
不能当作已经合并的官方修复。该讨论的主要测量针对397B，不是我们9B重复现象的
根因证明。本项目仍需实际9B固定输入数值/计划检查、多题与多seed结果验证。

## 验收顺序

- Transformers单卡、vLLM单卡、vLLM单卡+MTP三个组分别记录。
- 首先固定输入、预热、batch=1、相同输出长度，分开统计prefill/TTFT与decode token/s。
- 保留思考，使用一致提示词和采样配置，完整问答检查指标/变换、数值、引用及拒答。
- 引擎迁移评测先不混入1024 token思考预算，预算机制单独作为另一实验维度。
- MTP记录草稿接受率和实际decode收益，不能把理论并行预测步数当提速倍数。
- 驻留服务减少每次冷加载的等待，但这属于冷启动/整体延迟优化，不冒充decode token/s提升。
- 不以多并发总吞吐、其他模型/H100的结果，推断本机A800单请求效果。

当前结论：存在具体可行候选，优先级为vLLM基础执行优化，其次原生MTP；
环境可安装性、实际提速幅度及正确率仍需要独立环境和服务器实测。
