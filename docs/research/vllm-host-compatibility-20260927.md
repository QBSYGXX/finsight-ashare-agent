# vLLM 离线环境选择：实际服务器核验（2026-09-27）

当前状态：**离线安装包未完成；未安装vLLM、未运行vLLM推理、未测得引擎加速。**
Windows本机的全二进制依赖解析在llguidance处失败，退出码1。不能把单个vLLM轮子的
manylinux标签当成整个环境兼容性的证明。没有使用忽略依赖的强行安装。

## 服务器报告

报告包 `finsight-vllm-doctor-20260927-184723-vllm-probe-v011-3051073.zip`，
SHA256 `c510fd6539dc318295e8a659e134bd01a8adac515a95ca60165c538baef17ac1`。
旁附SHA256、ZIP CRC和诊断完成状态已核验。报告源码提交
`7544b3c4db4329ba63f29ce4e82c85411919debc`。

| 项目 | 实际报告 |
|---|---|
| 系统 | AlmaLinux8，glibc2.28，内核4.18.0-372.51.1.el8_6.x86_64 |
| Python | 3.11.16 |
| 基线 | Torch2.8.0+cu126、Transformers5.8.0、Triton3.4.0 |
| 显卡与驱动 | 4×A80080GB，算力8.0，驱动535.86.10 |
| 工具 | gcc/g++8.5.0；PATH中无nvcc、cmake |
| Docker | 26.1.3客户端存在，但连不上daemon；不能当作可用容器 |
| 其他容器 | PATH中无Apptainer、Singularity、Podman |
| CUDA兼容库 | 本次搜索只发现系统libcuda535.86.10；未发现前向兼容库 |
| 模型 | mtp_num_hidden_layers=1；索引15个MTP张量，对应分片都存在 |

OS文本中VERSION=8.10而VERSION_ID=8.9，不用该报告断言精确小版本。
GPU0当时约22.8GB被既有FinSight进程占用；GPU1—3空闲仅是报告时间点的快照。
本轮没有终止服务器进程。没有重新校验全部模型张量；分片存在不等于MTP实际运行通过。

## 实际依赖阻塞

读取了vLLM官方0.30.0 CUDA12.9轮子中的METADATA（HTTP范围读取），
其标签为`cp38-abi3-manylinux_2_28_x86_64`，要求`llguidance>=1.7.0,<1.8.0`。
官方资产SHA256为`e98cb69659bfcfc849cf11ce0781a7161d40b02b51a6c3636924a5909f2aabcc`，
**完整545,459,905字节轮子没有下载完成，所以未完成该文件的整体哈希核验。**

使用uv0.12.5针对Linux x86_64/manylinux2.28/Python3.11进行only-binary解析，
失败于没有可用的llguidance1.7.x轮子。PyPI元数据复核：

- llguidance1.7.0、1.7.6的普通CPython ABI3 x86_64 Linux轮子要求manylinux2.31。
- 这两个版本另有manylinux2.17的`cp314-cp314t`轮子，但不能给Python3.11使用。
- llguidance1.6.1有manylinux2.28普通ABI3轮子，却不满足候选vLLM的>=1.7约束。
- 0.23.0、0.25.0候选的PyPI依赖也要求llguidance>=1.7,<1.8；0.23依赖预解析同样失败。
  这两版尚未完成CUDA12.9变体的完整安装解析，不表示它们所有变体均已排除。
- vLLM0.17.1本体要求manylinux2.31；0.20.0要求manylinux2.35，均不适配本机glibc2.28。

本机uv提示缺少Python3.11执行器，尝试获取Windows3.11后未成功建立minor-version链接，
回退现有3.13.5作为可能的构建执行器。上述解析明确指定目标Python3.11且禁止源码构建；
llguidance的限制另外直接从官方wheel标签核验，不依赖这个回退警告。

vLLM0.30.0还声明Torch2.13.0和Torchaudio2.11.0等要求，后续需要进一步核对整套依赖，
不能从当前第一个解析失败就假定剩余依赖可用。新CUDA/PTX与R535驱动的兼容性同样未验证。

## 下一条可行路径

优先向管理员确认是否能提供**当前账户可用的NVIDIA GPU容器**：Docker+NVIDIA
Container Toolkit或Apptainer均可。容器中的较新用户空间可以解决glibc轮子标签问题，
但**容器并不会升级宿主机驱动**；仍要选择CUDA版本、检查官方前向兼容支持并做实际GPU探针。
普通`docker --version`不构成验收，至少要能运行GPU容器并在其中完成CUDA运算。

如果无法使用容器，继续研究独立源码构建或可移植用户空间：先解决llguidance，再解析
整套依赖，最后验证CUDA/PTX。当前尚未把这条路线做成可交付环境；不替换系统glibc、
不升级系统驱动、不覆盖`.runtime/envs/finsight`，也不把Windows环境打包充当Linux环境。

确认路线后，在独立环境依次对比单卡HF、单卡vLLM BF16、单卡vLLM+MTP1。
保持提示词、思考模式、采样参数一致；预热后分开记录TTFT、固定长度decode token/s，
再检查完整财务查询的语义、数值和来源。MTP接受率和实际加速需要测量，不能预承诺倍数。

## 可复制给管理员的说明（未代发）

> 我需要在Cisri-M-lgpu2上用单张A800运行Qwen3.5-9B/vLLM推理实验。现有系统
> glibc2.28，候选依赖需要更新的用户空间；docker客户端存在但当前账户连不上daemon。
> 能否提供可访问GPU的Docker或Apptainer运行方式，并说明离线镜像导入方法、GPU分配规则？
> 宿主驱动是535.86.10，后续还需核对所选CUDA镜像的驱动/前向兼容支持。
> 实验拟用独立目录和环境，不修改现有论文或训练环境。

## 官方依据

- https://github.com/vllm-project/vllm/releases/tag/v0.30.0
- https://api.github.com/repos/vllm-project/vllm/releases/assets/580601076
- https://pypi.org/pypi/llguidance/1.7.0/json
- https://pypi.org/pypi/llguidance/1.7.6/json
- https://pypi.org/pypi/llguidance/1.6.1/json
- https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.5-9B.yaml

当前保留候选元数据和下载片段在Git忽略的`.cache/vllm-offline`；已停止慢下载，
不留后台无限等待进程。现有可用推理入口继续运行；此次独立v0.10.1补丁仅修正语义预检。
