# v0.13：不改系统的 vLLM 离线安装与实测

目标是提高真实生成吞吐（token/s），保留原生 thinking，并区分加载、预热和生成耗时。
这是待服务器验收的候选环境；此前通过的 CUDA/PTX 探针不等于 vLLM 已运行成功。

## 已有证据

2026-09-27 用户返回的 native_probe_v012 终端日志显示：llguidance 1.7.6 语法编译通过；系统
libcuda 的 PTX 8.8 对照失败；项目内 CUDA compatibility 575.57.08 的 PTX 8.8/sm80 内核执行结果为42，
driver API 12090，实际加载路径指向项目目录。该结果包尚未在本机独立核验，以上依据为终端日志。

## 上传及第一步

将以下四个文件上传到 `/calc/users/geyuboao/workspace/FinSight`：

- finsight-vllm-v0.13.zip 及其 .sha256
- finsight-vllm-wheels-v0.13.zip 及其 .sha256

```bash
cd /calc/users/geyuboao/workspace/FinSight
sha256sum -c finsight-vllm-v0.13.zip.sha256 &&
sha256sum -c finsight-vllm-wheels-v0.13.zip.sha256 &&
unzip -n finsight-vllm-v0.13.zip &&
unzip -n finsight-vllm-wheels-v0.13.zip &&
bash vllm_v013/run.sh setup --gpu 1 --background
```

后台命令立即返回 PID 和 LAUNCHER_LOG。按它打印的 `tail -n 60 -f '路径'` 查看进度；
Ctrl+C 只退出 tail，不会终止后台任务。每30秒有进度心跳。`EXIT=0` 才表示该阶段完成；
遇错立即停止，输出错误文件、完整子进程日志和可下载的 EXPORT ZIP。

setup 在 `.runtime/envs/finsight-vllm-v013` 创建独立 venv，复用原环境的 Python 3.11 二进制及
底层共享库，但不继承其 site-packages，也不安装任何包到原环境。无需 conda 命令、sudo、容器或系统驱动升级。
预留至少35GB空闲空间用于上传ZIP（约6.2GB）、解压后的wheel、约13.2GB安装文件及编译缓存。重复 setup 仅允许同一锁文件拥有的新环境；
不会覆盖其他已有环境。原环境保留在 `.runtime/envs/finsight`。

## setup 成功后的验收

先运行一次完整推理，保留思考，不设置1024的思考截断：

```bash
bash vllm_v013/run.sh smoke --gpu 1 --backend vllm --background
```

首次引擎加载可能需要编译内核及 CUDA graph capture，比之后慢。成功条件包括正常停止、JSON计划合法、
两项指标的值/单位/事实ID与固定工具参考一致。单题通过并不说明开放问答准确率。

smoke 成功后才运行三引擎对照：

```bash
bash vllm_v013/run.sh compare --gpu 1 --background
```

HF → vLLM → vLLM-MTP 串行启动独立进程，使用同一物理GPU（默认1）；不同时占用四卡。
MTP每步最多投机1个token，必须通过其自身完整问答。MTP失败会保留已完成的HF/vLLM记录，不能声称三组全部完成。
遇到GPU被他人占用，换空闲卡 `--gpu 0/2/3`；脚本不终止他人进程。

## 测速口径

- 同一Qwen3.5-9B基础模型，BF16、TP=1、原生thinking、seed42、temperature0.6、top_p0.95、top_k20。
- 正式完整问答 max_new_tokens8192、context16384；不把思考过程中的JSON当最终答案。
- 每引擎2次64token预热，然后3次固定512token测试，报告中位token/s。
- 固定长度测速在两端都抑制EOS；这些截断文本不作为财务回答。每引擎另跑完整问答验收。
- 跨引擎校验输入token ID哈希完全一致；相同seed不保证输出token完全一致。
- 默认指标是生成调用耗时（含prefill）计算的token/s；引擎时间戳提供decode估算，MTP批量吐token时需谨慎解读。
- 独立记录加载耗时；vLLM关闭prefix cache。不同序列长度的完整回答耗时不能替代固定长度测速。
- 此实验同时升级Torch及CUDA运行库，结果代表整套推理环境收益，不能全归因于vLLM单一因素。

## 依赖及兼容边界

196个wheel精确锁版本和SHA256，`pip --no-index --only-binary=:all: --require-hashes`。
主链为 vLLM0.23.0+cu129、Torch2.11.0+cu129、Transformers5.8.0、Triton3.6.0。
llguidance1.7.6为已实测的conda-forge ABI3二进制原样重打包，保留许可；不是冒充官方PyPI wheel。
详见 WHEELS.json 和构建脚本 `scripts/build_vllm_offline.py`。

FlashInfer0.6.12强制依赖cuda-tile[tileiras]，因此完整依赖中还含CUDA13编译组件。
它们位于独立的`nvidia/cu13`路径，CUDA12文件在其他路径，无运行文件覆盖；脚本不将cu13目录加入PATH或
LD_LIBRARY_PATH。若实际运行尝试使用不兼容的CUDA13编译路径，应以错误记录继续定位，不宣称此项已经通过。
另有两个上游wheel同时附带顶层build_backend.py；已核对不是运行时导入，本项目只安装现成wheel，不执行该构建后端。
首次vLLM完整模型运行仍是必要的兼容性检查；本机没有完成Linux GPU验收。

## 产物与回退

所有输出在 `outputs/vllm_v013/`；每次正常结束或被捕获的失败都打包到 `exports/finsight-vllm-v013-*.zip`。
返回该ZIP及.sha256即可核验。没有基础权重、凭据或整个环境。
停止使用新环境即可回退：继续 `bash thinking_v0102/run.sh chat`。
本补丁不改原模型、LoRA、训练数据、事实库、旧脚本和既有评测成绩。
