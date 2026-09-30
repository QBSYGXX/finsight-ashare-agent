# 服务器操作手册

整个流程在你现有的服务器终端/桌面和文件上传界面完成，不需要在本机使用SSH。服务器项目与论文项目分开，工作目录为 `$HOME/projects/finsight-ashare-agent`。

## 1. 上传、解压环境并诊断

上传代码包 `finsight-server-v0.2.zip` 和环境包 `finsight-conda-linux-x86_64-cu126-v0.1.tar.gz` 到服务器同一目录。在该目录执行：

```bash
mkdir -p "$HOME/projects"
FINSIGHT_UPLOAD_DIR="$PWD"
unzip finsight-server-v0.2.zip -d "$HOME/projects"
cd "$HOME/projects/finsight-ashare-agent"
bash server/install_env.sh "$FINSIGHT_UPLOAD_DIR/finsight-conda-linux-x86_64-cu126-v0.1.tar.gz"
bash server/run.sh doctor
```

环境解压完全离线，不调用服务器 Miniconda。要求 Linux x86_64、glibc >= 2.28；具体检查与失败处理见 [便携环境说明](PORTABLE_ENV.md)。若以前已经上传 v0.1，先保留自己改过的配置，再允许 unzip 替换同名代码文件；数据和模型目录不在代码包中。

把 `exports/doctor.json` 下载回本机。诊断包含显卡、总/空闲显存、驱动、Python、磁盘和下载连通性，不采集密钥或环境变量内容。它不执行训练。退出码2表示环境与候选配置不符，但诊断文件仍生成。先用这个文件确认服务器适配情况。

## 2. 检查环境并下载模型

诊断通过且候选配置适合服务器后：

```bash
bash server/run.sh setup
```

安装过便携包时，setup 校验并复用 `.runtime/envs/finsight`，跳过 pip 安装。基础模型下载到 `models/Qwen3.5-9B`，embedding到 `models/bge-m3`；首次下载可能较久。环境包含Python3.11、PyTorch2.8.0/cu126、Transformers5.8.0及固定LlamaFactory版本。真实安装版本写入 `outputs/environment/requirements.lock.txt`。环境包不包含基础模型权重。

没有安装便携包、但服务器已有 Python3.11+ 的情况下，仍保留原来的 `.venv` 在线安装路径。不要在两种环境之间混用检查结果。

`doctor` 的网络探测成功只表示小文件可访问，不保证数十GB权重下载无阻。Hugging Face下载使用固定revision和可续传缓存；重跑setup会复用缓存。依赖冲突、驱动过旧、下载失败请回传日志，不要随意改论文项目依赖。

## 3. 第一个真实问题

```bash
bash server/run.sh smoke
bash server/run.sh smoke --count 10
```

没有事实库时，smoke自动下载贵州茅台2024年报，复核其SHA256和12条种子事实，再调用9B。第一条问题为营业成本换算；10条覆盖同比、毛利率及口径缺失。结果保存在 `outputs/smoke/时间-llm/`，包含 `cases.json` 和可在本机打开的 `report.html`。

只检查工具管线时使用 `smoke --replay --count 10`，此模式不调用模型，不能作为模型评测结果。

## 4. 财报与训练数据

```bash
bash server/run.sh prepare --stage fetch
bash server/run.sh prepare --stage ingest
```

按 `DATA_REVIEW.md` 复核 `data/audit.csv`。可以把CSV和需要核对的PDF取回本机处理，再上传CSV。然后：

```bash
bash server/run.sh prepare --stage audit
bash server/run.sh prepare --stage dataset
```

覆盖不足时停止并生成 `data/processed/missing_facts.json`，不会编造800条答案。只有首批资料时，可用 `prepare --stage dataset --pilot` 生成10条工具回放集；pilot不是完整训练数据。

正式数据写入 `data/frozen/`。验证和测试各100题以及文件哈希在此时冻结。先保存这份数据快照，再开展任何调参。变化的资料需要新的实验数据版本，不直接覆盖已有冻结实验。

## 5. 20步试训，再完整训练

确认4张卡未被其他任务占用。必要时在当前终端设置本项目可见显卡：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash server/run.sh train --probe --background
bash server/run.sh status
```

运行命令会打印PID、日志和状态路径。用其实际日志路径执行 `tail -f '实际路径/run.log'`。每30秒采样显存；60秒无新输出时给出心跳和最后阶段。出现无输出不等于一定卡死，也可能正在下载或初始化。

试训成功要求：训练子进程退出码0、20个optimizer step、损失有限、适配器文件存在、重新加载适配器能生成文本。生成文本的正确性仍另行评测。成功收据为 `outputs/training/probe.json`。

```bash
bash server/run.sh train --background
```

有效batch=4卡 × 每卡1条 × 梯度累积8=32。完整训练最多2轮，与试训分别使用新目录。正常结束后才生成 `outputs/training/production.json`；进程启动、某个checkpoint存在都不等于训练完成。重复执行训练会开新实验，不会静默接着一个未完成checkpoint跑。

本项目内的setup、推理、训练、评测互斥，避免互占显存。这个锁无法管理你的其他论文任务。需要停止本项目后台任务时，对打印的runner PID执行 `kill -TERM PID`；运行器会终止自己的子进程组并记录失败退出状态。请勿使用全局pkill。

## 6. 对照评测与问答

可先在验证集跑前10题检查管线：

```bash
bash server/run.sh evaluate --split validation --limit 10 --background
```

这不是正式100题结果。完整实验：

```bash
bash server/run.sh evaluate --split validation --background
```

验证集调好后再测试，禁止用测试成绩挑模型或修改提示词：

```bash
bash server/run.sh evaluate --split test --background
bash server/run.sh chat
```

直接检索、未微调Agent、微调Agent使用相同快照和模型revision，分别记录每道题的预测、标准答案、工具轨迹和错误。中断重跑会复用当前相同配置已完成题目的记录。首版以CPU运行bge-m3，推理速度较慢但减少显存竞争。`--search lexical` 可用于检索消融，必须作为不同实验报告。

终端chat默认未微调模型，使用训练收据中的实际适配器目录传 `chat --adapter outputs/training/full-实际时间` 可演示微调版。首版终端问答和离线报告已覆盖演示，不要求服务器开放Web端口。

## 7. 导出与回传

```bash
bash server/run.sh export
```

下载 `exports/finsight-results-时间.zip`。其中有训练配置、依赖记录、模型revision、数据划分、适配器、评测、失败案例、HTML和末120行日志；不含基础模型和原始PDF。结果ZIP用于回传分析，不是直接公开到GitHub的发布包。

报错时发对应 `runs/.../status.json` 和 `run.log` 的末100行。诊断阶段只需 `exports/doctor.json`。本机修复后提供补丁，手动上传替换对应代码，不删除服务器数据或权重目录。
