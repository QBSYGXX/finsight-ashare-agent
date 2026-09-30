# Linux 便携 Conda 环境

环境在本机 Docker/WSL2 的 Rocky Linux 8.10 容器中构建，使用 micromamba 创建真正的 Linux Conda 环境，再用 conda-pack 打包。服务器只需解压和运行包内 Python，不需要修复现有 Miniconda，也不需要 Docker、root 权限或本机 SSH。

## 使用

和 `finsight-server-v0.2.zip` 一起上传环境包 `finsight-conda-linux-x86_64-cu126-v0.1.tar.gz`。上传、解压和诊断命令见 [服务器手册](SERVER_GUIDE.md)。校验文件已经随代码包提供，不必单独上传。

安装目标固定为 `$HOME/projects/finsight-ashare-agent/.runtime/envs/finsight`，以项目的实际位置为准。无需执行 `conda activate`。`bash server/run.sh ...` 会优先使用这个环境；`setup` 不再下载和安装 Python 依赖，但仍需要下载 Qwen3.5-9B 和 bge-m3 权重。

环境包包含 Python 3.11、PyTorch 2.8.0/cu126、Transformers 5.8.0、PEFT 0.18.1、固定提交的 LlamaFactory、Sentence Transformers、PDF解析与测试依赖。不含9B基础模型、财报数据、NVIDIA内核驱动或用户凭据。

当前包大小4,088,207,720字节（约4.09 GB），已在2026-09-23构建并完成独立断网容器安装。SHA256及具体依赖版本见 `server/environment/artifact.json` 和同目录校验文件。

没有额外编译 `flash-linear-attention` / `causal-conv1d`。检查时可能提示使用PyTorch实现，这是当前配置的正常回退路径；真实GPU性能和显存仍由20步试训验收。

## 服务器要求

- Linux x86_64，glibc >= 2.28。脚本在解压前检查；旧 CentOS 7 的 glibc 2.17 不符合此包要求。
- 安装时至少30 GiB空闲；整个项目运行前的 doctor 要求至少80 GiB空闲。
- Bash、GNU tar、sha256sum、getconf、flock、awk、tee 等常见 Linux 工具。
- 驱动与 A800 的兼容性由 `doctor` 检查；当前候选 profile 使用驱动主版本 >= 560 的保守门槛。真实4卡训练仍须通过20步试训。

历史记录仅支持服务器曾使用 Linux x86_64 和项目独立环境；不能据此确认今天的 glibc、驱动版本及 Miniconda 具体故障。

## 安装行为与排错

脚本先校验 SHA256，再解压到最终目录并执行 `conda-unpack`，随后检查库导入和微型 Qwen3.5 的前向、反向、LoRA保存加载。最后才写入 `.finsight-ready.json`。这项检查不下载9B权重，也不表示A800训练通过。

同一个环境包在相同目录中重复安装，会检查已完成标记并退出；未完成安装可用相同命令重新解压。已有未知环境时拒绝覆盖。安装日志位于 `runs/install-env-时间.log`，最后有 `INSTALL_EXIT` 和日志路径。请把错误附近的输出发回。

不要在 `conda-unpack` 后移动或重命名项目目录。需要迁移时，在新位置从原始压缩包重新安装；不要直接复制已经重定位过的环境。

`SHA256 mismatch` 通常表示上传不完整或代码包与环境包不匹配。glibc过低或驱动诊断未通过时，回传报错或 `exports/doctor.json`，不要替换服务器系统库或修改论文环境。

## 构建与证据

Windows本机需要可运行Linux容器的Docker Desktop。构建命令：

```powershell
python scripts/build_portable_env.py --output-dir C:\Users\Lenovo\Desktop\code\_release_work
```

构建日志保存在输出目录，包含阶段、实时输出和无新输出时的心跳。Docker volume `finsight-conda-build-cache` 保留已下载依赖，失败后重跑可复用。没有执行任何服务器SSH命令。

发布时同时生成环境 SHA256、Conda包版本记录、pip依赖锁及 Linux验证结果。`finsight-conda-explicit.txt` 是micromamba的原始包URL清单，用于追溯；安装直接使用压缩包。基础模型不放进环境包；压缩包已由 `.gitignore` 和代码发布白名单排除。环境包用于手动传输，不通过 `git add` 上传。

本次验证：Linux容器42项测试通过；重新解压后的包导入、微型Qwen3.5前向/反向、LoRA保存/加载和LlamaFactory启动通过；另一个断网容器从零执行安装、重复安装和项目入口，均退出0。检查结果保存在 `server/environment/finsight-conda-environment-check.json` 和 `finsight-offline-install-check.json`。
