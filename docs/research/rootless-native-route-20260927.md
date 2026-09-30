# 无管理员权限路线：新增兼容性证据

本记录补充上一份`vllm-host-compatibility-20260927.md`：此前验证的是PyPI普通CPython
预编译轮子受glibc2.31限制，不能据此断言所有无管理员部署路径都不可用。
用户明确无法联系服务器管理员，因此改用用户目录中的原生库验证路线。

## 新发现

conda-forge提供`llguidance-1.7.6-py310hc9716df_0.conda`，包声明glibc>=2.17、
CPython>=3.10、libgcc>=14。SHA256：
`b280507d74e8372a16669e8280858e3951eaf84b8dfb4200d0fa95015392831b`。
下载及包内文件哈希核验通过。二进制是x86_64 ABI3，读取ELF所需版本符号，
最高GLIBC_2.14。服务器既有便携环境的构建清单包含libgcc16.2.0，实际导入仍需测试。
这不是把不兼容的PyPI轮子改名，也没有降级或忽略vLLM的llguidance版本要求。

NVIDIA RHEL8官方仓库`cuda-compat-12-9-575.57.08-1.el8.x86_64.rpm`，
SHA256 `89855e5ce2b80ed5fc48368000e1d540a126e8f30e04cd98f9d0e1cb00fc660b`，
54,679,936字节。通过仓库repomd→primary元数据→RPM哈希链核验，提取用户态兼容库；
ELF所需glibc版本均不超过2.28。这些库仅由探针子进程加载，不替换系统驱动或libc。
此静态检查不证明R535.86.10+A800的运行时兼容性。

把llguidance暂从输入需求中分离后，针对Linux x86_64/glibc2.28/Python3.11，
vLLM0.23.0的其余PyPI需求（CUDA12相关extras）与Transformers5.8.0能解析；
结果包含Torch2.11.0+cu129、Torchaudio2.11.0+cu129、Torchvision0.26.0+cu129、
xgrammar0.2.3。这是可行性初筛，不是完整环境锁：尚需读取0.23 CUDA12.9轮子
自身METADATA、合并已验证的llguidance构建、完成实际安装和全模型运行。
没有将缺失依赖的解析结果当作可安装环境发布。

## 本次交付门槛

`native_probe_v012`实际测试：

1. 导入本包llguidance并执行JSON语法编译和校验。
2. 使用系统libcuda尝试PTX8.8作为对照，失败可以记录。
3. 使用包内兼容libcuda和PTX编译库，在指定物理GPU上JIT编译sm80核、运行并验证输出42。

两个必要门槛为llguidance与compat_ptx，必须实际执行成功才返回native_gates_passed。
每个子进程限时180秒、保留stdout/stderr、每20秒输出当前阶段，失败也导出结果。
仅分配4字节的测试缓冲，CUDA上下文本身另有少量开销；不加载权重或启动训练。
本机测试只验证编排、损坏包拒绝、PCI选择、成功/失败路径与清理逻辑；无本地A800执行结论。

服务器由用户手动上传运行，不能本机SSH。通过上述门槛后才准备完整独立vLLM环境，
继续测单卡BF16及MTP1，不改变冻结事实、训练参数和历史成绩。

## 交互修复

用户v0.10.1日志显示输入期间PROCESS_RUNNING持续打印。v0.10.2暂停输入期心跳，
提交后回显实际QUESTION，按工作阶段重新计时，并增加MODEL_LOAD_COMPLETE耗时。
它不更改模型、提示词、预算或财务算法。终端中的“20年”确实是程序收到的输入；
仅凭日志无法证明丢字由心跳造成，因此不自动补成2025年。

这次生成记录1069token/49.433秒与既有预算模式相近；此前长等待出现在模型加载前后，
原进程总计时混入用户输入时间，不能直接当作精确冷启动耗时。新计时用于进一步定位。

## 来源

- https://api.anaconda.org/package/conda-forge/llguidance
- https://conda.anaconda.org/conda-forge/linux-64/llguidance-1.7.6-py310hc9716df_0.conda
- https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/repodata/repomd.xml
- https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html

本机Docker CLI启动尝试未就绪，已停止等待该命令；新路线不依赖本机Docker守护进程。
