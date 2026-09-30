# 无管理员权限的原生兼容性验证 v0.12

用途：在现有A800服务器验证两个实际运行门槛。此包**不是完整vLLM环境**，
不加载9B模型，不做训练，不改变系统驱动、系统glibc或现有Python环境。

1. 加载conda-forge的llguidance1.7.6 Linux ABI3扩展，实际编译并校验JSON语法。
   该构建声明glibc>=2.17，不同于此前检查的PyPI manylinux2.31轮子。
2. 在独立子进程加载NVIDIA官方cuda-compat-12-9/575.57.08用户态库，
   JIT编译PTX8.8/sm80，执行GPU核并验证输出42；使用旧系统库做对照。

无下载、无sudo、无pip安装、无环境激活。库搜索路径仅对子进程生效。
运行时在包内创建相对库链接，输出写入runs/outputs/exports。
包保留第三方来源、包哈希与许可出处。llguidance使用现有便携环境的libgcc；
是否满足运行时条件以实际导入结果为准，不以包标签代替执行验证。

## 操作

把ZIP和SHA256上传到`/calc/users/geyuboao/workspace/FinSight`，执行：

若收到的是`finsight-rootless-step1-v0.12.zip`合并包，请校验其同名SHA256并解压，
然后同样运行`bash native_probe_v012/run.sh --gpu 1`。合并包还包含交互修复入口
`thinking_v0102/run.sh chat`，两个组件均保留各自的文件哈希清单。

```bash
cd /calc/users/geyuboao/workspace/FinSight &&
sha256sum -c finsight-native-probe-v0.12.zip.sha256 &&
unzip -n finsight-native-probe-v0.12.zip &&
bash native_probe_v012/run.sh --gpu 1
```

`--gpu 1`表示nvidia-smi的物理GPU1，程序用PCI地址选择它，避免逻辑卡号混淆。
本测试只短暂创建CUDA上下文、分配4字节并运行1个线程，不终止任何已有进程。
如平台给你分配了其他GPU，请使用对应物理编号。

每阶段最多180秒，20秒无新输出会打印当前阶段心跳；GPU清单最多20秒。
退出后不论成功失败都尝试导出结果，下载`RESULTS=`的ZIP及SHA256回本机_release_work。
旧库的PTX8.8对照失败不代表整包失败。验收需要`llguidance`和`compat_ptx`都成功，
最终`STATUS=native_gates_passed EXIT=0`。

这只证明两个门槛通过。完整vLLM/Torch/Triton环境、全模型执行、MTP接受率与token/s
仍需随后验证，不能把本探针成功解释为vLLM已经可用。

## 回退

没有改写既有环境。运行结束后直接继续旧入口即可；无需卸载或恢复系统文件。
