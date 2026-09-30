# 新协议 LoRA v0.16

目的：训练模型生成当前 `agent-v015.5` 的 `plan → report_search/read_page/financial_metric → finish` 动作。现有Agent、公式、来源校验和诊断集保持冻结。旧LoRA不加载、不覆盖。本包准备好了训练流程，**本机没有启动9B训练，也没有产生新模型正确率**。

## 数据与训练方案

- 仅8家train公司：贵州茅台、五粮液、海尔智家、格力电器、宁德时代、隆基绿能、恒瑞医药、伊利股份。验证与独立测试公司不进入监督样本。
- 690条规则示范轨迹，已通过当前真实工具执行，转为2,273个下一动作样本。16类任务覆盖基本/派生指标、多指标、多年度、96条同比任务、缺失字段/口径、部分缺失、模糊需求拒绝、数字篡改请求、原文检索选句、混合任务及错误恢复。
- 21条原文和21条混合轨迹只教“选取包含指定业务词的完整原句”，不声称具备完整的经营分析监督。3个公司年度未找到合适短句，排除记录见data/audit.json。
- 8条错误恢复轨迹故意过早finish。该错误动作只留在上下文中，绝不作为学习目标；随后正确动作和结束动作才计算损失。
- 这些是执行过的模板教师轨迹，不是自然专家对话，也不是模型自主轨迹。数字来自财务工具，不能将工具执行通过当作独立准确率。
- 原生`apply_chat_template(enable_thinking=False)`与线上一致，历史和工具文本全部屏蔽损失；只监督当前JSON动作及EOS。
- BF16 LoRA，rank16、alpha32、dropout0.05、学习率1e-4、2轮、有效batch32、seed20260928。只训练语言层Linear；视觉、MTP、词嵌入及lm_head冻结。使用最终轮次，不按验证成绩挑选checkpoint。
- 每卡microbatch1；1/2/4卡对应梯度累积32/16/8。默认仅用物理GPU1，不默认占满4卡。
- 最大训练序列10,240 token，单动作最多1,024 token；超长直接停止，不静默截断。发布前本机tokenizers＋原生Jinja检查最长9,052 token，4个样本超过8,192，因此调整序列预算而非删掉工具证据；服务器仍会用实际AutoTokenizer重新检查。
- 仅计算动作尾部logits，避免给长历史每一位置都分配248k词表logits。服务器小型随机Qwen会先比较尾部损失与完整masked loss，再检查LoRA反向传播；不通过就不加载9B训练。

## 服务器执行

依赖服务器已有v0.15.5、原训练环境`.runtime/envs/finsight`、v0.13加速环境和validation-hybrid索引；无需管理员或联网安装。本包只新增`lora_v016/`。至少预留35GB可写磁盘空间，合并阶段还会检查可用空间。

上传`finsight-lora-v0.16.zip`与同名`.sha256`到项目目录。先看`nvidia-smi`，确认所选卡可用。单卡完整流程：

```bash
cd /calc/users/geyuboao/workspace/FinSight
sha256sum -c finsight-lora-v0.16.zip.sha256 &&
unzip -n finsight-lora-v0.16.zip &&
FINSIGHT_GPUS=1 bash lora_v016/run.sh all --background
```

若4张卡都空闲，可将最后一行改为下面这行。不要把单卡和四卡命令同时运行：

```bash
FINSIGHT_GPUS=0,1,2,3 bash lora_v016/run.sh all --background
```

程序打印RUN、PID、LOG及PROGRESS_LOG和`tail`命令。后台进程不依赖SSH会话持续连接。跟随打印的PROGRESS_LOG查看阶段、训练loss、退出码和最终EXPORT；心跳只表示进程还活着，连续180秒无输出会提示可能停滞。

完整顺序：输入/模型指纹 → 模板与标签检查 → 20步探针 → 新进程适配器重载 → 从基础模型重新开始2轮正式训练 → 再次重载 → 独立合并模型 → 基础模型41题 → 新模型41题 → 配对汇总 → 自动导出。

探针使用包含各类别最长样本的32条动作重复检查兼容性，不是质量评测，也不作为正式训练起点。任何阶段失败即停止后续阶段，自动导出诊断记录，最终`EXIT`非零。失败不能被汇报为训练完成。

也可分阶段执行；`prepare`打印RUN，后续用实际编号替换`RUN_ID`，整个实验保持相同GPU数量：

```bash
bash lora_v016/run.sh prepare
bash lora_v016/run.sh probe --run RUN_ID --background
# 上一步完成且EXIT=0后执行下一步，不能同时启动
bash lora_v016/run.sh train --run RUN_ID --background
bash lora_v016/run.sh merge --run RUN_ID --background
bash lora_v016/run.sh evaluate --run RUN_ID --background
bash lora_v016/run.sh export --run RUN_ID
```

输出位于`outputs/lora_v016/RUN_ID/`，包含prepare、probe、full、merged、evaluation/base、evaluation/tuned和comparison.json。旧输出不覆盖，不自动续训；失败后先导出定位，不手工删除完成凭证绕过门禁。

## 对照与模型保存

正式适配器在`full/adapter/`，可回退到原基础模型或旧LoRA；新合并权重在`merged/`，不覆盖`models/Qwen3.5-9B`。合并将LoRA增量加到原始分片的对应语言权重，保留所有原始张量及未训练的MTP，逐张量读回核验。这样无需依赖Qwen线性注意力与MTP组合的动态LoRA支持，也避免普通HF重存可能丢失MTP。

基础与合并模型分别在新进程中使用相同vLLM＋MTP、BF16、单卡、thinking off、seed42、每步1,024 token和10步上限。共享同一validation-hybrid索引、提示词、工具和原评分。基础模型重新实测，不能拿不同历史配置成绩冒充本次对照。保留原诊断/自然题/前置检查分组、逐题新增通过、退化和共同失败。

MTP草稿权重未训练，接受率或吞吐可能变化；候选仍由各自目标模型验证。模型合并存在通常的BF16舍入，这里的正式部署对照对象是合并后的新模型。

比亚迪、迈瑞独立测试仍不启用。现有41题已用于开发，结果只能说明开发诊断的变化；一次训练种子不能证明稳定泛化增益。训练loss下降、适配器重载成功、规则轨迹执行通过均不等于Agent正确率提高。

最终返回包包含配置、哈希、训练统计、重载结果和82题对照轨迹，不包含原始训练文本、模型权重或适配器权重；权重留在服务器。下载打印的ZIP及SHA256后再核验效果，不能预先宣称微调有提升。

## 读代码与验证

`corpus.py`定义任务并经当前Agent执行；`tokenization.py`负责训练与推理格式一致；`train.py`训练并验证保存重载；`merge.py`保留原始权重结构；`evaluate.py`复用原评分做公平对照；`launch.py`负责阶段、日志和进程清理。

可复用经验：监督微调的关键是**训练时看到的状态、学习的输出与部署时真正要求的行为一致**。这里用精确模板前缀校验、仅当前动作的标签掩码和真实工具回放验证它；同样适用于数据库查询、搜索和办公工具Agent。自查可以从一条recovery样本入手，确认错误finish在历史里但标签没有监督它。
