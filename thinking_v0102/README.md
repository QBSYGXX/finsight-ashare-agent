# v0.10.2 交互输入修复

独立派生自v0.10.1，模型、思考预算、公式、引用检查保持原版本。
等待输入时暂停进度心跳；提交后回显程序实际收到的QUESTION，并恢复工作阶段心跳。
模型首次加载另记MODEL_LOAD_COMPLETE耗时，不把输入等待时间算成加载耗时。
不会猜测补全缺失年份，也不会把“20年”自动改成“2025年”。

```bash
cd /calc/users/geyuboao/workspace/FinSight &&
sha256sum -c finsight-thinking-v0.10.2.zip.sha256 &&
unzip -n finsight-thinking-v0.10.2.zip &&
bash thinking_v0102/run.sh chat
```

输入时应不再夹入PROCESS_RUNNING；加载和生成时仍有日志。
查询请一次粘贴完整问题，例如`贵州茅台2025年毛利率同比增长率`。
新输出写入outputs/thinking_v0102；旧包保留，换回旧入口即可回退。
此补丁不是vLLM，不提高token/s。
