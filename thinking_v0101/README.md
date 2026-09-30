# v0.10.1：比率同比歧义预检

基于已发布 v0.10 的独立副本，只增加执行前的共同语义检查。原版对照实验保留。
`毛利率同比`、`净利率同比`、`资产负债率同比`等问题须明确同比增长率或百分点变化。
其他指标的“增长率”不能替本指标补全口径。原始思考、预算思考和规则模式采用同一检查。

此包不提高 token/s，不包含vLLM，不改变模型、提示词、预算、财务公式或事实快照。
正常问题保留原行为。歧义题在调用模型前直接澄清，不能计入模型准确率或推理提速。
检查基于已登记别名和有限表达模式，不等于任意自然语言歧义均可识别。

上传ZIP和旁附SHA256至服务器项目目录：

```bash
cd /calc/users/geyuboao/workspace/FinSight &&
sha256sum -c finsight-thinking-v0.10.1.zip.sha256 &&
unzip -n finsight-thinking-v0.10.1.zip &&
CUDA_VISIBLE_DEVICES=0 bash thinking_v0101/run.sh chat
```

输入`中兴通讯2024年毛利率同比`应直接返回needs_clarification，不加载模型。
输入`中兴通讯2024年毛利率、资产负债率`按原预算模式运行。
原v0.10六题的正确率不因本补丁而追溯修改。新实验另存outputs/thinking_v0101。
需要对照时运行`bash thinking_v0101/run.sh evaluate --background`，查看各题trace。
回退只需重新使用`thinking_v010/run.sh`；不覆盖旧目录。
