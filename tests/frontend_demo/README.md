# 前端手动实验

这些代码不是业务前端，也不会被后端推理流程调用。

- `runner.py`：命令行读取案例、调用真实前端、保存实验结果。
- `checks.py`：模拟缺失信息需求和字段覆盖统计，不判断语义是否正确。
- `sample_cases.jsonl`：实验输入，不是放进业务 Prompt 的 few-shot 示例。
- `requirements.txt`：独立前端加 embedding 实验的依赖。

在项目根目录，配置 LLM_BASE_URL、LLM_MODEL、LLM_API_KEY 后运行：

```bash
python -m tests.frontend_demo.runner --output frontend_results.json
```

此命令会调用真实模型。仅查看选项可使用 `--help`，不会加载模型或发送请求。
离线结构与集成测试仍在 `tests/test_semantic_frontend.py`，人工响应不代表真实 LLM 效果。
