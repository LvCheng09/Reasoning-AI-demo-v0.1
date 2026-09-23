# Event Reasoning Demo v0.1

[English README](README_EN.md)

这是一个可以直接运行的事件图推理演示版本：前端 LLM 只把自然语言翻译为规范 Event Graph；后端用 Knowledge 证明可补全条件、用 Rule 网络逐步推理；最后只把 Control 放行的结果改写为用户可读回答。

本项目采用 [Apache License 2.0](LICENSE)。

当前正式演示包含两个相互连接的领域：

- 财务：应收账款周转天数（DSO）计算。
- 库存与供应链：库存资金占用、缺货暴露、经营现金流压力。

前端只做语义规范化和 Event Graph 构建，不在前端做业务推理。此发布包只包含推理系统，不包含 Auto-Learn、训练或规则重构代码；每次演示都不会修改随仓库发布的 Rule / Knowledge 数据库。

## 技术原理

完整的设计动机、feature 表示、Knowledge/Rule 分工、严格匹配、Control、Calculator、Memory、Output Gate 和当前边界，请看：

- [技术原理说明（中文）](docs/TECHNICAL_EXPLANATION.CHINESE.md)
- [Technical overview (English)](docs/TECHNICAL_EXPLANATION.ENGLISH.md)
- [模块边界](docs/ARCHITECTURE.md)

## 快速开始

要求：Python 3.11+、可用的 PyTorch 环境，以及一个可访问 SiliconFlow 的 API Key。前端模型固定为 `Qwen/Qwen3-8B`。

```bash
cd Event-reasoning-demo-v1
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[demo,dev]"

# 编辑后只在当前终端加载；不要把 .env 上传 GitHub。
source .env.example
export LLM_API_KEY="你的真实密钥"

python demo.py --preset dso
```

第一次运行会下载默认的本地嵌入模型 `Qwen/Qwen3-Embedding-0.6B`。如果你已有本地模型快照，把 `MAIN_PART_EMBEDDING_MODEL` 设为该目录，并设置 `MAIN_PART_EMBEDDING_LOCAL_FILES_ONLY=true`。

可运行的内置问题：

```bash
python demo.py --preset inventory
python demo.py --preset stockout
python demo.py --preset cash-flow
python demo.py --preset missing-condition
python demo.py --question "某公司库存周转变慢，安全库存偏低，会有什么影响？" --json
```

将完整运行记录保存为 JSON：

```bash
python demo.py --preset dso --output demo-results/dso.json
```

`data/demo/db/knowledge.sqlite` 保存 core–relation–nature 知识网络，`data/demo/db/rule.sqlite` 保存 Rule 网络。Memory 是运行时经验层，不随 Demo 发布，也不会在演示中被写成学习结果。

## 验证

以下测试不调用真实 API，不下载模型：

```bash
python -m pytest tests/test_contracts.py tests/test_semantic_alignment.py tests/test_semantic_frontend.py tests/test_output_presentation.py tests/output_decoder/test_output_decoder.py
```

真实 API 演示需要你自行提供密钥，并受网络、供应商可用性与模型下载影响。一次正常演示会使用同一个 `Qwen/Qwen3-8B` 两次：第一次构建 Event Graph，第二次仅把已验证结果改写为自然语言；第二次失败或改写破坏数值/单位时，会自动显示 Decoder 的确定性草稿。典型单题耗时不能视为固定性能基准。

## 当前边界

- 这是两个领域的研究演示，不是通用问答产品。
- Rule / Knowledge 的质量与覆盖范围决定能力上限；不满足条件时系统应拒绝补齐或换路，而不是放宽匹配。
- 已验证的展示路径包括 DSO、库存资金占用、缺货暴露、库存对经营现金流的跨领域影响，以及一个缺条件的拒绝场景。真实 LLM 服务偶发失败仍会使单次演示失败，重试后可能恢复。
- 本仓库没有包含 API Key、模型权重、旧运行记录、训练日志或原工程 Git 历史。

## GitHub 上传前

```bash
git init
git add .
git status
git commit -m "Release demo v0.1"
```

确认 `git status` 中没有 `.env`、模型目录或 `demo-results/` 后，再在 GitHub 创建空仓库并按 GitHub 页面提示添加 remote、push。
