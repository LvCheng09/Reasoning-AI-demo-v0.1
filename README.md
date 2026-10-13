# Reasoning-AI-demo-v0.1

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
cd Reasoning-AI-demo-v0.1
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

这两个数据库是发布种子。Demo与Python默认入口均先创建 `runtime/inference_*` 独立副本，检索计数、索引和运行Memory只写副本；JSON报告会列出实际路径。不要把 `runtime/` 上传GitHub。若设置 `MAIN_PART_DB_DIR`，它必须是已有的可写运行库目录，不能指向发布种子。

报告分别记录初始化、前端与输入上下文、图接线、后端墙钟/进程CPU和输出层耗时，以及供应商返回的token用量。缺少用量记为未知，不能按免费计算。输出层回退会报告原因。退出码：`0`表示Control明确求解或明确阻断；`1`表示运行错误；`2`表示未完成/结果未确认。`blocked`不等于`solved`。

## 验证

以下测试不调用真实 API，不下载模型：

```bash
MAIN_PART_FRONTEND_VECTORIZE=0 python -m pytest
```

另有真实后端回归，从人工构造的Event图经过实际Launcher、Matcher、Executor和输出门，不调用外部LLM，也不使用伪向量。它要求本地已有嵌入模型，禁止联网下载：

```bash
python -m tools.validate_release_runtime --output demo-results/release-check.json
```

每次使用新的输出文件名，旧记录不覆盖。这不是前端LLM测试、盲测或泛化证明。GitHub自动测试执行全部离线测试，不把这些检查数当成真实题目数。

真实 API 演示需要你自行提供密钥，并受网络、供应商可用性与模型下载影响。一次正常演示会使用同一个 `Qwen/Qwen3-8B` 两次：第一次构建 Event Graph，第二次仅把已验证结果改写为自然语言；第二次失败或改写破坏数值/单位时，会自动显示 Decoder 的确定性草稿。典型单题耗时不能视为固定性能基准。

## 当前边界

- 这是两个领域的研究演示，不是通用问答产品。
- 最新动态中策略、执行中证据失效/重绑及成本对照仍属于隔离实验，不在此版本的默认入口中；不能把实验收益当成该仓库的性能承诺。
- Rule / Knowledge 的质量与覆盖范围决定能力上限；不满足条件时系统应拒绝补齐或换路，而不是放宽匹配。
- 已验证的展示路径包括 DSO、库存资金占用、缺货暴露、库存对经营现金流的跨领域影响，以及一个缺条件的拒绝场景。真实 LLM 服务偶发失败仍会使单次演示失败，重试后可能恢复。
- 本仓库没有包含 API Key、模型权重、旧运行记录、训练日志或原工程 Git 历史。

## 更新仓库

```bash
git add .
git status
git commit -m "Describe the change"
git push
```

确认 `git status` 中没有密钥、`.env`、模型目录、`runtime/`或`demo-results/`。仓库可以保持私有；公开不是运行或更新的前提。
