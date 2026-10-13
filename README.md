# Reasoning-AI-demo-v0.1

[English README](README_EN.md)

这是 ERS 显式事件状态与规则组合思想的可运行推理 Demo：前端 LLM 将自然语言翻译为 Event Graph，语义相似提出候选，明确条件决定能否执行。后端用 Knowledge 证明可补全性质，由 Control 组织可复用 Rule 的绑定与连接，执行结果和失败反馈再影响下一步；最终只展示经过输出门批准的结果。

本项目采用 [Apache License 2.0](LICENSE)。

当前正式演示包含两个相互连接的领域：

- 财务：应收账款周转天数（DSO）计算。
- 库存与供应链：库存资金占用、缺货暴露、经营现金流压力。

前端只做语义规范化和 Event Graph 构建，不在前端做业务推理。此发布包只包含推理系统，不包含 Auto-Learn、训练或规则重构代码；每次演示都不会修改随仓库发布的 Rule / Knowledge 数据库。

## 技术报告

完整说明按“设计原因 → 数据结构 → 模块调用 → 代码实现 → 发布边界”组织，涵盖语义候选与执行许可、Knowledge/Memory 分工、Control 反馈、Rule 组合、Calculator 和输出接受：

- [技术报告（中文）](docs/TECH_REPORT.CHINESE.md)
- [Tech Report (English)](docs/TECH_REPORT.ENGLISH.md)
- [模块边界](docs/ARCHITECTURE.md)

文档术语与 ERS v10 理论部分对齐，但以本仓库的实际实现为准。论文里的动态中策略、完整子目标调用／返回和通用依赖失效维护没有随此 Demo 发布；现有 selector 委托和来源记录不能当作这些能力已完成的证据。论文、图稿、实验结果和成本对照不在此次文档更新中上传。

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

显式传入 `db_paths` 也不能指向发布种子，包括符号链接和硬链接。文字入口只使用 `semantic_v1` 与 `Qwen/Qwen3-8B`，旧模型路由已删除；图输入不需要API配置。最终输出必须有输出门批准、明确的当前Event和有效结果引用，缺失或冲突则返回未解决。非数值、布尔和无单位的判断保留Decoder的确定结论，不交给LLM自由改写。

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

真实 API 演示需要你自行提供密钥，并受网络、供应商可用性与模型下载影响。前端使用 `Qwen/Qwen3-8B` 构建 Event Graph；有单位的数值结果可以用同一模型改写表达，失败或破坏数值/单位时回退到Decoder草稿。非数值、布尔和无单位结论直接保留确定输出，不进行自由改写。典型单题耗时不能视为固定性能基准。

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
