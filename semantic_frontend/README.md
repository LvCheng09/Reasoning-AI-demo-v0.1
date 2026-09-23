# 独立语义前端 v1

前端和后端在同一 Python 进程中工作，通过字典传递数据。没有 HTTP 服务、端口或额外进程。
只有前端调用外部固定 LLM 时使用原来的 OpenAI-compatible API。

## 代码边界

- `frontend.py`：构图和局部细化的业务流程。
- `graph.py`：文字图数据结构与解码。
- `validation.py`：格式、引用与图结构校验。
- `prompts.py`、`examples.py`：Prompt 和业务输入示例。
- `llm_client.py`：固定 LLM 调用。
- `embedding.py`：单独使用前端时可选的向量化。
- `tests/frontend_demo/checks.py`：测试用模拟需求和字段覆盖评分。
- `tests/frontend_demo/runner.py`：手动实验命令行入口；样例和实验依赖清单也在该目录。
- `tools/run_semantic_frontend_demo.py`：仅保留旧命令兼容入口，不含实现。
- `adapter.py`：将文字图转换成现有后端的 spans/events 格式，并保留补图前的后端数据。
- `main_part/frontend_event_builder.py`：后端已有的两个接口定义。
- `main_part/frontend_semantic.py`：配置与实例组装，不包含语义或推理逻辑。
- `main_part/model_launcher.py`：使用适配器，随后调用已有向量化、图契约和 Scheduler。

前端包不导入 main_part。后端不需要知道 Prompt、模型名称或前端的内部类。
以后更换前端，只需实现：

```python
build_event_graph(game_input) -> dict
patch_event_graph(*, game_input, current_event_graph, patches) -> dict
```

返回值使用现有 `event_graph.v1` 数据布局：`spans`、`events`、`evidence_span_ids`、`relations`。
主线映射为工作区，小事件映射为结构化 span；父子层级保留在 `metadata.semantic_event`。
跨主线的明示关系保留端点引用。前端内部文档版本为 `semantic_frontend.v1`。

## 使用

配置不包含默认服务器地址或可用模型假设，必须填写实际服务信息：

```bash
export MAIN_PART_FRONTEND_IMPLEMENTATION=semantic_v1
export LLM_BASE_URL=http://YOUR_LLM_SERVER:8000/v1
export LLM_MODEL=YOUR_FIXED_MODEL
export LLM_API_KEY=YOUR_KEY
```

然后沿用原来的 `MainPartModelLauncher(...).submit_input(text)`。
未设置 implementation 时保留原前端，不静默切换现有实验；设置 semantic_v1 但未配置模型则明确报错。
也可以直接注入，不需要环境变量：

```python
from semantic_frontend import SemanticFrontendAdapter, SimpleSemanticFrontend, SingleSmallLLM
from main_part.model_launcher import MainPartModelLauncher

llm = SingleSmallLLM(base_url, model_name, api_key)
adapter = SemanticFrontendAdapter(SimpleSemanticFrontend(llm, embedder=None))
launcher = MainPartModelLauncher(frontend_adapter=adapter, db_paths=db_paths)
result = launcher.submit_input(text)
```

`embedder=None` 表示前端只返回文字图；真实向量化由后端现有 encoder 完成。
补图保留未变化 span 的向量；后端依据原文字段、模型名和版本复用向量，改变的字段重新编码。
不要在前后端同时加载两个 embedding 模型。

## 翻译范围

只规范化原文表达，保留否定、时间、条件、归属、可能性、数值原始表述。
禁止推导新结论、计算派生数值、用外部知识补事实、把“随后”改成因果。
这些要求由 Prompt 约束；没有额外语义审查模型，也不把格式通过当成理解正确。

格式校验检查 ID、字段类型、引用、主线归属、父子循环以及原文片段是否存在。
原文引用检查只是字符串检查，不是语义证明。

Prompt/Schema 同时用于初次构图和局部细化，约束以下内容：

- 明示的天数单独生成 days，计算目标使用 modality=request，不生成答案。
- 金额保留原值和原单位，例如 100万元仍是 value=100、unit=万元，不换算成 millions。
- 普通销售收入用 total_sales，不擅自解释为净赊销收入 sales。
- measured_in 从金额指向单位/币种；目标与输入之间不补公式依赖。
- 每项引用尽量短且足够的连续原文，关系必须有原文明示支持。
- confidence 表示提取把握，不是预测成真的概率；没有证据的边不输出。

Schema 使用类型、枚举和范围描述，不再提供容易被照抄的零置信度占位输出。
它作为提示词传入，不是服务端强制 JSON Schema，也不新增语义判定器。
格式检查支持 request；适配器不自动补天数、改单位或猜业务含义。
严格提示词不能保证真实 LLM 不出错，仍需观察真实输出。

初次构图和局部细化 Prompt 都包含 `examples.py` 的 DSO 输入样例：货币、金额量级、
期间、天数、目标、平均应收账款、净赊销收入及明示关系。字段名来自当前 DSO 输入图，
数值使用独立示例，没有提供计算答案或执行路径。
小事件的可选 `span` 字段携带 core/relation/nature/feature_type/span_role/value/unit，
适配器原样映射为后端 span，不在适配器里猜测业务含义或计算数值。
Prompt 明确禁止把例子的币种、期间、数值等复制到新原文中，未知信息必须缺省。
离线验证只证明样例字段能够传入后端；真实 LLM 是否能照样生成、是否匹配业务规则仍待实测。

局部细化请求优先提供 `target_span_id`，也接受明确的小事件 ID。
只给主线 ID 时，该主线必须有唯一根事件；目标不明确则返回 unresolved，不默认改第一条主线。
LLM 原文中找不到所需信息时允许空 patch + unresolved。补图先在副本上检查，失败不改原图。
v1 不提供跨主线移动、删除或重写已有边的功能；这些要后续显式扩展协议，不能偷偷修改。
后端推导结果不会传入 LLM Prompt，但会保留在合并后的后端图中。

## 当前验证边界

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -m pytest \
  tests/test_semantic_frontend.py tests/test_contracts.py tests/test_runtime.py tests/test_calculator.py
```

离线测试使用明确标注的人工 JSON 响应、向量形状夹具，测试真实构图/补图入口、数据保留和格式处理。
这些不是 LLM，不证明语义理解、真实 embedding 效果或业务推理成功。
真实 LLM 的质量和对现有业务 Rule 的适配仍需接入实际服务后验证。
不会因为普通自然语言被包装成 span 就声称它满足 DSO 等业务规则需要的所有细粒度字段。

原始两个样例迁移到 `tests/frontend_demo/sample_cases.jsonl`；已去掉
“随后下调计划必须存在因果边”的不合理要求。旧 demo 的 field_coverage 仅表示字段覆盖，不是语义准确率。

单独运行前端 demo 的入口已改为（不再使用 `python -m semantic_frontend.frontend`）：

```bash
python -m tests.frontend_demo.runner --output frontend_results.json
```

该工具沿用 LLM_BASE_URL、LLM_MODEL、LLM_API_KEY、EMBEDDING_MODEL、EMBEDDING_DEVICE 配置。
只调用前端 LLM 而不运行 demo 时，继续使用 `SimpleSemanticFrontend.initial_parse()` 和 `.refine()`；
这些接口与后端的 `build_event_graph()`／`patch_event_graph()` 均未改变。
业务模块不导入 tests 或 tools。业务文件中的 DSO Prompt 样例参与真实 LLM 构图，因此保留在 examples.py，
不与用于批量实验的 sample_cases.jsonl 混放。旧的 tools 命令仍可用，只转发到新的测试入口。
