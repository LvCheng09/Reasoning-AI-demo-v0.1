# 技术原理：显式事件图推理系统

这份文档说明 Demo v0.1 的推理部分如何工作、为什么这样设计，以及它现在明确不做什么。安装和运行方式请看仓库根目录的 [README](../README.md)。

## 1. 要解决的核心问题

自然语言的表达很多样，同一个事实可以有不同说法；但一个可验证的问题，最终仍要依赖明确的条件、关系和计算过程。系统因此把任务拆成两层：

```text
语言理解：原文 -> 规范 Event Graph
显式推理：Event Graph -> Knowledge / Rule Path -> 结果
```

前端 LLM 负责第一层：理解不同表达、识别事实、整理结构。后端不把它的自然语言输出直接当答案，而是在图上严格匹配 Knowledge 和 Rule，逐步生成结论。

这不是“用规则取代 LLM”。两者处理不同问题：LLM 更适合把语言中的共识表达变为统一表示；规则更适合在条件成立时执行确定、可复查的推理。

## 2. 从语言到 feature

一个事物可以由一个或多个 feature 描述。描述越详细，feature 越多；语言中的名称则是人们对一组 feature 的高度抽象。例如“屏幕”可以只表示一个对象，也可以有“黑色”“发光”“显示画面”等更具体的性质。

人们通常用名称交流，但实际推理依赖这些更明确的性质与关系：某个 feature 是否出现、多个 feature 是否同时满足、一个性质能否推出另一个性质。Event Graph 的目的，就是把语言中的抽象名称和具体 feature 都保留下来，成为后续可检查的状态。

在本系统中，一个 span 的主要语义拆成三部分：

```text
core      ：是什么，例如 accounts_receivable、sales、inventory
relation  ：以什么关系出现，例如 temporal_role、measurement、scope
nature    ：该关系的性质，例如 beginning、annual、net_credit
```

“期初应收账款”可写作：

```text
core = accounts_receivable
relation = temporal_role
nature = beginning
```

同一 core 在不同 relation/nature 下可以代表不同含义。因此，系统不依赖简单的关键词相等，也不把所有相似文字都当成同一个概念。

## 3. 向量、文字与 LLM 的角色

文字字段用于调试、日志和面向用户的解释；core、relation、nature 的向量是系统进行语义检索和严格匹配的主要依据。当前 Demo 使用 Qwen embedding 将规范语义映射为向量。

Embedding 不是“事实正确性证明”，而是让不同文字表达能映射到同一个可比较的语义空间。真正允许一个条件进入推理，仍需满足 Rule 的结构、关系、单位、时间、实体和阈值约束。

前端 LLM 更像是现实语言的 encoder：它通过 Prompt 被要求提取何种信息、以何种粒度构图、区分明确事实和未明确事实。不同 Prompt 会影响它观察与提取信息的方式。后端发现原文信息确实不足时，可以构造定向补充请求，让前端只补某块文本事实；前端仍不做业务推理。

本 Demo 仅验证文本入口。相同的“先编码为结构、再显式推理”的边界也可以扩展到视觉等输入，但这不属于当前发布内容。

## 4. Event Graph 与 Contract

前端输出是一个 Event Graph，其中包含 span、span 间关系、来源证据、数值、单位、实体、时间和置信度等信息。后端先运行 Event Graph Contract，检查：

- JSON 与必填字段是否符合结构；
- ID、关系端点和 evidence 引用是否有效；
- 置信度、列表和对象字段是否可规范化；
- 向量字段是否符合约定。

Contract 只做结构和来源检查，不再额外假装判断 LLM 的“语义理解是否正确”。是否可以作为推理条件，由后续 Knowledge 和 Matcher 严格判断；前端不得擅自补造事实的要求则写在 Prompt 中。

## 5. Knowledge：证明缺失条件是否已经具备

Knowledge 保存的是概念内部及局部结构的定义，即 core、relation、nature 和相关 feature 如何组成某个可识别的 span。它不保存“从输入直接得到业务结论”的规则。

当某条 Rule 的 `require` 不满足时，系统不会降低匹配标准。Control 可以根据缺失 slot 查询 Knowledge：

```text
当前图中是否已经存在该定义要求的全部前提？
```

若能证明，系统添加带 provenance 的规范化 span 或 relation，并对原 Rule **重新严格匹配**；若不能证明，就拒绝这条 Rule，换路径或请求前端补充。

Knowledge schema 可以声明不同的激活方式：

- **单 core**：该 core 本身足以证明某种性质；
- **多个 core**：必须全部出现，才可补充性质；
- **relation context**：同一 core 需要由 relation 和邻接 core 一起确定含义。

因此，看到 `sales` 不会自动补成 `total_sales`；只有 `total_sales` 的定义前提在当前图中齐全时，才允许补全。这是“证明后补齐”，不是“相似就放行”。

## 6. Rule：从已成立条件到新结果

Rule 是 Knowledge span 之间的桥梁，其基本形式为：

```text
require  ->  produced
```

- `require`：需要哪些 span、关系、单位、期间、实体或前序结果；
- `produced`：条件满足后创建什么 span/关系、调用什么计算、进入哪条后续路径或生成输出候选。

例如 DSO 的局部路径可以是：

```text
期初 AR + 期末 AR -> 平均 AR
平均 AR + 净赊销 + 期间天数 -> DSO
DSO + 单位/证据检查 -> 输出候选
```

复杂问题由多条小 Rule 串起来。这样每一步都能看到输入、输出和依据，而不是把所有逻辑塞进一条难以检查的巨型规则。

## 7. Control、Matcher、Executor 的边界

```text
Control  ：决定当前目标、缺口和下一步策略
Matcher  ：严格判断一条 Rule 的 require 是否真的成立
Executor ：执行已经严格匹配的 produced，并把结果写回图
Calculator：完成 Rule 声明的确定性数值计算
```

Control 类似解决问题时的大方向：先找关键条件，再选择可行路径；没有路径时分析缺什么信息、能否用 Knowledge 证明、是否需要换路。它不直接替代领域计算。

Matcher 检查每个 required slot 的绑定、多个 slot 是否错误复用同一 span、关系端点、向量契约、阈值以及单位/时间/实体等限制。失败 Rule 会留下结构化缺口，但绝不会因“差不多”而被当作成功。

Executor 不解释业务含义，也不猜测下一步。它只把已命中的 Rule 变成 Event Graph 更新、计算调用、后续 selector 或输出候选。这样 Rule 的业务变化不会迫使 Executor 理解每个领域。

Calculator 与 Executor 分开，是为了让简单计算、未来可能加入的受控小模型和图写入逻辑各自独立。当前 Demo 的结论依赖确定性、可审计的计算；它不把一个通用神经网络当成业务推理的替代品。

## 8. Memory 的位置

Knowledge、Rule、Memory 保存三种不同层次的信息：

```text
Knowledge：常见概念及其局部 semantic structure
Rule     ：可复用的状态变换
Memory   ：某次具体问题中采用过什么路径、结果如何
```

历史 Memory 可以让 Control 优先尝试相似的成功路径，但不能跳过 Matcher。当前 Event Graph 必须重新严格匹配；路径失败时，系统回到普通候选路径。这避免把过去的个例直接套用到条件不同的新任务。

Demo v0.1 不随仓库发布历史 Memory，也不写入学习结果。

## 9. 多步运行与输出门

每条 Rule 的 `produced` 会写回 Working Memory/Event Graph，成为下一轮 Rule 的输入：

```text
Control -> Match -> Execute -> 更新 Event Graph -> Control
```

这让系统能够先得到中间结果，再用中间结果继续推理。最终结果也不会立即输出，而是先成为 output candidate。Output Gate 检查当前控制策略的完成条件、结果粒度、路径和必要约束；未通过的候选会被修订或阻断。

只有被放行的结果会交给 Output Decoder。Decoder 读取最终 result span 和支持事实；之后的 LLM Verbalizer 只将这份小型结构化结果改写为文字，不能访问 Knowledge/Rule 数据库或重新推理。若文字层改变了数值或单位，则退回确定性草稿。

## 10. 一个 DSO 示例

```text
输入：平均应收账款 120，净赊销额 960，全年 360 天，求 DSO。

1. 前端 LLM 构建带证据的 AR、net credit sales、days span。
2. Contract 检查图和证据结构。
3. Control 识别 DSO 目标并选择财务策略。
4. Matcher 严格绑定三个输入条件。
5. Calculator 执行：120 / 960 × 360 = 45。
6. Executor 写入 dso_result = 45 days，保留输入和 Rule 路径。
7. Output Gate 放行后，输出层生成用户可读回答。
```

若缺少净赊销额，普通销售额不会被默认当作净赊销额。系统只有在 Knowledge 能用当前图证明其 scope 时才补齐；否则换路、请求补充或返回缺条件。

## 11. 当前能力与边界

已接入并有工程测试支持的能力包括 Event Graph Contract、Knowledge 驱动补全后的严格重匹配、Rule 的逐步执行、Calculator、输出门与独立的文字输出层。DSO 和库存资金占用均已完成真实 API 的端到端演示。

仍需保持边界：

- 它是两个领域的研究 Demo，不是无限领域的通用问答产品；
- LLM 的前端翻译仍可能出错，Contract 不保证自然语言理解的绝对正确；
- 未覆盖的 Knowledge/Rule 不能靠放宽匹配解决；
- Control 能逐步组合和换路，但当前不能宣称已实现全局最优的多步规划；
- Auto-Learn、训练、规则重构与 PRS 编译器不在本发布版的范围内。

## 12. 代码定位

| 部分 | 实现 |
| --- | --- |
| 前端与主入口 | `main_part/model_launcher.py`、`semantic_frontend/` |
| Event Graph Contract | `main_part/event_graph_contract.py` |
| Scheduler、Control、Output Gate | `main_part/scheduler_new.py` |
| Knowledge 结构补全 | `main_part/semantic_alignment.py`、`main_part/retrieval.py` |
| 严格匹配 | `main_part/matcher/matching.py` |
| Rule 路径 | `main_part/rule_selector.py`、`main_part/rule_path_candidates.py` |
| 执行与计算 | `main_part/executor_new.py`、`main_part/calculator/` |
| 结果解码与文字输出 | `main_part/output_service.py`、`main_part/output_decoder.py`、`main_part/answer_verbalizer.py` |

更短的模块边界说明见 [ARCHITECTURE.md](ARCHITECTURE.md)。
