# 调度、文档定位、数据分析、答复

这份计划只交给另一位实现者。实现这四个代理，不要再加第五个，也不要改记忆代理的职责。

四个代理要拆开的是现在这条文档问答：定位、计算、成稿挤在同一个模型、同一份工具列表里，最多跑 15 轮。拆开之后，每个代理只碰自己的材料，主程序决定谁上场。代理之间不互相调用。

模型不按角色分配。四个代理和主对话、记忆代理走同一条 `chat_with_fallback`。本文不指定模型，也不要从 `README.md` 第 6 节抄模型名。

## 主程序里已经定下来的事实

仓库在 wsl：`/home/r/tencent-docs-web`。正在跑的服务在 tx主机，代码目录 `/home/ubuntu/tencent-docs-web`，Gradio 监听 `0.0.0.0:7860`，根路径 `/docs-chat`。腾讯文档 MCP 不在本仓库，在 tx主机的 `/home/ubuntu/tencent-docs-mcp/server.py`。本计划不改 MCP。

一轮用户消息的现有顺序：

1. `custom_ui.bot_response` 取出最后一条用户文字，调用 `chat_interface(message, history)`。界面占位文案和文件树保持不动。
2. `process_chat` 把历史收成 `{role, content}`，只保留 user / assistant。
3. 记忆代理先跑：`memory_agent.run(用户原话, 此前最多 6 轮)`。
   - `handled=True`：这句话到此结束，四个代理不要启动。
   - `handled=False`：`reply` 里若有文字，是记忆确认（规则写上了，但这句话还要查文档）。先存着，等最终回复由现有的 `deliver()` 拼到前面。四个代理不要自己再拼这一句。
4. 今天放行之后，是一个模型看见全部 MCP 工具，循环最多 15 次，用尽再转交宿主机的 `codebuddy --print`。这一段由四个代理替换掉，不要留成第二条并行链路。

记忆文件只由记忆代理写。路径规则已经在 `memory_agent.default_memory_path()`。四个代理只读 `format_memory_prompt()` 的那段文字，不打开文件，不调用 `update_memory_rule`。

SQLite 仍是一轮一行：`log_to_db(用户原话, 最终回复)`。中间计划、工具原文、代码不入库。库路径维持 `/home/ubuntu/tencent-docs-web/chat_history.db`。

`llm_client.chat_with_fallback(messages, tools=None, tool_choice=None)` 保持这一个入口。四个代理需要工具时，把 `tools` 和 `tool_choice` 传进去，和记忆代理一样：先强制指定自己的决策工具，这次调用失败再去掉 `tool_choice` 重试一次。不要新建客户端，不要一轮里把同一份表格发给两家供应商。

Gradio 并发仍是 2。一次 `process_chat` 只拉起一个 MCP stdio 会话，定位和数据分析共用这次会话。会话对象只放在主程序里，按需要传给这两个代理。调度和答复拿不到会话，因此没有调用工具的路径。

## 一轮里谁上场

主程序是唯一的调度执行者。调度代理只交出一份计划，不负责去调用另外三个。

```
用户原话
  → 记忆代理（已有）
       handled → 返回它的 reply，结束
       否则记下 memory_note
  → 调度代理（无工具会话，一次模型）
       chat / clarify → 用它的 reply 作为正文，deliver，结束
       lookup         → 文档定位 → 答复
       analyze        → 文档定位 → 数据分析 → 答复
  → deliver(正文)    # 有 memory_note 就加在正文前
```

定位结果若是找不到、或不止一份，主程序用固定句子问用户，不再调用数据分析和答复。固定句子见下文「失败时谁对用户说话」。

任一代理抛异常：记日志，用该代理的固定失败句结束本轮。不要吞掉异常后再掉进旧的 15 轮循环，也不要转交 CodeBuddy。

## 代理之间交什么

用普通数据结构交接，可以 `asdict` 打进日志。不要把模型的 message 对象、MCP 会话、整表二维数组传给下一个代理。

调度计划：

| 字段 | 含义 |
| --- | --- |
| `intent` | `chat`、`clarify`、`lookup`、`analyze` 之一 |
| `reply` | 只有 `chat` / `clarify` 会展示给用户 |
| `question` | 去掉寒暄后的问题，保留用户的用词 |
| `doc_hint` | 用户用来指文档的那一段原话。不要在这里换成记忆里的正式名称 |
| `sheet_hint` | 用户用来指子表的原话，没有就是空字符串 |
| `calc_goal` | `analyze` 时要算的那件事。`lookup` 必须是空字符串 |

定位结果：

| 字段 | 含义 |
| --- | --- |
| `status` | `found`、`ambiguous`、`not_found`、`error` |
| `doc_title` | 文档列表里的完整标题，逐字 |
| `file_id` | 列表返回的 id，仅主程序和定位使用 |
| `sheet_title` / `sheet_id` | 选定的那一张子表 |
| `columns` | 表头 |
| `read_range` | 实际读过的 A1 范围 |
| `sheet_row_count` | `list_sheets` 给出的行数；接口没有就空着 |
| `rows` | 表头之外的样例。统计路径最多 3 行，查阅路径最多 20 行 |
| `truncated` | 还有更多行没有放进 `rows` |
| `candidates` | 不止一份时，候选的完整标题或子表名 |
| `note` | 给答复或固定句子用的短说明，例如截断 |

数据结果：

| 字段 | 含义 |
| --- | --- |
| `status` | `ok` 或 `error` |
| `doc_title` / `sheet_title` | 与定位结果逐字相同 |
| `code` | 最后一次真正执行的代码 |
| `output` | 工具打印，主程序截到 4000 字 |
| `attempts` | 已消耗的次数，含被拒绝的代码 |

答复只返回一段用户能看的字符串。

## 调度

负责判断这轮是闲聊、追问、查表，还是要统计，并写出上面的计划。不查文档，不算数，不写给用户的查表结论。

模型只看到：

- 一段系统说明：你是调度。必须调用 `schedule_decision`。闲聊才填 `reply`。查表问题不要在 `reply` 里回答。`doc_hint` 保持用户原话。别名对照只供理解，不要改写进 `doc_hint`。
- `format_memory_prompt()` 的原文。
- 最近 6 轮 user / assistant，每条正文截到 200 字，与记忆代理的历史裁剪同一尺度。
- 用户本轮原话。

看不到 MCP 工具列表，看不到表格。

决策工具 `schedule_decision` 的参数就是计划里的六个字段，`intent` 与 `question` 必填。

主程序收到结果后按下面改写，再决定叫谁。改写是代码，不是再问一次模型：

- `intent` 不在四种之内：同一上下文重试一次。仍不对，本轮用调度的失败句结束。
- `calc_goal` 去掉空白后非空：按 `analyze` 执行。模型如果在 `reply` 里已经写了数字或结论，丢掉，不展示。
- `intent` 是 `analyze` 但 `calc_goal` 为空：改成 `lookup`。
- `intent` 是 `chat` 或 `clarify`：不调用定位、分析、答复。`reply` 为空就重试一次，仍空则用失败句。
- `intent` 是 `lookup` 或 `analyze`：丢掉 `reply`，避免调度抢在答复之前把结论说死。

这样闲聊不会惊动文档接口，查表也不会被调度用一句话带过。

## 文档定位

负责把 `doc_hint` / `sheet_hint` 收成唯一的一份文档、一张子表、表头和少量行。不写 pandas，不组织最终答复。

别名只在这里落地。记忆说「封边条」等于「封边条250424」时，用来和文档标题比对的是正式名称。`doc_hint` 仍保留用户原话，方便对不上时原样问回去。两个字段都参与比对：正式名称优先；正式名称对上不止一份，再退回用户原话。不要对用户整句做盲目替换。

工具白名单只有这四个，由主程序在调用前截断，不靠提示词：

- `list_docs`
- `list_sheets`
- `read_sheet`
- `search_and_read_sheet`

调用形状以 tx主机上的 MCP 为准：

- `list_docs(folder_id="/", limit=100, file_type="sheet", is_owner=0)`。定位先由主程序调一次，不交给模型决定页大小。接口没有偏移，就这一页，最多 100 份表格。
- `list_sheets(file_id)`
- `read_sheet(file_id, sheet_id, cell_range)`。限制是行 ≤ 1000、列 ≤ 200、单元格 ≤ 10000。
- `search_and_read_sheet(doc_title, sheet_name="", max_rows=200, max_cols=30)`。实现时 `max_rows` 必须改小，见下。这个工具用「标题包含即命中列表中的第一个」，子表对不上就默默读第一张。定位不要靠这个默认。

其余 MCP 工具一律拒绝，包括 `analyze_sheet_pandas`、`update_memory_rule`、`write_sheet`、`create_doc`、`get_doc_content`、`export_doc`、`get_export_progress`、`get_auth_url`、`exchange_code`、`get_user_info`。拒绝时回给模型的工具结果是固定一句「这个工具不在文档定位的范围内」，并且计入下面的次数。`get_doc_content` 和导出每天只有很少次数，四个代理都不许碰。用户问的是在线文档正文而不是表格时，调度用 `clarify`，`reply` 说明目前只查表格。不要为了正文去导出。

定位的步骤：

1. 主程序已经拿到最多 100 条表格的 `id` 和 `title`。模型只看见标题列表、用户问题、两个 hint、记忆对照。必须调用 `locate_decision`。
2. `locate_decision` 的字段：`status`（`found` / `ambiguous` / `not_found`）、`doc_title`、`sheet_hint`、`candidates`（字符串数组）。
3. 主程序验收，不信任模型的 `status`：
   - `doc_title` 必须与列表中某一条完整标题逐字相等，否则当作 `not_found`。
   - 用正式名称和用户原话分别做包含比对。比对结果多于一条，改为 `ambiguous`，`candidates` 用那些完整标题。禁止取列表里的第一个。
   - 零条是 `not_found`。
4. 只有唯一文档才 `list_sheets`。子表同样要求唯一：`sheet_hint` 为空且只有一张子表，用那一张；为空且有多张，`ambiguous`，候选是子表名；有 hint 但匹配数不是 1，同样 `ambiguous` 或 `not_found`。禁止落到「第一张」。
5. 唯一子表之后才读单元格。统计路径 `max_rows=4`（表头加 3 行）。查阅路径 `max_rows=21`（表头加 20 行）。`max_cols=30`。优先 `search_and_read_sheet(完整标题, 完整子表名, 上述行数, 30)`。返回的 `data` 在写入定位结果前就裁剪，模型的下一轮看不到全量。
6. 这一步最多再给模型 1 次修正机会（例如子表名几乎匹配但差一个字）。整个定位模型回合连同拒绝的工具，上限 4。到顶还没唯一命中，就按当前最接近的状态结束，不再加轮次。

`search_and_read_sheet` 内部仍会按自己的默认去读，主程序只把裁剪后的结构交给后面的代理。文件 id、原始二维数组留在主程序的这一轮局部变量里，函数返回后丢掉，不写入磁盘。

## 数据分析

负责对已经定位的那一张表做统计。不重新搜索文档，不改表，不把样例行心算成答案。

只在调度 `intent=analyze` 且定位 `status=found` 时启动。

模型只看到：用户问题、`calc_goal`、完整文档标题、完整子表名、列名、最多 3 行样例、此前几次的报错（如果有）。看不到聊天记录，看不到查阅路径那种 20 行，看不到文件 id。

白名单只有 `analyze_sheet_pandas(doc_title, python_code, sheet_name)`。`doc_title` 和 `sheet_name` 由主程序改成定位结果里的完整标题，忽略模型自己填的别的名字。这样即使用户说的是简称，工具拿到的也是已经唯一确定的标题，避免宏工具再次「包含即第一个」。

次数上限 3，含下面被拒绝的代码。

现有工具在 MCP 进程里对 `python_code` 做 `exec`，变量是 `df` 和 `pd`，没有沙盒。本计划不改 MCP。主程序在 `call_tool` 之前拒绝这段代码，拒绝也占 1 次，并把原因作为工具结果还给模型重写：

- 必须含有 `print`。
- 整段出现下列任一内容就拒绝：`import`、`__`、`open(`、`exec(`、`eval(`、`compile(`、`globals(`、`locals(`、`getattr(`、`setattr(`、`input(`、`breakpoint(`、`os`、`sys`、`subprocess`、`socket`、`pathlib`、`shutil`、`requests`、`pickle`、`ctypes`。
- 拒绝句只说明违反了哪一条，不把其余表格数据附带回去。

工具返回的 `code_output` 截到 4000 字放入数据结果。`error` 键存在，或输出以「执行代码出错」开头，都算这一次失败。3 次都失败：`status=error`，把最后一次输出交给答复，由答复说明没算出来。不要让答复根据那 3 行样例补一个数。

## 答复

负责把已经拿到的材料写成给用户的一段话。不调用工具，不发起新的读取或计算。

只在定位 `found` 之后启动。找不到或不止一份时不启动。

模型只看到：

- 系统说明：只用材料里有的事实。材料没给出的数字不要写。文档标题和子表名与材料逐字相同。不要描述代理、工具或代码。记忆确认句不要写，主程序会加。
- 用户原话和 `question`。
- 定位的标题、子表、列名、`read_range`、`truncated`、`note`。
- 查阅路径：最多 20 行。统计路径：不给这 20 行，只给数据结果的 `output`。统计失败时给 `status=error` 和那段错误输出。

一次模型调用，不传 `tools`。若返回里仍带工具调用，忽略工具，只用文字；文字为空就用失败句。

`truncated=True` 时，答复要说明展示的是前若干行，不是全表。统计结果里已经有总数的，用工具输出里的总数，不要用样例行数代替。

## 主程序怎么接

改动集中在 `process_chat` 里记忆代理放行之后的那一段。`chat_interface` 的签名、Gradio 组件、`custom_ui.py` 的文件树不改。

建议新文件，都在仓库根目录，和 `memory_agent.py` 并列：

| 文件 | 内容 |
| --- | --- |
| `agent_types.py` | 上面三个结构、白名单常量、代码拒绝函数、裁剪函数 |
| `scheduler_agent.py` | `run(user_text, prior_turns, memory_text)` |
| `doc_locator_agent.py` | `run(session, plan, memory_text)` |
| `data_analyst_agent.py` | `run(session, plan, located)` |
| `answer_agent.py` | `run(user_text, plan, located, analysis)` |
| `test_subagents.py` | 注入假的 `chat`，不联网 |

`app.py` 在记忆代理之后的伪代码：

```
memory_text = format_memory_prompt()
plan = await scheduler.run(clean_user_input, prior[-6:], memory_text)
if plan.intent in {"chat", "clarify"}:
    return deliver(plan.reply)

async with 同一个 MCP 会话:
    located = await locator.run(session, plan, memory_text)
    if located.status != "found":
        return deliver(固定句子(located))
    analysis = None
    if plan.intent == "analyze":
        analysis = await analyst.run(session, plan, located)
    text = await answer.run(clean_user_input, plan, located, analysis)
    return deliver(text)
```

`deliver()` 继续负责两件事：memory_note 不在正文里才加到前面；然后 `log_to_db`。代理返回的字符串里不要预写 memory_note。

工具调用的唯一入口放在主程序（或 `agent_types` 里的一个函数），定位和分析都走它：

- 名字不在该代理白名单：不 `call_tool`，回固定拒绝句，次数加一。
- 名字在 `MEMORY_TOOL_NAMES`：同样拒绝。这条和今天 `process_chat` 里的拒绝并存到旧循环被删掉为止；删掉旧循环后仍保留这一处。
- 参数不是合法 JSON：当作这一次工具失败，不抛到进程外。

删掉的现有行为：`max_turns = 15` 的全能工具循环，以及随后的 `codebuddy --print`。宿主机上的 CodeBuddy 不是这四个代理的兜底。不要用环境变量把旧循环留在默认路径后面。

侧边栏文件树继续由 `custom_ui.fetch_file_tree` 直接拉。不要把目录交给调度当第二份文档列表，避免和 `list_docs` 的 100 条结果不一致。

## 失败时谁对用户说话

| 情况 | 谁说话 | 其余代理 |
| --- | --- | --- |
| 记忆 `handled` | 记忆代理原文 | 四个都不跑 |
| 记忆写上了规则且还要查表 | 最终正文前加记忆那句 | 调度照常看用户原话 |
| 调度模型两次都没有合法计划 | 固定句：这轮没有分清是闲聊还是查表，没有查文档。 | 不打开 MCP |
| 闲聊、追问 | 调度的 `reply` | 不打开 MCP |
| 100 份表格里没有 | 固定句，带上用户的 `doc_hint` | 不读单元格，不计算，不叫答复 |
| 多份文档或多张子表 | 固定句列出 `candidates`，问要哪一份 | 同上 |
| 定位的模型或 MCP 报错 | 固定句：文档列表没有取到，这轮没有查表。 | 不计算，不叫答复 |
| 代码被拒绝或 pandas 报错未到 3 次 | 不直接对用户说 | 把短原因还给数据分析重写 |
| 3 次仍失败 | 答复，说明没算出来，可附最后一段错误 | 答复不得补数字 |
| 答复空文本 | 固定句：已经定位到《标题》的「子表」，但没有组织出回答。 | 统计路径再附上截断后的 `output` |
| 任一代理未捕获异常 | 该代理一句固定失败句 | 不转交旧循环，不转交 CodeBuddy |

固定句用代码拼接标题和候选，不再调用模型，避免失败路径上编出一份不存在的文档。

## 和主程序对齐的几件具体事

记忆代理的强提示、弱提示、`continue_chat`、损坏文件不覆盖，都保持原样。四个代理不要再判断「这句是不是在记规则」。

用户说「记住 A 就是 B，另外 B 的合计是多少」：记忆代理 `handled=False` 且 `reply` 为确认句；调度的 `doc_hint` 仍是用户说的 A 或 B；定位用记忆把 A 收成 B 的正式标题；确认句只由 `deliver()` 加一次。

同一轮的 MCP 连接失败：定位返回 `error`，走定位失败句。不要在失败后再开第二个 MCP 进程重试整条链。

`analyze_sheet_pandas` 内部会再次搜索并读取最多 10000 行。这是 MCP 的现有行为，数据留在 MCP 进程里。数据分析代理只接收打印文本。不要为了「少读一次」去改 MCP，也不要把那 10000 行拉回主程序塞进模型。

查阅路径展示上限 20 行，统计路径样例上限 3 行。用户要「全部列出来」时，答复说明只展示了前 20 行。不要把 `search_and_read_sheet` 的默认 200 行放进答复的上下文。

日志用一行写清代理名、`intent`、定位状态、工具名和次数。表格正文不要打在 INFO 里。

测试按 `test_memory_agent.py` 的做法，注入假 `chat`，wsl 上不需要装在线密钥也能跑。至少覆盖：

- 无记忆提示的普通查表句会进入调度；记忆 `handled` 时调度函数不被调用。
- `calc_goal` 非空时忽略调度自己写的答复。
- `lookup` 不调用数据分析。
- 两个标题都包含 hint 时不调用 `search_and_read_sheet`，候选进固定句。
- 定位发出 `analyze_sheet_pandas` 或 `write_sheet` 时，MCP 的 `call_tool` 未被调用。
- 数据分析发出另一个 `doc_title` 时，真正提交的是定位给出的完整标题。
- 含 `import` 或没有 `print` 的代码不提交，并占用一次。
- 统计失败后的答复材料里没有样例行，只有错误输出。
- `memory_note` 只出现一次，且在最前。

## 不要做的事

- 不实现写入代理。MCP 虽已有 `write_sheet` 和 `create_doc`，这四个代理不调用。要写入时另开计划，并先经过界面确认。
- 不按 MCP 工具各做一个代理。定位可以碰四个只读工具，分析只碰 pandas 那一个。
- 不做文件树代理，不做单元格向量检索，不做第二个模型对话来验算。验算如果以后要做，是同一段 pandas 再跑一条断言，不是新代理。
- 不改记忆代理的对外结果：`handled`、`updated`、`reply`。
- 不给四个代理分别指定模型或供应商。阿里云若要接入，只作为 `chat_with_fallback` 在腾讯云这条失败之后的下一家，本计划不要求现在接上。
- 不部署。部署仍是人在 wsl 里对 tx主机执行现有的 `dpcd`，不在本计划内。
- 不改 `README.md`。那里关于单一模型循环和 CodeBuddy 兜底的描述，以本文替换进 `process_chat` 之后的行为为准。
