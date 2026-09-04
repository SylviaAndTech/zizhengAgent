"""
AI 助手对话代理：用 LangChain 的 create_agent（内部基于 LangGraph）驱动工具调用循环，
把原来"案例工作台"里手动点按钮做的事情，变成对话里说一句话就能做，
也能在聊天里直接对素材库做语义检索（不用精确知道case_code/素材ID）。

对外的会话存储格式保持不变（还是 role/content 的简单结构，content可以是字符串，
也可以是 [{"type":"text"/"tool_use"/"tool_result", ...}] 这种block列表）——
这样不管内部用什么agent框架，前端的聊天渲染逻辑完全不用跟着改。
"""
import json
import queue
import threading

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from db import (
    SessionLocal, RawMaterial, Case, CaseKnowledgeMapping, BackgroundJob, DIMENSIONS,
)
from job_queue import submit_generate_job, submit_enrich_job
from material_index import search_materials as _search_materials_impl
from qwen_client import get_langchain_llm, require_api_key, MAX_TOOL_ROUNDS
from audit import log_case_change

SYSTEM_PROMPT = f"""你是"思政案例生成工作台"的AI助手，帮助用户完成案例草稿的生成与编辑，
也可以帮用户在素材库里检索相关素材。

严格规则：
1. 生成或编辑案例正文时，只能使用素材库里已经抓取/上传成功的原始材料里的事实，不能编造任何时间、地点、人物、数据。
2. 叙事表达必须原创改写，不能照抄原文整段。
3. 编辑已有案例前，先用 get_case 工具看当前完整内容，只改用户要求的部分，其余字段原样保留（update_case 只需要传你要修改的字段）。
4. 修改内容后必须调用 update_case 把结果写回数据库，不要只在聊天里描述而不落库。
5. 如果用户是在描述内容而不是给出具体案例编号/素材ID（比如"有没有关于XX的素材"），用 search_materials 做语义检索，别瞎猜material_id。
6. 如果用户要求"用已采纳的知识点补充案例的适用课程举例/教学设计"，直接调用 enrich_case_with_accepted_knowledge 工具（不要自己瞎编内容）。
7. generate_case_draft 和 enrich_case_with_accepted_knowledge 这两个工具是"提交后台任务"，
   调用后立刻返回一个job_id，任务还在后台跑，**不代表已经做完了**。这时要如实告诉用户
   "已经提交、大概要多久、页面顶部有进度条"，绝对不能说成"已经生成好了"，也不要编造进度。
   用户之后问"生成得怎么样了/好了吗"，用 check_job 工具查真实状态再回答，不要凭感觉猜。
8. 回复用简短的中文说明你做了什么、案例现在是什么状态，不要大段罗列JSON。

可用的思政维度：{", ".join(DIMENSIONS)}
"""


@tool
def list_materials(case_code: str) -> str:
    """查询指定案例编号下、已导入素材库的原始素材列表（含抓取状态与正文预览）"""
    db = SessionLocal()
    try:
        materials = db.query(RawMaterial).filter(RawMaterial.case_code == case_code).all()
        result = [
            {"id": m.id, "title": m.source_title or m.url, "status": m.fetch_status, "preview": (m.fetched_text or "")[:150]}
            for m in materials
        ]
        return json.dumps({"materials": result}, ensure_ascii=False)
    finally:
        db.close()


@tool
def search_materials(query: str, case_code: str = "") -> str:
    """按自然语言语义检索素材库里的相关内容，适合用户没给出具体素材ID、只描述了个大概内容的场景。
    case_code留空则搜全部素材库，填了就只搜这个案例编号下的素材"""
    try:
        hits = _search_materials_impl(query, top_k=5, case_code=case_code or None)
        return json.dumps({"hits": hits}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def list_cases(case_code: str = "") -> str:
    """查询案例列表，可按案例编号过滤，返回每个案例的id/标题/状态"""
    db = SessionLocal()
    try:
        query = db.query(Case)
        if case_code:
            query = query.filter(Case.case_code == case_code)
        cases = query.order_by(Case.id.desc()).all()
        result = [{"id": c.id, "case_code": c.case_code, "title": c.title, "status": c.status} for c in cases]
        return json.dumps({"cases": result}, ensure_ascii=False)
    finally:
        db.close()


@tool
def get_case(case_id: int) -> str:
    """获取某个案例的完整内容（七段式全部字段），用于编辑前先看清楚现状"""
    db = SessionLocal()
    try:
        case = db.query(Case).filter(Case.id == case_id).first()
        if not case:
            return json.dumps({"error": "案例不存在"}, ensure_ascii=False)
        return json.dumps(case.to_dict(), ensure_ascii=False)
    finally:
        db.close()


@tool
def generate_case_draft(case_code: str, material_ids: list[int]) -> str:
    """用指定的素材ID列表，提交一个"生成七段式案例草稿"的后台任务。
    不需要指定思政维度——每个案例都要求对五个官方思政维度（政治认同/家国情怀/文化素养/
    宪法法治意识/道德修养）逐一给出摘录表述+解释，案例所属的书稿章节分类（也是这五个维度之一）
    由模型结合案例内容自主判断给出。

    **这个工具只负责提交任务，立刻就返回，返回时案例还没生成好**——正文要走写作→评审→修订
    的多轮循环，通常5-20分钟。返回的job_id可以之后用 check_job 工具查进度。回复用户时要说
    "已经提交、大概5-20分钟、页面顶部有进度条、完成后案例库里会出现"，不能说成已经生成完了。

    如果这个case_code在案例库里已经生成过（不管用的是不是同一批素材），不会覆盖已有案例，
    而是自动落库成 case_code-1、case_code-2……这样的新版本编号，方便同一个编号反复试生成、
    审核时再挑一个采纳。具体用了哪个版本号要等生成完成后才能确定，不要提前告诉用户。"""
    db = SessionLocal()
    try:
        materials = db.query(RawMaterial).filter(RawMaterial.id.in_(material_ids)).all()
        success_materials = [m for m in materials if m.fetch_status == "success"]
        if not success_materials:
            return json.dumps({"error": "所选素材均不可用（未抓取/解析成功），无法生成"}, ensure_ascii=False)
        job_id = submit_generate_job(case_code, [m.id for m in success_materials])
        return json.dumps({
            "job_id": job_id,
            "submitted": True,
            "message": f"案例{case_code}的生成任务已提交（任务号{job_id}），预计5-20分钟。"
                       f"任务在后台跑，页面顶部会显示进度，也可以随时问我进度。",
        }, ensure_ascii=False)
    finally:
        db.close()


@tool
def check_job(job_id: int) -> str:
    """查询一个后台任务（案例生成/知识点匹配/知识点补充）的真实进度。
    用户问"生成得怎么样了""好了吗""还要多久"这类问题时用这个工具查了再回答，
    不要凭空猜测或编造进度。
    返回的status含义：pending=排队中，running=正在跑（current_stage是当前步骤），
    done=已完成，failed=失败（error字段是原因）。"""
    db = SessionLocal()
    try:
        job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()
        if not job:
            return json.dumps({"error": f"没有找到任务号{job_id}"}, ensure_ascii=False)
        return json.dumps(job.to_dict(include_result_case=True), ensure_ascii=False)
    finally:
        db.close()


@tool
def update_case(
    case_id: int,
    title: str = None,
    full_narrative: str = None,
    teaching_objectives: dict = None,
    sizheng_elements: dict = None,
    applicable_courses: list = None,
    teaching_design: dict = None,
    further_reading: list = None,
    status: str = None,
) -> str:
    """更新某个案例的字段（比如修改标题、完整案例正文、教学设计等），只传需要修改的字段。
    适用课程举例/教学设计通常由知识点匹配采纳后自动生成，不建议在这里手工瞎编；
    status 取值范围：草稿/待审核/已采纳/已驳回"""
    db = SessionLocal()
    try:
        case = db.query(Case).filter(Case.id == case_id).first()
        if not case:
            return json.dumps({"error": "案例不存在"}, ensure_ascii=False)

        fields = {
            "title": title, "full_narrative": full_narrative,
            "teaching_objectives": teaching_objectives, "sizheng_elements": sizheng_elements,
            "applicable_courses": applicable_courses, "teaching_design": teaching_design,
            "further_reading": further_reading, "status": status,
        }
        changes = {}
        for field, value in fields.items():
            if value is None:
                continue
            old_value = getattr(case, field, None)
            new_value = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            if old_value != new_value:
                changes[field] = {"old": old_value, "new": new_value}
            setattr(case, field, new_value)

        db.commit()
        db.refresh(case)
        if changes:
            log_case_change(db, case.id, "AI助手", changes)
        return json.dumps(case.to_dict(), ensure_ascii=False)
    finally:
        db.close()


@tool
def enrich_case_with_accepted_knowledge(case_id: int) -> str:
    """用某个案例在「知识点匹配」页面已经人工标记为"已采纳"的知识点关联，提交一个后台任务，
    调用模型补充/更新这个案例的"适用课程举例"与"教学设计"两个字段。这个案例必须已经跑过
    知识点匹配、且至少有一条被标记为"已采纳"，否则会返回error，如果报错就把这个情况告诉用户，
    不要自己凭空编造适用课程举例/教学设计的内容。

    **这个工具只负责提交任务，立刻就返回，返回时内容还没补充好**（模型调用通常几十秒到
    一分钟），返回的job_id可以之后用 check_job 查进度。回复用户时不要说成已经补充完了。

    注意：现在用户在页面上采纳/取消采纳知识点关联时后台会自动触发同样的补充任务，这个工具
    通常不需要手动调用，只有用户明确要求"重新生成一次"之类的场景才用得上。"""
    db = SessionLocal()
    try:
        case = db.query(Case).filter(Case.id == case_id).first()
        if not case:
            return json.dumps({"error": "案例不存在"}, ensure_ascii=False)

        has_accepted = (
            db.query(CaseKnowledgeMapping)
            .filter(CaseKnowledgeMapping.case_id == case_id, CaseKnowledgeMapping.status == "已采纳")
            .first()
        )
        if not has_accepted:
            return json.dumps(
                {"error": "这个案例还没有已采纳的知识点关联，请先在「知识点匹配」页面运行匹配并采纳至少一条"},
                ensure_ascii=False,
            )

        job_id = submit_enrich_job(case_id)
        return json.dumps({
            "job_id": job_id,
            "submitted": True,
            "message": f"已提交补充任务（任务号{job_id}），预计几十秒到一分钟，完成后案例的"
                       f"适用课程举例和教学设计会自动更新。",
        }, ensure_ascii=False)
    finally:
        db.close()


TOOLS = [
    list_materials, search_materials, list_cases, get_case, generate_case_draft, update_case,
    enrich_case_with_accepted_knowledge, check_job,
]

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = create_agent(get_langchain_llm(), tools=TOOLS, system_prompt=SYSTEM_PROMPT)
    return _agent


def _to_langchain_messages(stored: list[dict]) -> list:
    """把我们自己的稳定存储格式（前端认的那套）转成LangChain消息对象"""
    messages = []
    for m in stored:
        role = m.get("role")
        content = m.get("content")
        if role == "user":
            if isinstance(content, str):
                messages.append(HumanMessage(content=content))
            else:
                for block in content or []:
                    if block.get("type") == "tool_result":
                        messages.append(ToolMessage(content=block.get("content", ""), tool_call_id=block.get("tool_use_id", "")))
        elif role == "assistant":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            tool_calls = [
                {"name": b["name"], "args": b.get("input", {}), "id": b["id"]}
                for b in blocks if b.get("type") == "tool_use"
            ]
            messages.append(AIMessage(content=text, tool_calls=tool_calls))
    return messages


def _from_langchain_messages(messages: list) -> list[dict]:
    """把LangChain消息对象转回我们自己的稳定存储格式，前端的聊天渲染逻辑靠的就是这个格式"""
    stored = []
    for msg in messages:
        if msg.type == "human":
            stored.append({"role": "user", "content": msg.content})
        elif msg.type == "ai":
            blocks = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for tc in (msg.tool_calls or []):
                blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["args"]})
            stored.append({"role": "assistant", "content": blocks})
        elif msg.type == "tool":
            stored.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": msg.tool_call_id, "content": msg.content}],
            })
    return stored


def stream_chat(stored_messages: list[dict], user_message: str):
    """
    流式版对话：逐token把AI助手正在生成的文字yield出去，供接口边生成边推给前端（打字机效果），
    而不是等模型把整轮工具调用+回复都跑完才一次性返回。

    用LangGraph的双模式流：
    - stream_mode="messages" 拿到的是模型节点逐token吐出的AIMessageChunk，用来做实时展示；
    - stream_mode="values" 拿到的是每个步骤结束后的完整图状态（含工具调用/工具结果），
      用它的最后一次取值作为最终要落库的完整对话历史——这样不用自己拼工具调用参数/结果，
      直接复用和非流式版一样的、LangGraph已经组装好的准确状态。

    真正的.stream()迭代放在后台线程（_run_agent）里跑，本函数只从队列里取事件转成yield。

    注意：所有耗时的LLM操作（案例生成、知识点补充）现在都是"工具提交后台任务后立刻返回"，
    工具调用本身都是秒级的，这条SSE连接不会再被占用十几分钟。那些长任务的进度改成写数据库、
    前端轮询 /api/jobs 拿（见job_queue.py），不走这条流。

    yield出的事件：
      {"type": "token", "text": ...}                          —— 增量文本，直接拼到当前气泡上
      {"type": "tool_call", "name": ...}                       —— 检测到一次工具调用，前端展示一个工具chip
      {"type": "done", "messages": [...], "reply": ...}        —— 流结束，附最终要落库的完整历史
      {"type": "error", "message": ...}                        —— 出错，调用方决定要不要落库当前进度
    """
    require_api_key()

    history = _to_langchain_messages(stored_messages) if stored_messages else []
    history.append(HumanMessage(content=user_message))

    q: queue.Queue = queue.Queue()

    def _run_agent():
        try:
            for mode, chunk in _get_agent().stream(
                {"messages": history},
                config={"recursion_limit": MAX_TOOL_ROUNDS * 2 + 1},
                stream_mode=["messages", "values"],
            ):
                q.put(("event", mode, chunk))
        except GraphRecursionError:
            q.put(("recursion_error", None))
        except Exception as e:
            q.put(("exception", e))
        finally:
            q.put(("sentinel", None))

    thread = threading.Thread(target=_run_agent, daemon=True)
    thread.start()

    final_state = None
    seen_tool_call_ids = set()

    while True:
        item = q.get()
        kind = item[0]

        if kind == "sentinel":
            break

        if kind == "recursion_error":
            reply = "（工具调用次数过多，已中止，请换个更具体的说法重试）"
            yield {"type": "token", "text": reply}
            yield {"type": "done", "messages": _from_langchain_messages(history), "reply": reply}
            return

        if kind == "exception":
            _, exc = item
            yield {"type": "error", "message": str(exc)}
            return

        # kind == "event"：item是("event", mode, chunk)，跟原来for循环里拿到的一样
        _, mode, chunk = item
        if mode == "values":
            final_state = chunk
            continue

        # mode == "messages"：chunk 是 (message_chunk, metadata)
        msg_chunk, meta = chunk
        if meta.get("langgraph_node") != "model":
            continue  # 跳过tools节点吐出的ToolMessage整块内容，那个不是给用户逐字看的

        content = getattr(msg_chunk, "content", None)
        if content:
            yield {"type": "token", "text": content}

        for tc in (getattr(msg_chunk, "tool_call_chunks", None) or []):
            tc_id = tc.get("id")
            name = tc.get("name")
            if name and tc_id and tc_id not in seen_tool_call_ids:
                seen_tool_call_ids.add(tc_id)
                yield {"type": "tool_call", "name": name}

    thread.join()

    updated_messages = final_state["messages"] if final_state else history
    reply = ""
    for msg in reversed(updated_messages):
        if msg.type == "ai" and msg.content:
            reply = msg.content
            break

    yield {"type": "done", "messages": _from_langchain_messages(updated_messages), "reply": reply}
