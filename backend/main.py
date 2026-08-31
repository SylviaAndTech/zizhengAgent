"""
FastAPI 主入口
提供：素材导入、案例生成、案例查询/编辑/审核 等接口
运行方式: uvicorn main:app --reload --port 8000
"""
import io
import json
import logging
import os
import shutil
import tempfile

import docx
from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import or_, func
from sqlalchemy.orm import Session

from db import (
    init_db, get_db, SessionLocal, RawMaterial, Case, CaseMaterial, DIMENSIONS,
    KnowledgePoint, CaseKnowledgeMapping, ChatSession, CaseAuditLog, User,
)
from auth import (
    AUTH_ENABLED, DEFAULT_ADMIN_USERNAME, verify_password,
    create_session_token, get_user_from_token, delete_session_token,
)
from fetch_material import fetch_url_text
from generate_case import generate_case_draft
from parse_document import parse_uploaded_document
from knowledge_matching import (
    extract_knowledge_points, match_case_to_knowledge, index_knowledge_point,
    reindex_knowledge_point, remove_knowledge_point_from_index, enrich_case_from_accepted_mappings,
)
from chat_agent import stream_chat
from audit import log_case_change
from doc_writer import write_case_section, set_default_font
from knowledge_graph import build_graph, render_graph_png, render_graph_html
from book_export import build_book_docx
from material_index import index_material

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="思政案例生成工作台 - 原型")

# ---------- 登录鉴权：所有/api/*接口默认都要求登录，登录本身这个接口除外 ----------
# AUTH_ENABLED=false（.env里配）可以整体关掉这层校验，方便测试阶段不用每次都登录；
# 上线前要确认.env里这个开关是打开的（默认就是true，不用特意配也行）。
#
# 这个中间件必须在下面app.add_middleware(CORSMiddleware...)之前注册——Starlette的中间件栈是
# "后注册的在最外层"，如果CORS中间件先注册、auth中间件后注册，auth中间件会被包在CORS外面，
# 它直接返回的401 JSONResponse就不会经过CORS中间件处理、不会带Access-Control-Allow-Origin
# 响应头。前端(:5500)跨源请求受保护接口时，浏览器看到没有CORS头的401会直接判定为CORS错误、
# 把响应体和状态码都藏起来不让JS读到，导致前端apiFetch()里"检查res.status===401"这行代码
# 永远走不到（fetch()的promise会在CORS阶段就reject掉）。实测验证过这个顺序错了会复现这个问题，
# 调换成现在这个顺序（auth先注册、在内层；CORS后注册、在外层）之后401响应能正确带上CORS头。
_PUBLIC_API_PATHS = {"/api/auth/login"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not AUTH_ENABLED:
        return await call_next(request)
    path = request.url.path
    if request.method == "OPTIONS" or not path.startswith("/api/") or path in _PUBLIC_API_PATHS:
        return await call_next(request)
    auth_header = request.headers.get("Authorization", "")
    token = auth_header[7:] if auth_header.startswith("Bearer ") else ""
    db = SessionLocal()
    try:
        user = get_user_from_token(db, token)
    finally:
        db.close()
    if not user:
        return JSONResponse({"detail": "未登录或登录已过期，请重新登录"}, status_code=401)
    request.state.user = user
    return await call_next(request)


# 默认只放开本机常用的静态服务端口；部署到其他地址时通过 CORS_ORIGINS（逗号分隔）覆盖
_default_origins = "http://localhost:5500,http://127.0.0.1:5500,http://localhost:3000,http://127.0.0.1:3000"
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", _default_origins).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
# ChromaDB/LlamaIndex 的连接是懒加载的（第一次真正建索引/检索时才连），
# 这里不用单独预热，连不上也不会拖垮后端启动，只会在用到时报错。


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == req.username, User.is_active.is_(True)).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "用户名或密码错误")
    token = create_session_token(db, user)
    return {"token": token, "username": user.username}


@app.post("/api/auth/logout")
def logout(request: Request, db: Session = Depends(get_db)):
    auth_header = request.headers.get("Authorization", "")
    token = auth_header[7:] if auth_header.startswith("Bearer ") else ""
    delete_session_token(db, token)
    return {"logged_out": True}


@app.get("/api/auth/me")
def get_me(request: Request):
    if not AUTH_ENABLED:
        return {"username": DEFAULT_ADMIN_USERNAME, "auth_disabled": True}
    user = request.state.user  # 走到这里说明auth_middleware已经校验通过，user一定存在
    return {"username": user.username, "auth_disabled": False}


# ---------- 请求体模型 ----------

class ImportMaterialsRequest(BaseModel):
    case_code: str
    urls: list[str]


class GenerateCaseRequest(BaseModel):
    case_code: str
    material_ids: list[int]


class UpdateCaseRequest(BaseModel):
    title: str | None = None
    full_narrative: str | None = None
    teaching_objectives: dict | None = None
    sizheng_elements: dict | None = None
    applicable_courses: list | None = None
    teaching_design: dict | None = None
    further_reading: list | None = None
    status: str | None = None


class ExportCasesRequest(BaseModel):
    case_ids: list[int]


class ChatSendRequest(BaseModel):
    session_id: int
    message: str


class UpdateMappingRequest(BaseModel):
    status: str | None = None
    suggestion_text: str | None = None


class UpdateKnowledgePointRequest(BaseModel):
    course_name: str | None = None
    chapter: str | None = None
    description: str | None = None


class CreateKnowledgePointRequest(BaseModel):
    course_name: str
    chapter: str | None = None
    description: str


# ---------- 素材相关接口 ----------

@app.post("/api/materials/import")
def import_materials(req: ImportMaterialsRequest, db: Session = Depends(get_db)):
    """批量导入URL并抓取正文，落盘存档；同一案例下已经存过的URL不会重复入库"""
    results = []
    for url in req.urls:
        url = url.strip()
        if not url:
            continue

        existing = (
            db.query(RawMaterial)
            .filter(RawMaterial.case_code == req.case_code, RawMaterial.url == url)
            .first()
        )
        if existing:
            d = _material_to_dict(existing)
            d["duplicate"] = True
            results.append(d)
            continue

        fetched = fetch_url_text(url)
        material = RawMaterial(
            case_code=req.case_code,
            url=url,
            source_title=fetched.get("title"),
            fetched_text=fetched["text"],
            fetch_status=fetched["status"],
            fetch_error=fetched["error"],
        )
        db.add(material)
        db.commit()
        db.refresh(material)
        _index_material_best_effort(material)
        results.append(_material_to_dict(material))
    return {"imported": results}


def _index_material_best_effort(material: RawMaterial):
    """把素材内容切块存进检索向量库，供AI助手语义搜索；索引失败不应该让素材导入本身失败"""
    if material.fetch_status != "success":
        return
    try:
        index_material(material)
    except Exception as e:
        logger.warning(f"素材(id={material.id})索引失败，AI助手暂时搜不到它: {e}")


def _save_upload_to_tempfile(file: UploadFile, filename: str) -> str:
    """把上传文件流式落盘到临时文件，返回路径。调用方处理完后负责用 os.unlink 清理。
    大纲/素材文档可能到几百MB，用 file.file.read() 整个读进内存在小内存服务器上容易顶不住，
    这里改成分块拷贝到磁盘，全程不在Python里持有完整文件的bytes对象"""
    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return tmp.name


@app.post("/api/materials/upload")
def upload_material(
    case_code: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """上传Word/PDF，提取正文并尝试按标题拆分成多个候选案例，分别落入素材库"""
    filename = file.filename or ""
    lower_name = filename.lower()
    if not (lower_name.endswith(".docx") or lower_name.endswith(".pdf")):
        raise HTTPException(400, "仅支持 .docx 或 .pdf 文件（不支持旧版 .doc）")

    source_type = "docx" if lower_name.endswith(".docx") else "pdf"

    tmp_path = _save_upload_to_tempfile(file, filename)
    try:
        segments = parse_uploaded_document(filename, tmp_path)
    except Exception as e:
        raise HTTPException(400, f"文档解析失败: {str(e)}")
    finally:
        os.unlink(tmp_path)

    if not segments:
        raise HTTPException(400, "没有从文档中提取到任何正文内容")

    results = []
    for idx, seg in enumerate(segments):
        material = RawMaterial(
            case_code=case_code,
            url=None,
            source_title=seg.get("title") or f"{filename} 片段{idx + 1}",
            fetched_text=seg["text"],
            fetch_status="success",
            fetch_error=None,
            source_type=source_type,
            source_filename=filename,
            segment_index=idx,
        )
        db.add(material)
        db.commit()
        db.refresh(material)
        _index_material_best_effort(material)
        d = _material_to_dict(material)
        d["fallback"] = seg.get("fallback", False)
        results.append(d)

    return {"imported": results, "segment_count": len(results)}


def _material_to_dict(m: RawMaterial) -> dict:
    return {
        "id": m.id,
        "case_code": m.case_code,
        "url": m.url,
        "source_title": m.source_title,
        "source_type": m.source_type,
        "source_filename": m.source_filename,
        "segment_index": m.segment_index,
        "fetch_status": m.fetch_status,
        "fetch_error": m.fetch_error,
        "text_preview": (m.fetched_text or "")[:200],
        "full_text": m.fetched_text,
    }


@app.get("/api/materials")
def list_materials(
    case_code: str | None = None, q: str | None = None, db: Session = Depends(get_db)
):
    """查询素材库：case_code不填时返回全部案例的素材，填了就按案例编号过滤；
    q是关键词搜索，匹配标题/URL/来源文件名/正文内容"""
    query = db.query(RawMaterial)
    if case_code:
        query = query.filter(RawMaterial.case_code == case_code)
    if q:
        like = f"%{q}%"
        query = query.filter(
            or_(
                RawMaterial.source_title.like(like),
                RawMaterial.url.like(like),
                RawMaterial.source_filename.like(like),
                RawMaterial.fetched_text.like(like),
            )
        )
    materials = query.order_by(RawMaterial.id.desc()).all()
    return [_material_to_dict(m) for m in materials]


def _compute_next_case_code(case_codes: list[str]) -> str | None:
    """给"新增素材默认归到哪个案例编号"提供一个合理的默认值：取现有案例编号里最大的一个，
    把它最后一段数字加1（如 "3.4" -> "3.5"，"5" -> "6"）。没有任何现有编号、或编号格式不是
    纯数字点分（如"3.4"）就返回None，交给用户自己填，不瞎猜一个可能没意义的默认值"""

    def sort_key(code: str):
        try:
            return tuple(int(p) for p in code.split("."))
        except ValueError:
            return None

    numeric = [(sort_key(c), c) for c in case_codes]
    numeric = [(k, c) for k, c in numeric if k is not None]
    if not numeric:
        return None
    numeric.sort(key=lambda x: x[0])
    max_code = numeric[-1][1]
    parts = max_code.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    return ".".join(parts)


@app.get("/api/materials/case_codes")
def list_material_case_codes(db: Session = Depends(get_db)):
    """素材库里已经出现过的案例编号（供前端做筛选下拉），外加一个建议的"下一个案例编号"默认值"""
    rows = db.query(RawMaterial.case_code).distinct().all()
    codes = sorted({r[0] for r in rows if r[0]})
    return {"case_codes": codes, "next_case_code": _compute_next_case_code(codes)}


@app.delete("/api/materials/{material_id}")
def delete_material(material_id: int, db: Session = Depends(get_db)):
    """删除一条素材；同时清掉它在案例证据链里的关联记录"""
    material = db.query(RawMaterial).filter(RawMaterial.id == material_id).first()
    if not material:
        raise HTTPException(404, "素材不存在")
    db.query(CaseMaterial).filter(CaseMaterial.material_id == material_id).delete()
    db.delete(material)
    db.commit()
    return {"deleted": True}


# ---------- 案例生成/审核相关接口 ----------

@app.get("/api/dimensions")
def get_dimensions():
    return DIMENSIONS


@app.post("/api/cases/generate")
def generate_case(req: GenerateCaseRequest, db: Session = Depends(get_db)):
    """根据选定的素材，调用LLM生成7段式案例草稿"""
    materials = (
        db.query(RawMaterial)
        .filter(RawMaterial.id.in_(req.material_ids))
        .all()
    )
    if not materials:
        raise HTTPException(400, "未找到指定的素材ID")

    success_materials = [m for m in materials if m.fetch_status == "success"]
    if not success_materials:
        raise HTTPException(
            400, "所选素材均抓取失败，没有可供生成的真实内容，请先补充有效素材"
        )

    material_payload = [
        {"id": m.id, "url": m.url, "title": None, "text": m.fetched_text}
        for m in success_materials
    ]

    try:
        draft = generate_case_draft(req.case_code, material_payload)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"生成失败: {str(e)}")

    case = Case(
        case_code=req.case_code,
        dimension=(draft.get("sizheng_elements") or {}).get("对应维度"),
        title=draft.get("title"),
        full_narrative=draft.get("full_narrative"),
        full_narrative_draft=draft.get("full_narrative_draft"),
        teaching_objectives=json.dumps(draft.get("teaching_objectives"), ensure_ascii=False),
        sizheng_elements=json.dumps(draft.get("sizheng_elements"), ensure_ascii=False),
        # 适用课程举例/教学设计不在初次生成时产出了，等知识点匹配采纳后才会有内容，
        # 这里留空（真正的NULL，不是json.dumps(None)的字符串"null"）
        applicable_courses=None,
        teaching_design=None,
        further_reading=json.dumps(draft.get("further_reading"), ensure_ascii=False),
        status="待审核",
    )
    db.add(case)
    db.commit()
    db.refresh(case)

    for m in success_materials:
        db.add(CaseMaterial(case_id=case.id, material_id=m.id))
    db.commit()

    return case.to_dict()


@app.get("/api/cases")
def list_cases(case_code: str | None = None, db: Session = Depends(get_db)):
    query = db.query(Case)
    if case_code:
        query = query.filter(Case.case_code == case_code)
    cases = query.order_by(Case.id.desc()).all()
    return [c.to_dict() for c in cases]


@app.get("/api/cases/{case_id}")
def get_case(case_id: int, db: Session = Depends(get_db)):
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(404, "案例不存在")
    return case.to_dict()


@app.get("/api/cases/{case_id}/materials")
def get_case_materials(case_id: int, db: Session = Depends(get_db)):
    """案例正文里的[素材N:定位短语]引用标注，N是生成时素材的排列序号；这里按当初关联的顺序
    （CaseMaterial.id升序，即生成时插入的先后顺序）把素材原文一并返回，前端才能在hover时
    用"素材N"精确对上是哪一条、再用定位短语去全文里找具体位置"""
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(404, "案例不存在")
    links = (
        db.query(CaseMaterial)
        .filter(CaseMaterial.case_id == case_id)
        .order_by(CaseMaterial.id)
        .all()
    )
    materials = []
    for link in links:
        m = link.material
        if not m:
            continue
        materials.append({
            "id": m.id,
            "url": m.url,
            "title": m.source_title,
            "full_text": m.fetched_text,
        })
    return {"materials": materials}


@app.put("/api/cases/{case_id}")
def update_case(case_id: int, req: UpdateCaseRequest, db: Session = Depends(get_db)):
    """人工编辑/审核通过案例"""
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(404, "案例不存在")

    update_data = req.model_dump(exclude_unset=True)
    changes = {}
    for field, value in update_data.items():
        old_value = getattr(case, field, None)
        new_value = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        if old_value != new_value:
            changes[field] = {"old": old_value, "new": new_value}
        setattr(case, field, new_value)

    db.commit()
    db.refresh(case)
    if changes:
        log_case_change(db, case.id, "用户", changes)
    return case.to_dict()


@app.delete("/api/cases/{case_id}")
def delete_case(case_id: int, db: Session = Depends(get_db)):
    """删除一个案例；连带清掉它的证据链关联、知识点关联建议、修改记录，避免留下悬空外键"""
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(404, "案例不存在")
    db.query(CaseMaterial).filter(CaseMaterial.case_id == case_id).delete()
    db.query(CaseKnowledgeMapping).filter(CaseKnowledgeMapping.case_id == case_id).delete()
    db.query(CaseAuditLog).filter(CaseAuditLog.case_id == case_id).delete()
    db.delete(case)
    db.commit()
    return {"deleted": True}


@app.get("/api/cases/{case_id}/audit_log")
def get_case_audit_log(case_id: int, db: Session = Depends(get_db)):
    """查看某个案例的修改记录：谁在什么时候改了哪些字段"""
    logs = (
        db.query(CaseAuditLog)
        .filter(CaseAuditLog.case_id == case_id)
        .order_by(CaseAuditLog.created_at.desc())
        .all()
    )
    return [log.to_dict() for log in logs]


@app.post("/api/cases/export")
def export_cases(req: ExportCasesRequest, db: Session = Depends(get_db)):
    """把勾选的案例按七段式结构拼装成一份 Word 文档，供下载"""
    cases = (
        db.query(Case)
        .filter(Case.id.in_(req.case_ids))
        .order_by(Case.case_code)
        .all()
    )
    if not cases:
        raise HTTPException(400, "没有找到要导出的案例")

    doc = docx.Document()
    set_default_font(doc)
    doc.add_heading("思政案例集（导出）", level=0)

    for c in cases:
        write_case_section(doc, c.to_dict(), heading_level=1)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=cases_export.docx"},
    )


# ---------- AI 助手（agent对话，替代原来的案例工作台手动编辑） ----------

@app.get("/api/chat/sessions")
def list_chat_sessions(db: Session = Depends(get_db)):
    """会话列表，按最近更新排序，供左侧栏展示历史记录"""
    sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
    return [
        {
            "id": s.id,
            "title": s.title,
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            "message_count": len(json.loads(s.messages)) if s.messages else 0,
        }
        for s in sessions
    ]


@app.post("/api/chat/sessions")
def create_chat_session(db: Session = Depends(get_db)):
    session = ChatSession(title="新对话", messages="[]")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session.to_dict()


@app.get("/api/chat/sessions/{session_id}")
def get_chat_session(session_id: int, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(404, "会话不存在")
    return session.to_dict()


@app.delete("/api/chat/sessions/{session_id}")
def delete_chat_session(session_id: int, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(404, "会话不存在")
    db.delete(session)
    db.commit()
    return {"deleted": True}


@app.post("/api/chat")
def chat(req: ChatSendRequest, db: Session = Depends(get_db)):
    """给指定会话追加一条用户消息，以SSE流式推送AI助手逐token生成的回复；
    流结束时（事件类型done）把完整历史落回这个会话，同一条SSE事件里带上刷新后的session，
    前端不用再单独发一次请求同步会话标题/消息数"""
    session = db.query(ChatSession).filter(ChatSession.id == req.session_id).first()
    if not session:
        raise HTTPException(404, "会话不存在，请先创建一个新对话")

    history = json.loads(session.messages) if session.messages else []
    session_id = req.session_id
    user_message = req.message

    def event_stream():
        # StreamingResponse的生成器是在这个路由函数return之后才被逐步消费的，
        # 那时FastAPI已经把Depends(get_db)这个请求作用域的db session关掉了（ORM对象也随之脱管），
        # 所以落库这一步不能复用上面注入的db，要单独开一个新session
        try:
            for event in stream_chat(history, user_message):
                if event["type"] == "done":
                    write_db = SessionLocal()
                    try:
                        s = write_db.query(ChatSession).filter(ChatSession.id == session_id).first()
                        if s:
                            s.messages = json.dumps(event["messages"], ensure_ascii=False)
                            if s.title == "新对话" and user_message.strip():
                                s.title = user_message.strip()[:20]
                            write_db.commit()
                            write_db.refresh(s)
                            event = {**event, "session": s.to_dict()}
                    finally:
                        write_db.close()
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'AI助手出错: {e}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------- 知识点抽取与案例-知识点匹配 ----------

@app.post("/api/knowledge/upload")
def upload_knowledge(
    files: list[UploadFile] = File(...),
    course_name: str = Form(""),
    db: Session = Depends(get_db),
):
    """
    批量上传课程教学大纲（Word/PDF），逐份文档拆解出知识点条目。
    course_name 留空时，每份文档各自从标题/文件名里自动识别课程名（适合一次上传多门课程的大纲）；
    填写时，作为这一批全部文件的统一课程名（适合同一门课程的大纲拆成多个文件上传）。
    单个文件解析失败不影响其余文件继续处理。
    """
    created = []
    errors = []
    skipped_duplicate_points = 0

    for file in files:
        filename = file.filename or "(未命名文件)"
        if not (filename.lower().endswith(".docx") or filename.lower().endswith(".pdf")):
            errors.append(f"{filename}：仅支持 .docx 或 .pdf 文件")
            continue

        if db.query(KnowledgePoint).filter(KnowledgePoint.source_filename == filename).first():
            errors.append(f"{filename}：这个文件名已经上传过了，跳过（如果是新版本，请先删掉旧的知识点再传）")
            continue

        tmp_path = _save_upload_to_tempfile(file, filename)
        try:
            detected_course_name, points = extract_knowledge_points(filename, tmp_path, course_name)
        except Exception as e:
            errors.append(f"{filename}：解析失败 {str(e)}")
            continue
        finally:
            os.unlink(tmp_path)

        if not points:
            errors.append(f"{filename}：没有提取到可用的知识点条目")
            continue

        for p in points:
            # 同一门课程、同一章节下，知识点描述完全一样就不重复入库
            # （同一份大纲重复解析、或者不同文件里覆盖了相同内容时很容易出现）
            exists = (
                db.query(KnowledgePoint)
                .filter(
                    KnowledgePoint.course_name == p["course_name"],
                    KnowledgePoint.chapter == p["chapter"],
                    KnowledgePoint.description == p["description"],
                )
                .first()
            )
            if exists:
                skipped_duplicate_points += 1
                continue

            kp = KnowledgePoint(
                course_name=p["course_name"],
                chapter=p["chapter"],
                description=p["description"],
                source_filename=filename,
            )
            db.add(kp)
            db.commit()
            db.refresh(kp)
            try:
                index_knowledge_point(kp)
            except Exception as e:
                # 索引失败不能就这样留着这条MySQL记录——它会正常出现在知识点列表里，
                # 看起来一切正常，但向量粗筛永远召回不到它，是一种更隐蔽的"看得见、
                # 用不到"的不一致，跟"MySQL删了、向量库没删干净"的孤儿向量是同一类问题
                # 反过来的版本。这里直接把这条MySQL记录也回滚掉，当成这条知识点没导入
                # 成功处理，让用户能看到明确的失败提示、之后可以重新上传
                db.delete(kp)
                db.commit()
                errors.append(f"{filename}：知识点「{p['description'][:20]}...」建立向量索引失败，已跳过：{e}")
                continue
            created.append({
                "id": kp.id, "course_name": kp.course_name,
                "chapter": kp.chapter, "description": kp.description,
                "source_filename": kp.source_filename,
            })

    if not created and errors:
        raise HTTPException(400, "；".join(errors))

    return {
        "imported": created, "count": len(created), "errors": errors,
        "skipped_duplicate_points": skipped_duplicate_points,
    }


@app.get("/api/knowledge")
def list_knowledge(
    course_name: str | None = None,
    q: str | None = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
):
    """q 非空时按关键字模糊匹配 课程名/章节/知识点描述 任一字段（跨课程搜索，忽略course_name）；
    分页给前端"按课程折叠、展开后翻页"和"搜索结果扁平分页"两种视图共用"""
    query = db.query(KnowledgePoint)
    if q:
        like = f"%{q}%"
        query = query.filter(
            or_(
                KnowledgePoint.course_name.like(like),
                KnowledgePoint.chapter.like(like),
                KnowledgePoint.description.like(like),
            )
        )
    elif course_name:
        query = query.filter(KnowledgePoint.course_name == course_name)

    total = query.count()
    points = (
        query.order_by(KnowledgePoint.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    point_ids = [k.id for k in points]
    cases_by_point: dict[int, list[dict]] = {pid: [] for pid in point_ids}
    if point_ids:
        accepted = (
            db.query(CaseKnowledgeMapping)
            .filter(
                CaseKnowledgeMapping.knowledge_point_id.in_(point_ids),
                CaseKnowledgeMapping.status == "已采纳",
            )
            .all()
        )
        for m in accepted:
            if not m.case:
                continue
            cases_by_point[m.knowledge_point_id].append({
                "mapping_id": m.id, "case_id": m.case_id,
                "case_code": m.case.case_code, "title": m.case.title,
            })

    items = [
        {
            "id": k.id, "course_name": k.course_name, "chapter": k.chapter,
            "description": k.description, "source_filename": k.source_filename,
            "cases": cases_by_point.get(k.id, []),
        }
        for k in points
    ]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.get("/api/knowledge/courses")
def list_knowledge_courses(db: Session = Depends(get_db)):
    """课程名+各自知识点数量，给知识点库折叠列表的课程标题行用——不用为了拿这份清单
    去加载全部知识点正文"""
    rows = (
        db.query(KnowledgePoint.course_name, func.count(KnowledgePoint.id))
        .group_by(KnowledgePoint.course_name)
        .order_by(KnowledgePoint.course_name)
        .all()
    )
    return [{"course_name": name, "count": count} for name, count in rows]


@app.post("/api/knowledge")
def create_knowledge_point(req: CreateKnowledgePointRequest, db: Session = Depends(get_db)):
    """手动添加一条知识点，跟批量上传大纲解析出来的条目走同一张表，只是来源不是文件解析"""
    course_name = req.course_name.strip()
    description = req.description.strip()
    if not course_name or not description:
        raise HTTPException(400, "课程名称和知识点描述不能为空")

    kp = KnowledgePoint(
        course_name=course_name,
        chapter=(req.chapter or "").strip() or None,
        description=description,
        source_filename=None,
    )
    db.add(kp)
    db.commit()
    db.refresh(kp)
    try:
        index_knowledge_point(kp)
    except Exception as e:
        logger.warning(f"知识点(id={kp.id})索引失败，匹配时暂时召回不到它: {e}")
    return {
        "id": kp.id, "course_name": kp.course_name,
        "chapter": kp.chapter, "description": kp.description,
        "source_filename": kp.source_filename,
    }


def _blocking_mappings_detail(db: Session, point_ids: list[int]) -> str | None:
    """知识点被某个案例的知识点关联(case_knowledge_mappings)引用着的话，数据库外键约束
    不允许直接删除该知识点——这里在真正执行删除之前先查一遍，返回一句人话说明是被哪个
    案例的哪条关联卡住了，没有就返回None；不然任由IntegrityError原始报错抛给前端，
    用户完全看不出是哪个案例、哪条知识点导致删不掉"""
    mappings = (
        db.query(CaseKnowledgeMapping)
        .filter(CaseKnowledgeMapping.knowledge_point_id.in_(point_ids))
        .all()
    )
    if not mappings:
        return None
    # 同一个知识点跟同一个案例之间可能同时存在"推荐"和"已采纳"两条关联记录，
    # 按(知识点,案例)去重，不然消息里会把同一个案例重复报好几遍
    seen = set()
    parts = []
    for m in mappings:
        key = (m.knowledge_point_id, m.case_id)
        if key in seen:
            continue
        seen.add(key)
        kp = m.knowledge_point
        case = m.case
        kp_desc = kp.description if kp else "(知识点已不存在)"
        if len(kp_desc) > 30:
            kp_desc = kp_desc[:30] + "…"
        case_label = f"案例{case.case_code}《{case.title or ''}》" if case else f"案例(id={m.case_id})"
        parts.append(f"知识点「{kp_desc}」(id={m.knowledge_point_id}) 关联着 {case_label}")
    return (
        "；".join(parts)
        + "。如果确实要删除，可以强制删除——会连同这些案例对应的知识点关联一并解除，"
        "这些案例会因此缺失对应的知识点关联信息。"
    )


@app.delete("/api/knowledge")
def delete_knowledge_by_course(course_name: str, force: bool = False, db: Session = Depends(get_db)):
    """删掉一门课程下的全部知识点（连带向量索引），用于整门课程的大纲要重新导入或者传错课程了的场景。
    force=True 时，如果知识点被某个案例的知识点关联引用着，连同这些关联记录一并删除
    （否则数据库外键约束不允许删除被引用的知识点）"""
    points = db.query(KnowledgePoint).filter(KnowledgePoint.course_name == course_name).all()
    if not points:
        raise HTTPException(404, "没有找到这门课程的知识点")
    point_ids = [k.id for k in points]

    blocking_detail = _blocking_mappings_detail(db, point_ids)
    if blocking_detail and not force:
        raise HTTPException(409, f"这门课程下有知识点已经被案例引用，无法删除：{blocking_detail}")

    # 先删向量库、全部成功了再删MySQL，理由同delete_knowledge_point：避免"MySQL删了、
    # 向量库没删干净"这种不报错的孤儿向量
    failed_ids = []
    for point_id in point_ids:
        try:
            remove_knowledge_point_from_index(point_id)
        except Exception as e:
            failed_ids.append(point_id)
            logger.warning(f"知识点(id={point_id})从向量库删除失败: {e}")
    if failed_ids:
        raise HTTPException(
            500,
            f"有{len(failed_ids)}条知识点从向量库删除失败，为避免留下孤儿向量，本次删除已取消，可以重试：涉及id {failed_ids}",
        )

    removed_mappings = 0
    if force:
        removed_mappings = (
            db.query(CaseKnowledgeMapping)
            .filter(CaseKnowledgeMapping.knowledge_point_id.in_(point_ids))
            .delete(synchronize_session=False)
        )

    db.query(KnowledgePoint).filter(KnowledgePoint.id.in_(point_ids)).delete(synchronize_session=False)
    db.commit()
    return {"deleted": True, "count": len(point_ids), "removed_mappings": removed_mappings}


@app.put("/api/knowledge/{point_id}")
def update_knowledge_point(point_id: int, req: UpdateKnowledgePointRequest, db: Session = Depends(get_db)):
    """人工修正一条知识点（抽取难免有错，特别是扫描件OCR来的）；描述改了要重新建索引，
    不然向量检索命中的还是修改前的旧文本"""
    kp = db.query(KnowledgePoint).filter(KnowledgePoint.id == point_id).first()
    if not kp:
        raise HTTPException(404, "知识点不存在")

    update_data = req.model_dump(exclude_unset=True)
    description_changed = "description" in update_data and update_data["description"] != kp.description
    for field, value in update_data.items():
        setattr(kp, field, value)
    db.commit()
    db.refresh(kp)

    if description_changed:
        try:
            reindex_knowledge_point(kp)
        except Exception as e:
            logger.warning(f"知识点(id={kp.id})重新索引失败，语义检索暂时还是旧文本: {e}")

    return {
        "id": kp.id, "course_name": kp.course_name, "chapter": kp.chapter,
        "description": kp.description, "source_filename": kp.source_filename,
    }


@app.delete("/api/knowledge/{point_id}")
def delete_knowledge_point(point_id: int, force: bool = False, db: Session = Depends(get_db)):
    """force=True 时，如果这条知识点被某个案例的知识点关联引用着，连同这条关联记录一并删除"""
    kp = db.query(KnowledgePoint).filter(KnowledgePoint.id == point_id).first()
    if not kp:
        raise HTTPException(404, "知识点不存在")

    blocking_detail = _blocking_mappings_detail(db, [point_id])
    if blocking_detail and not force:
        raise HTTPException(409, f"这条知识点已经被案例引用，无法删除：{blocking_detail}")

    # 先删向量库、成功了再删MySQL：万一向量库这一步失败，整个请求直接报错、MySQL记录
    # 原样保留，用户能看到失败提示并重试；如果反过来先删MySQL再删向量库，向量库这步
    # 一旦失败就会留下"MySQL已经没有这条记录、向量库里还留着"的孤儿向量——不报错、
    # 不易察觉，之前排查向量粗筛问题时就真的挖出过82条这样的历史孤儿
    try:
        remove_knowledge_point_from_index(point_id)
    except Exception as e:
        raise HTTPException(500, f"知识点(id={point_id})从向量库删除失败，为避免留下孤儿向量，本次删除已取消，可以重试：{e}")

    removed_mappings = 0
    if force:
        removed_mappings = (
            db.query(CaseKnowledgeMapping)
            .filter(CaseKnowledgeMapping.knowledge_point_id == point_id)
            .delete(synchronize_session=False)
        )

    db.delete(kp)
    db.commit()
    return {"deleted": True, "removed_mappings": removed_mappings}


@app.post("/api/cases/{case_id}/match_knowledge")
def match_knowledge(case_id: int, db: Session = Depends(get_db)):
    """
    对已审核/草稿案例，跑一遍与知识点库的匹配：向量粗筛+Qwen复核精排，生成候选关联建议
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(404, "案例不存在")

    if db.query(KnowledgePoint).count() == 0:
        raise HTTPException(400, "还没有导入任何知识点，请先在「知识点匹配」标签页上传课程教学大纲")

    try:
        matches = match_case_to_knowledge(db, case)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # 清空旧的"推荐"记录避免重复堆积；已人工采纳/拒绝的决定予以保留
    db.query(CaseKnowledgeMapping).filter(
        CaseKnowledgeMapping.case_id == case_id,
        CaseKnowledgeMapping.status == "推荐",
    ).delete()
    db.commit()

    created = []
    for m in matches:
        mapping = CaseKnowledgeMapping(
            case_id=case_id,
            knowledge_point_id=m["knowledge_point"].id,
            relevance_score=m["relevance_score"],
            suggestion_text=m["suggestion_text"],
            status="推荐",
        )
        db.add(mapping)
        db.commit()
        db.refresh(mapping)
        created.append(mapping.to_dict())
    return {"mappings": created}


@app.get("/api/cases/{case_id}/knowledge_mappings")
def get_knowledge_mappings(case_id: int, db: Session = Depends(get_db)):
    mappings = (
        db.query(CaseKnowledgeMapping)
        .filter(CaseKnowledgeMapping.case_id == case_id)
        .all()
    )
    return [m.to_dict() for m in mappings]


@app.put("/api/knowledge_mappings/{mapping_id}")
def update_mapping(mapping_id: int, req: UpdateMappingRequest, db: Session = Depends(get_db)):
    """人工修改：采纳/拒绝某条知识点关联，或编辑融入方式建议文字。
    只要这次修改让案例的"已采纳"知识点集合发生了变化（新采纳一条、或者把已采纳的改成别的
    状态），就自动重新生成一遍"适用课程举例"/"教学设计"——不用再手动点按钮触发，见
    knowledge_matching.enrich_case_from_accepted_mappings()。这一步涉及真实LLM调用，
    接口响应会比单纯改个状态慢一些。"""
    mapping = db.query(CaseKnowledgeMapping).filter(CaseKnowledgeMapping.id == mapping_id).first()
    if not mapping:
        raise HTTPException(404, "关联记录不存在")

    old_status = mapping.status
    if req.status is not None:
        mapping.status = req.status
    if req.suggestion_text is not None:
        mapping.suggestion_text = req.suggestion_text
    db.commit()
    db.refresh(mapping)

    result = mapping.to_dict()
    accepted_set_changed = req.status is not None and "已采纳" in (old_status, req.status) and old_status != req.status
    if accepted_set_changed:
        case = db.query(Case).filter(Case.id == mapping.case_id).first()
        if case:
            try:
                changes = enrich_case_from_accepted_mappings(db, case)
                if changes:
                    db.commit()
                    db.refresh(case)
                    log_case_change(db, case.id, "知识点匹配(自动)", changes)
            except ValueError as e:
                # 重新生成失败（比如API key没配好）不应该让"采纳/取消采纳"这个状态改动本身失败——
                # 状态已经落库了，只是适用课程举例/教学设计这次没能自动刷新；用一个附加字段告诉
                # 前端这个情况，而不是把整个请求判定为失败（返回体形状要跟正常情况一致，前端才能
                # 正常解析出mapping数据，不能直接raise一个跟mapping.to_dict()完全不同形状的错误体）
                result["enrich_warning"] = f"状态已更新，但自动生成适用课程举例/教学设计失败：{e}"

    return result


@app.delete("/api/knowledge_mappings/{mapping_id}")
def delete_knowledge_mapping(mapping_id: int, db: Session = Depends(get_db)):
    """解绑一条案例↔知识点关联。如果这条关联当时是"已采纳"状态，解绑后案例的"已采纳"集合
    也跟着变了，同样要重新生成一遍"适用课程举例"/"教学设计"（如果解绑后已采纳集合空了，
    这两个字段会被清空，不会留着解绑前的旧内容）。"""
    mapping = db.query(CaseKnowledgeMapping).filter(CaseKnowledgeMapping.id == mapping_id).first()
    if not mapping:
        raise HTTPException(404, "关联记录不存在")

    was_accepted = mapping.status == "已采纳"
    case_id = mapping.case_id
    db.delete(mapping)
    db.commit()

    result = {"unbound": True}
    if was_accepted:
        case = db.query(Case).filter(Case.id == case_id).first()
        if case:
            try:
                changes = enrich_case_from_accepted_mappings(db, case)
                if changes:
                    db.commit()
                    log_case_change(db, case.id, "知识点匹配(自动)", changes)
            except ValueError as e:
                result["enrich_warning"] = f"已解绑，但自动生成适用课程举例/教学设计失败：{e}"

    return result


# ---------- 知识图谱 ----------

@app.get("/api/knowledge_graph.png")
def get_knowledge_graph_png(db: Session = Depends(get_db)):
    """维度→案例→知识点 三层关系图的静态图片，只统计已采纳的案例与已采纳的知识点关联"""
    graph = build_graph(db)
    png_bytes = render_graph_png(graph)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/api/knowledge_graph.html")
def get_knowledge_graph_html(db: Session = Depends(get_db)):
    """同一份图谱的可交互HTML版本（可拖拽缩放），单独打开查看"""
    graph = build_graph(db)
    html = render_graph_html(graph)
    return HTMLResponse(content=html)


# ---------- 成书编译 ----------

@app.get("/api/book/export")
def export_book(status: str = "已采纳", db: Session = Depends(get_db)):
    """
    按"前言 + 按维度分章 + 附录一/二/三 + 知识图谱"的固定版式，
    把指定审核状态（默认"已采纳"）的案例编译成一份完整Word文档
    """
    docx_bytes = build_book_docx(db, status_filter=status)
    return StreamingResponse(
        io.BytesIO(docx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=sizheng_case_book.docx"},
    )


@app.get("/")
def health_check():
    return {"status": "ok", "message": "思政案例生成工作台 API 运行中"}
