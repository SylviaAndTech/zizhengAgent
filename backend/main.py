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
    KnowledgePoint, CaseKnowledgeMapping, ChatSession, CaseAuditLog, User, BackgroundJob,
)
from auth import (
    AUTH_ENABLED, DEFAULT_ADMIN_USERNAME, verify_password,
    create_session_token, get_user_from_token, delete_session_token,
)
from fetch_material import fetch_url_text
from parse_document import parse_uploaded_document
from knowledge_matching import (
    extract_knowledge_points, index_knowledge_point,
    reindex_knowledge_point, remove_knowledge_point_from_index,
)
from job_queue import (
    submit_generate_job, submit_match_knowledge_job, submit_enrich_job,
    reset_stale_jobs_on_startup,
)
from chat_agent import stream_chat
from audit import log_case_change
from doc_writer import write_case_section, set_default_font
from mermaid_tree import build_applicable_courses_mermaid
from mermaid_render import render_mermaid_batch
from knowledge_graph import build_graph, render_graph_png, render_graph_html
from book_export import build_book_docx
from material_index import index_material

logger = logging.getLogger("uvicorn.error")

# root_path="/api"：生产环境 nginx 用 `proxy_pass http://api:8000/`（末尾带斜杠）把 /api 前缀
# 剥掉之后才转发过来，所以本应用的路由都注册成不带 /api 的路径（比如 /cases）。
# root_path 不参与路由匹配，只告诉 FastAPI"我实际被挂在 /api 下面"，让 Swagger/OpenAPI 生成的
# 请求地址带上这个前缀——不设的话 /api/docs 能打开，但页面里 Try it out 会打到 /cases（少了
# /api），直接 404。
app = FastAPI(title="思政案例生成工作台 - 原型", root_path=os.environ.get("ROOT_PATH", "/api"))

# ---------- 登录鉴权：默认全部接口都要求登录，只有白名单里的放行 ----------
# AUTH_ENABLED=false（.env里配）可以整体关掉这层校验，方便测试阶段不用每次都登录；
# 上线前要确认.env里这个开关是打开的（默认就是true，不用特意配也行）。
#
# 【为什么是白名单而不是按路径前缀判断】
# 旧写法是 `not path.startswith("/api/") -> 放行`。那套逻辑依赖"所有业务接口都在 /api/ 下"，
# 一旦放到 nginx 后面、前缀被剥掉，后端收到的路径全都不以 /api/ 开头，于是每个请求都命中放行
# 分支——整站鉴权直接失效、所有接口变成匿名可访问。改成"默认拦、白名单放行"之后是 fail-closed 的：
# 将来新增接口忘了配也只会变成"需要登录"，而不是"意外公开"。
# 白名单同时写了带前缀和不带前缀两种形式：不带前缀是生产（nginx 剥掉后）的实际路径，带前缀是
# 本地直连 uvicorn 调试时的路径，两种跑法都能正常登录。
_PUBLIC_PATHS = {
    "/auth/login", "/api/auth/login",     # 登录接口本身，不能要求先登录
    "/health", "/api/health",             # 容器 healthcheck，不带凭证
    "/", "/api/",                         # 根路径的存活检查
    "/docs", "/api/docs", "/redoc", "/api/redoc",
    "/openapi.json", "/api/openapi.json",
}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not AUTH_ENABLED:
        return await call_next(request)
    # 用 scope["path"] 而不是 url.path：两者在有 root_path 时含义可能不同，scope["path"] 始终是
    # 应用实际用来匹配路由的那个路径，跟白名单比对才不会错位
    path = request.scope.get("path", request.url.path)
    if request.method == "OPTIONS" or path in _PUBLIC_PATHS:
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


# 【CORS 默认关闭】生产环境 nginx 把前端静态页和 /api 反代放在同一个 origin 下，浏览器视角下
# 前后端同源，压根不需要 CORS。所以这里改成"只有显式设置了 CORS_ORIGINS 环境变量才注册这个
# 中间件"——生产不设=零 CORS 开销，也不会因为配错白名单把接口暴露给别的站点。
# 保留这个开关是为了本地开发时可以不走 nginx、直接 uvicorn + 另起一个静态服务器调试，
# 那种跑法是跨源的，设一下 CORS_ORIGINS 就能用。
#
# 注册顺序有讲究：必须在上面 auth 中间件之后注册（Starlette 是"后注册的在更外层"），这样
# auth 返回的 401 才会经过 CORS 中间件、带上 Access-Control-Allow-Origin 头。否则跨源场景下
# 浏览器会把没有 CORS 头的 401 判成 CORS 错误，连状态码都不给 JS 读，前端就没法识别"要重新登录"。
_cors_origins_env = os.environ.get("CORS_ORIGINS", "").strip()
if _cors_origins_env:
    CORS_ORIGINS = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(f"已启用 CORS，允许的来源: {CORS_ORIGINS}")

init_db()
# 后台任务是进程内线程池跑的，进程一重启就都没了；把数据库里还停在pending/running的旧任务
# 标记成失败，免得前端轮询到一个永远不会变的"进行中"
reset_stale_jobs_on_startup()
# ChromaDB/LlamaIndex 的连接是懒加载的（第一次真正建索引/检索时才连），
# 这里不用单独预热，连不上也不会拖垮后端启动，只会在用到时报错。


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/auth/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == req.username, User.is_active.is_(True)).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "用户名或密码错误")
    token = create_session_token(db, user)
    return {"token": token, "username": user.username}


@app.post("/auth/logout")
def logout(request: Request, db: Session = Depends(get_db)):
    auth_header = request.headers.get("Authorization", "")
    token = auth_header[7:] if auth_header.startswith("Bearer ") else ""
    delete_session_token(db, token)
    return {"logged_out": True}


@app.get("/auth/me")
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

@app.post("/materials/import")
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


@app.post("/materials/upload")
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


@app.get("/materials")
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


@app.get("/materials/case_codes")
def list_material_case_codes(db: Session = Depends(get_db)):
    """素材库里已经出现过的案例编号（供前端做筛选下拉），外加一个建议的"下一个案例编号"默认值"""
    rows = db.query(RawMaterial.case_code).distinct().all()
    codes = sorted({r[0] for r in rows if r[0]})
    return {"case_codes": codes, "next_case_code": _compute_next_case_code(codes)}


@app.delete("/materials/{material_id}")
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

@app.get("/dimensions")
def get_dimensions():
    return DIMENSIONS


@app.post("/cases/generate", status_code=202)
def generate_case(req: GenerateCaseRequest, request: Request, db: Session = Depends(get_db)):
    """提交一个案例生成任务，立刻返回job_id，不等生成跑完。

    案例正文要走writer/judge/reviser评审循环，通常5-20分钟，同步等待会把成败绑死在这条
    HTTP连接上（断线/刷新就再也拿不到结果）。这里只做参数校验+入队，实际生成在
    job_queue的线程池里跑，调用方之后轮询 GET /api/jobs/{job_id} 看进度。"""
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

    user = getattr(request.state, "user", None)
    job_id = submit_generate_job(
        req.case_code, [m.id for m in success_materials],
        requested_by=user.id if user else None,
    )
    return {"job_id": job_id, "message": "已提交生成任务，预计5-20分钟，可轮询任务状态查看进度"}


@app.get("/cases")
def list_cases(case_code: str | None = None, db: Session = Depends(get_db)):
    query = db.query(Case)
    if case_code:
        query = query.filter(Case.case_code == case_code)
    cases = query.order_by(Case.id.desc()).all()
    return [c.to_dict() for c in cases]


@app.get("/cases/{case_id}")
def get_case(case_id: int, db: Session = Depends(get_db)):
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(404, "案例不存在")
    return case.to_dict()


@app.get("/cases/{case_id}/materials")
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


@app.put("/cases/{case_id}")
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


@app.delete("/cases/{case_id}")
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


@app.get("/cases/{case_id}/audit_log")
def get_case_audit_log(case_id: int, db: Session = Depends(get_db)):
    """查看某个案例的修改记录：谁在什么时候改了哪些字段"""
    logs = (
        db.query(CaseAuditLog)
        .filter(CaseAuditLog.case_id == case_id)
        .order_by(CaseAuditLog.created_at.desc())
        .all()
    )
    return [log.to_dict() for log in logs]


@app.post("/cases/export")
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

    case_dicts = [c.to_dict() for c in cases]
    tree_pngs = render_mermaid_batch([build_applicable_courses_mermaid(cd) for cd in case_dicts])

    doc = docx.Document()
    set_default_font(doc)
    doc.add_heading("思政案例集（导出）", level=0)

    for cd, png in zip(case_dicts, tree_pngs):
        write_case_section(doc, cd, heading_level=1, course_tree_png=png)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=cases_export.docx"},
    )


# ---------- AI 助手（agent对话，替代原来的案例工作台手动编辑） ----------

@app.get("/chat/sessions")
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


@app.post("/chat/sessions")
def create_chat_session(db: Session = Depends(get_db)):
    session = ChatSession(title="新对话", messages="[]")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session.to_dict()


@app.get("/chat/sessions/{session_id}")
def get_chat_session(session_id: int, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(404, "会话不存在")
    return session.to_dict()


@app.delete("/chat/sessions/{session_id}")
def delete_chat_session(session_id: int, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(404, "会话不存在")
    db.delete(session)
    db.commit()
    return {"deleted": True}


@app.post("/chat")
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

@app.post("/knowledge/upload")
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


@app.get("/knowledge")
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


@app.get("/knowledge/courses")
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


@app.post("/knowledge")
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


@app.delete("/knowledge")
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


@app.put("/knowledge/{point_id}")
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


@app.delete("/knowledge/{point_id}")
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


@app.post("/cases/{case_id}/match_knowledge", status_code=202)
def match_knowledge(case_id: int, request: Request, db: Session = Depends(get_db)):
    """提交一个知识点匹配任务（向量+关键词混合粗筛 + LLM复核精排），立刻返回job_id。

    实际匹配在后台线程池里跑，完成后结果直接写进case_knowledge_mappings表；调用方轮询到
    这个job变成done之后，用 GET /api/cases/{case_id}/knowledge_mappings 读结果即可
    （不在任务表里重复存一份匹配结果）。"""
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(404, "案例不存在")

    if db.query(KnowledgePoint).count() == 0:
        raise HTTPException(400, "还没有导入任何知识点，请先在「知识点匹配」标签页上传课程教学大纲")

    user = getattr(request.state, "user", None)
    job_id = submit_match_knowledge_job(case_id, requested_by=user.id if user else None)
    return {"job_id": job_id, "message": "已提交匹配任务，完成后候选列表会自动刷新"}


@app.get("/jobs")
def list_jobs(status: str = "pending,running", limit: int = 20, db: Session = Depends(get_db)):
    """列出后台任务，默认只列还在进行中的（前端每隔几秒轮询这个接口刷新顶部进度条）。
    status可以传逗号分隔的多个状态，比如 "pending,running" 或 "done,failed"。
    不带result_case完整内容——轮询很频繁，案例正文两三千字，每次都带上纯属浪费带宽。"""
    wanted = [s.strip() for s in status.split(",") if s.strip()]
    query = db.query(BackgroundJob)
    if wanted:
        query = query.filter(BackgroundJob.status.in_(wanted))
    jobs = query.order_by(BackgroundJob.created_at.desc()).limit(limit).all()
    return [j.to_dict() for j in jobs]


@app.get("/jobs/{job_id}")
def get_job(job_id: int, db: Session = Depends(get_db)):
    """查单个后台任务的状态。done的生成任务会带上完整的result_case内容。"""
    job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "任务不存在")
    return job.to_dict(include_result_case=True)


@app.get("/cases/{case_id}/knowledge_mappings")
def get_knowledge_mappings(case_id: int, db: Session = Depends(get_db)):
    mappings = (
        db.query(CaseKnowledgeMapping)
        .filter(CaseKnowledgeMapping.case_id == case_id)
        .all()
    )
    return [m.to_dict() for m in mappings]


@app.put("/knowledge_mappings/{mapping_id}")
def update_mapping(mapping_id: int, req: UpdateMappingRequest, request: Request, db: Session = Depends(get_db)):
    """人工修改：采纳/拒绝某条知识点关联，或编辑融入方式建议文字。
    只要这次修改让案例的"已采纳"知识点集合发生了变化（新采纳一条、或者把已采纳的改成别的
    状态），就自动重新生成一遍"适用课程举例"/"教学设计"。

    这个重新生成涉及真实LLM调用（几十秒），所以是提交成后台任务、本接口立刻返回，不再让
    "改个下拉框状态"这么个小操作卡在一次模型调用上。触发了的话返回体里会多一个
    enrich_job_id，前端拿它去轮询，完成后再刷新案例内容。"""
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
        user = getattr(request.state, "user", None)
        result["enrich_job_id"] = submit_enrich_job(
            mapping.case_id, requested_by=user.id if user else None
        )

    return result


@app.delete("/knowledge_mappings/{mapping_id}")
def delete_knowledge_mapping(mapping_id: int, request: Request, db: Session = Depends(get_db)):
    """解绑一条案例↔知识点关联。如果这条关联当时是"已采纳"状态，解绑后案例的"已采纳"集合
    也跟着变了，同样要重新生成一遍"适用课程举例"/"教学设计"（如果解绑后已采纳集合空了，
    这两个字段会被清空，不会留着解绑前的旧内容）——跟PUT那个接口一样走后台任务，
    返回体里带enrich_job_id给前端轮询。"""
    mapping = db.query(CaseKnowledgeMapping).filter(CaseKnowledgeMapping.id == mapping_id).first()
    if not mapping:
        raise HTTPException(404, "关联记录不存在")

    was_accepted = mapping.status == "已采纳"
    case_id = mapping.case_id
    db.delete(mapping)
    db.commit()

    result = {"unbound": True}
    if was_accepted:
        user = getattr(request.state, "user", None)
        result["enrich_job_id"] = submit_enrich_job(case_id, requested_by=user.id if user else None)

    return result


# ---------- 知识图谱 ----------

@app.get("/knowledge_graph.png")
def get_knowledge_graph_png(db: Session = Depends(get_db)):
    """维度→案例→知识点 三层关系图的静态图片，只统计已采纳的案例与已采纳的知识点关联"""
    graph = build_graph(db)
    png_bytes = render_graph_png(graph)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/knowledge_graph.html")
def get_knowledge_graph_html(db: Session = Depends(get_db)):
    """同一份图谱的可交互HTML版本（可拖拽缩放），单独打开查看"""
    graph = build_graph(db)
    html = render_graph_html(graph)
    return HTMLResponse(content=html)


# ---------- 成书编译 ----------

@app.get("/book/export")
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


@app.get("/health")
def health():
    """容器存活探针。compose 的 healthcheck 直接打容器内的 http://localhost:8000/health
    （不经过 nginx，所以路径不带 /api）；从浏览器则是 https://域名/api/health。
    故意只返回静态 JSON、不碰数据库——这个探针要回答的是"进程还活着吗"，如果把数据库
    连通性也算进来，MySQL 抖一下就会导致 api 容器被判定不健康、进而被 restart，
    但其实进程本身好好的，重启只会让问题更糟。"""
    return {"status": "ok"}
