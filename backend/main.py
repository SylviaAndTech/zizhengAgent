"""
FastAPI 主入口
提供：素材导入、案例生成、案例查询/编辑/审核 等接口
运行方式: uvicorn main:app --reload --port 8000
"""
import io
import json
import logging
import os

import docx
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import (
    init_db, get_db, RawMaterial, Case, CaseMaterial, DIMENSIONS,
    KnowledgePoint, CaseKnowledgeMapping, ChatSession, CaseAuditLog,
)
from fetch_material import fetch_url_text
from generate_case import generate_case_draft, enrich_case_with_knowledge
from parse_document import parse_uploaded_document
from knowledge_matching import extract_knowledge_points, match_case_to_knowledge, index_knowledge_point
from chat_agent import run_chat
from audit import log_case_change
from doc_writer import write_case_section
from knowledge_graph import build_graph, render_graph_png, render_graph_html
from book_export import build_book_docx
from material_index import index_material

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="思政案例生成工作台 - 原型")

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


# ---------- 请求体模型 ----------

class ImportMaterialsRequest(BaseModel):
    case_code: str
    urls: list[str]


class GenerateCaseRequest(BaseModel):
    case_code: str
    dimension: str
    material_ids: list[int]


class UpdateCaseRequest(BaseModel):
    title: str | None = None
    full_narrative: str | None = None
    teaching_objectives: dict | None = None
    sizheng_elements: dict | None = None
    applicable_courses: list | None = None
    teaching_design: dict | None = None
    evaluation: dict | None = None
    further_reading: list | None = None
    status: str | None = None


class ExportCasesRequest(BaseModel):
    case_ids: list[int]


class ChatSendRequest(BaseModel):
    session_id: int
    message: str
    case_code: str | None = None
    dimension: str | None = None


class UpdateMappingRequest(BaseModel):
    status: str | None = None
    suggestion_text: str | None = None


# ---------- 素材相关接口 ----------

@app.post("/api/materials/import")
def import_materials(req: ImportMaterialsRequest, db: Session = Depends(get_db)):
    """批量导入URL并抓取正文，落盘存档"""
    results = []
    for url in req.urls:
        url = url.strip()
        if not url:
            continue
        fetched = fetch_url_text(url)
        material = RawMaterial(
            case_code=req.case_code,
            url=url,
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


@app.post("/api/materials/upload")
def upload_material(
    case_code: str = Form(...),
    dimension: str = Form(""),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """上传Word/PDF，提取正文并尝试按标题拆分成多个候选案例，分别落入素材库"""
    filename = file.filename or ""
    lower_name = filename.lower()
    if not (lower_name.endswith(".docx") or lower_name.endswith(".pdf")):
        raise HTTPException(400, "仅支持 .docx 或 .pdf 文件（不支持旧版 .doc）")

    file_bytes = file.file.read()
    source_type = "docx" if lower_name.endswith(".docx") else "pdf"

    try:
        segments = parse_uploaded_document(filename, file_bytes)
    except Exception as e:
        raise HTTPException(400, f"文档解析失败: {str(e)}")

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
def list_materials(case_code: str, db: Session = Depends(get_db)):
    """按案例编号查询已导入的素材"""
    materials = db.query(RawMaterial).filter(RawMaterial.case_code == case_code).all()
    return [_material_to_dict(m) for m in materials]


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
        draft = generate_case_draft(req.case_code, req.dimension, material_payload)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"生成失败: {str(e)}")

    case = Case(
        case_code=req.case_code,
        dimension=req.dimension,
        title=draft.get("title"),
        full_narrative=draft.get("full_narrative"),
        teaching_objectives=json.dumps(draft.get("teaching_objectives"), ensure_ascii=False),
        sizheng_elements=json.dumps(draft.get("sizheng_elements"), ensure_ascii=False),
        applicable_courses=json.dumps(draft.get("applicable_courses"), ensure_ascii=False),
        teaching_design=json.dumps(draft.get("teaching_design"), ensure_ascii=False),
        evaluation=json.dumps(draft.get("evaluation"), ensure_ascii=False),
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
    """给指定会话追加一条用户消息，跑一轮agent对话，把结果落回这个会话"""
    session = db.query(ChatSession).filter(ChatSession.id == req.session_id).first()
    if not session:
        raise HTTPException(404, "会话不存在，请先创建一个新对话")

    history = json.loads(session.messages) if session.messages else []
    context_note = f"[当前案例编号：{req.case_code or '未指定'}，思政维度：{req.dimension or '未指定'}]\n"

    try:
        result = run_chat(history, context_note + req.message)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"AI助手出错: {str(e)}")

    session.messages = json.dumps(result["messages"], ensure_ascii=False)
    if session.title == "新对话" and req.message.strip():
        session.title = req.message.strip()[:20]
    db.commit()
    db.refresh(session)

    return {"session": session.to_dict(), "reply": result["reply"]}


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

    for file in files:
        filename = file.filename or "(未命名文件)"
        if not (filename.lower().endswith(".docx") or filename.lower().endswith(".pdf")):
            errors.append(f"{filename}：仅支持 .docx 或 .pdf 文件")
            continue

        file_bytes = file.file.read()
        try:
            detected_course_name, points = extract_knowledge_points(filename, file_bytes, course_name)
        except Exception as e:
            errors.append(f"{filename}：解析失败 {str(e)}")
            continue

        if not points:
            errors.append(f"{filename}：没有提取到可用的知识点条目")
            continue

        for p in points:
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
                logger.warning(f"知识点(id={kp.id})索引失败，匹配时暂时召回不到它: {e}")
            created.append({
                "id": kp.id, "course_name": kp.course_name,
                "chapter": kp.chapter, "description": kp.description,
                "source_filename": kp.source_filename,
            })

    if not created and errors:
        raise HTTPException(400, "；".join(errors))

    return {"imported": created, "count": len(created), "errors": errors}


@app.get("/api/knowledge")
def list_knowledge(course_name: str | None = None, db: Session = Depends(get_db)):
    query = db.query(KnowledgePoint)
    if course_name:
        query = query.filter(KnowledgePoint.course_name == course_name)
    points = query.order_by(KnowledgePoint.id.desc()).all()
    return [
        {
            "id": k.id, "course_name": k.course_name, "chapter": k.chapter,
            "description": k.description, "source_filename": k.source_filename,
        }
        for k in points
    ]


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
    """人工修改：采纳/拒绝某条知识点关联，或编辑融入方式建议文字"""
    mapping = db.query(CaseKnowledgeMapping).filter(CaseKnowledgeMapping.id == mapping_id).first()
    if not mapping:
        raise HTTPException(404, "关联记录不存在")
    if req.status is not None:
        mapping.status = req.status
    if req.suggestion_text is not None:
        mapping.suggestion_text = req.suggestion_text
    db.commit()
    db.refresh(mapping)
    return mapping.to_dict()


@app.post("/api/cases/{case_id}/enrich")
def enrich_case(case_id: int, db: Session = Depends(get_db)):
    """根据已采纳的知识点关联，调用模型补充案例的「适用课程举例」与「教学设计」"""
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(404, "案例不存在")

    accepted = (
        db.query(CaseKnowledgeMapping)
        .filter(CaseKnowledgeMapping.case_id == case_id, CaseKnowledgeMapping.status == "已采纳")
        .all()
    )
    if not accepted:
        raise HTTPException(400, "还没有已采纳的知识点关联，请先在匹配表格里勾选采纳")

    accepted_payload = [
        {
            "course_name": m.knowledge_point.course_name,
            "chapter": m.knowledge_point.chapter,
            "description": m.knowledge_point.description,
            "suggestion_text": m.suggestion_text,
        }
        for m in accepted
    ]

    try:
        enrichment = enrich_case_with_knowledge(case.to_dict(), accepted_payload)
    except ValueError as e:
        raise HTTPException(400, str(e))

    changes = {}
    if "applicable_courses" in enrichment:
        new_value = json.dumps(enrichment["applicable_courses"], ensure_ascii=False)
        if case.applicable_courses != new_value:
            changes["applicable_courses"] = {"old": case.applicable_courses, "new": new_value}
        case.applicable_courses = new_value
    if "teaching_design" in enrichment:
        new_value = json.dumps(enrichment["teaching_design"], ensure_ascii=False)
        if case.teaching_design != new_value:
            changes["teaching_design"] = {"old": case.teaching_design, "new": new_value}
        case.teaching_design = new_value
    db.commit()
    db.refresh(case)
    if changes:
        log_case_change(db, case.id, "AI助手(知识点补充)", changes)
    return case.to_dict()


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
