"""
数据库模型定义
对应设计文档中的：RawMaterial（原始素材）、Case（案例）
关系型数据存 MySQL；知识点/素材内容的向量另外存 ChromaDB（见 knowledge_matching.py / material_index.py，
用LlamaIndex管理），这里只留主键关联。
"""
import datetime
import json
import logging
import os

from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Boolean, text
)
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

load_dotenv()

DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = os.environ.get("DB_PORT", "3306")
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "sizheng_cases")

DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)  # pool_pre_ping避免MySQL断开空闲连接后报错
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 五维度固定值，对应大纲第二至六章
DIMENSIONS = ["政治认同", "家国情怀", "文化素养", "宪法法治意识", "道德修养"]


class RawMaterial(Base):
    """原始素材：一条URL对应的抓取记录，是案例真实性的证据链"""
    __tablename__ = "raw_materials"

    id = Column(Integer, primary_key=True, index=True)
    case_code = Column(String(20), index=True)  # 如 "3.4"
    url = Column(Text, nullable=True)  # 上传文件拆分出的素材没有URL
    source_title = Column(Text)  # 用户粘贴时附带的原始标题，或从文档中拆分出的段落标题
    fetched_text = Column(LONGTEXT)  # 抓取/提取到的正文快照（保留全文，不做截断），供追溯核实
    fetch_status = Column(String(20), default="pending")  # pending/success/failed
    fetch_error = Column(Text, nullable=True)
    source_type = Column(String(10), default="url")  # url/docx/pdf
    source_filename = Column(Text, nullable=True)  # 上传文件的原始文件名
    segment_index = Column(Integer, nullable=True)  # 同一份文档拆分出的第几段案例（从0开始）
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    cases = relationship("CaseMaterial", back_populates="material")


class Case(Base):
    """案例：对应大纲里"案例X.X"的完整7段式结构"""
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, index=True)
    case_code = Column(String(20), index=True)  # 如 "3.4"
    dimension = Column(String(20))  # 五维度之一
    title = Column(Text)  # 生成的案例标题（要求不能与网上原文重复）

    # 以下字段存 JSON 字符串，简化SQLite schema，读取时用 json.loads
    full_narrative = Column(Text)          # 完整案例（去AI味改写后的定稿，故事化叙述，含句级引用标注）
    full_narrative_draft = Column(Text)    # 去AI味改写之前的正文初稿，跟full_narrative一起给用户对照查看
    teaching_objectives = Column(Text)     # 案例教学目标 {"知识":..,"能力":..,"素养":..}
    sizheng_elements = Column(Text)        # 课程思政元素，按五维度摘录 {dimension: text}
    applicable_courses = Column(Text)      # 适用课程举例 [{课程名称,适用章节,融入方式建议}]
    teaching_design = Column(Text)         # 教学设计 {"课前":..,"课中":..,"课后":..}
    evaluation = Column(Text)              # 课程评价与成效 {"达成度":..,"参与度":..,"教学反思":..}
    further_reading = Column(Text)         # 延伸阅读 [{"type":..,"title":..,"url":..}]

    status = Column(String(20), default="草稿")  # 草稿/待审核/已采纳/已驳回
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    materials = relationship("CaseMaterial", back_populates="case")

    def to_dict(self):
        def _load(field):
            try:
                return json.loads(field) if field else None
            except (json.JSONDecodeError, TypeError):
                return field

        return {
            "id": self.id,
            "case_code": self.case_code,
            "dimension": self.dimension,
            "title": self.title,
            "full_narrative": self.full_narrative,
            "full_narrative_draft": self.full_narrative_draft,
            "teaching_objectives": _load(self.teaching_objectives),
            "sizheng_elements": _load(self.sizheng_elements),
            "applicable_courses": _load(self.applicable_courses),
            "teaching_design": _load(self.teaching_design),
            "evaluation": _load(self.evaluation),
            "further_reading": _load(self.further_reading),
            "status": self.status,
            "source_material_ids": [m.material_id for m in self.materials],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class CaseMaterial(Base):
    """案例 与 原始素材 的多对多关联表——即案例的证据链"""
    __tablename__ = "case_materials"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"))
    material_id = Column(Integer, ForeignKey("raw_materials.id"))

    case = relationship("Case", back_populates="materials")
    material = relationship("RawMaterial", back_populates="cases")


class KnowledgePoint(Base):
    """从上传的课程教学大纲里拆解出的知识点条目"""
    __tablename__ = "knowledge_points"

    id = Column(Integer, primary_key=True, index=True)
    course_name = Column(String(100), index=True)  # 课程名称，如"人工智能导论"
    chapter = Column(Text, nullable=True)  # 所属章节/单元标题
    description = Column(Text, nullable=False)  # 知识点描述原文
    source_filename = Column(Text, nullable=True)  # 来源大纲文件名
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    # 描述文本的向量不存在这里，存在 ChromaDB 的 knowledge_point_vectors 集合里，用本表的 id 作为主键对应


class CaseKnowledgeMapping(Base):
    """案例 与 知识点 的关联建议，对应大纲附录二的表格，需人工审核"""
    __tablename__ = "case_knowledge_mappings"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"))
    knowledge_point_id = Column(Integer, ForeignKey("knowledge_points.id"))
    relevance_score = Column(Integer, default=0)  # 0-100，基于字符相似度的粗略打分
    suggestion_text = Column(Text, nullable=True)  # 融入方式建议，可人工编辑
    status = Column(String(20), default="推荐")  # 推荐/已采纳/已拒绝
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    case = relationship("Case")
    knowledge_point = relationship("KnowledgePoint")

    def to_dict(self):
        return {
            "id": self.id,
            "case_id": self.case_id,
            "knowledge_point_id": self.knowledge_point_id,
            "course_name": self.knowledge_point.course_name if self.knowledge_point else None,
            "chapter": self.knowledge_point.chapter if self.knowledge_point else None,
            "description": self.knowledge_point.description if self.knowledge_point else None,
            "relevance_score": self.relevance_score,
            "suggestion_text": self.suggestion_text,
            "status": self.status,
        }


class ChatSession(Base):
    """AI助手的一段对话历史，支持多个会话并行存在、随时切换查看"""
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(100), default="新对话")
    messages = Column(Text, default="[]")  # JSON序列化的Anthropic messages格式历史
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "messages": json.loads(self.messages) if self.messages else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class CaseAuditLog(Base):
    """案例修改的操作留痕：谁（AI助手/用户直接编辑等）在什么时候改了哪些字段"""
    __tablename__ = "case_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), index=True)
    actor = Column(String(50), default="用户")  # 例如："用户" / "AI助手"
    changes = Column(Text, nullable=False)  # JSON: {字段名: {"old":.., "new":..}, ...}
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "case_id": self.case_id,
            "actor": self.actor,
            "changes": json.loads(self.changes) if self.changes else {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class User(Base):
    """登录账号。现在只有一个固定的admin账号，但按多用户设计——以后加新用户只是插入
    一行数据，不用改代码或表结构。"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(200), nullable=False)  # 格式见 auth.py 的 hash_password()
    is_active = Column(Boolean, default=True)  # 停用账号用这个字段软删除，不用真的删行
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {"id": self.id, "username": self.username, "is_active": self.is_active}


class UserSession(Base):
    """登录会话：服务端保存token，前端登录后把token放Authorization请求头里带上
    （不用cookie——前端和后端在开发环境是两个不同origin，跨源cookie在纯HTTP下受
    SameSite/Secure策略限制很难用；token方案不受这个限制，实现也更简单）。"""
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)


def init_db():
    Base.metadata.create_all(bind=engine)
    _migrate_fetched_text_to_longtext()
    _migrate_add_full_narrative_draft()
    _seed_default_admin()


def _seed_default_admin():
    """确保至少有一个admin账号能登录——幂等，重复调用（每次启动都会调）不会出错，
    也不会覆盖已存在账号的密码，避免每次重启服务器都把管理员自己改过的密码悄悄重置回默认值。
    实际的哈希逻辑在auth.py（这里延迟import，避免db.py和auth.py出现循环依赖：
    auth.py里也需要用到本文件的User/UserSession）。"""
    from auth import DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD, hash_password
    db = SessionLocal()
    try:
        exists = db.query(User).filter(User.username == DEFAULT_ADMIN_USERNAME).first()
        if not exists:
            db.add(User(username=DEFAULT_ADMIN_USERNAME, password_hash=hash_password(DEFAULT_ADMIN_PASSWORD)))
            db.commit()
    finally:
        db.close()


def _migrate_fetched_text_to_longtext():
    """create_all不会修改已存在表的列类型；旧库的fetched_text可能还是TEXT(65535字节上限，
    正文长一点的网页会被截断)，这里补一次ALTER把它放宽到LONGTEXT。已经是LONGTEXT时该语句是空操作。"""
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE raw_materials MODIFY COLUMN fetched_text LONGTEXT"))
    except Exception as e:
        logging.getLogger("uvicorn.error").warning(f"fetched_text列类型迁移失败（不影响启动）: {e}")


def _migrate_add_full_narrative_draft():
    """create_all不会给已存在的表补新列——旧库的cases表还没有full_narrative_draft这一列，
    这里补一次ALTER；已经有这一列时ALTER会报错，用try/except吞掉（跟上面fetched_text那个
    迁移一样的写法），不影响启动。"""
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE cases ADD COLUMN full_narrative_draft LONGTEXT"))
    except Exception as e:
        logging.getLogger("uvicorn.error").info(f"full_narrative_draft列已存在或迁移失败（不影响启动）: {e}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
