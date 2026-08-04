"""
案例修改的操作留痕：小工具函数，供 main.py 的直接编辑接口和 chat_agent.py 的AI助手工具共用。
"""
import json

from sqlalchemy.orm import Session

from db import CaseAuditLog


def log_case_change(db: Session, case_id: int, actor: str, changes: dict):
    """changes: {字段名: {"old":.., "new":..}, ...}，空dict时不写日志"""
    if not changes:
        return
    log = CaseAuditLog(case_id=case_id, actor=actor, changes=json.dumps(changes, ensure_ascii=False))
    db.add(log)
    db.commit()
