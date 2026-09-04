"""
三类慢LLM操作的后台任务队列：案例生成、知识点匹配、用已采纳知识点补充案例。

为什么要有这个模块：这三个操作都要调大模型，快的几十秒、慢的（案例生成的写作-评审循环）
十几分钟。原来它们都是"在处理HTTP请求的那个线程里同步跑完"，结果是整条链路的成败绑死在
一条HTTP/SSE连接的生命周期上——用户刷新页面、网络抖一下、关掉标签页，前端就再也拿不到
结果了（后台其实还在跑、数据也可能已经写进去了，但用户完全看不到，容易误以为失败而重复
触发，白白再花一次时间和模型调用成本）。改成任务队列后：提交立刻返回job_id，实际执行在
线程池里跑，进度写数据库，前端/聊天agent之后随时轮询——不怕断连、刷新、关浏览器。

为什么用进程内线程池而不是Celery/RQ+Redis：这个项目就一两个用户，为此多运维一套消息中间件
不划算。代价是后端进程重启会丢失正在跑的任务（不支持断点续跑），所以启动时会把数据库里还
停在pending/running的旧任务统一标记成failed（见reset_stale_jobs_on_startup），不让它们在
前端显示成一个永远不动的"进行中"。

DB session的用法有两条讲究，都跟"这是在线程池的工作线程里跑"有关：
1. 绝对不能复用调用方（HTTP请求）的session——那个session的生命周期跟着请求走，请求早就
   结束关掉了，这里还在用就会报错。每个任务函数自己开自己的。
2. 真正耗时的那段LLM调用期间不要占着session（尤其是十几分钟的案例生成）——一直占着等于
   一直占着连接池里的一条MySQL连接，纯属浪费。所以generate任务是"短开一次读素材→关掉→
   跑LLM（不持有session）→再短开一次写结果"，而不是从头到尾抱着一个session。
"""
import concurrent.futures
import json
import logging
import os

from audit import log_case_change
from db import (
    SessionLocal, BackgroundJob, Case, CaseMaterial, CaseKnowledgeMapping, RawMaterial,
    allocate_case_version_code,
)

logger = logging.getLogger("uvicorn.error")

# 默认2：案例生成单次就是好几个max_tokens=8000的重模型调用，并发开太多既容易触发服务商的
# 限流，也会让本来就慢的每个任务互相拖慢。真需要更高并发再调这个环境变量。
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_GENERATION_JOBS", "2"))

_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_JOBS, thread_name_prefix="bg-job"
)


def _update_job(job_id: int, **fields):
    """短开一个session更新任务状态就立刻关掉——任务本身可能要跑十几分钟，不能为了偶尔
    改一次状态就一直占着连接。这个函数自己吞掉异常：状态更新失败不该让整个任务崩掉
    （任务本身的业务结果比这条进度记录重要得多）。"""
    db = SessionLocal()
    try:
        job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()
        if not job:
            return
        for k, v in fields.items():
            setattr(job, k, v)
        db.commit()
    except Exception as e:
        logger.warning(f"更新任务{job_id}状态失败（不影响任务本身继续跑）: {e}")
    finally:
        db.close()


def reset_stale_jobs_on_startup():
    """进程重启后，之前pending/running的任务已经没有工作线程在跑了（线程池是进程内的，
    随进程一起没了），统一标成failed，避免前端轮询到一个永远不会变的"进行中"。
    main.py启动时调用一次。"""
    db = SessionLocal()
    try:
        stale = (
            db.query(BackgroundJob)
            .filter(BackgroundJob.status.in_(["pending", "running"]))
            .all()
        )
        for job in stale:
            job.status = "failed"
            job.error = "后端服务重启，这个任务被中断了，请重新发起一次"
        if stale:
            db.commit()
            logger.info(f"启动清理：{len(stale)}个中断的后台任务已标记为失败")
    except Exception as e:
        logger.warning(f"启动清理旧任务失败（不影响启动）: {e}")
    finally:
        db.close()


def _create_job(job_type: str, **fields) -> int:
    db = SessionLocal()
    try:
        job = BackgroundJob(job_type=job_type, status="pending", **fields)
        db.add(job)
        db.commit()
        db.refresh(job)
        return job.id
    finally:
        db.close()


# ---------- 提交入口 ----------

def submit_generate_job(case_code: str, material_ids: list[int], requested_by: int | None = None) -> int:
    job_id = _create_job(
        "generate", case_code=case_code,
        payload=json.dumps({"material_ids": material_ids}), requested_by=requested_by,
    )
    _executor.submit(_run_generate, job_id, case_code, material_ids)
    return job_id


def submit_match_knowledge_job(case_id: int, requested_by: int | None = None) -> int:
    job_id = _create_job("match_knowledge", case_id=case_id, requested_by=requested_by)
    _executor.submit(_run_match_knowledge, job_id, case_id)
    return job_id


def submit_enrich_job(case_id: int, requested_by: int | None = None) -> int:
    job_id = _create_job("enrich", case_id=case_id, requested_by=requested_by)
    _executor.submit(_run_enrich, job_id, case_id)
    return job_id


# ---------- 实际执行（都跑在线程池的工作线程里） ----------

def _run_generate(job_id: int, case_code: str, material_ids: list[int]):
    """案例生成：读素材 → 跑写作-评审循环 → 分配版本号建Case行 → 关联素材 → 记审计日志。

    "用draft dict建Case行"这段逻辑原来在main.py的generate_case端点和chat_agent.py的
    generate_case_draft工具里各写了一遍（两份很容易改一处漏一处），现在两个入口都改成提交
    任务，这段逻辑就只剩这里一份了。
    """
    _update_job(job_id, status="running", current_stage="准备素材")
    try:
        # 第一段：短开session读素材，读完立刻关——下面跑LLM那十几分钟不持有任何数据库连接
        db = SessionLocal()
        try:
            materials = db.query(RawMaterial).filter(RawMaterial.id.in_(material_ids)).all()
            success_materials = [m for m in materials if m.fetch_status == "success"]
            if not success_materials:
                _update_job(job_id, status="failed", error="所选素材均不可用（未抓取/解析成功），无法生成")
                return
            payload = [
                {"id": m.id, "url": m.url, "title": m.source_title, "text": m.fetched_text}
                for m in success_materials
            ]
            success_ids = [m.id for m in success_materials]
        finally:
            db.close()

        # 第二段：真正耗时的LLM流水线，全程不持有session。on_stage每次都是独立的短事务，
        # 前端轮询就是靠它看到"事实提炼→正文初稿→AI评审(第1轮)→..."的实时变化
        from generate_case import generate_case_draft  # 延迟import，避免模块级循环依赖

        def _on_stage(stage: str):
            _update_job(job_id, current_stage=stage)

        draft = generate_case_draft(case_code, payload, on_stage=_on_stage)

        # 第三段：再短开一次session把结果落库
        _update_job(job_id, current_stage="写入数据库")
        db = SessionLocal()
        try:
            case = Case(
                case_code=allocate_case_version_code(db, case_code),
                dimension=(draft.get("sizheng_elements") or {}).get("对应维度"),
                title=draft.get("title"),
                full_narrative=draft.get("full_narrative"),
                full_narrative_draft=draft.get("full_narrative_draft"),
                teaching_objectives=json.dumps(draft.get("teaching_objectives"), ensure_ascii=False),
                sizheng_elements=json.dumps(draft.get("sizheng_elements"), ensure_ascii=False),
                # 适用课程举例/教学设计不在初次生成时产出，等知识点匹配采纳后才会有内容，
                # 这里留真正的NULL（不是json.dumps(None)那个字符串"null"）
                applicable_courses=None,
                teaching_design=None,
                further_reading=json.dumps(draft.get("further_reading"), ensure_ascii=False),
                status="待审核",
            )
            db.add(case)
            db.commit()
            db.refresh(case)
            for mid in success_ids:
                db.add(CaseMaterial(case_id=case.id, material_id=mid))
            db.commit()
            log_case_change(db, case.id, "AI助手", {"__created__": {"old": None, "new": "generate_case_draft"}})
            new_case_id = case.id
        finally:
            db.close()

        _update_job(job_id, status="done", current_stage=None, result_case_id=new_case_id)

    except Exception as e:
        logger.exception(f"案例生成任务{job_id}失败")
        _update_job(job_id, status="failed", error=str(e))


def _run_match_knowledge(job_id: int, case_id: int):
    """知识点匹配：向量+BM25混合粗筛 → LLM复核精排 → 清掉旧的"推荐"记录、写入新的。
    已人工采纳/拒绝的记录不动（这段保留策略跟原来main.py端点里的逻辑一致，只是搬了个地方）。
    """
    _update_job(job_id, status="running")
    db = SessionLocal()
    try:
        from knowledge_matching import match_case_to_knowledge  # 延迟import，避免循环依赖

        case = db.query(Case).filter(Case.id == case_id).first()
        if not case:
            _update_job(job_id, status="failed", error="案例不存在")
            return

        matches = match_case_to_knowledge(db, case)
        # match_case_to_knowledge内部可能顺带生成/刷新了case.topic_keywords（向量检索查询
        # 文本的缓存），这里落库；不是用户可见的内容变更，不写审计日志
        db.commit()

        # 清空旧的"推荐"记录避免重复堆积；已人工采纳/拒绝的决定予以保留
        db.query(CaseKnowledgeMapping).filter(
            CaseKnowledgeMapping.case_id == case_id,
            CaseKnowledgeMapping.status == "推荐",
        ).delete()
        db.commit()

        for m in matches:
            db.add(CaseKnowledgeMapping(
                case_id=case_id,
                knowledge_point_id=m["knowledge_point"].id,
                relevance_score=m["relevance_score"],
                suggestion_text=m["suggestion_text"],
                status="推荐",
            ))
        db.commit()

        _update_job(job_id, status="done")
    except Exception as e:
        logger.exception(f"知识点匹配任务{job_id}失败")
        _update_job(job_id, status="failed", error=str(e))
    finally:
        db.close()


def _run_enrich(job_id: int, case_id: int):
    """用已采纳的知识点关联，补充/更新案例的"适用课程举例"与"教学设计"两个字段。
    原来这段逻辑在三个地方各调用一次（chat_agent的工具、PUT/DELETE两个mapping端点），
    现在统一收敛到这里。"""
    _update_job(job_id, status="running")
    db = SessionLocal()
    try:
        from knowledge_matching import enrich_case_from_accepted_mappings  # 延迟import

        case = db.query(Case).filter(Case.id == case_id).first()
        if not case:
            _update_job(job_id, status="failed", error="案例不存在")
            return

        changes = enrich_case_from_accepted_mappings(db, case)
        if changes:
            db.commit()
            db.refresh(case)
            log_case_change(db, case.id, "知识点匹配(自动)", changes)

        _update_job(job_id, status="done")
    except Exception as e:
        logger.exception(f"知识点补充任务{job_id}失败")
        _update_job(job_id, status="failed", error=str(e))
    finally:
        db.close()
