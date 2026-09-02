"""
Writer→Judge→Reviser 多Agent循环的共享状态定义。

LangGraph的核心是一个在各节点间流转的state（dict）。所有agent读写同一个state，
这样"写作者→评审者→修订者"之间才能传递信息（正文、打分、修改清单等）。

跟参考设计里的 agent_state.py 相比，这里按生产环境实际的输入输出做了调整：
- 用已经结构化好的 facts/narrative_arc（在 generate_case.py 里作为独立步骤先跑一次）
  作为输入，而不是一整段拼接好的 materials_text——事实提炼不是这个循环的职责。
- 额外加了 first_draft：单独保留写作者第一次的输出，因为 Case 表的
  full_narrative_draft 字段语义一直是"改写前的初稿"，循环引入后要继续用这个字段
  承接同样的语义，不能跟收敛后的最终 narrative 混在一起覆盖掉。
- 额外加了 structure_errors：纯Python结构校验（引用标注格式、两段式结构、篇幅）的
  结果，跟 LLM judge 的 rubric 打分分开存，因为这层检查不依赖模型自觉，路由逻辑里
  两者要分别判断。
- scores/issues/veto_triggered/... 等评审细节按约定不落库，只存在这个内存态的
  state 里，供进度回调（history）展示用，生成流程结束后就跟着这次调用一起丢弃。
"""
from typing import TypedDict, Optional
from typing_extensions import Annotated
import operator


class CaseState(TypedDict, total=False):
    # ---- 输入 ----
    case_code: str
    facts: dict                # 事实骨架（generate_case._extract_facts的输出，循环开始前就已经算好）
    narrative_arc: Optional[str]   # 行文脉络建议（generate_case._suggest_narrative_arc的输出）

    # ---- Writer 的产出 ----
    narrative: str              # 当前正文（初稿或修订/重写后的最新版本）
    first_draft: str            # 写作者第一次输出，供Case.full_narrative_draft字段使用，不随后续修订变化

    # ---- Judge 的产出 ----
    scores: dict                 # 各维度打分 {criterion_id: 1-5}
    issues: dict                 # 各维度的具体问题
    veto_triggered: bool         # 是否触发事实忠实性一票否决
    suspected_fabrications: list  # 疑似编造内容清单
    revision_checklist: list     # 修订清单
    weighted_ratio: Optional[float]  # 加权得分比例
    passed: bool                 # rubric是否通过
    structure_errors: list       # validate_structure()的结果，跟rubric打分分开存

    # ---- 流程控制 ----
    iteration: Annotated[int, operator.add]  # 迭代轮次（每轮+1，累加）
    max_iterations: int          # 最大迭代轮次（防止无限循环）
    history: Annotated[list, operator.add]   # 过程记录，供进度回调展示，不落库
