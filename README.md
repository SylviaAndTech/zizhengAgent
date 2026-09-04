# 思政案例生成工作台 - 原型 v0.5

"案例生成工作台"模块的可运行原型，覆盖完整闭环：

**素材导入（URL抓取/Word/PDF拆分，含扫描件OCR） → 登录后在AI助手对话生成/编辑7段式案例（正文走写作者-评审者-修订者多轮打磨） → 知识点混合检索匹配补充 → 审核 → 知识图谱/成书导出（含Word内嵌图片）**

严格按照《讲好数字中国故事：人工智能类课程思政案例集》大纲里"每一个案例的框架"
（完整案例/教学目标/思政元素/适用课程/教学设计/延伸阅读）来生成，
并强制"不编造事实、不照抄原文、保留来源引用"三条规则。

## 技术栈

- **后端**：FastAPI + SQLAlchemy
- **关系数据库**：MySQL（素材/案例/知识点/会话/审计日志/用户与登录会话）
- **向量数据库**：ChromaDB（知识点描述、素材正文切块后的向量，供语义检索）
- **大模型**：走硅基流动（SiliconFlow）的 OpenAI 兼容协议，默认模型是 DeepSeek 系列（正文写作单独用更强的 Pro 版本，其余用 Flash 版本）。**注意**：环境变量历史上还叫 `DASHSCOPE_API_KEY`（最早接的是阿里云原生 DashScope，后来换了服务商但变量名没跟着改），实际要去 [硅基流动](https://cloud.siliconflow.cn) 申请 key，**不是阿里云 DashScope 控制台**
- **知识库检索**：LlamaIndex（文档切块、向量存取），封装在 ChromaDB 之上；知识点↔案例匹配是向量语义检索 + BM25关键词检索两路粗筛做RRF融合，再交给LLM复核精排（不是纯向量召回）
- **AI 助手 Agent**：LangChain（`create_agent`，内部基于 LangGraph）驱动工具调用循环，处理案例的生成/编辑/检索等业务操作
- **案例正文写作**：独立于上面AI助手agent的另一套 LangGraph 状态图——写作者(Writer)出初稿 → 评审者(Judge)按rubric打分+事实核查 → 不达标则修订者(Reviser)局部精修或打回写作者重写，循环到通过或达到轮次上限（`CASE_NARRATIVE_MAX_ITERATIONS`），所以案例生成比早期版本耗时更长（通常几分钟到十几分钟）
- **登录鉴权**：简单的账号密码+会话token机制（`auth.py`），首次启动自动建一个admin账号
- **文档处理**：`python-docx`（读写Word）、`pymupdf`（读PDF，比pypdf对中文更稳）、扫描版PDF/图片走Tesseract OCR或Qwen视觉模型兜底
- **知识图谱**：`networkx` 建图，`matplotlib` 出静态图片（含中文字体），`pyvis` 出可交互HTML
- **Mermaid图导出**：案例详情页"适用课程举例"树状图用Mermaid.js画，导出Word/成书时用Playwright起无头Chromium把同一张图渲染成图片嵌进文档
- **前端**：无框架、无构建步骤，单文件 `frontend/index.html`
- **部署**：Docker Compose 一键起 MySQL + ChromaDB + 后端 + 前端(nginx反向代理)

## 目录结构

```
prototype/
├── docker-compose.yml       # MySQL + ChromaDB + 后端 + 前端(nginx)，完整栈一键起
├── deploy/
│   └── setup_swap.sh        # 小内存云主机（2核4G这类）加swap防OOM的一次性脚本
├── backend/
│   ├── main.py                    # FastAPI 入口，所有API端点
│   ├── db.py                      # SQLAlchemy 模型定义 + MySQL 连接
│   ├── auth.py                    # 登录鉴权：账号密码校验、会话token
│   ├── qwen_client.py             # 统一的大模型客户端（硅基流动/OpenAI兼容协议）与模型配置
│   ├── llama_index_setup.py       # LlamaIndex 全局配置(Settings.llm/embed_model)
│   ├── generate_case.py           # 案例草稿生成主流程：事实提炼→写作评审循环→结构化字段
│   ├── case_agent_state.py        # 写作-评审循环的共享状态定义(LangGraph CaseState)
│   ├── case_agent_graph.py        # 写作者/评审者/修订者三节点状态图 + 路由逻辑
│   ├── style_rubric.py            # 评审用的文笔rubric打分表 + 引用格式/篇幅的硬性结构校验
│   ├── case_narrative_examples.py # 案例正文写作的few-shot范文库
│   ├── prompts.py                 # 所有系统提示词模板
│   ├── chat_agent.py              # AI助手：LangChain agent + 业务工具(生成/编辑/检索案例)
│   ├── fetch_material.py          # URL 正文抓取（trafilatura）
│   ├── parse_document.py          # 上传的 Word/PDF 解析 + 按标题拆分候选案例
│   ├── ocr_utils.py               # 扫描版PDF/图片判断 + OCR调用的公共工具
│   ├── syllabus_table_ocr.py      # 教学大纲表格版式的OCR拆解
│   ├── syllabus_vision_ocr.py     # 教学大纲用Qwen视觉模型做整页OCR的兜底方案
│   ├── knowledge_matching.py      # 知识点抽取 + 向量/BM25混合粗筛 + LLM复核精排
│   ├── material_index.py          # 素材正文切块/向量化/存入ChromaDB + 语义检索（供AI助手用）
│   ├── knowledge_graph.py         # 总知识图谱：networkx建图 + 静态图片/可交互HTML
│   ├── mermaid_tree.py            # 单个案例"课程↔知识点"树状图的Mermaid定义生成
│   ├── mermaid_render.py          # 用Playwright无头浏览器把Mermaid图渲染成PNG
│   ├── book_export.py             # 成书编译：前言+按维度分章+附录+图谱+Mermaid图
│   ├── book_front_matter.py       # 成书前言等固定编排文案（占位模板，出版前需人工替换）
│   ├── doc_writer.py              # 案例七段式写入Word的共用逻辑
│   ├── audit.py                   # 案例修改留痕的写入工具
│   ├── Dockerfile                 # 后端镜像：Python 3.12 + tesseract + 中文字体 + Playwright
│   ├── .dockerignore
│   ├── requirements.txt
│   └── .env / .env.example
└── frontend/
    ├── index.html                # 单页前端：💬AI助手 / 📂素材库 / 📚案例库 / 🧩知识点匹配
    ├── Dockerfile                # 前端镜像：nginx托管静态页面
    └── nginx.conf                # 静态文件 + /api反向代理到后端容器
```

## 快速开始

有两种跑法：**方式一**适合正式部署到服务器（前后端都在容器里，一条命令起完整栈）；
**方式二**适合本地改代码调试（后端`--reload`热更新，前端直接改完刷新页面就看到）。
不管哪种方式，第一步都是先去申请Key、填好 `.env`。

### 0. 准备API Key + 配置 `.env`

```bash
cd backend
cp .env.example .env
```

打开 `backend/.env`，至少要填一个东西：

```bash
DASHSCOPE_API_KEY=你的密钥
```

**这个key要去 [硅基流动 SiliconFlow](https://cloud.siliconflow.cn) 申请，不是阿里云** ——
变量名叫`DASHSCOPE_API_KEY`是历史遗留（详见上面"技术栈"里的说明），去阿里云申请的key在这里用不了。

### 方式一：Docker 一键部署（推荐用于服务器）

```bash
cd prototype
docker compose up -d --build
```

会拉起4个容器：`mysql`、`chroma`、`backend`、`frontend`。首次构建要下载依赖+Playwright的
Chromium内核，会比较慢。跑起来后确认一下：

```bash
docker compose ps          # 4个容器都应该是 healthy/running
```

浏览器直接访问 **http://<服务器IP或localhost>/** 就是完整应用（前端nginx托管静态页面，
`/api`自动反代到后端容器，浏览器视角下前后端同源，不用管CORS）。

MySQL/ChromaDB的数据分别存在 `./volumes/mysql`、`./volumes/chroma`，`docker compose down`
不会清掉这两个目录（加 `-v` 才会，谨慎使用）。生产环境这两个服务默认不对公网暴露端口，
只有 `frontend` 的80端口对外，如需临时连数据库调试，去 `docker-compose.yml` 里按注释打开
`127.0.0.1:xxxx:xxxx`形式的端口映射（不要绑`0.0.0.0`）。

如果你的服务器是2核4G这类小内存云主机，建议先跑一次
`sudo bash deploy/setup_swap.sh` 加一块swap，防止大文件上传/扫描件OCR这类瞬时内存
峰值被OOM killer杀掉进程。

### 方式二：本地开发模式（分别起服务，方便改代码）

#### 1. 启动 MySQL + ChromaDB

```bash
cd prototype
docker compose up -d mysql chroma
```

只起这两个依赖服务，不建后端/前端镜像。首次拉镜像可能比较慢，跑起来后确认：

```bash
docker compose ps mysql chroma   # 都应该是 healthy
```

MySQL 的 `sizheng_cases` 库、账号 `sizheng`/密码 `sizheng_pass` 会在容器首次启动时自动建好。
ChromaDB 的两个 collection（`knowledge_point_vectors`、`material_chunks`）不用手动建，后端
第一次真正用到时会自动建。

注意 ChromaDB 容器内部固定监听 8000 端口，跟咱们自己的 FastAPI 后端端口冲突，所以 compose
里把它映射到了 host 的 **8001**，`.env` 里的 `CHROMA_PORT` 要跟这个对上（默认值已经对好了）。
如果你不用这份 compose、自己装的 MySQL/ChromaDB，跳过这步，直接确保这两个服务能连上就行。

#### 2. 安装后端依赖（建议用虚拟环境）

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate      # Windows 用 .venv\Scripts\activate
pip install -r requirements.txt
```

如果要在本地测试"导出Word时把Mermaid图渲染成图片"这个功能，还需要单独下载一次浏览器内核
（Playwright的Python包本身已经在requirements.txt里了）：

```bash
python -m playwright install chromium
```

没装这一步不影响其余功能，只是导出的Word里对应案例的树状图会跳过（文字版表格还在）。

#### 3. 启动后端

```bash
# 还在 backend/ 目录下，虚拟环境已激活
uvicorn main:app --reload --port 8000
```

看到 `Application startup complete.` 说明 MySQL 连接成功、表已自动建好。访问
http://localhost:8000 应返回 `{"status":"ok",...}`。

如果这一步报 `Can't connect to MySQL server`，说明第1步的MySQL没起来或者 `.env` 里的连接
信息不对。

#### 4. 打开前端

```bash
cd frontend
python3 -m http.server 5500
# 访问 http://localhost:5500
```

不建议直接双击打开 `index.html` 文件——本地开发模式下前端(`index.html`里的`API_BASE`)
是直接连后端的固定地址，且后端默认的 CORS 白名单是按 `http://localhost:5500` 这类地址配的，
直接用 `file://` 打开会跨域失败。

#### 5. 登录

首次启动会自动创建用户名`admin`、密码为 `.env` 里 `AUTH_DEFAULT_ADMIN_PASSWORD` 的管理员
账号（默认密码见 `.env.example`，**生产环境务必自己改掉**）。打开前端后先用这个账号登录。

### 验证部署是否成功

登录后，去「🧩 知识点匹配」标签页上传一份 Word/PDF 教学大纲——这一步不需要调用大模型，
纯本地解析，能测出前后端联通、MySQL 写入是否正常。然后去「💬 AI 助手」发一句话，能收到
回复就说明 `DASHSCOPE_API_KEY`、ChromaDB 也都通了。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | 空，**必填** | 硅基流动的 API Key（变量名历史遗留，不是阿里云） |
| `QWEN_MODEL` | `deepseek-ai/DeepSeek-V4-Flash` | 对话/结构化字段生成用的模型 |
| `QWEN_NARRATIVE_MODEL` | `deepseek-ai/DeepSeek-V4-Pro` | 案例正文写作/修订专用的更强模型 |
| `QWEN_EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-8B` | 向量模型，知识点/案例语义匹配用 |
| `QWEN_VISION_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | 扫描版教学大纲整页OCR兜底用的视觉模型 |
| `QWEN_MAX_TOOL_ROUNDS` | `6` | AI助手一轮对话最多允许的工具调用轮次 |
| `CASE_NARRATIVE_MAX_ITERATIONS` | `2` | 案例正文写作-评审循环的最大轮次，调大更精细但更耗时耗成本 |
| `DB_HOST`/`DB_PORT`/`DB_USER`/`DB_PASSWORD`/`DB_NAME` | 见 `.env.example` | MySQL 连接信息（Docker部署时`DB_HOST`会被compose自动覆盖成`mysql`，不用手动改） |
| `CHROMA_HOST`/`CHROMA_PORT` | `127.0.0.1`/`8001` | ChromaDB 连接信息（Docker部署时`CHROMA_HOST`会被compose自动覆盖成`chroma`） |
| `AUTH_ENABLED` | `true` | 是否要求登录才能访问接口，测试联调可临时设`false`，正式环境务必`true` |
| `AUTH_DEFAULT_ADMIN_PASSWORD` | 见 `.env.example` | 首次启动自动建的admin账号密码，账号已存在时改这个值不会覆盖已有密码 |
| `AUTH_SESSION_TTL_HOURS` | `168`（7天） | 登录会话有效期，过期需重新登录 |
| `CORS_ORIGINS`（可选） | `localhost:5500`等 | 本地开发模式下允许跨域访问后端的前端地址，逗号分隔；Docker部署走nginx同源反代，不需要设 |

## 哪些功能不需要 Key 就能用，哪些必须配好大模型Key才能跑通

| 功能 | 需要 `DASHSCOPE_API_KEY`？ |
|---|---|
| 登录 | 不需要 |
| URL批量抓取素材 / 上传Word\PDF拆分素材（含扫描件OCR，如果用Tesseract路径） | 不需要 |
| 知识点大纲上传拆解 | 不需要（但拆出来的知识点不会被索引进ChromaDB，AI助手/匹配功能会找不到它，直到配了key） |
| 案例库列表查看、审核状态修改、修改记录查看 | 不需要 |
| 知识图谱查看、成书Word导出（含Mermaid图，需另装Playwright浏览器内核） | 不需要（前提是库里已经有案例数据） |
| AI助手对话（生成/编辑案例、语义检索素材） | **需要** |
| 案例草稿生成（写作-评审循环） | **需要** |
| 扫描版教学大纲走视觉模型OCR兜底 | **需要**（走Tesseract本地OCR的路径不需要） |
| 知识点↔案例 混合检索匹配 + LLM复核 | **需要** |
| 用已采纳知识点补充案例（教学设计/适用课程） | **需要** |

## 使用流程

1. 登录（首次用 `.env` 里配置的admin账号）
2. 「📂 素材库」：粘贴URL批量抓取，或上传Word/PDF自动拆分成候选案例素材（扫描件会自动走OCR）
3. 「💬 AI 助手」：直接对话——"帮我生成案例3.4的初稿""把开头写得更有画面感""审核通过这个案例"，左侧栏可以新建/切换/删除历史对话。案例正文生成会经过写作-评审多轮打磨，通常要几分钟到十几分钟，聊天界面会实时显示当前跑到第几轮
4. 「📚 案例库」：审阅生成的案例（草稿/待审核/已采纳/已驳回），勾选后可以"导出勾选为Word"或"导出成书"，导出的Word里"适用课程举例"会带Mermaid树状图截图
5. 「🧩 知识点匹配」：批量上传课程教学大纲自动拆解知识点，对某个案例运行匹配（选中案例会自动带出之前的匹配记录，不用重新跑），在候选表格里采纳/拒绝、编辑融入建议，采纳后可以让AI补充案例的"适用课程举例"和"教学设计"
6. 知识点匹配标签页里也能直接查看总知识图谱（图片/可交互两种）

## 已知限制（原型阶段，非最终产品）

- 网页抓取用的是 `trafilatura`，对强反爬/需要JS渲染的页面（如微信公众号）提取效果有限，会明确标记"失败"而不是编造内容
- 成书导出的"前言"（`book_front_matter.py`）是模板占位文字，不是大纲原文，出版前需要人工替换
- 素材/知识点上传后，索引失败（比如没配key）只会在后端日志里留警告，前端目前不会明确提示"这条其实还搜不到"
- 案例正文写作-评审循环的评审打分/否决原因/修订历史不落库，只在生成过程中通过进度提示展示，生成结束后不可追溯查看当时评审给了什么反馈
- 登录鉴权是简单的账号密码+会话token，没有找回密码/多角色权限区分这些完整账号体系该有的功能

## 下一步可以做的事

- 给登录鉴权加角色/权限区分（目前只有单一admin账号概念），审计日志的`actor`字段可以更精细
- 案例写作-评审循环的评审过程如果落库，可以在案例详情页展示"AI自评历史"供人工审核参考
- 素材/知识点索引失败时，前端给出更明确的提示（而不是只在后端日志里）
- HTTPS/域名证书、CI/CD自动构建镜像目前都还没配，需要的话可以在现有Docker资产基础上加
