# 思政案例生成工作台 - 原型 v0.4

"案例生成工作台"模块的可运行原型，覆盖完整闭环：

**素材导入（URL抓取/Word/PDF拆分） → AI助手对话生成/编辑7段式案例 → 知识点向量匹配补充 → 审核 → 知识图谱/成书导出**

严格按照《讲好数字中国故事：人工智能类课程思政案例集》大纲里"每一个案例的框架"
（完整案例/教学目标/思政元素/适用课程/教学设计/课程评价/延伸阅读）来生成，
并强制"不编造事实、不照抄原文、保留来源引用"三条规则。

## 技术栈

- **后端**：FastAPI + SQLAlchemy
- **关系数据库**：MySQL（素材/案例/知识点/会话/审计日志）
- **向量数据库**：ChromaDB（知识点描述、素材正文切块后的向量，供语义检索）
- **大模型**：阿里云 DashScope（Qwen 系列），对话生成走 OpenAI 兼容协议
- **知识库检索**：LlamaIndex（文档切块、向量存取、检索器），封装在 ChromaDB 之上
- **AI 助手 Agent**：LangChain（`create_agent`，内部基于 LangGraph）驱动工具调用循环
- **文档处理**：`python-docx`（读写Word）、`pypdf`（读PDF）
- **知识图谱**：`networkx` 建图，`matplotlib` 出静态图片，`pyvis` 出可交互HTML
- **前端**：无框架、无构建步骤，单文件 `frontend/index.html`

## 目录结构

```
prototype/
├── docker-compose.yml      # 开发环境的 MySQL + ChromaDB
├── backend/
│   ├── main.py                 # FastAPI 入口，所有API端点
│   ├── db.py                   # SQLAlchemy 模型定义 + MySQL 连接
│   ├── qwen_client.py          # 统一的 Qwen(DashScope) 客户端与模型配置
│   ├── llama_index_setup.py    # LlamaIndex 全局配置(Settings.llm/embed_model → Qwen)
│   ├── generate_case.py        # 调用 Qwen 生成案例草稿 / 补充适用课程+教学设计
│   ├── chat_agent.py           # AI助手：LangChain agent + 6个业务工具
│   ├── prompts.py              # 所有系统提示词模板
│   ├── fetch_material.py       # URL 正文抓取（trafilatura）
│   ├── parse_document.py       # 上传的 Word/PDF 解析 + 按标题拆分候选案例
│   ├── knowledge_matching.py   # 知识点抽取 + 向量粗筛(LlamaIndex/ChromaDB) + Qwen复核精排
│   ├── material_index.py       # 素材正文切块/向量化/存入ChromaDB + 语义检索（供AI助手用）
│   ├── knowledge_graph.py      # 知识图谱：networkx建图 + 静态图片/可交互HTML
│   ├── book_export.py          # 成书编译：前言+按维度分章+附录+图谱
│   ├── doc_writer.py           # 案例7段式写入Word的共用逻辑
│   ├── audit.py                # 案例修改留痕的写入工具
│   ├── requirements.txt
│   └── .env / .env.example
└── frontend/
    └── index.html               # 单页前端：💬AI助手 / 📂素材库 / 📚案例库 / 🧩知识点匹配
```

## 快速开始（第一次部署）

### 前置条件

- Docker（跑 MySQL + ChromaDB）
- Python 3.10+（代码里用了 `str | None` 这类新语法，3.9 跑不了；建议 3.12）
- 一个阿里云 DashScope 的 API Key：https://dashscope.console.aliyun.com/apiKey

### 1. 启动 MySQL + ChromaDB

项目根目录已经准备好 `docker-compose.yml`，账号密码/端口都和后端的 `.env` 默认值对上了：

```bash
cd prototype
docker compose up -d
```

会拉起2个容器：`mysql`、`chroma`。ChromaDB 本身不依赖额外的元数据/对象存储服务，比之前的方案简单。

首次拉镜像可能比较慢。跑起来后确认一下：

```bash
docker compose ps          # 2个容器都应该是 Up/running
```

MySQL 的 `sizheng_cases` 库、账号 `sizheng`/密码 `sizheng_pass` 会在容器首次启动时自动建好（`MYSQL_DATABASE`/`MYSQL_USER`/`MYSQL_PASSWORD` 环境变量）。ChromaDB 的两个 collection（`knowledge_point_vectors`、`material_chunks`）不用手动建，后端第一次真正用到时会自动建。

注意 ChromaDB 容器内部固定监听 8000 端口，跟咱们自己的 FastAPI 后端端口冲突，所以 compose 里把它映射到了 host 的 **8001**，`.env` 里的 `CHROMA_PORT` 要跟这个对上（默认值已经对好了）。

如果你不用这份 compose、自己装的 MySQL/ChromaDB，跳过这步，直接确保这两个服务能连上就行。

### 2. 安装后端依赖（建议用虚拟环境）

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate      # Windows 用 .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 配置 `.env`

```bash
cp .env.example .env
```

打开 `backend/.env`，至少要填一个东西：

```bash
DASHSCOPE_API_KEY=你的密钥
```

其余变量如果你就用第1步的 docker-compose，默认值不用改，照抄一遍供参考：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | 空，**必填** | 阿里云 DashScope 的 API Key |
| `QWEN_MODEL` | `qwen-plus` | 对话/生成模型，需支持 function calling |
| `QWEN_EMBEDDING_MODEL` | `text-embedding-v3` | 向量模型 |
| `QWEN_MAX_TOOL_ROUNDS` | `6` | AI助手一轮对话最多允许的工具调用轮次 |
| `DB_HOST`/`DB_PORT`/`DB_USER`/`DB_PASSWORD`/`DB_NAME` | 见 `.env.example` | MySQL 连接信息 |
| `CHROMA_HOST`/`CHROMA_PORT` | `127.0.0.1`/`8001` | ChromaDB 连接信息 |
| `CORS_ORIGINS`（可选，代码里有默认值不用特意设） | `localhost:5500`等 | 允许跨域访问后端的前端地址，逗号分隔 |

### 4. 启动后端

```bash
# 还在 backend/ 目录下，虚拟环境已激活
uvicorn main:app --reload --port 8000
```

看到 `Application startup complete.` 说明 MySQL 连接成功、表已自动建好。访问
http://localhost:8000 应返回 `{"status":"ok",...}`。

如果这一步报 `Can't connect to MySQL server`，说明第1步的MySQL没起来或者 `.env` 里的连接信息不对。

### 5. 打开前端

```bash
cd frontend
python3 -m http.server 5500
# 访问 http://localhost:5500
```

不建议直接双击打开 `index.html` 文件——后端默认的 CORS 白名单是按 `http://localhost:5500` 这类地址配的，直接用 `file://` 打开会跨域失败。

### 6. 验证部署是否成功

打开前端后，去「🧩 知识点匹配」标签页上传一份 Word/PDF 教学大纲——这一步不需要调用 Qwen，纯本地解析，能测出前后端联通、MySQL 写入是否正常。然后去「💬 AI 助手」发一句话，能收到回复就说明 DASHSCOPE_API_KEY、ChromaDB 也都通了。

## 哪些功能不需要 Key 就能用，哪些必须配好 Qwen 才能跑通

| 功能 | 需要 DASHSCOPE_API_KEY？ |
|---|---|
| URL批量抓取素材 / 上传Word\PDF拆分素材 | 不需要 |
| 知识点大纲上传拆解 | 不需要（但拆出来的知识点不会被索引进ChromaDB，AI助手/匹配功能会找不到它，直到配了key） |
| 案例库列表查看、审核状态修改、修改记录查看 | 不需要 |
| 知识图谱查看、成书Word导出 | 不需要（前提是库里已经有案例数据） |
| AI助手对话（生成/编辑案例、语义检索素材） | **需要** |
| 案例草稿生成 | **需要** |
| 知识点↔案例 向量匹配 + LLM复核 | **需要** |
| 用已采纳知识点补充案例（教学设计/适用课程） | **需要** |

## 使用流程

1. 「📂 素材库」：粘贴URL批量抓取，或上传Word/PDF自动拆分成候选案例素材
2. 「💬 AI 助手」：直接对话——"帮我生成案例3.4的初稿""把开头写得更有画面感""审核通过这个案例"，左侧栏可以新建/切换/删除历史对话
3. 「📚 案例库」：审阅生成的案例（草稿/待审核/已采纳/已驳回），勾选后可以"导出勾选为Word"或"导出成书"
4. 「🧩 知识点匹配」：批量上传课程教学大纲自动拆解知识点，对某个案例运行向量匹配，在候选表格里采纳/拒绝、编辑融入建议，采纳后可以让AI补充案例的"适用课程举例"和"教学设计"
5. 知识点匹配标签页里也能直接查看知识图谱（图片/可交互两种）

## 已知限制（原型阶段，非最终产品）

- 前端未做登录鉴权，所有操作匿名，审计日志的 `actor` 字段只是粗粒度区分"用户"/"AI助手"，不是真正的账号体系
- 网页抓取用的是 `trafilatura`，对强反爬/需要JS渲染的页面（如微信公众号）提取效果有限，会明确标记"失败"而不是编造内容
- 知识点↔案例的向量检索目前是纯向量召回，没做混合检索（向量+关键词），字面精确匹配的场景可能不如预期
- 成书导出的"前言"是模板占位文字，不是大纲原文，出版前需要人工替换
- 素材/知识点上传后，索引失败（比如没配key）只会在后端日志里留警告，前端目前不会明确提示"这条其实还搜不到"

## 下一步可以做的事

- 知识点匹配加混合检索（ChromaDB本身只做稠密向量检索，没有Milvus那种原生稀疏向量融合能力，
  真要做混合检索得在应用层自己实现关键词检索再和向量结果做排序融合，比如RRF）
- 给前端加简单的账号体系，让审计日志真正对应到人
- 素材/知识点索引失败时，前端给出更明确的提示（而不是只在后端日志里）
