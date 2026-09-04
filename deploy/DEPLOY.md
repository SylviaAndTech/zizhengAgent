# 腾讯云部署手册

思政案例生成工作台 — 从零到上线的完整步骤。站点地址：**https://civics.randomgen.tech**

## 架构速览

```
                         ┌─────────── CVM 服务器 ───────────┐
浏览器                    │                                  │
  │  https://civics...    │  ┌────────┐      ┌────────┐     │
  └──────────────────────►│  │  web   │─────►│  api   │     │
     443                  │  │(nginx) │/api/ │(FastAPI)│    │
                          │  └────────┘      └───┬────┘     │
                          │   托管前端静态页        │          │
                          │   反代并剥掉/api前缀    │          │
                          │                      ▼          │
                          │            ┌───────┐ ┌────────┐ │
                          │            │ mysql │ │ chroma │ │
                          │            └───────┘ └────────┘ │
                          │              app/     infra/     │
                          └──────────────────────────────────┘
                                   全部通过 appnet 网络互通
```

- `infra/` = 数据层（mysql + chroma），很少变动，发版时不碰
- `app/` = 应用层（api + web），每次发版更新这一层
- 镜像由 GitHub Actions 构建后推到 TCR，服务器只负责 `pull`

---

## 阶段零：开工前检查（不做完后面白搭）

### 0.1 ⚠️ ICP 备案（国内地域的硬性前提）

你的 CVM 在国内地域，**域名必须完成 ICP 备案，否则运营商会直接封掉 80/443 端口**，
不管服务跑得多好，外网都访问不了。

```bash
# 查备案状态：腾讯云控制台 → 备案 → 我的备案
# 或工信部官方查询：https://beian.miit.gov.cn/
```

- **已备案** → 直接进入阶段一
- **未备案 / 备案中** → 备案周期通常 1–3 周。这期间你仍然可以先把整套服务部署起来做功能验证，
  只是**先别用 80/443**，改用高位端口（下面「附录 A」有临时方案），等备案通过再切回来。

### 0.2 安全组放行端口

腾讯云控制台 → 云服务器 → 安全组 → 修改规则，**入站规则**需要放行：

| 协议端口 | 来源 | 用途 |
|---|---|---|
| TCP:22 | 你的办公 IP（**不要填 0.0.0.0/0**） | SSH 管理 |
| TCP:80 | 0.0.0.0/0 | HTTP（会 301 跳转到 443） |
| TCP:443 | 0.0.0.0/0 | HTTPS 正式访问 |

> 数据库端口 3306 和 chroma 的 8000 **不要放行**——它们只在容器内网通信，
> 对公网暴露数据库是没必要的攻击面。

### 0.3 确认 DNS 解析已生效

在**你自己电脑**上执行：

```bash
dig +short civics.randomgen.tech
# 预期输出：你的 CVM 公网 IP，比如 123.45.67.89
```

如果没输出或 IP 不对，去腾讯云 DNSPod 检查 A 记录。解析生效可能要几分钟到几小时。

---

## 阶段一：服务器基础环境

SSH 登录服务器后执行。以下假设你用的是 Ubuntu 22.04；CentOS 请把 `apt` 换成 `yum`。

### 1.1 安装 Docker

```bash
# 国内地域直连 Docker 官方源很慢，用腾讯云的镜像源
curl -fsSL https://mirrors.cloud.tencent.com/docker-ce/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /usr/share/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker.gpg] \
https://mirrors.cloud.tencent.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 让当前用户免 sudo 用 docker（执行完要重新登录才生效）
sudo usermod -aG docker $USER
newgrp docker
```

**验证**：

```bash
docker --version           # 预期：Docker version 2x.x.x
docker compose version     # 预期：Docker Compose version v2.x.x
docker run --rm hello-world   # 预期：Hello from Docker!
```

### 1.2 加 swap（2核4G 这类小机器强烈建议）

仓库里已经带了脚本。上传 Word/PDF、跑 OCR、导出 Word 时内存会瞬时冲高，
没有 swap 容易被 OOM killer 直接杀进程。

```bash
sudo bash deploy/setup_swap.sh    # 默认加 4G
free -h                            # 预期：Swap 那一行不再是 0B
```

### 1.3 配置 Docker 镜像加速（拉取官方镜像用）

```bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<'EOF'
{
  "registry-mirrors": ["https://mirror.ccs.tencentyun.com"]
}
EOF
sudo systemctl daemon-reload && sudo systemctl restart docker
```

---

## 阶段二：把两个基础镜像转推到 TCR

`infra/docker-compose.yml` 引用的是 `ccr.ccs.tencentyun.com/ideological/mysql:8.0` 和
`.../chroma:1.5.9`。你的 TCR 里现在**还没有**这两个镜像，必须先放进去，否则阶段五起不来。

在**服务器上**执行（已配好镜像加速器，拉官方镜像会比较快）：

```bash
# 1) 登录 TCR
docker login ccr.ccs.tencentyun.com
# 用户名 = 腾讯云账号ID，密码 = 镜像仓库独立密码，见「阶段三 3.1」

# 2) 拉官方镜像
docker pull mysql:8.0
docker pull chromadb/chroma:1.5.9

# 3) 打上 TCR 的标签
docker tag mysql:8.0            ccr.ccs.tencentyun.com/ideological/mysql:8.0
docker tag chromadb/chroma:1.5.9 ccr.ccs.tencentyun.com/ideological/chroma:1.5.9

# 4) 推上去
docker push ccr.ccs.tencentyun.com/ideological/mysql:8.0
docker push ccr.ccs.tencentyun.com/ideological/chroma:1.5.9
```

**验证**：腾讯云控制台 → 容器镜像服务 → 镜像仓库，应该能看到 `mysql` 和 `chroma` 两个仓库。

> 如果 `docker pull` 卡住/超时，说明镜像加速器没生效，回头检查 1.3；
> 或者在你自己电脑（能正常连 Docker Hub）上执行这四步，效果一样。

---

## 阶段三：配置 GitHub 并触发首次构建

### 3.1 拿到镜像仓库登录凭证

本项目用的是 `ccr.ccs.tencentyun.com`，这是**容器镜像服务个人版**的地址。
个人版**没有**「长期访问凭证」这个功能（那是企业版 TCR 才有的），用账号 ID + 仓库密码登录：

| 项 | 在哪拿 |
|---|---|
| **用户名** | 腾讯云**账号 ID**（纯数字，如 `100012345678`）<br>控制台右上角头像 → 账号信息 → 账号ID |
| **密码** | 镜像仓库的**独立登录密码**<br>控制台 → 容器镜像服务 → 个人版 → 实例信息 → 设置/重置密码 |

> **这个密码不是你的腾讯云账号登录密码**，是专门给镜像仓库用的、单独设置的一个密码。
> 它只对 `ccr.ccs.tencentyun.com` 生效，泄露了不影响腾讯云账号本身，也可以随时在控制台重置。
> 所以放进 GitHub Secrets 是可以接受的。

**个人版的安全限制要知道**（企业版才有的能力这里没有）：

- 不能按命名空间限权，这一个密码对你账号下所有镜像仓库都有推拉权限
- 不能签发多个可独立吊销的凭证，只有"重置密码"这一个手段
- 所以：**给它设一个别处没用过的独立强密码**，只放在 GitHub Secrets 里
  （Secrets 是加密存储的，且会在 Actions 日志里自动打码）；
  一旦怀疑泄露，立刻去控制台重置密码即可

如果以后要更细的权限控制（子账号、只读凭证、按命名空间授权），需要升级到企业版 TCR。

### 3.2 配置 GitHub Secrets

GitHub 仓库 → Settings → Secrets and variables → Actions → New repository secret：

| Secret 名 | 值 |
|---|---|
| `TCR_USERNAME` | 上一步的凭证用户名 |
| `TCR_PASSWORD` | 上一步的凭证密码 |

### 3.3 推代码触发构建

```bash
git add -A
git commit -m "feat: 容器化部署配置"
git push origin main
```

去 GitHub 仓库的 **Actions** 标签页看构建过程。首次构建要装系统依赖、Python 依赖和
Chromium 浏览器内核，**大约 10–20 分钟**，之后有缓存会快很多。

> ⚠️ **首次构建请留意这两步**：`Dockerfile.api` 里的
> `apt-get install tesseract-ocr fonts-noto-cjk` 和 `playwright install --with-deps chromium`。
> 这两层在本地开发环境因网络限制没能实际验证过（用的是标准包名和 Playwright 官方参数，
> GitHub Actions 网络正常应该没问题）。如果在这里报错，把日志发出来。

**验证**：构建成功后，TCR 控制台里 `sizheng-api` 和 `sizheng-web` 两个仓库都应该出现
`latest` 和一个 40 位 commit sha 标签。

---

## 阶段四：服务器上落地配置

### 4.1 获取部署文件

服务器上其实**不需要完整源码**（镜像已经在 TCR 里了），只需要 compose 文件和证书。
但直接 clone 整个仓库最省事，后续更新配置也方便：

```bash
cd ~
git clone <你的仓库地址> prototype
cd prototype
```

### 4.2 创建共享网络（只需一次）

```bash
docker network create appnet
docker network ls | grep appnet    # 预期：能看到 appnet
```

> `infra/` 和 `app/` 两套 compose 都声明这个网络为 `external: true`，靠它互通。
> 不建的话 compose 会报 `network appnet declared as external, but could not be found`。

### 4.3 配置数据层环境变量

```bash
cd ~/prototype/infra
cp .env.example .env
vi .env
```

把两个密码改成**强密码**：

```ini
MYSQL_ROOT_PASSWORD=<改成你的强密码>
MYSQL_DATABASE=sizheng_cases
MYSQL_USER=sizheng
MYSQL_PASSWORD=<改成你的强密码>
```

> ⚠️ 这几个变量**只在 MySQL 数据目录为空（第一次启动）时生效**。之后再改这里不会修改
> 已存在的密码——那需要进容器执行 `ALTER USER`，或者删掉 `data/mysql` 重新初始化（会丢数据）。
> **所以第一次就要定好密码。**

### 4.4 配置应用层环境变量

```bash
cd ~/prototype/app
cp .env.example .env
vi .env
```

必须改的几项：

```ini
# 硅基流动的 API Key —— 注意是 https://cloud.siliconflow.cn 申请，不是阿里云
DASHSCOPE_API_KEY=<你的key>

# 下面三项必须跟 infra/.env 里的完全一致，否则连不上数据库
DB_USER=sizheng
DB_PASSWORD=<跟 infra/.env 的 MYSQL_PASSWORD 一模一样>
DB_NAME=sizheng_cases

# 首次启动会用这个密码创建 admin 账号，务必改成强密码
AUTH_DEFAULT_ADMIN_PASSWORD=<改成你的强密码>
```

**两个最容易配错的地方**（模板里已经是对的，别手滑改坏）：

| 变量 | 正确值 | 常见错误 |
|---|---|---|
| `CHROMA_PORT` | `8000` | 填成 8001。8001 是本地开发时映射到宿主机的端口，容器之间通信要用容器内实际监听的 8000 |
| `DB_HOST` / `CHROMA_HOST` | `mysql` / `chroma` | 填成 IP。容器 IP 每次重启都会变，必须用 compose 服务名让 Docker DNS 解析 |

### 4.5 放置 SSL 证书 ⚠️ 文件名必须改

腾讯云下载的 Nginx 证书包解压后是这样的名字：

```
civics.randomgen.tech_bundle.crt
civics.randomgen.tech.key
```

但 `deploy/nginx.conf` 里写死的路径是 `fullchain.crt` 和 `private.key`，**必须重命名**：

```bash
mkdir -p ~/prototype/app/certs
cd ~/prototype/app/certs

# 把证书上传到这个目录后重命名（文件名按你实际下载到的改）
mv civics.randomgen.tech_bundle.crt  fullchain.crt
mv civics.randomgen.tech.key          private.key

# 私钥权限收紧
chmod 600 private.key
ls -l    # 预期：fullchain.crt 和 private.key 两个文件都在
```

> 从本地上传证书到服务器：
> `scp 证书目录/* ubuntu@<服务器IP>:~/prototype/app/certs/`

### 4.6 服务器登录 TCR

```bash
docker login ccr.ccs.tencentyun.com
# 输入阶段三 3.1 创建的凭证
```

不登录的话下一步 `docker compose pull` 会报 `pull access denied`。

---

## 阶段五：启动服务（顺序不能反）

### 5.1 先起数据层

```bash
cd ~/prototype/infra
docker compose up -d

# 等两个容器变成 healthy（大约 30 秒）
watch -n 2 'docker compose ps'
# 预期：sizheng-mysql 和 sizheng-chroma 的 STATUS 都是 Up (healthy)
# 看到 healthy 后按 Ctrl+C 退出 watch
```

> 必须先起数据层：`app` 里的 api 容器健康检查依赖数据库连通，
> 数据库没起来 api 会一直不健康，web 也就一直不启动。

### 5.2 再起应用层

```bash
cd ~/prototype/app
docker compose pull
docker compose up -d --force-recreate

# api 冷启动要 import 一大堆重依赖再建表，大约 1-2 分钟
watch -n 3 'docker compose ps'
# 预期：sizheng-api 是 Up (healthy)，sizheng-web 是 Up
```

如果 api 一直不 healthy，看日志：`docker compose logs -f api`

### 5.3 确认四个容器全部就位

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
```

预期看到 4 个：`sizheng-mysql`、`sizheng-chroma`（都不映射端口）、
`sizheng-api`（不映射端口）、`sizheng-web`（映射 80、443）。

---

## 阶段六：验证

### 6.1 服务器上自检

```bash
# 1) 健康检查（验证 nginx 剥前缀 + 后端路由对齐）
curl -k https://localhost/api/health
# 预期：{"status":"ok"}

# 2) 前端页面
curl -k -o /dev/null -w "%{http_code}\n" https://localhost/
# 预期：200

# 3) 【安全项·最重要】未登录访问受保护接口，必须 401
curl -k -w " [%{http_code}]\n" https://localhost/api/cases
# 预期：{"detail":"未登录或登录已过期，请重新登录"} [401]
# 如果这里返回 200，说明鉴权没生效，立刻停下来排查，不要对外开放

# 4) 登录拿 token 再访问
TOKEN=$(curl -sk -X POST -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"<你在app/.env里设的密码>"}' \
  https://localhost/api/auth/login | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
curl -k -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $TOKEN" https://localhost/api/cases
# 预期：200
```

### 6.2 HTTPS 与证书

在**你自己电脑**上执行：

```bash
# HTTP 应该 301 跳转到 HTTPS
curl -I http://civics.randomgen.tech
# 预期：HTTP/1.1 301 Moved Permanently + Location: https://civics.randomgen.tech/

# HTTPS 证书链验证（不带 -k，让 curl 真的去校验证书）
curl -I https://civics.randomgen.tech
# 预期：HTTP/2 200，且没有证书错误
```

如果报证书错误，通常是用了单证书而不是证书链（`_bundle.crt` 才是完整链）。

### 6.3 浏览器完整走查

打开 https://civics.randomgen.tech ，用 admin 账号登录后逐项确认：

| 功能 | 怎么测 | 验证的是什么 |
|---|---|---|
| 登录 | 用 admin + 你设的密码 | 鉴权链路、数据库写入 |
| 上传大纲 | 「🧩 知识点匹配」上传一个 Word/PDF | 文件上传（20m 限制）、MySQL 写入 |
| 上传扫描件 PDF | 传一个纯图片的 PDF | 容器里的 tesseract OCR 可用 |
| AI 助手对话 | 发一句话 | 大模型 key 通、**SSE 打字机效果**（逐字出而不是等一整段） |
| 案例生成 | 让 AI 生成一个案例 | 后台任务队列、顶部进度条 |
| 知识图谱 | 「查看图谱（图片）」 | matplotlib 中文字体（**不能是方框**） |
| 导出 Word | 案例库勾选后导出 | 容器里的 Playwright/Chromium（Word 里应有 Mermaid 树状图） |

---

## 阶段七：日常运维

### 发布新版本

代码 push 到 main 后 GitHub Actions 会自动构建镜像。在 Actions 的运行摘要里能看到这次的
commit sha，然后在服务器上：

```bash
cd ~/prototype/app
export TAG=<Actions 里显示的 commit sha>
docker compose pull
docker compose up -d --force-recreate
```

> ⚠️ **`--force-recreate` 不能省。** nginx 在加载配置时就把 `api` 这个主机名解析成 IP 并
> 一直缓存。如果只重建 api 容器，它会拿到新 IP 而 nginx 还在打旧 IP，导致**持续 502**
> （这个已经实测复现过）。`--force-recreate` 让两个容器一起重建，nginx 重新解析。
> 代价是几秒钟中断。

### 回滚

```bash
cd ~/prototype/app
export TAG=<上一个正常版本的 commit sha>
docker compose pull && docker compose up -d --force-recreate
```

用 sha 而不是 `latest` 的好处就在这里：`latest` 会随下次构建漂移，sha 是不可变的，回滚精确。

### 看日志

```bash
cd ~/prototype/app
docker compose logs -f api          # 后端日志（报错主要看这个）
docker compose logs -f web          # nginx 访问/错误日志
docker compose logs --tail 100 api  # 只看最近 100 行

cd ~/prototype/infra
docker compose logs -f mysql
```

### 进容器排查

```bash
docker exec -it sizheng-api bash                    # 进后端容器
docker exec -it sizheng-mysql mysql -uroot -p       # 进 MySQL 命令行
```

### 备份数据

数据都在宿主机 `~/prototype/data/` 下（绑定挂载），备份就是打包这个目录：

```bash
cd ~/prototype
# 停机备份最稳妥（避免备份到写了一半的文件）
cd app && docker compose stop && cd ../infra && docker compose stop
sudo tar czf ~/backup-$(date +%F).tar.gz data/
cd ../infra && docker compose start && cd ../app && docker compose start
```

建议配个 cron 每天自动备份，并把备份包传到腾讯云 COS。

### 证书续期

腾讯云免费证书有效期 3 个月（或按你申请的类型）。到期前重新申请后：

```bash
cd ~/prototype/app/certs
# 覆盖这两个文件（文件名必须还是这两个）
mv 新证书_bundle.crt fullchain.crt
mv 新证书.key         private.key
chmod 600 private.key
cd .. && docker compose restart web    # 重启 nginx 加载新证书
```

### 开启 CI 自动部署（第二阶段）

确认手动部署一切正常后，可以放开自动部署：

1. 在 GitHub 补配 Secrets：`SSH_HOST`、`SSH_USER`、`SSH_KEY`（私钥全文）、`SSH_PORT`
2. 编辑 `.github/workflows/deploy.yml`，把文件末尾 `deploy:` 那一整段的注释去掉
3. push 一次，观察 Actions 是否自动完成部署

---

## 附录 A：备案期间的临时验证方案

备案没下来之前 80/443 不可用，但可以用高位端口先验证功能：

```bash
# 1) 安全组临时放行 TCP:8443
# 2) 改端口映射
cd ~/prototype/app
vi docker-compose.yml
```

把 web 服务的 ports 改成：

```yaml
    ports:
      - "8080:80"
      - "8443:443"
```

然后用 `https://<服务器IP>:8443` 访问（浏览器会警告证书域名不匹配，点继续即可）。
**备案通过后记得改回 `80:80` 和 `443:443`，并关掉安全组里的 8443。**

---

## 附录 B：常见问题排查

| 现象 | 可能原因 | 怎么解决 |
|---|---|---|
| 域名打不开、连接超时 | ①ICP 未备案 ②安全组没放行 80/443 ③DNS 没生效 | 按 0.1–0.3 逐项查。先在服务器上 `curl -k https://localhost/` 确认服务本身正常，正常就是网络层问题 |
| 502 Bad Gateway | api 容器没起来，或 nginx 缓存了 api 旧 IP | `docker compose ps` 看 api 是否 healthy；健康就 `docker compose up -d --force-recreate` |
| 所有接口 404 | nginx 剥前缀和后端路由没对齐 | 确认 `deploy/nginx.conf` 里 `proxy_pass http://api:8000/;` **末尾有斜杠** |
| Swagger 能开但 Try it out 404 | `root_path` 没生效 | 确认 `app/.env` 里没把 `ROOT_PATH` 改错（默认 `/api` 就对） |
| 未登录也能访问接口 | 鉴权没生效 | 检查 `app/.env` 里 `AUTH_ENABLED=true`；**这是严重问题，先下线再排查** |
| 上传大文件失败 / 413 | nginx 请求体大小限制 | `deploy/nginx.conf` 里 `client_max_body_size 20m`，需要更大就调这里再重建 web |
| AI 对话没有打字机效果 | nginx 缓冲了 SSE 流 | 确认 `deploy/nginx.conf` 里有 `proxy_buffering off` |
| 知识图谱中文显示成方框 | 容器缺中文字体 | 确认镜像构建时装了 `fonts-noto-cjk`（`Dockerfile.api`），重新构建 |
| 导出的 Word 里没有 Mermaid 图 | 容器里 Chromium 没装好 | `docker exec -it sizheng-api python -m playwright install chromium`；或看 api 日志里的 warning |
| 改了 MySQL 密码但连不上 | `MYSQL_*` 只在首次初始化时生效 | 进容器 `ALTER USER` 改，或删掉 `data/mysql` 重新初始化（**会丢数据**） |
| `network appnet not found` | 忘了建共享网络 | `docker network create appnet` |
| `pull access denied` | 服务器没登录 TCR | `docker login ccr.ccs.tencentyun.com` |
| api 一直不 healthy | 连不上数据库 | `docker compose logs api` 看具体报错；检查 `app/.env` 的 `DB_*` 和 `CHROMA_PORT=8000` |
| 磁盘满了 | 旧镜像堆积 | `docker image prune -a -f`（会删掉所有未被容器使用的镜像） |

---

## 附录 C：关键路径速查

| 项目 | 值 |
|---|---|
| 站点地址 | https://civics.randomgen.tech |
| 镜像仓库 | `ccr.ccs.tencentyun.com/ideological/` |
| 应用镜像 | `sizheng-api`、`sizheng-web` |
| 基础镜像 | `mysql:8.0`、`chroma:1.5.9`（需自行转推） |
| 数据目录 | `~/prototype/data/mysql`、`~/prototype/data/chroma` |
| 证书目录 | `~/prototype/app/certs/{fullchain.crt,private.key}` |
| 数据层配置 | `~/prototype/infra/.env` |
| 应用层配置 | `~/prototype/app/.env` |
| 共享网络 | `appnet` |
| 容器名 | `sizheng-mysql`、`sizheng-chroma`、`sizheng-api`、`sizheng-web` |
