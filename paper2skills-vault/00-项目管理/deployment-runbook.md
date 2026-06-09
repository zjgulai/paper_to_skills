---
name: deployment-runbook
description: paper2skills Playbook 生产部署运维手册。包含服务器信息、更新部署命令、nginx结构说明、SSL续签步骤、回滚方法。当执行生产部署或运维操作时使用。
---

# paper2skills Playbook 部署运维手册

> 最后更新：2026-06-08

## 服务器信息

| 项目 | 值 |
|---|---|
| 服务器 IP | 101.34.52.232 |
| 系统 | Ubuntu 22.04 LTS |
| 用户名 | ubuntu |
| SSH 密钥 | `ai_video.pem`（项目根目录，不提交 git） |
| 域名 | skills.lute-tlz-dddd.top |
| 静态文件路径 | /opt/paper2skills/html/ |

## 快速连接

```bash
ssh -i ai_video.pem ubuntu@101.34.52.232
```

## 架构概述

```
nginx 容器（ai_video_nginx）
  ├── 80  → 301 重定向到 HTTPS
  ├── 443 → SSL（Let's Encrypt，/etc/letsencrypt/live/lute-tlz-dddd.top/）
  └── /var/www/skills → /opt/paper2skills/html（bind mount，持久化）
```

nginx 配置文件位置（宿主机）：
`/opt/ai-video/deploy/lighthouse/nginx.conf`

## 手动更新部署（临时方案）

```bash
# 1. 本地重新 build
cd /path/to/paper_to_skills
python3 paper2skills-skills/playbook-generator/scripts/build_playbook.py \
  --root . --vault paper2skills-vault --out playbook

# 2. 打包（拆成两包避免超时）
cd playbook
tar -czf /tmp/playbook_core.tar.gz \
    assets/ domains/ graph/ playbooks/ topics/ workflows/ \
    agents.html ai-roadmap.html index.html build-report.json README.md
tar -czf /tmp/playbook_skills.tar.gz skills/

# 3. 上传
scp -i ai_video.pem /tmp/playbook_core.tar.gz /tmp/playbook_skills.tar.gz \
    ubuntu@101.34.52.232:/tmp/

# 4. 服务器解压（bind mount 实时生效，无需 nginx reload）
ssh -i ai_video.pem ubuntu@101.34.52.232 "
    rm -rf /opt/paper2skills/html/* && \
    tar -xzf /tmp/playbook_core.tar.gz -C /opt/paper2skills/html/ && \
    tar -xzf /tmp/playbook_skills.tar.gz -C /opt/paper2skills/html/ && \
    echo '更新完成:' \$(find /opt/paper2skills/html -type f | wc -l) files && \
    rm /tmp/playbook_*.tar.gz
"
```

## 验证部署

```bash
python3 -c "
import urllib.request, ssl
ctx = ssl.create_default_context()
for url, name in [
    ('https://skills.lute-tlz-dddd.top/index.html', '首页'),
    ('https://skills.lute-tlz-dddd.top/ai-roadmap.html', 'CEO白皮书'),
    ('https://skills.lute-tlz-dddd.top/assets/graph-data.json', '图谱数据'),
]:
    try:
        resp = urllib.request.urlopen(url, timeout=10, context=ctx)
        print(f'OK {resp.status}  {name}')
    except Exception as e:
        print(f'ERR {name}: {e}')
"
```

## SSL 证书管理

证书路径（服务器）：`/etc/letsencrypt/live/lute-tlz-dddd.top/`

**查看证书信息**：
```bash
ssh -i ai_video.pem ubuntu@101.34.52.232 \
    "sudo openssl x509 -in /etc/letsencrypt/live/lute-tlz-dddd.top/fullchain.pem -noout -subject -enddate"
```

**续签（添加新子域）**：
```bash
ssh -i ai_video.pem ubuntu@101.34.52.232 \
    "sudo certbot certonly --webroot --webroot-path /var/www/certbot \
    --non-interactive --agree-tos --expand \
    -d lute-tlz-dddd.top -d skills.lute-tlz-dddd.top \
    -d video.lute-tlz-dddd.top [其他域名...]"
```

## nginx 热更新

```bash
# 1. 编辑配置
ssh -i ai_video.pem ubuntu@101.34.52.232
sudo vim /opt/ai-video/deploy/lighthouse/nginx.conf

# 2. 测试配置语法
docker exec ai_video_nginx nginx -t

# 3. 热 reload（不中断连接）
docker exec ai_video_nginx nginx -s reload
```

## 回滚

```bash
# 上传旧版本包到服务器并解压即可（文件覆盖，无需重启nginx）
# 建议在更新前备份：
ssh -i ai_video.pem ubuntu@101.34.52.232 \
    "cp -r /opt/paper2skills/html /opt/paper2skills/html.bak.\$(date +%Y%m%d)"
```

## 现有 Docker 容器（不可污染）

| 容器名 | 用途 | 端口 |
|---|---|---|
| ai_video_nginx | Nginx 反向代理（所有域名共用） | 80, 443 |
| ai_video_frontend | AI Video 前端 | 3000 (内部) |
| ai_video_backend | Lighthouse 后端 | 8001 (内部) |
| medical_audit_app | 医疗审计应用 | 18080 (内部) |

**重要**: paper2skills 通过 bind mount 挂载到现有 `ai_video_nginx`，不新建容器。
