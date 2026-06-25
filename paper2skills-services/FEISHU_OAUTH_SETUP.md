# 飞书 OAuth 配置说明

## 1. 创建飞书自建应用

1. 打开 https://open.feishu.cn/app
2. 点击「创建企业自建应用」
3. 应用名称：paper2skills
4. 记录 App ID 和 App Secret

## 2. 开启网页应用能力

在应用管理页面：
- 「添加应用能力」→「网页」
- 桌面端主页 URL：`https://skills.lute-tlz-dddd.top`

## 3. 配置 OAuth 回调

「安全设置」→「重定向 URL」添加：
```text
https://skills.lute-tlz-dddd.top/auth/callback
```

## 4. 申请权限

「权限管理」→ 开启：
- `contact:user.id:readonly`
- `authen:user_info:readonly`

## 5. 发布应用

「版本管理与发布」→ 创建版本 → 申请线上发布

## 6. 配置服务器环境变量

在服务器 `/opt/paper2skills/service/.env` 添加：
```env
FEISHU_APP_ID=cli_xxxxxxxxxxxxxxxx
FEISHU_APP_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
JWT_SECRET=$(openssl rand -hex 32)
SITE_URL=https://skills.lute-tlz-dddd.top
```

## 7. 重启服务

```bash
systemctl restart p2s-service
```
