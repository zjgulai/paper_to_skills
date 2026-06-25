# p2s-service

paper2skills 的 FastAPI 飞书回调与每日巡检服务。

## 运行

```bash
uvicorn app:app --reload --port 8765
```

## 环境变量

- `P2S_INSPECT_SECRET`
- `DEEPSEEK_API_KEY`
- `FEISHU_WEBHOOK_URL`

## API

- `POST /api/feishu-callback`：飞书卡片按钮回调
- `POST /api/feishu-event`：飞书事件订阅，文档分享后生成 Skill 草稿
- `POST /api/daily-inspect`：每日巡检入口，需 secret 校验

## 生产部署

- systemd 服务名：`p2s-service`
