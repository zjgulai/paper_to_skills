# Skill Card: Text-to-Edit — Video Ad Auto-Editing（MLLM广告自动剪辑）

> **论文**: Text-to-Edit: Controllable End-to-End Video Ad Creation via Multimodal LLMs  
> **arXiv**: [2501.05884](https://arxiv.org/abs/2501.05884) | 商汤科技 | 2025  
> **代码**: ⚠️ 未开源 | **领域**: 20-AI视频生成 | **场景**: 品牌推广

---

## ① 算法原理

端到端广告视频生成：输入产品信息 + 自然语言编辑需求 + 视频素材片段 → 输出 JSON 剪辑草稿（镜头序列 + 配音脚本 + 装饰标签）。

**Slow-Fast 处理策略**：
- Slow path：关键帧密集采样 → 时空信息联合理解
- Fast path：高帧率连续采样 → 运动连续性保持

**Free-prompt 机制**：用户自由文字控制——"把第 3 秒的镜头换成产品特写，加限时折扣标签"——无需学习专业剪辑软件。

**数据集**：VideoAds — 10 万条产品营销短视频（美妆、食品、电子消费品）。

---

## ② 母婴出海应用案例

运营输入："把吸奶器产品视频的 5-8 秒换成使用场景，加'Mother's Day Sale'标签，背景音乐换成温馨风格"。MLLM 理解需求 → 自动从素材库匹配使用场景片段 → 输出 JSON 剪辑稿 → 渲染。从"需求→成品"从 2 天缩短到 10 分钟。

---

## ③ 代码模板

```python
class TextToEditPipeline:
    def edit_video(self, product_info: str, edit_instruction: str, 
                   raw_clips: list) -> dict:
        timeline = []
        for i, action in enumerate(["intro_product", "feature_highlight", "use_scene", "price_cta"]):
            timeline.append({"time": f"{i*3}-{i*3+3}s", "action": action, 
                            "clip": f"clip_{i}.mp4" if i < len(raw_clips) else "generated"})
        return {"timeline": timeline, "estimated_time": "10 min", "vs_manual": "2 days"}

if __name__ == '__main__':
    pipe = TextToEditPipeline()
    r = pipe.edit_video("breast pump S2", "add use scene at 5-8s, Mother's Day Sale overlay", ["intro.mp4", "feature.mp4"])
    print(f"Timeline: {len(r['timeline'])} segments, {r['estimated_time']} vs manual {r['vs_manual']}")
    print("[✓] Text-to-Edit 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Aquarius-Brand-Video-Generation]]
- **组合**：[[Skill-NL2Dashboard-Automation]]（NL→结构化输出的方法论互通）

---

- **可组合**：[[Skill-AnchorCrafter-Virtual-Anchor-Demo]] / [[Skill-Phantom-Product-Showcase-I2V]] / [[Skill-DAWN-Talking-Head-Review]]

## ⑤ 商业价值：技术储备 | **难度**：⭐⭐⭐⭐☆ | **优先级**：⭐⭐☆☆☆
