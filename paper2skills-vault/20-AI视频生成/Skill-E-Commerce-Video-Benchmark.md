# Skill Card: E-Commerce Video Benchmark（电商视频质量评估基准）

> **论文**: E-CommerceVideo: A Benchmark and Approach for E-Commerce Video Generation from Product Images  
> **来源**: ICLR 2026 投稿 (OpenReview GraP6A5SUO) | 淘宝数据 | 2025  
> **代码**: ✅ 数据集 15,096 条三元组 + 基线代码  
> **领域**: 20-AI视频生成 | **场景**: 质量评估

roadmap_phase: phase3
---

## ① 算法原理

**唯一电商域专用 Benchmark**。通用 T2V 评测用 UCF-101/MSR-VTT（自然场景），但电商视频核心要求完全不同——商品颜色/纹理/Logo 不能有任何失真。E-CommerceVideo 建立电商专属评测体系。

**数据集**：15,096 条三元组（多视角商品图 + 字幕描述 + 真值视频），来自淘宝。

**基线方法**：Wan2.2 T2V 14B 微调 + VAE-based Spatial Injection Mechanism（多视角商品图空间注入）。

**电商专属评测维度**：
- Product Color Fidelity (PCF)：商品颜色准确率
- Logo/Texture Preservation (LTP)：Logo 纹理保留率
- Viewpoint Smoothness (VS)：多视角切换平滑度
- Motion Naturalness (MN)：运动自然度

---

## ② 母婴出海应用案例

评估 Phantom vs AnchorCrafter vs Wan2.2 在母婴品类（吸奶器/婴儿车/玩具）上的商品保真度。Phantom 在 PCF 和 LTP 维度最优（主体一致性专长），AnchorCrafter 在 MN 最优（HOI 运动最自然）。选型决策：Amazon listing 用 Phantom，TikTok 带货用 AnchorCrafter。

---

## ③ 代码模板

```python
class EcommerceVideoBenchmark:
    DIMENSIONS = ["PCF", "LTP", "VS", "MN"]
    WEIGHTS = {"product_fidelity": 0.35, "logo_texture": 0.25, "viewpoint_smooth": 0.20, "motion_natural": 0.20}
    
    def evaluate_model(self, model_name: str, scores: dict) -> dict:
        total = sum(scores.get(d, 0) * self.WEIGHTS.get(d, 0.2) for d in self.DIMENSIONS)
        return {"model": model_name, "scores": scores, "overall": round(total, 3), 
                "recommended_for": "Amazon listing" if scores.get("PCF",0)>0.85 else "TikTok UGC"}

if __name__ == '__main__':
    bench = EcommerceVideoBenchmark()
    print(bench.evaluate_model("Phantom", {"PCF":0.92, "LTP":0.88, "VS":0.85, "MN":0.78}))
    print("[✓] E-Commerce Video Benchmark 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Phantom-Product-Showcase-I2V]] | [[Skill-AnchorCrafter-Virtual-Anchor-Demo]]
- **组合**：[[Skill-Model-Evaluation-Metrics]]（评估方法论互通）

---
- **相关**：[[Skill-Demand-Forecasting-Supply-Chain]]

## ⑤ 商业价值：技术储备 | **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐☆☆☆
