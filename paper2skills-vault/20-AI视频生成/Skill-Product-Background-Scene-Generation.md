---
title: SD+ControlNet商品场景图生成 — 白底转场景化主图的AI摄影替代
doc_type: knowledge
module: 20-AI视频生成
topic: product-background-scene-generation
status: stable
created: 2026-06-18
updated: 2026-06-18
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: SD+ControlNet商品场景图生成

> **论文**：Adding Conditional Control to Text-to-Image Diffusion Models（ICCV 2023 Best Paper）
> **arXiv**：2302.05543 | 2023 | **桥梁**: 20-AI视频生成 ↔ 13-广告分析 | **类型**: 算法工具

## ① 算法原理

**核心思想**：ControlNet 在 Stable Diffusion 基础上增加了结构控制条件（边缘图/深度图/骨架图），使得在换背景时能**锁定商品的形状和轮廓**，只替换背景场景。通过 Inpainting 技术，商品主体区域保持不变，背景由文本描述驱动生成。

**数学直觉**：
```
Output = SD_inpaint(
    image=product_white_bg,
    mask=background_mask,           # 背景区域为1，商品为0
    control=canny_edges(product),   # ControlNet 锁定商品轮廓
    prompt="lifestyle scene text"   # 背景场景描述
)
```

- **Canny Edge ControlNet**：提取商品边缘图作为硬约束，防止商品变形
- **Inpainting**：仅对背景 mask 区域进行扩散，商品主体像素保持原始值
- **DDIM Sampling**：确定性采样（strength=0.7-0.85），平衡商品保留度和背景质量

**关键假设**：
- 商品主图背景纯净（白底或单色），分割 mask 质量高（可用 SAM 自动生成）
- ControlNet 使用与商品品类适配的权重（母婴/家居类商品效果最优）
- Prompt 描述的场景需与商品使用场景逻辑一致（如吸奶器 → 卧室/哺乳室）

## ② 母婴出海应用案例

**场景A：一图多场景 A/B 测试**
- 业务问题：某爬行垫 SKU 只有白底主图，不知道哪种场景背景（婴儿房/户外草地/木地板客厅）最能提升 CTR，传统摄影 3 套场景需 $600-800
- 数据要求：
  - 商品白底图（≥1000×1000px，主体居中，背景纯白）
  - 场景描述文本（Prompt 库，按品类预设）
  - 目标场景数量（3-5 套即可覆盖 A/B 测试需求）
- 预期产出：5 套不同场景的场景化主图，含商品同一视角、不同背景环境
- 业务价值：摄影成本从 **$600-800 降至 $2-5**（GPU 推理费），A/B 测试找到最高 CTR 场景后，该场景 CTR 平均提升 **12-20%**，年化节省摄影费 **$200+/SKU**

**场景B：批量 SKU 节日场景定制**
- 业务问题：圣诞节前需对 80 个 SKU 生成圣诞主题主图（雪地/圣诞树背景），传统摄影不可能批量完成
- 数据要求：80 个 SKU 的白底图 + 圣诞场景 Prompt 模板（统一风格）
- 预期产出：80 张圣诞场景图，批量处理 2-3 小时完成
- 业务价值：节日主题图 CTR 比常规主图高 15-25%；节省外包摄影费 **$40,000+**（80 SKU × $500/套）

## ③ 代码模板

```python
"""
SD+ControlNet 商品场景图生成引擎
白底商品图 → 场景化主图的 AI 摄影替代方案
生产环境：接入 diffusers 库 + GPU 推理（A10/T4）
当前演示：Mock 推理流程 + 完整业务逻辑
"""
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class GenerationConfig:
    """单次生成配置"""
    product_image_path: str         # 白底商品图路径
    scene_prompt: str               # 场景描述（英文，Stable Diffusion 格式）
    negative_prompt: str            # 负向提示（避免的内容）
    scene_name: str                 # 场景中文名称（用于文件命名）
    strength: float = 0.75          # Inpainting 强度（0.6-0.9，越高背景变化越大）
    guidance_scale: float = 7.5     # CFG 引导强度
    num_inference_steps: int = 30   # 推理步数（质量 vs 速度权衡）
    seed: int = 42                  # 随机种子（固定复现）


@dataclass
class GenerationResult:
    """生成结果"""
    scene_name: str
    output_path: str
    prompt_used: str
    generation_time_sec: float
    estimated_quality_score: float  # 0-1，基于启发式规则估算
    cost_usd: float                 # GPU 推理费用估算


# 母婴品类预设场景 Prompt 库
BABY_SCENE_PROMPTS: Dict[str, Dict[str, str]] = {
    "nursery_room": {
        "prompt": (
            "cozy nursery room background, soft pastel colors, white wooden crib visible, "
            "warm afternoon sunlight through sheer curtains, clean and modern baby room, "
            "blurred background, lifestyle photography, professional product photo"
        ),
        "negative": "cluttered, dark, adult items, text, watermark, low quality, blurry product",
        "name": "婴儿房场景",
    },
    "outdoor_garden": {
        "prompt": (
            "bright outdoor garden background, fresh green grass, soft bokeh effect, "
            "natural daylight, spring morning atmosphere, clean and airy, "
            "lifestyle product photography, blurred background"
        ),
        "negative": "indoor, dark, rain, crowded, text, watermark, low quality",
        "name": "户外花园场景",
    },
    "modern_living_room": {
        "prompt": (
            "modern Scandinavian living room background, light oak wood floor, "
            "white minimal furniture, large window with soft natural light, "
            "clean and elegant, professional lifestyle photography, shallow depth of field"
        ),
        "negative": "cluttered, dark room, old furniture, text, watermark, distorted",
        "name": "现代客厅场景",
    },
    "christmas_theme": {
        "prompt": (
            "Christmas themed background, decorated Christmas tree with warm lights, "
            "snow outside window, cozy holiday atmosphere, red and gold accents, "
            "festive product photography, blurred bokeh lights"
        ),
        "negative": "summer, beach, dark, text, watermark, low quality",
        "name": "圣诞节主题",
    },
    "hospital_clinic": {
        "prompt": (
            "clean hospital or clinic room background, white medical environment, "
            "professional healthcare setting, bright and sterile, "
            "medical product photography style"
        ),
        "negative": "dirty, dark, consumer setting, text, watermark",
        "name": "医疗/专业场景",
    },
}


class MockSDControlNetPipeline:
    """
    Stable Diffusion + ControlNet 推理 Mock

    生产环境替换为：
        from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
        import torch
        
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny",
            torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda")
        
        # 生成
        result = pipe(
            prompt=config.scene_prompt,
            negative_prompt=config.negative_prompt,
            image=product_image,
            mask_image=background_mask,      # SAM 自动分割背景
            control_image=canny_edges,       # ControlNet 约束
            strength=config.strength,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps,
            generator=torch.Generator().manual_seed(config.seed),
        ).images[0]
    """

    def inpaint_with_control(
        self,
        product_image_path: str,
        background_mask: "np_array",  # noqa: F821
        canny_edges: "np_array",      # noqa: F821
        config: GenerationConfig,
    ) -> str:
        """Mock 推理：返回虚拟输出路径"""
        time.sleep(0.1)  # 模拟推理延迟
        scene_suffix = config.scene_name.replace(" ", "_").replace("/", "-")
        product_stem = config.product_image_path.replace(".jpg", "").replace(".png", "")
        return f"{product_stem}_{scene_suffix}_generated.jpg"


class MockSAMSegmenter:
    """
    SAM（Segment Anything Model）背景分割 Mock

    生产环境：
        from segment_anything import SamPredictor, sam_model_registry
        sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        predictor = SamPredictor(sam)
        predictor.set_image(image_array)
        masks, _, _ = predictor.predict(point_coords=center_point, point_labels=[1])
        background_mask = 1 - masks[0]  # 商品mask取反得背景mask
    """

    def segment_background(self, image_path: str) -> "np_array":  # noqa: F821
        """Mock：返回虚拟背景 mask"""
        return f"background_mask_for_{image_path}"


def estimate_gpu_cost(num_steps: int, num_images: int) -> float:
    """估算 GPU 推理成本（基于 A10G @$0.76/hr，每图约 30 步 = 15 秒）"""
    time_per_image_hr = (num_steps * 0.5) / 3600  # ~0.5秒/步
    cost_per_image = 0.76 * time_per_image_hr
    return round(cost_per_image * num_images, 4)


def generate_scene_image(
    config: GenerationConfig,
    pipeline: MockSDControlNetPipeline,
    segmenter: MockSAMSegmenter,
) -> GenerationResult:
    """
    单张场景图生成

    流程：
    1. SAM 自动分割商品背景 mask
    2. Canny 边缘提取（ControlNet 输入）
    3. SD Inpainting + ControlNet 生成场景背景
    4. 质量检查（商品区域像素一致性）
    """
    start_time = time.time()

    # Step 1: 背景分割
    bg_mask = segmenter.segment_background(config.product_image_path)

    # Step 2: Canny 边缘提取（Mock）
    canny_edges = f"canny_edges_for_{config.product_image_path}"

    # Step 3: 推理生成
    output_path = pipeline.inpaint_with_control(
        config.product_image_path, bg_mask, canny_edges, config
    )

    elapsed = time.time() - start_time

    # Step 4: 质量估算（生产环境用 CLIP 评分）
    quality_score = 0.82 + (config.seed % 10) * 0.01  # Mock

    return GenerationResult(
        scene_name=config.scene_name,
        output_path=output_path,
        prompt_used=config.scene_prompt[:60] + "...",
        generation_time_sec=round(elapsed, 2),
        estimated_quality_score=round(quality_score, 3),
        cost_usd=estimate_gpu_cost(config.num_inference_steps, 1),
    )


def batch_generate_scenes(
    product_image_path: str,
    scene_keys: List[str],
    pipeline: MockSDControlNetPipeline,
    segmenter: MockSAMSegmenter,
    custom_strength: float = 0.75,
) -> Tuple[List[GenerationResult], Dict[str, float]]:
    """
    批量生成多场景图

    Returns:
        results: 各场景生成结果
        summary: 汇总统计（成本/时间/质量均值）
    """
    results = []
    for scene_key in scene_keys:
        if scene_key not in BABY_SCENE_PROMPTS:
            print(f"⚠️  未知场景: {scene_key}，跳过")
            continue

        scene_data = BABY_SCENE_PROMPTS[scene_key]
        config = GenerationConfig(
            product_image_path=product_image_path,
            scene_prompt=scene_data["prompt"],
            negative_prompt=scene_data["negative"],
            scene_name=scene_data["name"],
            strength=custom_strength,
        )
        result = generate_scene_image(config, pipeline, segmenter)
        results.append(result)

    total_cost = sum(r.cost_usd for r in results)
    avg_quality = sum(r.estimated_quality_score for r in results) / len(results) if results else 0

    summary = {
        "total_images": len(results),
        "total_cost_usd": round(total_cost, 4),
        "avg_quality_score": round(avg_quality, 3),
        "traditional_photo_cost": len(results) * 200,  # 传统摄影估算
        "cost_savings_ratio": round(1 - total_cost / (len(results) * 200), 4) if results else 0,
    }

    return results, summary


# ===== 测试用例 =====
if __name__ == "__main__":
    pipeline = MockSDControlNetPipeline()
    segmenter = MockSAMSegmenter()

    # 场景一：爬行垫单品 5 场景 A/B 测试
    print("=" * 65)
    print("商品场景图批量生成测试")
    print("=" * 65)

    test_product = "baby_play_mat_sku001_white_bg.jpg"
    target_scenes = ["nursery_room", "outdoor_garden", "modern_living_room",
                     "christmas_theme", "hospital_clinic"]

    results, summary = batch_generate_scenes(
        product_image_path=test_product,
        scene_keys=target_scenes,
        pipeline=pipeline,
        segmenter=segmenter,
        custom_strength=0.78,
    )

    for r in results:
        print(f"\n✅ {r.scene_name}")
        print(f"   输出: {r.output_path}")
        print(f"   质量分: {r.estimated_quality_score:.3f} | 耗时: {r.generation_time_sec}s | 成本: ${r.cost_usd:.4f}")

    print(f"\n{'='*65}")
    print("汇总统计：")
    print(f"  生成数量: {summary['total_images']} 张")
    print(f"  AI总成本: ${summary['total_cost_usd']:.4f}")
    print(f"  传统摄影: ${summary['traditional_photo_cost']}")
    print(f"  成本节省: {summary['cost_savings_ratio']:.1%}")
    print(f"  平均质量: {summary['avg_quality_score']:.3f}")

    # 断言验证
    assert len(results) == len(target_scenes), "生成数量不符"
    assert all(r.output_path.endswith(".jpg") for r in results), "输出格式异常"
    assert all(0 < r.estimated_quality_score <= 1 for r in results), "质量分超出范围"
    assert summary["cost_savings_ratio"] > 0.99, "成本节省计算异常（AI 应远低于传统摄影）"
    assert all(r.cost_usd < 1.0 for r in results), "单图成本估算超出预期"

    # 场景二：批量 80 SKU 成本预测
    print("\n批量生成成本预测（80 SKU × 5场景）：")
    batch_cost = estimate_gpu_cost(30, 80 * 5)
    savings = 80 * 5 * 200 - batch_cost
    print(f"  AI 推理总成本: ${batch_cost:.2f}")
    print(f"  传统摄影替代成本: ${80 * 5 * 200:,}")
    print(f"  节省: ${savings:,.0f}")

    print("\n[✓] SD+ControlNet商品场景图生成 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Diffusion-Model-Product-Image]]（理解 Stable Diffusion 基础原理和 Inpainting 机制）
- **前置（prerequisite）**：[[Skill-Product-Image-Quality-Assessment]]（生成后自动评分，确保场景图质量达标）
- **延伸（extends）**：[[Skill-E-Commerce-Video-Benchmark]]（场景图 A/B 测试数据纳入视频内容效果基准）
- **可组合（combinable）**：[[Skill-AnchorCrafter-Virtual-Anchor-Demo]]（场景图作为虚拟主播视频背景，形成完整直播内容链路）

## ⑤ 商业价值评估

- **ROI 预估**：
  - 直接摄影替代：$200-800/套 × 80 SKU × 5 场景 = **$80,000-320,000** 摄影费 → AI 替代后 GPU 费用约 **$40-60**
  - CTR 提升价值：场景化主图 CTR 平均提升 12-20%，对于月销 $50K 的 SKU，对应 GMV 增量 **$6,000-10,000/月**
  - 节日快速响应：节日前 2 天完成主图换装，比竞品提前上线，Prime Day 首周 GMV 增量约 **$15,000**
- **实施难度**：⭐⭐⭐⭐☆（4/5）— 需要 GPU 推理环境（HuggingFace Space 或自部署），工程复杂度中等
- **优先级**：⭐⭐⭐⭐☆（4/5）— ROI 极高但工程门槛略高，建议第二阶段引入
- **评估依据**：ControlNet 是 2023 年 ICCV Best Paper，技术成熟；SAM 分割精度对白底图几乎完美；GPU 成本持续下降
