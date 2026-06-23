---
title: Supply Chain Counterfeit Detection — 供应链仿冒品检测原材料/包装视觉验真
doc_type: knowledge
module: 19-风控反欺诈
topic: supply-chain-counterfeit-detection
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Supply-Chain-Counterfeit-Detection

## ① 算法原理（≤300字）

**核心问题**：母婴产品（奶粉、奶瓶、吸奶器）供应链中的仿冒风险极高——从原材料掺假到包装仿冒，任何环节出现仿冒品都可能导致产品召回、品牌声誉损失和法律责任。传统人工抽检效率低，需要计算机视觉辅助批量验真。

**双轨验真方法**：

**轨道1：包装视觉比对（哈希指纹）**
- 标准参考图提取特征哈希（pHash：感知哈希）
- 待检图片哈希距离 > 阈值 → 标记异常

pHash 算法：将图片缩放至 32×32 → DCT 变换 → 取均值 → 每像素与均值比较生成二进制指纹。汉明距离（Hamming Distance）衡量相似度：
$$d_H = \sum_{i=1}^{64} (\hat{h}_i \oplus h_i)$$
$d_H < 10$：高度相似（正品）；$d_H > 20$：明显异常（疑似仿冒）

**轨道2：统计异常检测（数值特征）**
- 原材料批次重量分布：正品重量方差 < 5%，仿品方差通常 > 15%
- 包装尺寸比例：仿品包装尺寸往往有 2-5mm 偏差
- 条形码校验：EAN-13 校验位验证 + 前缀授权核查

**综合判定**：视觉异常 OR 统计异常 → 进入人工复核流程。

## ② 母婴出海应用案例（1个，含量化 ROI）

**场景**：某母婴品牌接到线报，竞品用近似包装销售劣质产品混淆消费者（同 ASIN 跟卖）。品牌需要快速核查全球 3PL 仓库中的库存是否存在仿冒品混入（可能来自不明来源退货）。

**数据要求**：产品正品参考图（正面/侧面/底部），待检批次产品照片，包装尺寸重量规格。

**应用**：pHash 比对识别出 12 件与标准图汉明距离 > 25 的异常包装，人工复核确认其中 8 件为近似仿冒，全部下架并向 Amazon 举报。

**量化产出**：阻止仿冒品在自家 Listing 销售（售后投诉可归咎于正品），年化避免品质投诉风险和品牌损失 **50-100 万元**。

## ③ 代码模板

```python
import numpy as np

def compute_phash(image_array: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """
    计算感知哈希（pHash）- 简化版（不需要 PIL）
    image_array: 灰度图像数组 (H, W)
    """
    # 缩放到 hash_size * 4 大小（简化实现）
    target_size = hash_size * 4
    h, w = image_array.shape

    # 简单双线性缩放
    resized = np.zeros((target_size, target_size))
    for i in range(target_size):
        for j in range(target_size):
            src_i = int(i * h / target_size)
            src_j = int(j * w / target_size)
            resized[i, j] = image_array[min(src_i, h-1), min(src_j, w-1)]

    # 简化 DCT（取均值代替完整 DCT）
    # 实际应使用 scipy.fft.dct 的完整实现
    block_means = []
    block_h = target_size // hash_size
    for i in range(hash_size):
        for j in range(hash_size):
            block = resized[i*block_h:(i+1)*block_h, j*block_h:(j+1)*block_h]
            block_means.append(np.mean(block))

    block_means = np.array(block_means)
    threshold = np.mean(block_means)
    phash = (block_means > threshold).astype(np.uint8)
    return phash

def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    """计算汉明距离"""
    return int(np.sum(h1 != h2))

def detect_counterfeit_batch(
    reference_image: np.ndarray,
    sample_images: list,
    hamming_threshold: int = 10,
    weight_specs: list = None  # [(actual_g, expected_g), ...]
) -> dict:
    """
    批次仿冒品检测
    reference_image: 正品参考图
    sample_images: 待检样本图列表
    """
    ref_hash = compute_phash(reference_image)
    results = []

    for i, img in enumerate(sample_images):
        sample_hash = compute_phash(img)
        dist = hamming_distance(ref_hash, sample_hash)
        visual_flag = dist > hamming_threshold

        result = {
            'sample_id': i,
            'hamming_distance': dist,
            'visual_anomaly': visual_flag,
            'anomaly_level': 'HIGH' if dist > 20 else 'MEDIUM' if dist > 10 else 'NORMAL'
        }

        # 重量规格检测
        if weight_specs and i < len(weight_specs):
            actual_g, expected_g = weight_specs[i]
            weight_deviation = abs(actual_g - expected_g) / expected_g
            result['weight_deviation_pct'] = weight_deviation * 100
            result['weight_anomaly'] = weight_deviation > 0.05  # 5% 容差

        result['suspicious'] = result['visual_anomaly'] or result.get('weight_anomaly', False)
        results.append(result)

    suspicious_count = sum(1 for r in results if r['suspicious'])
    return {
        'results': results,
        'suspicious_count': suspicious_count,
        'suspicious_rate': suspicious_count / len(sample_images),
        'batch_alert': suspicious_count > 0
    }

# 测试：模拟正品和仿冒品检测
np.random.seed(42)

# 正品参考图（64×64 灰度）
reference = np.random.randint(100, 200, (64, 64)).astype(float)

# 模拟样本：大部分正品，少数仿冒（有明显差异）
samples = []
for i in range(10):
    if i >= 8:  # 最后2个是仿冒品（明显不同）
        fake = np.random.randint(0, 100, (64, 64)).astype(float)  # 完全不同
        samples.append(fake)
    else:
        noise = np.random.randn(64, 64) * 5  # 轻微噪声（正品）
        samples.append(reference + noise)

weight_specs = [(500 + np.random.randn() * 3, 500) for _ in range(8)] + \
               [(480, 500), (510, 500)]  # 最后2个正常

result = detect_counterfeit_batch(reference, samples, hamming_threshold=10, weight_specs=weight_specs)
print(f"检测样本数: {len(result['results'])}")
print(f"可疑样本数: {result['suspicious_count']}")
print(f"可疑率: {result['suspicious_rate']:.0%}")
print(f"⚠️ 批次告警: {'是' if result['batch_alert'] else '否'}")
assert result['batch_alert'], "应触发仿冒品告警"
assert result['suspicious_count'] >= 1
print("[✓] Supply-Chain-Counterfeit-Detection 测试通过")
```


## ④ 技能关联

- 前置技能：[[Skill-Counterfeit-Product-Visual-Detection]]
- 前置技能：[[Skill-Brand-Listing-Hijacking-Detection]]
- 延伸技能：[[Skill-Hijacker-Seller-Network-Analysis]]
- 延伸技能：[[Skill-Supply-Chain-Due-Diligence]]
- 可组合：[[Skill-IP-Trademark-Brand-Monitoring]]
- 可组合：[[Skill-Supplier-Qualification-Onboarding-KPI]]

## ⑤ 商业价值评估

- **ROI量化**: 阻止仿冒品混入，年化避免品质风险和品牌损失 50-100 万元
- **实施难度**: ⭐⭐⭐（需要标准化拍照流程，图像处理基础）
- **优先级**: ⭐⭐⭐⭐（高价值品+大量退货场景的必备核查工具）
