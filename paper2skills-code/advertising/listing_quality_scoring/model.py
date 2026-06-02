"""
Skill-Listing-Quality-Scoring
基于 KDD'23 Amazon + EMNLP'24 IPL + IEEE Big Data 2025 MetaSynth
母婴跨境电商 Amazon Listing 质量评分工具
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ── 数据结构 ───────────────────────────────────────────────
@dataclass
class ListingInput:
    asin: str
    title: str
    bullet_points: list[str]          # 最多5条
    description: str = ""
    main_image_url: Optional[str] = None
    has_aplus: bool = False
    has_video: bool = False
    price: float = 0.0
    category: str = "baby"
    review_count: int = 0
    rating: float = 0.0


@dataclass
class ListingScore:
    asin: str
    total_score: float                # 0-100
    title_score: float
    bullets_score: float
    description_score: float
    image_score: float
    completeness_score: float
    grade: str                        # A/B/C/D
    top_issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


# ── 标题质量评分 ───────────────────────────────────────────
def score_title(title: str, category: str = "baby") -> tuple[float, list[str]]:
    """
    评分维度（参考 KDD'23 Amazon 框架）：
    - 长度合规性（Amazon 推荐 150-200 字符）
    - 关键词密度（品类核心词覆盖率）
    - 数字/规格信息（量化描述）
    - 可读性（避免堆砌关键词）
    """
    issues = []
    score = 100.0

    # 1. 长度检查
    length = len(title)
    if length < 80:
        score -= 20
        issues.append(f"标题过短 ({length}字符)，建议 150-200 字符，补充规格/场景描述")
    elif length > 200:
        score -= 10
        issues.append(f"标题过长 ({length}字符)，超出 Amazon 推荐上限，可能被截断")

    # 2. 关键词覆盖（母婴消毒器品类示例）
    CATEGORY_KEYWORDS = {
        "baby": ["baby", "sterilizer", "BPA-free", "UV", "steam",
                 "bottle", "pacifier", "infant", "newborn", "safe"],
        "general": ["product", "quality", "premium", "set", "pack"],
    }
    keywords = CATEGORY_KEYWORDS.get(category, CATEGORY_KEYWORDS["general"])
    title_lower = title.lower()
    covered = sum(1 for kw in keywords if kw.lower() in title_lower)
    keyword_ratio = covered / len(keywords)
    if keyword_ratio < 0.3:
        score -= 15
        issues.append(f"品类核心关键词覆盖率低 ({keyword_ratio:.0%})，建议加入: {', '.join(keywords[:3])}")
    elif keyword_ratio < 0.5:
        score -= 7

    # 3. 量化描述（数字/规格）
    numbers = re.findall(r'\d+', title)
    if not numbers:
        score -= 10
        issues.append("标题缺少量化描述（如容量/时间/件数），建议加入规格数字增强说服力")

    # 4. 可读性（避免全大写/特殊符号堆砌）
    caps_ratio = sum(1 for c in title if c.isupper()) / max(len(title), 1)
    if caps_ratio > 0.5:
        score -= 10
        issues.append("大写字母比例过高，影响可读性")

    special_chars = len(re.findall(r'[|!@#$%^&*]', title))
    if special_chars > 3:
        score -= 8
        issues.append(f"特殊符号过多 ({special_chars}个)，建议使用逗号或短横线替代")

    return max(0.0, score), issues


# ── Bullet Points 质量评分 ─────────────────────────────────
def score_bullets(bullets: list[str]) -> tuple[float, list[str]]:
    """
    评分维度：
    - 数量（5条满分）
    - 每条长度（Amazon 推荐 100-150字符）
    - 利益点而非特性（feature vs benefit）
    - 量化描述覆盖
    """
    issues = []
    score = 100.0

    # 数量
    count = len(bullets)
    if count < 3:
        score -= 30
        issues.append(f"Bullet points 仅 {count} 条，强烈建议补充至 5 条")
    elif count < 5:
        score -= 10
        issues.append(f"Bullet points {count}/5 条，补充至5条可提升展示完整度")

    if count == 0:
        return 0.0, issues

    # 逐条评估
    short_count = 0
    no_benefit_count = 0
    no_number_count = 0

    benefit_words = ["ensures", "helps", "prevents", "reduces", "protects",
                     "eliminates", "provides", "delivers", "supports", "keeps"]

    for i, bullet in enumerate(bullets[:5], 1):
        if len(bullet) < 50:
            short_count += 1
        if not any(bw in bullet.lower() for bw in benefit_words):
            no_benefit_count += 1
        if not re.search(r'\d+', bullet):
            no_number_count += 1

    if short_count >= 2:
        score -= 15
        issues.append(f"{short_count} 条 bullet 过短，建议每条 100-150 字符，展开描述使用场景")
    if no_benefit_count >= 3:
        score -= 12
        issues.append("多数 bullet 描述特性而非利益，建议以 'Ensures/Protects/Eliminates' 等动词开头")
    if no_number_count >= 3:
        score -= 10
        issues.append("缺少量化数据（如 '99.9% bacteria elimination in 3 min'），量化描述显著提升可信度")

    return max(0.0, score), issues


# ── 内容完整度评分（IPL 11维框架简化版）─────────────────────
def score_completeness(listing: ListingInput) -> tuple[float, list[str]]:
    """
    参考 EMNLP'24 IPL 的多维加权评分框架
    quality_score = Σ(wᵢ × featureᵢ)
    """
    issues = []
    components = {}

    # 描述完整度 (w=0.15)
    desc_score = min(100.0, len(listing.description) / 1000 * 100) if listing.description else 0
    components["description"] = desc_score * 0.15
    if desc_score < 50:
        issues.append("Product Description 内容不足，建议 500+ 字符，覆盖使用场景和安全认证")

    # A+ 内容 (w=0.15)
    components["aplus"] = 100.0 * 0.15 if listing.has_aplus else 0
    if not listing.has_aplus:
        issues.append("缺少 A+ Content，竞争激烈品类中 A+ 可提升 CVR 3-10%")

    # 视频 (w=0.10)
    components["video"] = 100.0 * 0.10 if listing.has_video else 0
    if not listing.has_video:
        issues.append("缺少产品视频，母婴品类视频可显著降低退货率")

    # Review 质量 (w=0.15)
    review_score = 0.0
    if listing.review_count >= 50 and listing.rating >= 4.0:
        review_score = 100.0
    elif listing.review_count >= 10:
        review_score = min(100.0, listing.review_count * 1.5)
        if listing.rating < 4.0:
            review_score *= 0.7
    components["reviews"] = review_score * 0.15

    total = sum(components.values()) / 0.55  # 归一化到 0-100
    return min(100.0, total), issues


# ── 图片质量估算（无 CV 模型的轻量代理指标）──────────────────
def score_image_proxy(listing: ListingInput) -> tuple[float, list[str]]:
    """
    无 CV 模型时的代理评分（参考 Mercari Image Score 框架的维度定义）
    实际部署建议替换为 CLIP 微调模型（arXiv:2408.11349）
    """
    issues = []
    score = 60.0  # 基础分（有图片即得）

    if listing.main_image_url is None:
        return 0.0, ["缺少主图 URL，无法评估图片质量"]

    # URL 格式代理指标
    url = listing.main_image_url.lower()
    if any(ext in url for ext in ['.jpg', '.jpeg', '.png', '.webp']):
        score += 10
    if 'SL1500' in listing.main_image_url or 'SL2000' in listing.main_image_url:
        score += 20  # Amazon 高分辨率图片标志
        score += 10  # 推测白底合规

    if score < 80:
        issues.append("图片质量代理评分偏低，建议确认: ①白色背景 ②≥1000×1000px ③产品占图面积≥85%")

    return min(100.0, score), issues


# ── 综合评分 ──────────────────────────────────────────────
def score_listing(listing: ListingInput) -> ListingScore:
    """
    综合评分（参考 IPL 权重体系 + KDD'23 Amazon 双模态框架）

    权重分配：
    - 标题质量:     25% （搜索排名 + 首印象）
    - Bullet points: 25% （转化决策核心）
    - 图片质量:     25% （视觉吸引力）
    - 内容完整度:   15% （A+/视频/描述）
    - Review 质量:  10% （社会证明，间接）
    """
    title_score, title_issues = score_title(listing.title, listing.category)
    bullets_score, bullet_issues = score_bullets(listing.bullet_points)
    image_score, image_issues = score_image_proxy(listing)
    completeness_score, complete_issues = score_completeness(listing)

    # 描述单独提取
    desc_score = min(100.0, len(listing.description) / 1000 * 100) if listing.description else 0

    # 加权综合
    total = (
        title_score * 0.25 +
        bullets_score * 0.25 +
        image_score * 0.25 +
        completeness_score * 0.15 +
        min(100.0, listing.review_count * 2) * 0.10
    )

    # 等级划分
    if total >= 85:
        grade = "A"
    elif total >= 70:
        grade = "B"
    elif total >= 55:
        grade = "C"
    else:
        grade = "D"

    # 汇总问题并按影响力排序
    all_issues = title_issues + bullet_issues + image_issues + complete_issues
    top_issues = all_issues[:5]

    # 生成推荐（基于最低分维度）
    recommendations = _generate_recommendations(
        title_score, bullets_score, image_score, completeness_score, listing
    )

    return ListingScore(
        asin=listing.asin,
        total_score=round(total, 1),
        title_score=round(title_score, 1),
        bullets_score=round(bullets_score, 1),
        description_score=round(desc_score, 1),
        image_score=round(image_score, 1),
        completeness_score=round(completeness_score, 1),
        grade=grade,
        top_issues=top_issues,
        recommendations=recommendations,
    )


def _generate_recommendations(title_s, bullets_s, image_s, complete_s, listing: ListingInput) -> list[str]:
    recs = []
    scores = {
        "标题优化": title_s,
        "Bullet Points 改写": bullets_s,
        "图片升级": image_s,
        "内容完善 (A+/视频)": complete_s,
    }
    # 按分数升序，取最低的3个
    for dim, s in sorted(scores.items(), key=lambda x: x[1])[:3]:
        if dim == "标题优化" and title_s < 70:
            recs.append(f"[高优] 重写标题：加入品类核心词 + 量化规格，目标长度 150-200 字符")
        elif dim == "Bullet Points 改写" and bullets_s < 70:
            recs.append(f"[高优] 改写 Bullet Points：以利益点开头 (Ensures/Eliminates)，每条加入量化数据")
        elif dim == "图片升级" and image_s < 70:
            recs.append(f"[中优] 主图升级：白底 + ≥1500px + 产品占面积 85%+，参考竞品 BSR Top 3")
        elif dim == "内容完善 (A+/视频)" and not listing.has_aplus:
            recs.append(f"[中优] 制作 A+ Content：竞争品类中 A+ 平均提升 CVR 5-8%")
    return recs


# ── 批量评分 + 优先级排序 ─────────────────────────────────
def batch_score_and_prioritize(listings: list[ListingInput]) -> list[ListingScore]:
    """批量评分，按优化 ROI 降序排列（分数越低优化空间越大）"""
    scores = [score_listing(l) for l in listings]
    return sorted(scores, key=lambda s: s.total_score)


# ── 示例：baby sterilizer SKU 评分 ────────────────────────
if __name__ == "__main__":
    # 模拟两个 baby sterilizer SKU
    listings = [
        ListingInput(
            asin="B0EXAMPLE1",
            title="Baby Bottle Sterilizer UV-C Sterilizer for Baby Bottles Pacifiers Toys",
            bullet_points=[
                "UV-C sterilization eliminates 99.9% of harmful bacteria and viruses in just 3 minutes",
                "Fits all standard baby bottles including Dr. Brown's, Philips Avent, and Tommee Tippee",
                "BPA-free food-grade materials ensure complete safety for your newborn",
                "Compact design with auto-shutoff protects against overheating",
                "USB-C rechargeable — lasts 30+ cycles per charge, perfect for travel",
            ],
            description="Keep your baby safe with hospital-grade UV-C sterilization. " * 10,
            main_image_url="https://m.media-amazon.com/images/I/example_SL1500_.jpg",
            has_aplus=True,
            has_video=False,
            price=49.99,
            category="baby",
            review_count=284,
            rating=4.6,
        ),
        ListingInput(
            asin="B0EXAMPLE2",
            title="UV Sterilizer Baby",
            bullet_points=[
                "Good for bottles",
                "Safe material",
            ],
            description="",
            main_image_url="https://m.media-amazon.com/images/I/example2.jpg",
            has_aplus=False,
            has_video=False,
            price=39.99,
            category="baby",
            review_count=12,
            rating=3.8,
        ),
    ]

    print("=" * 60)
    print("Baby Sterilizer SKU Listing 质量评分报告")
    print("=" * 60)

    results = batch_score_and_prioritize(listings)

    for r in results:
        print(f"\nASIN: {r.asin}  综合评分: {r.total_score}/100  等级: {r.grade}")
        print(f"  标题: {r.title_score:.0f} | Bullets: {r.bullets_score:.0f} | "
              f"图片: {r.image_score:.0f} | 完整度: {r.completeness_score:.0f}")
        if r.top_issues:
            print("  主要问题:")
            for issue in r.top_issues[:3]:
                print(f"    ⚠️  {issue}")
        if r.recommendations:
            print("  优化建议:")
            for rec in r.recommendations:
                print(f"    → {rec}")

    # 广告投放门控示例
    print("\n" + "=" * 60)
    print("广告投放质量门控")
    print("=" * 60)
    for r in results:
        status = "✅ 达标，可投放" if r.total_score >= 70 else "❌ 未达标，需先优化 Listing"
        print(f"  {r.asin}: {r.total_score:.0f}分 ({r.grade}级) → {status}")
