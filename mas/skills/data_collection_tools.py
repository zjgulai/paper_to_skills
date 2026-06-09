"""22-数据采集工程 Skill 工具.

涵盖:
  - 竞品信息爬取 (品类竞品价格/销量/评分)
  - 实时市场信号检测 (关键词趋势/搜索量/热度)
  - 数据质量评估 (完整性/一致性/异常值)
  - 商品信息提取 (URL解析/结构化字段)
  - 假评论检测 (机器生成/刷单检测)
  - 价格历史采集 (ASIN价格时序)
"""

from __future__ import annotations

from typing import Any, Dict, List


def web_crawl_competitors(category: str, max_results: int = 10) -> Dict[str, Any]:
    """爬取竞品信息，基于 Skill-Competitor-Intelligence-Crawl.

    当前使用 mock 数据；生产环境替换为真实爬虫或 API。
    返回: competitors 列表，含价格/评分/月销量/关键词
    """
    mock_competitors = [
        {
            "asin": f"B0COMP{i:04d}",
            "title": f"{category} Competitor Product {i}",
            "price_usd": round(19.99 + i * 4.5, 2),
            "avg_rating": round(3.8 + (i % 3) * 0.2, 1),
            "review_count": 100 + i * 50,
            "monthly_sales_est": 1000 + i * 300,
            "bsr": 5000 - i * 200,
            "seller_type": "FBA" if i % 2 == 0 else "FBM",
        }
        for i in range(1, min(max_results + 1, 11))
    ]

    avg_price = sum(c["price_usd"] for c in mock_competitors) / len(mock_competitors) if mock_competitors else 0.0
    avg_rating = sum(c["avg_rating"] for c in mock_competitors) / len(mock_competitors) if mock_competitors else 0.0

    return {
        "skill": "data_collection_web_crawl",
        "category": category,
        "competitor_count": len(mock_competitors),
        "competitors": mock_competitors,
        "market_avg_price_usd": round(avg_price, 2),
        "market_avg_rating": round(avg_rating, 2),
        "confidence": 0.80,
    }


def detect_market_signal(keywords: List[str]) -> Dict[str, Any]:
    """实时市场信号检测，基于 Skill-Market-Signal-Realtime-Collection.

    当前使用 mock 数据；生产环境替换为 Google Trends / Amazon API。
    返回: 每个关键词的搜索趋势、热度指数、季节性因子
    """
    signals = []
    for i, kw in enumerate(keywords):
        trend_score = round(60.0 + (i % 5) * 8.0, 1)
        signals.append({
            "keyword": kw,
            "search_volume_monthly": 5000 + i * 1200,
            "trend_score": trend_score,             # 0-100，越高越热
            "trend_direction": "rising" if trend_score >= 70 else ("stable" if trend_score >= 50 else "declining"),
            "seasonality_peak_month": (i % 12) + 1,
            "competition_index": round(0.3 + (i % 4) * 0.15, 2),  # 0-1，越高竞争越激烈
        })

    hot_keywords = [s for s in signals if s["trend_direction"] == "rising"]

    return {
        "skill": "data_collection_market_signal",
        "keyword_count": len(keywords),
        "signals": signals,
        "hot_keywords": [s["keyword"] for s in hot_keywords],
        "avg_trend_score": round(sum(s["trend_score"] for s in signals) / max(len(signals), 1), 1),
        "confidence": 0.78,
    }


def assess_data_quality(data: Dict[str, Any]) -> Dict[str, Any]:
    """数据质量评估，基于 Skill-Data-Quality-Assessment.

    检查完整性、一致性、异常值。
    返回: quality_score, issues 列表, pass/fail
    """
    issues = []
    checks = {}

    # 完整性检查
    required_fields = ["asin", "price_usd", "avg_rating", "review_count"]
    missing = [f for f in required_fields if f not in data or data[f] is None]
    checks["completeness"] = len(missing) == 0
    if missing:
        issues.append({"type": "missing_field", "fields": missing, "severity": "high"})

    # 范围合理性检查
    price = data.get("price_usd", 0)
    rating = data.get("avg_rating", 0)
    review_count = data.get("review_count", 0)

    checks["price_valid"] = 0 < price < 10000
    checks["rating_valid"] = 0 <= rating <= 5
    checks["review_count_valid"] = review_count >= 0

    if not checks["price_valid"]:
        issues.append({"type": "out_of_range", "field": "price_usd", "value": price, "severity": "medium"})
    if not checks["rating_valid"]:
        issues.append({"type": "out_of_range", "field": "avg_rating", "value": rating, "severity": "medium"})

    # 一致性检查：高评分应有足够评论支撑
    if rating >= 4.5 and review_count < 10:
        issues.append({"type": "inconsistency", "detail": "high_rating_low_reviews", "severity": "low"})
        checks["consistency"] = False
    else:
        checks["consistency"] = True

    pass_count = sum(1 for v in checks.values() if v)
    quality_score = round(pass_count / max(len(checks), 1), 2)

    return {
        "skill": "data_collection_quality_assessment",
        "quality_score": quality_score,
        "checks": checks,
        "issues": issues,
        "issue_count": len(issues),
        "passed": quality_score >= 0.8 and not any(i["severity"] == "high" for i in issues),
        "confidence": 0.90,
    }


def extract_product_info(url: str) -> Dict[str, Any]:
    """商品信息提取，基于 Skill-Product-Info-Extraction.

    当前使用 mock 数据；生产环境替换为真实爬虫。
    返回: 结构化商品字段 (ASIN, 标题, 价格, 图片, 描述, 变体)
    """
    # 从 URL mock 提取 ASIN
    asin = "B0EXTRACT001"
    if "/dp/" in url:
        parts = url.split("/dp/")
        if len(parts) > 1:
            asin = parts[1].split("/")[0].split("?")[0][:10]

    mock_product = {
        "asin": asin,
        "title": f"Mock Product Title for {asin}",
        "brand": "MockBrand",
        "price_usd": 29.99,
        "currency": "USD",
        "avg_rating": 4.3,
        "review_count": 856,
        "availability": "In Stock",
        "bsr": 1234,
        "category_path": ["Baby", "Feeding", "Bottles"],
        "bullet_points": [
            "BPA-free materials",
            "Anti-colic design",
            "Easy to clean",
        ],
        "description": "High quality baby product with advanced features.",
        "image_count": 6,
        "variant_count": 3,
        "seller_type": "FBA",
        "fulfillment": "Amazon",
    }

    return {
        "skill": "data_collection_product_extraction",
        "source_url": url,
        "product": mock_product,
        "extraction_success": True,
        "fields_extracted": len(mock_product),
        "confidence": 0.85,
    }


def detect_fake_reviews(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    """假评论检测，基于 Skill-Fake-Review-Detection.

    检测机器生成评论、刷单行为、异常评论模式。
    返回: fake_count, fake_rate, suspicious 列表
    """
    suspicious = []
    for i, review in enumerate(reviews):
        flags = []
        text = review.get("text", "")
        rating = review.get("rating", 3)
        verified = review.get("verified_purchase", False)

        # 文本长度异常短
        if len(text) < 20:
            flags.append("too_short")

        # 未经验证购买的极端评分
        if not verified and rating in (1, 5):
            flags.append("unverified_extreme_rating")

        # 重复词语检测（简化：连续重复字符）
        if text and len(set(text.lower())) < len(text.lower()) * 0.3:
            flags.append("low_vocabulary_diversity")

        # 日期异常（同一天大量5星评论，mock 检测）
        if review.get("helpful_votes", 0) == 0 and rating == 5 and not verified:
            flags.append("zero_helpful_unverified_5star")

        if flags:
            suspicious.append({
                "review_index": i,
                "review_id": review.get("id", f"review_{i}"),
                "flags": flags,
                "risk_score": round(len(flags) / 4.0, 2),
            })

    fake_rate = len(suspicious) / max(len(reviews), 1)
    quality_level = "good" if fake_rate < 0.1 else ("moderate" if fake_rate < 0.3 else "poor")

    return {
        "skill": "data_collection_fake_review_detection",
        "total_reviews": len(reviews),
        "suspicious_count": len(suspicious),
        "fake_rate": round(fake_rate, 3),
        "quality_level": quality_level,
        "suspicious_reviews": suspicious,
        "confidence": 0.82,
    }


def collect_price_history(asin: str, days: int = 30) -> Dict[str, Any]:
    """价格历史采集，基于 Skill-Price-History-Collection.

    当前使用 mock 数据；生产环境替换为 Keepa API 或类似服务。
    返回: daily_prices 时序、最高/最低/均价、波动率
    """
    import math

    # 生成 mock 价格时序（带正弦波动）
    base_price = 29.99
    daily_prices = []
    for d in range(days):
        noise = math.sin(d * 0.5) * 2.0 + (d % 7) * 0.3  # 周期性波动
        price = round(base_price + noise, 2)
        daily_prices.append({
            "day_offset": d,
            "price_usd": price,
            "was_on_sale": price < base_price - 1.5,
        })

    prices = [p["price_usd"] for p in daily_prices]
    min_price = min(prices)
    max_price = max(prices)
    avg_price = sum(prices) / len(prices)
    volatility = round((max_price - min_price) / avg_price, 3)

    sale_days = [p for p in daily_prices if p["was_on_sale"]]

    return {
        "skill": "data_collection_price_history",
        "asin": asin,
        "days_collected": days,
        "daily_prices": daily_prices,
        "price_stats": {
            "min_usd": round(min_price, 2),
            "max_usd": round(max_price, 2),
            "avg_usd": round(avg_price, 2),
            "current_usd": daily_prices[-1]["price_usd"],
            "volatility_rate": volatility,
        },
        "sale_events": len(sale_days),
        "price_trend": "stable" if volatility < 0.1 else ("volatile" if volatility > 0.2 else "moderate"),
        "confidence": 0.88,
    }
