from __future__ import annotations

from mas.skills.mcp_servers.data_collection_server import DataCollectionServer
from mas.skills.mcp_servers.client import MultiServerMCPClient
from mas.skills.registry import SkillRegistry
from mas.skills.data_collection_tools import (
    web_crawl_competitors,
    detect_market_signal,
    assess_data_quality,
    extract_product_info,
    detect_fake_reviews,
    collect_price_history,
)


def test_data_collection_server_tool_count():
    server = DataCollectionServer()
    assert len(server.list_tools()) == 6


def test_data_collection_server_list_tools_format():
    server = DataCollectionServer()
    for t in server.list_tools():
        assert "name" in t
        assert "description" in t
        assert t["domain"] == "data_collection"


def test_mcp_client_includes_data_collection():
    client = MultiServerMCPClient()
    domains = {t["domain"] for t in client.list_all_tools()}
    assert "data_collection" in domains


def test_mcp_client_total_tools_increased():
    client = MultiServerMCPClient()
    assert client.total_tools >= 33


def test_web_crawl_competitors_direct():
    result = web_crawl_competitors("baby_bottle", max_results=5)
    assert result["skill"] == "data_collection_web_crawl"
    assert result["category"] == "baby_bottle"
    assert result["competitor_count"] == 5
    assert len(result["competitors"]) == 5
    assert isinstance(result["market_avg_price_usd"], float)
    assert 0 < result["confidence"] <= 1.0


def test_web_crawl_via_mcp():
    server = DataCollectionServer()
    result = server.call_tool("data_collection_web_crawl", category="diaper", max_results=3)
    assert result["skill"] == "data_collection_web_crawl"
    assert result["competitor_count"] == 3


def test_detect_market_signal_direct():
    keywords = ["organic baby food", "anti-colic bottle", "BPA free sippy cup"]
    result = detect_market_signal(keywords)
    assert result["skill"] == "data_collection_market_signal"
    assert result["keyword_count"] == 3
    assert len(result["signals"]) == 3
    assert isinstance(result["avg_trend_score"], float)
    for s in result["signals"]:
        assert s["trend_direction"] in ("rising", "stable", "declining")


def test_detect_market_signal_via_mcp():
    server = DataCollectionServer()
    result = server.call_tool("data_collection_market_signal", keywords=["stroller", "carseat"])
    assert result["skill"] == "data_collection_market_signal"
    assert result["keyword_count"] == 2


def test_assess_data_quality_pass():
    data = {"asin": "B001XXXX", "price_usd": 29.99, "avg_rating": 4.2, "review_count": 350}
    result = assess_data_quality(data)
    assert result["skill"] == "data_collection_quality_assessment"
    assert result["quality_score"] >= 0.8
    assert result["passed"] is True
    assert result["issue_count"] == 0


def test_assess_data_quality_fail_missing_fields():
    data = {"price_usd": 29.99}
    result = assess_data_quality(data)
    assert result["passed"] is False
    assert any(i["type"] == "missing_field" for i in result["issues"])


def test_assess_data_quality_via_mcp():
    server = DataCollectionServer()
    data = {"asin": "B002YYYY", "price_usd": 15.0, "avg_rating": 3.9, "review_count": 100}
    result = server.call_tool("data_collection_quality_assessment", data=data)
    assert result["skill"] == "data_collection_quality_assessment"
    assert "quality_score" in result


def test_extract_product_info_direct():
    result = extract_product_info("https://www.amazon.com/dp/B0TESTPROD/ref=sr")
    assert result["skill"] == "data_collection_product_extraction"
    assert result["extraction_success"] is True
    product = result["product"]
    assert "asin" in product
    assert "price_usd" in product
    assert "avg_rating" in product


def test_extract_product_info_via_mcp():
    server = DataCollectionServer()
    result = server.call_tool("data_collection_product_extraction", url="https://www.amazon.com/dp/B00EXAMPLE")
    assert result["skill"] == "data_collection_product_extraction"
    assert result["extraction_success"] is True


def test_detect_fake_reviews_clean():
    reviews = [
        {"id": "R1", "text": "Great product, my baby loves it! Very easy to clean and durable.", "rating": 5, "verified_purchase": True, "helpful_votes": 12},
        {"id": "R2", "text": "Good quality but a bit pricey compared to alternatives on the market.", "rating": 4, "verified_purchase": True, "helpful_votes": 3},
    ]
    result = detect_fake_reviews(reviews)
    assert result["skill"] == "data_collection_fake_review_detection"
    assert result["total_reviews"] == 2
    assert result["quality_level"] in ("good", "moderate", "poor")


def test_detect_fake_reviews_suspicious():
    reviews = [
        {"id": "R1", "text": "ok", "rating": 5, "verified_purchase": False, "helpful_votes": 0},
        {"id": "R2", "text": "Bad", "rating": 5, "verified_purchase": False, "helpful_votes": 0},
    ]
    result = detect_fake_reviews(reviews)
    assert result["suspicious_count"] > 0
    assert result["fake_rate"] > 0


def test_detect_fake_reviews_via_mcp():
    server = DataCollectionServer()
    reviews = [{"id": "R1", "text": "Excellent product highly recommended", "rating": 5, "verified_purchase": True, "helpful_votes": 5}]
    result = server.call_tool("data_collection_fake_review_detection", reviews=reviews)
    assert result["skill"] == "data_collection_fake_review_detection"


def test_collect_price_history_direct():
    result = collect_price_history("B0ASIN0001", days=7)
    assert result["skill"] == "data_collection_price_history"
    assert result["asin"] == "B0ASIN0001"
    assert result["days_collected"] == 7
    assert len(result["daily_prices"]) == 7
    stats = result["price_stats"]
    assert stats["min_usd"] <= stats["avg_usd"] <= stats["max_usd"]
    assert result["price_trend"] in ("stable", "moderate", "volatile")


def test_collect_price_history_default_days():
    result = collect_price_history("B0DEFAULT")
    assert result["days_collected"] == 30
    assert len(result["daily_prices"]) == 30


def test_collect_price_history_via_mcp():
    server = DataCollectionServer()
    result = server.call_tool("data_collection_price_history", asin="B0VIA0MCP", days=14)
    assert result["skill"] == "data_collection_price_history"
    assert result["asin"] == "B0VIA0MCP"
    assert result["days_collected"] == 14


def test_registry_has_data_collection_domain():
    reg = SkillRegistry()
    assert "data_collection" in reg.all_domains()
    dc_tools = {t.name for t in reg.get_tools_for_domains(["data_collection"])}
    assert "data_collection_web_crawl" in dc_tools
    assert "data_collection_market_signal" in dc_tools
    assert "data_collection_quality_assessment" in dc_tools
    assert "data_collection_product_extraction" in dc_tools
    assert "data_collection_fake_review_detection" in dc_tools
    assert "data_collection_price_history" in dc_tools
    assert len(dc_tools) == 6


def test_mcp_tool_not_found_still_works():
    client = MultiServerMCPClient()
    result = client.call_tool("data_collection_nonexistent_tool")
    assert result["status"] == "tool_not_found"


def test_client_can_route_all_data_collection_tools():
    client = MultiServerMCPClient()
    tool_names = [t["name"] for t in client.list_all_tools() if t["domain"] == "data_collection"]
    assert len(tool_names) == 6
    result = client.call_tool("data_collection_web_crawl", category="toy", max_results=2)
    assert result["skill"] == "data_collection_web_crawl"


def run_all():
    tests = [
        test_data_collection_server_tool_count,
        test_data_collection_server_list_tools_format,
        test_mcp_client_includes_data_collection,
        test_mcp_client_total_tools_increased,
        test_web_crawl_competitors_direct,
        test_web_crawl_via_mcp,
        test_detect_market_signal_direct,
        test_detect_market_signal_via_mcp,
        test_assess_data_quality_pass,
        test_assess_data_quality_fail_missing_fields,
        test_assess_data_quality_via_mcp,
        test_extract_product_info_direct,
        test_extract_product_info_via_mcp,
        test_detect_fake_reviews_clean,
        test_detect_fake_reviews_suspicious,
        test_detect_fake_reviews_via_mcp,
        test_collect_price_history_direct,
        test_collect_price_history_default_days,
        test_collect_price_history_via_mcp,
        test_registry_has_data_collection_domain,
        test_mcp_tool_not_found_still_works,
        test_client_can_route_all_data_collection_tools,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"  ❌ {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ❌ {t.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n数据采集 MCP Server 集成测试: {len(tests) - len(failures)}/{len(tests)} 通过")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(run_all())
