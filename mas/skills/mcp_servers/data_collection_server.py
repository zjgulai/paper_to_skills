from __future__ import annotations

from mas.skills.mcp_servers.base import BaseMCPServer
from mas.skills.data_collection_tools import (
    web_crawl_competitors,
    detect_market_signal,
    assess_data_quality,
    extract_product_info,
    detect_fake_reviews,
    collect_price_history,
)


class DataCollectionServer(BaseMCPServer):
    server_name = "data_collection_server"
    domain = "data_collection"

    def _register_tools(self) -> None:
        self._add(
            "data_collection_web_crawl",
            "竞品信息爬取,品类竞品价格/销量/评分/BSR结构化采集",
            lambda **kw: web_crawl_competitors(
                kw.get("category", ""),
                kw.get("max_results", 10),
            ),
        )
        self._add(
            "data_collection_market_signal",
            "实时市场信号检测,关键词趋势/搜索量/热度/季节性因子",
            lambda **kw: detect_market_signal(kw.get("keywords", [])),
        )
        self._add(
            "data_collection_quality_assessment",
            "数据质量评估,完整性/一致性/异常值检测+quality_score",
            lambda **kw: assess_data_quality(kw.get("data", kw)),
        )
        self._add(
            "data_collection_product_extraction",
            "商品信息提取,URL解析→结构化字段(ASIN/价格/评分/变体)",
            lambda **kw: extract_product_info(kw.get("url", "")),
        )
        self._add(
            "data_collection_fake_review_detection",
            "假评论检测,机器生成/刷单/异常评论模式识别",
            lambda **kw: detect_fake_reviews(kw.get("reviews", [])),
        )
        self._add(
            "data_collection_price_history",
            "价格历史采集,ASIN价格时序+统计(最高/最低/均价/波动率)",
            lambda **kw: collect_price_history(
                kw.get("asin", ""),
                kw.get("days", 30),
            ),
        )
