"""
数据接入模块
用于加载广告数据和销量数据
支持文件导入，可扩展到数据库
"""

import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class AdDataLoader:
    """广告数据加载器"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "paper2skills-data" / "ad_data"
        self.data_dir = Path(data_dir)

    def load_daily_stats(self, filename: str = "daily_creative_stats.csv") -> pd.DataFrame:
        """
        加载每日广告素材统计

        Returns:
            DataFrame with columns: date, creative_id, impressions, clicks, conversions, spend
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"数据文件不存在: {filepath}")

        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def get_creatives(self) -> List[str]:
        """获取所有素材ID"""
        df = self.load_daily_stats()
        return df['creative_id'].unique().tolist()

    def get_ctr(self, creative_id: str = None) -> Dict[str, float]:
        """
        计算 CTR

        Args:
            creative_id: 指定素材ID，None则返回所有

        Returns:
            {creative_id: ctr}
        """
        df = self.load_daily_stats()
        df['ctr'] = df['clicks'] / df['impressions']

        if creative_id:
            return {creative_id: df[df['creative_id'] == creative_id]['ctr'].mean()}

        return df.groupby('creative_id')['ctr'].mean().to_dict()

    def get_conversion_rate(self, creative_id: str = None) -> Dict[str, float]:
        """
        计算转化率

        Args:
            creative_id: 指定素材ID，None则返回所有

        Returns:
            {creative_id: cvr}
        """
        df = self.load_daily_stats()
        df['cvr'] = df['conversions'] / df['clicks']

        if creative_id:
            return {creative_id: df[df['creative_id'] == creative_id]['cvr'].mean()}

        return df.groupby('creative_id')['cvr'].mean().to_dict()


class SalesDataLoader:
    """销量数据加载器"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "paper2skills-data" / "sales_data"
        self.data_dir = Path(data_dir)

    def load_daily_sales(self, filename: str = "daily_sales.csv") -> pd.DataFrame:
        """
        加载每日销量数据

        Returns:
            DataFrame with columns: date, sku, category, sales_quantity, revenue, promotion_flag, holiday_flag
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"数据文件不存在: {filepath}")

        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def get_skus(self) -> List[str]:
        """获取所有SKU"""
        df = self.load_daily_sales()
        return df['sku'].unique().tolist()

    def get_category_sales(self, category: str = None) -> pd.DataFrame:
        """
        获取品类销量

        Args:
            category: 指定品类，None则返回所有

        Returns:
            按日期和品类汇总的销量
        """
        df = self.load_daily_sales()
        if category:
            df = df[df['category'] == category]

        return df.groupby(['date', 'category'])['sales_quantity'].sum().reset_index()

    def get_sku_time_series(self, sku: str = None) -> pd.DataFrame:
        """
        获取SKU时间序列

        Args:
            sku: 指定SKU，None则返回所有

        Returns:
            时间序列数据
        """
        df = self.load_daily_sales()
        if sku:
            df = df[df['sku'] == sku]

        return df[['date', 'sku', 'sales_quantity']].sort_values('date')


# ==================== 示例代码 ====================

def main():
    """主函数：演示数据加载"""
    print("=" * 60)
    print("数据加载测试")
    print("=" * 60)

    # 加载广告数据
    print("\n[1] 加载广告数据...")
    ad_loader = AdDataLoader()
    ad_data = ad_loader.load_daily_stats()
    print(f"   数据行数: {len(ad_data)}")
    print(f"   素材列表: {ad_loader.get_creatives()}")
    print(f"   CTR: {ad_loader.get_ctr()}")
    print(f"   CVR: {ad_loader.get_conversion_rate()}")

    # 加载销量数据
    print("\n[2] 加载销量数据...")
    sales_loader = SalesDataLoader()
    sales_data = sales_loader.load_daily_sales()
    print(f"   数据行数: {len(sales_data)}")
    print(f"   SKU列表: {sales_loader.get_skus()}")

    # 获取时间序列
    print("\n[3] 获取时间序列...")
    ts = sales_loader.get_sku_time_series('奶粉_1段')
    print(f"   奶粉_1段销量序列:")
    print(ts.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("数据加载完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()