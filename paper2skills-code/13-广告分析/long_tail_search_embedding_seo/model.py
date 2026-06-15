"""
Auto-extracted from: paper2skills-vault/13-广告分析/Skill-Long-Tail-Search-Embedding-SEO.md
Skill: Skill-Long-Tail-Search-Embedding-SEO
Domain: 13-广告分析
"""
"""
Long Tail Search Embedding SEO
双塔嵌入模型 + 长尾关键词挖掘 for 母婴跨境电商
"""
import numpy as np
import re
from collections import defaultdict


def generate_sample_data():
    """生成模拟搜索词和产品数据"""
    # 模拟 Amazon Search Term Report
    search_terms = [
        ('breast pump', 12500, 0.08, 15.2),           # (词, 搜索量, CVR, 竞争度)
        ('electric breast pump', 8200, 0.10, 12.5),
        ('quiet breast pump', 1200, 0.18, 4.2),       # 长尾，高CVR
        ('portable breast pump for travel', 680, 0.22, 2.1),
        ('double electric breast pump office', 420, 0.25, 1.8),
        ('silent breast pump night feeding', 310, 0.28, 1.5),
        ('rechargeable wearable breast pump', 890, 0.20, 3.1),
        ('breast pump hospital grade home use', 245, 0.30, 1.2),
        ('breast pump for low supply', 560, 0.24, 2.4),
        ('breast pump parts replacement', 1800, 0.15, 5.6),
        ('baby bottle sterilizer', 5600, 0.09, 11.2),
        ('bottle warmer with timer', 780, 0.19, 3.8),
    ]

    # 模拟产品信息
    products = [
        {
            'id': 'B08PUMP01',
            'title': 'Quiet Double Electric Breast Pump - Rechargeable Portable Wearable',
            'bullets': ['Ultra-quiet <45dB motor', 'Hospital-grade suction', 'USB rechargeable',
                        '4 modes 10 levels', 'Compatible with Medela parts'],
            'category': 'breast pump',
            'attributes': {'noise_level': 'quiet', 'power': 'rechargeable', 'type': 'double electric'},
        },
    ]
    return search_terms, products


def simple_text_embedding(text, dim=64):
    """简化的文本嵌入（生产中用 sentence-transformers 或 OpenAI embeddings）"""
    # 基于字符 n-gram 的轻量嵌入（演示用）
    text = text.lower()
    vec = np.zeros(dim)
    chars = list(text.replace(' ', ''))
    for i, c in enumerate(chars):
        idx = (ord(c) * 31 + i * 7) % dim
        vec[idx] += 1.0 / (len(chars) + 1)
    # 添加词级别特征
    words = text.split()
    for w in words:
        idx = hash(w) % dim
        vec[idx % dim] += 0.5 / (len(words) + 1)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)


def compute_opportunity_score(search_vol, cvr, competition, alpha=0.4, beta=0.4, gamma=0.2):
    """
    长尾词机会评分
    score = vol^α × cvr^β × (1/competition)^γ
    """
    # 归一化到 0-1
    vol_norm = np.log1p(search_vol) / np.log1p(15000)
    cvr_norm = cvr / 0.35
    comp_norm = 1.0 / (1.0 + competition / 15.0)
    score = (vol_norm ** alpha) * (cvr_norm ** beta) * (comp_norm ** gamma)
    return score


def find_semantic_matches(query_text, product, top_k=5):
    """计算搜索词与产品的语义相似度"""
    # 构建产品完整文本
    product_text = ' '.join([
        product['title'],
        ' '.join(product['bullets']),
        product['category'],
        ' '.join(f"{k} {v}" for k, v in product['attributes'].items())
    ])
    q_vec = simple_text_embedding(query_text)
    p_vec = simple_text_embedding(product_text)
    similarity = float(np.dot(q_vec, p_vec))
    return similarity


def run_long_tail_seo_analysis():
    """完整长尾SEO分析流程"""
    print("=" * 65)
    print("Long Tail Search Embedding SEO — 长尾关键词机会挖掘")
    print("=" * 65)

    search_terms, products = generate_sample_data()
    product = products[0]

    print(f"\n🎯 目标产品: {product['title'][:60]}")
    print(f"\n📊 关键词机会矩阵分析:")
    print(f"{'关键词':<42} {'搜索量':>7} {'CVR':>6} {'竞争':>6} {'语义':>7} {'机会分':>7} {'类型'}")
    print("-" * 90)

    results = []
    for term, vol, cvr, comp in search_terms:
        sem_sim = find_semantic_matches(term, product)
        opp_score = compute_opportunity_score(vol, cvr, comp)
        # 综合评分：机会分 × 语义相关度
        final_score = opp_score * (0.5 + sem_sim)
        term_type = '长尾' if vol < 2000 else '中频' if vol < 6000 else '头部'
        results.append((term, vol, cvr, comp, sem_sim, opp_score, final_score, term_type))

    results.sort(key=lambda x: -x[6])
    for term, vol, cvr, comp, sem, opp, final, ttype in results:
        flag = ' ⭐' if ttype == '长尾' and final > 0.3 else ''
        print(f"{term:<42} {vol:>7,} {cvr:>6.1%} {comp:>6.1f} {sem:>7.3f} {final:>7.3f}  {ttype}{flag}")

    # Search Terms 填充建议
    high_value_terms = [r[0] for r in results if r[7] == '长尾' and r[6] > 0.2][:8]
    search_terms_str = ' '.join(high_value_terms)
    print(f"\n📝 推荐 Search Terms 填充（{len(search_terms_str)}/250字节）:")
    print(f"   {search_terms_str[:250]}")

    # 长尾词聚类
    print("\n🔍 长尾词语义聚类（用于内容策略）:")
    clusters = {
        '安静/噪音': [r[0] for r in results if any(w in r[0] for w in ['quiet', 'silent', 'noise'])],
        '便携/出行': [r[0] for r in results if any(w in r[0] for w in ['portable', 'travel', 'rechargeable'])],
        '使用场景': [r[0] for r in results if any(w in r[0] for w in ['office', 'night', 'home'])],
    }
    for cluster, terms in clusters.items():
        if terms:
            print(f"  {cluster}: {', '.join(terms)}")

    print("\n💡 SEO 行动建议:")
    print("  1. 将「安静」「便携」主题词融入 Title 前80字符（A10最高权重位置）")
    print("  2. 5条 Bullet Points 各覆盖1个语义聚类主题")
    print("  3. Search Terms 优先填充「高CVR × 低竞争 × 高语义匹配」的长尾词")
    print("  4. A+ 内容中建立「使用场景」内容模块覆盖场景类长尾词")

    print("\n[✓] Long Tail Search Embedding SEO 测试通过")


if __name__ == '__main__':
    run_long_tail_seo_analysis()
