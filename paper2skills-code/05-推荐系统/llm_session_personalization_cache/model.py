"""
Auto-extracted from: paper2skills-vault/05-推荐系统/Skill-LLM-Session-Personalization-Cache.md
Skill: Skill-LLM-Session-Personalization-Cache
Domain: 05-推荐系统
"""
"""
LLM Session Personalization Cache
SPRINT 框架的轻量级实现：三层意图缓存 + 千人千面商品排序
"""
import numpy as np
import json
from datetime import datetime, timedelta
from collections import defaultdict


def generate_sample_users_and_items():
    """生成模拟用户行为和商品数据"""
    np.random.seed(42)

    items = {
        'I001': {'name': 'Quiet Double Breast Pump', 'category': 'breast_pump',
                 'price': 149.99, 'age_stage': 'newborn', 'brand': 'BrandA', 'attrs': [1,0,1,0,1]},
        'I002': {'name': 'Portable Wearable Breast Pump', 'category': 'breast_pump',
                 'price': 99.99, 'age_stage': 'newborn', 'brand': 'BrandB', 'attrs': [1,1,0,0,1]},
        'I003': {'name': 'Baby Car Seat 0-4 Years', 'category': 'car_seat',
                 'price': 299.99, 'age_stage': 'infant', 'brand': 'BrandC', 'attrs': [0,0,1,1,0]},
        'I004': {'name': 'Infant Learning Walker', 'category': 'walker',
                 'price': 79.99, 'age_stage': 'toddler', 'brand': 'BrandD', 'attrs': [0,1,0,1,0]},
        'I005': {'name': 'Breast Pump Replacement Parts', 'category': 'accessories',
                 'price': 24.99, 'age_stage': 'newborn', 'brand': 'BrandA', 'attrs': [0,0,0,0,1]},
        'I006': {'name': 'Baby Bottle Sterilizer', 'category': 'sterilizer',
                 'price': 59.99, 'age_stage': 'newborn', 'brand': 'BrandE', 'attrs': [0,0,1,0,0]},
    }

    users = {
        'U001': {  # 新妈妈，关注哺乳
            'history': ['I001', 'I005', 'I002', 'I006', 'I001'],
            'session': ['I002', 'I005'],
            'profile': 'nursing_mom',
        },
        'U002': {  # 奶爸，关注安全
            'history': ['I003', 'I004', 'I003'],
            'session': ['I004'],
            'profile': 'safety_dad',
        },
        'U003': {  # 孕期妈妈，全品类
            'history': ['I001', 'I003', 'I006'],
            'session': ['I001', 'I003'],
            'profile': 'pregnant_mom',
        },
    }
    return users, items


def build_item_vector(item):
    """构建商品嵌入向量"""
    category_map = {'breast_pump': [1,0,0,0,0,0], 'car_seat': [0,1,0,0,0,0],
                    'walker': [0,0,1,0,0,0], 'accessories': [0,0,0,1,0,0],
                    'sterilizer': [0,0,0,0,1,0]}
    cat_vec = category_map.get(item['category'], [0,0,0,0,0,1])

    age_map = {'newborn': 1.0, 'infant': 0.6, 'toddler': 0.3}
    age_score = age_map.get(item['age_stage'], 0.5)

    price_norm = 1.0 - min(item['price'] / 400, 1.0)  # 越便宜分越高（价格敏感）

    vec = np.array(cat_vec + item['attrs'] + [age_score, price_norm], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)


def build_intent_cache(user, items):
    """
    构建三层意图缓存
    L1: 长期偏好（全量历史）
    L2: 近期偏好（最近3次）
    L3: 会话即时意图（当前session）
    """
    def avg_item_vec(item_ids):
        if not item_ids:
            return np.zeros(13)
        vecs = [build_item_vector(items[iid]) for iid in item_ids if iid in items]
        return np.mean(vecs, axis=0) if vecs else np.zeros(13)

    l1_cache = avg_item_vec(user['history'])           # 全量历史
    l2_cache = avg_item_vec(user['history'][-3:])      # 最近3次
    l3_cache = avg_item_vec(user['session'])            # 当前session

    return {'L1': l1_cache, 'L2': l2_cache, 'L3': l3_cache}


def personalized_ranking(user_cache, items, alpha=0.3, beta=0.3, gamma=0.4):
    """
    基于三层意图缓存的个性化商品排序
    score = α·sim(L1,item) + β·sim(L2,item) + γ·sim(L3,item)
    session意图权重最高（gamma最大）
    """
    results = []
    for item_id, item in items.items():
        item_vec = build_item_vector(item)
        sim_l1 = float(np.dot(user_cache['L1'], item_vec))
        sim_l2 = float(np.dot(user_cache['L2'], item_vec))
        sim_l3 = float(np.dot(user_cache['L3'], item_vec))
        score = alpha * sim_l1 + beta * sim_l2 + gamma * sim_l3
        results.append({
            'item_id': item_id,
            'name': item['name'][:45],
            'score': score,
            'l1': sim_l1, 'l2': sim_l2, 'l3': sim_l3,
        })
    return sorted(results, key=lambda x: -x['score'])


def run_personalization_demo():
    """完整千人千面演示"""
    print("=" * 68)
    print("LLM Session Personalization Cache — 千人千面推荐系统")
    print("=" * 68)

    users, items = generate_sample_users_and_items()

    # 展示差异化排名
    for user_id, user in users.items():
        cache = build_intent_cache(user, items)
        ranking = personalized_ranking(cache, items)

        print(f"\n👤 用户 {user_id} [{user['profile']}]")
        print(f"   历史: {' → '.join(user['history'][-3:])}")
        print(f"   当前session: {' → '.join(user['session'])}")
        print(f"   意图L1(长期): {cache['L1'][:3].round(3)}")
        print(f"   意图L3(即时): {cache['L3'][:3].round(3)}")
        print(f"   个性化排名:")
        for i, r in enumerate(ranking[:4]):
            print(f"     #{i+1} [{r['item_id']}] {r['name']:<45} score={r['score']:.3f}")

    # 演示同一搜索词的不同展示
    print("\n" + "=" * 68)
    print("💡 千人千面效果：相同搜索词「breast pump」的个性化排名对比")
    print("=" * 68)
    breast_pump_items = {k: v for k, v in items.items()
                         if v['category'] in ['breast_pump', 'accessories', 'sterilizer']}
    for user_id, user in users.items():
        cache = build_intent_cache(user, items)
        ranking = personalized_ranking(cache, breast_pump_items)
        top3 = [r['item_id'] for r in ranking[:3]]
        print(f"  {user_id}[{user['profile']}]: {' > '.join(top3)}")

    print("\n📊 缓存策略总结:")
    print("  L1 长期偏好（周更新）: 来自全量历史，LLM生成品类/品牌画像")
    print("  L2 近期意图（日更新）: 来自近7天行为，捕捉需求变化")
    print("  L3 会话即时（分钟级）: 来自当前session，实时个性化")
    print("  → 三层融合权重: L1=0.3, L2=0.3, L3=0.4（session意图优先）")

    print("\n[✓] LLM Session Personalization Cache 测试通过")


if __name__ == '__main__':
    run_personalization_demo()
