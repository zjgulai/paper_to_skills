"""标签字典扩充脚本

基于零标签文本挖掘结果，执行：
1. 现有标签关键词扩展（同义词、变体、高频表达）
2. 新增通用标签（覆盖高频零标签主题）
3. 输出扩充后的标签字典 xlsx

输出文件: SGCS_VOC标签字典_V3.3_expanded.xlsx
"""

import json
import re
from copy import deepcopy
from pathlib import Path

import pandas as pd

OUTPUT_BASE = Path("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/labeling-latest")
ORIGINAL_XLSX = "/Users/pray/project/sgcs/20_insights/22_insight_reports/03_voc分析/01_内部VOC精细化运营洞察框架/SGCS_VOC标签字典_喂养电器V3.2_final.xlsx"
EXPANDED_XLSX = OUTPUT_BASE / "SGCS_VOC标签字典_V3.3_expanded.xlsx"

# ──────────────────────────────────────────────
# 关键词扩展映射: 标签ID → 新增关键词列表
# ──────────────────────────────────────────────

KEYWORD_EXPANSIONS = {
    # AIPL 节点标签扩展
    "TAG_A_001": ["dont know this brand", "who is this company", "never seen this brand", "unknown manufacturer"],
    "TAG_A_002": ["looks legit", "professional packaging", "well presented", "high end look", "premium packaging"],
    "TAG_A_005": ["false claims", "exaggerated", "misleading advertising", "not as advertised", "deceptive marketing"],
    "TAG_A_006": ["mom recommended", "parent approved", "trusted by moms", "community favorite"],

    # 价格相关
    "worth_the_price": [
        "good value", "great value", "excellent value", "bang for buck", "bang for the buck",
        "worth every penny", "worth the money", "worth buying", "good investment",
        "reasonably priced", "fair price", "competitive price", "great deal",
        "worth it", "worthwhile", "money well spent", "great purchase",
        "happy with price", "satisfied with cost", "good quality for price",
    ],
    "price_concern": [
        "too expensive", "overpriced", "pricey", "steep price", "high price",
        "costly", "price is high", "price too high", "not cheap",
        "cant afford", "expensive for what it is", "over priced",
    ],
    "poor_value_for_money": [
        "not worth it", "not worth the price", "not worth the money", "waste of money",
        "overpriced for quality", "expensive and bad", "costs too much for",
        "rip off", "ripoff", "money down the drain",
    ],

    # 产品性能
    "general_core_product_performance_issue": [
        "doesnt work", "not working", "stopped working", "quit working", "no longer works",
        "broke after", "broke within", "failed after", "died after", "died within",
        "malfunction", "malfunctioning", "defective unit", "faulty unit",
        "poor performance", "underperformed", "does not perform", "performance issue",
    ],
    "too_noisy": [
        "too loud", "very loud", "extremely loud", "unbearably loud", "loud noise",
        "noisy motor", "motor noise", "grinding noise", "buzzing", "humming loudly",
        "wakes baby", "wakes up baby", "disturbing noise", "annoying sound",
        "can hear it from", "hear it across", "louder than expected",
    ],
    "compatibility_issue": [
        "doesnt fit", "not compatible", "wont fit", "does not fit", "incompatible with",
        "wrong size", "wrong adapter", "wrong fitting", "doesnt match",
        "doesnt connect", "wont connect", "connection issue", "fitting issue",
    ],
    "size_runs_small": [
        "too small", "runs small", "size too small", "smaller than expected",
        "tight fit", "too tight", "squeezed", "compresses", "uncomfortably tight",
        "order size up", "should have sized up", "go up a size", "runs tight",
    ],
    "size_runs_large": [
        "too big", "too large", "runs big", "runs large", "size too big",
        "larger than expected", "bigger than expected", "loose fit", "too loose",
        "order size down", "should have sized down", "go down a size", "runs loose",
    ],

    # 材质/设计
    "instruction_user_manual": [
        "confusing instructions", "unclear directions", "hard to follow", "difficult to understand",
        "poorly written manual", "missing instructions", "no instructions", "vague instructions",
        "diagram unclear", "pictures not helpful", "manual is useless",
    ],
    "would_buy_again": [
        "buy again", "repurchase", "recommend to friend", "recommend to others",
        "would purchase again", "buy another", "get another one", "second one",
        "gift for friend", "bought for sister", "bought for friend", "spread the word",
    ],

    # 客服/物流
    "fast_support_response": [
        "quick response", "responded quickly", "fast reply", "same day response",
        "helpful support", "friendly customer service", "courteous staff",
        "excellent service", "outstanding service", "prompt response",
    ],
    "wrong_payment_card_account": [
        "charged wrong", "wrong charge", "double charged", "charged twice",
        "billing issue", "payment problem", "card declined", "transaction failed",
    ],
    "wrong_missing_extra_parts": [
        "missing piece", "missing part", "part missing", "incomplete set",
        "not all pieces", "fewer parts", "wrong part", "incorrect part",
        "extra piece", "spare part included", "additional part",
    ],

    # 负面体验
    "poor_usage_experience": [
        "difficult to use", "hard to use", "complicated", "not user friendly",
        "awkward to use", "uncomfortable to use", "tricky to use", "frustrating",
        "cant figure out", "confusing to use", "not intuitive",
    ],
    "burnt_smell": [
        "burning smell", "smells burnt", "burning odor", "smoky smell",
        "chemical smell", "plastic smell", "rubber smell", "weird smell",
        "strange odor", "foul smell", "acrid smell", "smells like burning",
    ],
    "melted_smoking_burnt_smell": [
        "melting", "melted", "deformed from heat", "warped from heat",
        "smoking", "smoke", "burning up", "caught fire", "fire hazard",
        "overheated", "burnt plastic", "melted plastic",
    ],

    # 品牌相关
    "misleading_ad_claim": [
        "false advertising", "false claim", "not as described", "does not match description",
        "different from picture", "not what i expected", "misrepresented",
        "deceptive", "dishonest marketing", "overpromised and underdelivered",
    ],
    "authentic_mom_recommendation": [
        "real mom", "actual user", "fellow mom", "mom friend", "mom group",
        "parent review", "honest review", "unbiased review", "real experience",
    ],
}

# ──────────────────────────────────────────────
# 新增通用标签定义
# ──────────────────────────────────────────────

# 格式: (tag_id, tag_en, tag_cn, aipl_node, sentiment, keywords, consumer_keywords, applicable_line, applicable_source)
NEW_GENERAL_TAGS = [
    # ===== 通用正向体验 (L1) =====
    (
        "TAG_GEN_001", "ease_of_use", "易于使用", "L1", "positive",
        "easy to use;easy use;simple to use;straightforward;user friendly;intuitive;effortless;hassle free;no hassle;foolproof;childs play;piece of cake;breezy;smooth sailing;works right out of the box;plug and play;no learning curve;user intuitive",
        "so easy to use;super easy;couldnt be easier;easiest thing;no brainer;anyone can use it;my grandma could use it;idiot proof;so simple;intuitive design;self explanatory;no instructions needed",
        "通用", "电商评论,评论/社媒,测评内容",
    ),
    (
        "TAG_GEN_002", "product_quality_perception", "产品质量感知", "L1", "positive",
        "high quality;well made;well built;solid construction;premium quality;top quality;excellent craftsmanship;quality material;quality build;durable construction;sturdy built;built to last;quality feel;feels expensive;feels premium;high end quality;superior quality;outstanding quality;impressive quality;remarkable quality",
        "great quality;good quality;nice quality;decent quality;quality is great;quality is good;feels high quality;feels well made;seems durable;looks durable;built solid;solidly built;quality product;quality item;not cheaply made;doesnt feel cheap;feels substantial;substantial feel",
        "通用", "电商评论,评论/社媒,测评内容",
    ),
    (
        "TAG_GEN_003", "comfort_experience", "舒适体验", "L1", "positive",
        "comfortable;comfy;cozy;soft;cushiony;plush;velvety;silky smooth;gentle on skin;snug;cozy fit;comfortable fit;feels good on;pleasant to wear;relaxing feel;soothing feel;gentle feel;luxurious feel;cloud like;pillow soft;butter soft;ultra soft;super soft;dreamy soft",
        "so comfortable;very comfortable;extremely comfortable;incredibly comfortable;super comfy;so soft;very soft;like a cloud;feels like heaven;like butter;so cozy;very cozy;comfortable to wear;comfortable to use;comfortable to hold;gentle on body;doesnt hurt;no discomfort;feels amazing",
        "通用", "电商评论,评论/社媒,测评内容",
    ),
    (
        "TAG_GEN_004", "product_functionality", "产品功能性", "L1", "positive",
        "works well;works great;works perfectly;functions properly;operates smoothly;performs well;does the job;gets the job done;effective;efficient;reliable;dependable;consistent performance;as expected;meets expectations;exceeds expectations;lives up to hype;does what it says;does what its supposed to;fulfills purpose;serves purpose",
        "works as advertised;works as described;does exactly what it says;does what it promises;functions as expected;operates as described;performs as expected;gets the job done right;does the trick;exactly what i needed;just what i needed;perfect for my needs;meets my needs;serves its purpose;does its job;functions well;operates well",
        "通用", "电商评论,评论/社媒,测评内容",
    ),
    (
        "TAG_GEN_005", "design_appearance", "设计外观", "L1", "positive",
        "good design;nice design;beautiful design;elegant design;sleek design;modern design;stylish design;attractive design;eye catching;cute design;adorable design;pretty design;lovely design;aesthetic;visually appealing;good looking;nice looking;beautiful look;gorgeous;stunning;elegant;classy;sophisticated;tasteful",
        "looks great;looks nice;looks beautiful;looks cute;looks adorable;looks pretty;looks lovely;looks amazing;looks stunning;looks elegant;looks modern;looks sleek;so cute;very cute;super cute;absolutely adorable;so pretty;very pretty;beautiful color;nice color;love the color;love the design;love the look;love how it looks;stylish looking",
        "通用", "电商评论,评论/社媒,测评内容",
    ),
    (
        "TAG_GEN_006", "material_texture", "材质触感", "L1", "positive",
        "good material;quality material;premium material;nice fabric;soft fabric;smooth fabric;buttery soft;silky smooth;velvety soft;plush material;cotton feel;breathable material;lightweight material;flexible material;stretchy material;durable fabric;thick material;thin material;see through material;opaque material",
        "feels soft;feels nice;feels smooth;feels silky;feels luxurious;feels premium;feels cheap;feels thin;feels flimsy;feels sturdy;feels thick;feels heavy;feels light;nice texture;good texture;smooth texture;rough texture;weird texture;strange texture;love the material;love the fabric;hate the material",
        "通用", "电商评论,评论/社媒,测评内容",
    ),
    (
        "TAG_GEN_007", "size_accuracy", "尺码准确性", "L1", "positive",
        "true to size;runs true;accurate sizing;correct size;right size;perfect size;exact size;as sized;sizing is accurate;size is correct;size is right;fits as expected;fits perfectly;fits well;fits great;fits nicely;good fit;perfect fit;snug fit;comfortable fit;secure fit",
        "true to size;runs true to size;exactly my size;just my size;perfect sizing;spot on sizing;sizing is perfect;size chart accurate;followed size chart;size guide accurate;ordered my usual size;usual size fits;normal size fits;fits like a glove;fits perfectly;couldnt fit better;ideal fit;dream fit",
        "通用", "电商评论,评论/社媒,测评内容",
    ),
    (
        "TAG_GEN_008", "portability_convenience", "便携便利性", "L1", "positive",
        "portable;compact;lightweight;light weight;easy to carry;travel friendly;travel ready;on the go;portable design;compact design;space saving;foldable;collapsible;easy to pack;fits in bag;fits in purse;fits in diaper bag;fits in luggage;convenient size;convenient shape;handy size;pocket sized",
        "easy to carry around;take it anywhere;take on the go;perfect for travel;great for travel;ideal for travel;travel essential;travel must have;fits in my bag;fits in my purse;small enough to carry;light enough to carry;doesnt take up space;saves space;compact enough;portable enough;convenient to carry;handy to have around",
        "通用", "电商评论,评论/社媒,测评内容",
    ),
    (
        "TAG_GEN_009", "cleaning_maintenance", "清洁维护", "L1", "positive",
        "easy to clean;easy cleaning;simple to clean;quick to clean; effortless cleaning;hassle free cleaning;low maintenance;easy maintenance;simple maintenance;dishwasher safe;machine washable;wipe clean;rinse clean;self cleaning;quick rinse;easy wipe down;easy to sanitize;easy to sterilize;removable parts;detachable parts",
        "cleans easily;washes easily;cleans up nicely;wipes clean easily;rinses clean;comes apart easily;easy to take apart;easy to put back together; dishwasher friendly;machine wash friendly;no special cleaning;minimal cleaning;low upkeep;easy upkeep;simple upkeep;doesnt stain;stain resistant;odor resistant;doesnt retain smell",
        "通用", "电商评论,评论/社媒,测评内容",
    ),
    (
        "TAG_GEN_010", "noise_level_acceptable", "噪音水平可接受", "L1", "positive",
        "quiet;whisper quiet;virtually silent;silent operation;quiet motor;low noise;minimal noise;negligible noise;barely audible;hardly noticeable;quiet enough;acceptably quiet;surprisingly quiet;quieter than expected;quieter than anticipated;quieter than others;white noise level;soothing sound;gentle hum;soft hum",
        "so quiet;very quiet;extremely quiet;incredibly quiet;super quiet;pleasantly quiet;quiet as a mouse;cant hear it;dont notice the noise;noise is minimal;sound is minimal;barely makes a sound;hardly any noise;negligible sound;white noise like;gentle white noise;soothing white noise;doesnt wake baby;doesnt disturb;sleep friendly",
        "通用", "电商评论,评论/社媒,测评内容",
    ),

    # ===== 通用负向体验 (L1/L2) =====
    (
        "TAG_GEN_011", "general_dissatisfaction", "一般不满", "L1", "negative",
        "disappointed;disappointing;unsatisfied;dissatisfied;regret buying;regret purchasing;waste of money;waste of time;not happy;unhappy;not pleased;not impressed;let down;underwhelmed;not what expected;falls short;doesnt meet expectations;failed expectations;poor experience;bad experience;terrible experience;awful experience;horrible experience;worst experience",
        "so disappointed;very disappointed;extremely disappointed;incredibly disappointed;totally disappointed;deeply disappointed;seriously disappointed;majorly disappointed;completely let down;totally let down;waste of money;complete waste;total waste;money down the drain;threw money away;should have saved my money;regret this purchase;regret buying this;wish i hadnt bought;not what i expected;not what i wanted;not as good as expected;fell short of expectations;did not live up to hype",
        "通用", "电商评论,评论/社媒,测评内容,客服工单",
    ),
    (
        "TAG_GEN_012", "difficult_to_use", "使用困难", "L1", "negative",
        "difficult to use;hard to use;complicated to use;not user friendly;not intuitive;awkward to use;uncomfortable to use;tricky to use;frustrating to use;confusing to operate;hard to operate;difficult to operate;not straightforward;counterintuitive;poorly designed;bad design;awkward design;clunky design;cumbersome;bulky;unwieldy",
        "too difficult;too hard;too complicated;too confusing;not easy to use;not simple;not straightforward;cant figure out how to;struggling to use;having trouble with;difficult to figure out;hard to figure out;complicated instructions;confusing buttons;awkward button placement;hard to reach;difficult to reach;not ergonomic;poor ergonomics;uncomfortable to hold",
        "通用", "电商评论,评论/社媒,测评内容,客服工单",
    ),
    (
        "TAG_GEN_013", "durability_concern", "耐用性担忧", "L1", "negative",
        "not durable;poor durability;falls apart;breaks easily;broke quickly;wore out;worn out;wearing out;doesnt last;wont last;short lifespan;short life;poor quality material;cheap material;flimsy;fragile;delicate;thin material;thin fabric;cheap plastic;cheap construction;weak construction;poorly made;cheaply made;shoddy",
        "fell apart;broke after first use;broke within a week;didnt last long;only lasted;lasted less than;started falling apart;coming apart at seams;stitching came undone;zipper broke;button fell off;material tore;fabric tore;started tearing;started fraying;wore out quickly;showing wear;looks worn;cheap feel;feels cheap;feels flimsy;feels fragile;not sturdy",
        "通用", "电商评论,评论/社媒,测评内容,客服工单",
    ),

    # ===== 服务体验 (L2) =====
    (
        "TAG_GEN_014", "positive_customer_service", "客服好评", "L2", "positive",
        "good customer service;great customer service;excellent customer service;outstanding customer service;amazing customer service;exceptional customer service;friendly service;helpful staff;responsive team;quick resolution;problem solved;issue resolved;satisfied with service;happy with support;grateful for help;appreciate the help;thank you support team;quick response;fast response;same day response",
        "customer service was great;support was amazing;team was so helpful;they went above and beyond;outstanding support;incredible customer care;friendly and helpful;quick to respond;responded within hours;resolved my issue;solved my problem;took care of me;replaced immediately;refunded quickly;no hassle return;hassle free exchange;very accommodating;extremely helpful;couldnt be happier with service;best customer service ever",
        "通用", "电商评论,评论/社媒,客服工单,CRM反馈",
    ),
    (
        "TAG_GEN_015", "fast_shipping_delivery", "快速发货配送", "L2", "positive",
        "fast shipping;quick shipping;fast delivery;quick delivery;arrived early;arrived ahead of schedule;arrived before expected;arrived quickly;came quickly;came fast;fast arrival;quick arrival;speedy delivery;express delivery;next day delivery;two day shipping; Prime shipping;arrived in two days;arrived next day;shipped same day;shipped quickly;shipped fast",
        "came so fast;arrived so quickly;got it in two days;next day arrival;super fast shipping;lightning fast delivery;came earlier than expected;before the estimated date;ahead of delivery window;right on time;on schedule;perfect timing;well packaged;nicely packaged;carefully packaged;secure packaging;box in good condition;no damage in shipping;intact upon arrival",
        "通用", "电商评论,评论/社媒,测评内容",
    ),

    # ===== 推荐/复购 (L3) =====
    (
        "TAG_GEN_016", "strong_recommendation", "强烈推荐", "L3", "positive",
        "highly recommend;strongly recommend;definitely recommend;absolutely recommend;cant recommend enough;would recommend to everyone;would recommend to anyone;recommend to all;recommend to every;recommend without reservation; enthusiastically recommend;wholeheartedly recommend;must have;essential item;must buy;must own;game changer;life changing;transformed my life;changed everything",
        "i highly recommend;we strongly recommend;definitely would recommend;absolutely would recommend;couldnt recommend more;wont regret buying;best purchase ever;best investment;money well spent;worth every cent;essential for new moms;every mom needs this;every parent needs this;wish i had this sooner;where has this been all my life;couldnt live without;cant imagine life without;game changing product;life saver;life changing product",
        "通用", "电商评论,评论/社媒,测评内容,社群讨论",
    ),
    (
        "TAG_GEN_017", "gift_purchase_intent", "礼品购买意向", "L3", "positive",
        "bought as gift;gift for;present for;baby shower gift;registry item;baby registry;baby shower present;gift idea;perfect gift;great gift;ideal gift;thoughtful gift;new mom gift;expecting mom gift;pregnancy gift;new parent gift;gift worthy;giftable;comes in nice box;nice packaging for gift;ready to gift",
        "got this as a gift;received as gift;gifted to me;my friend got me this;my sister recommended;my mom bought this;perfect baby shower gift;great for baby registry;added to registry;on my registry;bought for my daughter;bought for my sister;bought for friend;gift for new mom;gift for expecting mom;present for pregnant friend;new parent gift idea;ideal baby shower present;lovely gift idea;would make a great gift",
        "通用", "电商评论,评论/社媒,测评内容",
    ),

    # ===== 包装/开箱 (L1/L2) =====
    (
        "TAG_GEN_018", "packaging_quality", "包装质量", "L1", "positive",
        "good packaging;nice packaging;beautiful packaging;premium packaging;luxury packaging;well packaged;securely packaged;carefully packaged;nicely boxed;elegant box;sturdy box;protective packaging;safe packaging;eco friendly packaging;minimal packaging;recyclable packaging;beautiful box;cute packaging;adorable packaging;presentable packaging",
        "came in nice box;arrived well packaged;packaging was beautiful;love the packaging;impressed by packaging;nicely presented;beautifully packaged;securely wrapped;protected well;no damage to box;box was intact;packaging was sturdy;loved the box design;cute box;adorable packaging;perfect for gifting;ready to give as gift;would make nice present;beautiful unboxing experience;lovely presentation",
        "通用", "电商评论,评论/社媒,测评内容",
    ),
]


def load_zero_label_mining():
    """加载零标签挖掘结果"""
    path = OUTPUT_BASE / "zero_label_mining.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def expand_existing_tags(df_dict: dict) -> dict:
    """扩展现有标签的关键词"""
    print("\n" + "=" * 70)
    print("扩展现有标签关键词")
    print("=" * 70)

    expanded_count = 0
    total_new_keywords = 0

    for sheet_name, df in df_dict.items():
        if sheet_name.startswith("00_"):
            continue

        if "标签ID" not in df.columns or "英文关键词/典型表达" not in df.columns:
            continue

        for idx, row in df.iterrows():
            tag_id = str(row.get("标签ID", "")).strip()
            if not tag_id:
                continue

            # 检查是否有扩展映射
            new_keywords = KEYWORD_EXPANSIONS.get(tag_id, [])
            if not new_keywords:
                # 也尝试用标签英文名匹配
                tag_en = str(row.get("VOC标签（英文）", "")).strip()
                if tag_en:
                    new_keywords = KEYWORD_EXPANSIONS.get(tag_en, [])

            if new_keywords:
                # 获取当前关键词
                current_kw = str(row.get("英文关键词/典型表达", ""))
                current_list = [k.strip() for k in current_kw.split(";") if k.strip()]

                # 合并新旧关键词（去重）
                merged = list(dict.fromkeys(current_list + new_keywords))
                new_kw_str = ";".join(merged)

                # 更新 DataFrame
                df.at[idx, "英文关键词/典型表达"] = new_kw_str
                expanded_count += 1
                total_new_keywords += len(new_keywords)

    print(f"  扩展了 {expanded_count} 个标签")
    print(f"  新增关键词: {total_new_keywords} 个")
    return df_dict


def create_new_tags_df() -> pd.DataFrame:
    """创建新增通用标签 DataFrame"""
    print("\n" + "=" * 70)
    print("创建新增通用标签")
    print("=" * 70)

    rows = []
    for tag in NEW_GENERAL_TAGS:
        tag_id, tag_en, tag_cn, aipl, sentiment, keywords, consumer_kw, line, source = tag
        rows.append({
            "标签ID": tag_id,
            "AIPL节点": aipl,
            "标签主题": "general_experience",
            "VOC标签（中文）": tag_cn,
            "VOC标签（英文）": tag_en,
            "英文关键词/典型表达": keywords,
            "消费者习惯关键词/原话短语": consumer_kw,
            "标签定义": f"通用标签: {tag_cn}",
            "情感极性": sentiment,
            "是否AI可抽取": "是",
            "来源类型": "通用",
            "适用产品品线": line,
            "适用VOC载体": source,
            "适用用户画像": "",
            "对应原子指标": "",
            "MetricDirection": "",
            "Proxy NPS贡献": "promoter" if sentiment == "positive" else "detractor" if sentiment == "negative" else "neutral",
            "是否通用标签": "是",
            "故事线关联": "",
            "策略包": "",
            "业务动作/责任部门": "",
            "主责部门": "",
            "协同部门": "",
            "默认优先级": "P2",
            "备注": "V3.3 新增通用标签",
            "品类特异性指数": "",
            "共性/特性分类": "共性",
            "主导品类": "",
            "暖奶器评论数": "",
            "消毒器评论数": "",
            "合理性评分": "",
            "风险等级": "",
            "问题诊断": "",
            "优化建议": "",
            "优化优先级": "",
            "适用喂养电器": "是",
            "V3.1优化记录": "",
            "V3.2优化记录": "",
        })

    print(f"  新增 {len(rows)} 个通用标签")
    return pd.DataFrame(rows)


def merge_into_sheets(df_dict: dict, new_tags_df: pd.DataFrame) -> dict:
    """将新增标签合并到现有 sheet 中"""
    # 将新增标签追加到 "01_通用标签主表"
    if "01_通用标签主表" in df_dict:
        original = df_dict["01_通用标签主表"]
        merged = pd.concat([original, new_tags_df], ignore_index=True)
        df_dict["01_通用标签主表"] = merged
        print(f"\n  通用标签主表: {len(original)} → {len(merged)} 行")
    else:
        df_dict["01_通用标签主表"] = new_tags_df

    return df_dict


def save_expanded_dictionary(df_dict: dict):
    """保存扩充后的标签字典"""
    print("\n" + "=" * 70)
    print("保存扩充后的标签字典")
    print("=" * 70)

    with pd.ExcelWriter(EXPANDED_XLSX, engine="openpyxl") as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  Sheet: {sheet_name} ({len(df)} 行)")

    print(f"\n  输出文件: {EXPANDED_XLSX}")
    return EXPANDED_XLSX


def generate_expansion_report(df_dict: dict):
    """生成扩充报告"""
    print("\n" + "=" * 70)
    print("扩充报告")
    print("=" * 70)

    total_tags = 0
    total_keywords = 0
    for sheet_name, df in df_dict.items():
        if sheet_name.startswith("00_"):
            continue
        if "英文关键词/典型表达" not in df.columns:
            continue
        count = len(df)
        total_tags += count
        kw_count = df["英文关键词/典型表达"].dropna().apply(
            lambda x: len([k.strip() for k in str(x).split(";") if k.strip()])
        ).sum()
        total_keywords += kw_count
        avg_kw = kw_count / count if count > 0 else 0
        print(f"  {sheet_name}: {count} 标签, 平均 {avg_kw:.1f} 关键词/标签")

    print(f"\n  总计: {total_tags} 标签")
    print(f"  总计关键词: {total_keywords}")
    print(f"  平均关键词/标签: {total_keywords/total_tags:.1f}")


def main():
    print("=" * 70)
    print("VOC 标签字典扩充 (V3.2 → V3.3)")
    print("=" * 70)

    # 1. 加载原始字典
    print("\n加载原始标签字典...")
    df_dict = pd.read_excel(ORIGINAL_XLSX, sheet_name=None)
    original_tag_count = sum(len(df) for name, df in df_dict.items() if not name.startswith("00_"))
    print(f"  原始标签数: {original_tag_count}")

    # 2. 加载零标签挖掘结果
    mining = load_zero_label_mining()
    print(f"  零标签样本: {mining['sample_size']:,}")

    # 3. 扩展现有标签关键词
    df_dict = expand_existing_tags(df_dict)

    # 4. 创建新增通用标签
    new_tags_df = create_new_tags_df()

    # 5. 合并到主表
    df_dict = merge_into_sheets(df_dict, new_tags_df)

    # 6. 生成报告
    generate_expansion_report(df_dict)

    # 7. 保存
    save_expanded_dictionary(df_dict)

    print("\n" + "=" * 70)
    print("扩充完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
