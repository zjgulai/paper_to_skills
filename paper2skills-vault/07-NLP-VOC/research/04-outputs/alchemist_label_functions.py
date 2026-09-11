"""ALCHEmist Label Functions (Auto-Generated)

为标签字典 v3.4 新增候选标签生成的标注规则。
每个函数遵循: (text) -> (matched: bool, confidence: float)

Usage:
    from alchemist_label_functions import lf_bath_bombs, lf_wipe_warmer
    matched, conf = lf_bath_bombs("These bath bombs smell amazing!")
"""

NEGATION_WORDS = {
    "not", "no", "never", "none", "nobody", "nothing",
    "n't", "dont", "doesnt", "didnt", "wouldnt", "couldnt",
    "shouldnt", "wont", "cant", "isnt", "arent", "wasnt",
    "werent", "hasnt", "havent", "hadnt",
}


def lf_bath_bombs(text: str) -> tuple[bool, float]:
    """Label Function: [沐浴炸弹] bath bombs (沐浴炸弹) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "bath bombs", "bath bomb"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["bath bombs", "bath bomb"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_wipe_warmer(text: str) -> tuple[bool, float]:
    """Label Function: [湿巾加热盒] wipe warmer (湿巾加热盒) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "wipe warmer", "wipe warmers"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["wipe warmer", "wipe warmers"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_nasal_aspirator(text: str) -> tuple[bool, float]:
    """Label Function: [手持式吸鼻器] nasal aspirator (手持式吸鼻器) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "nasal aspirator", "nasal aspirators"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["nasal aspirator", "nasal aspirators"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_drying_rack(text: str) -> tuple[bool, float]:
    """Label Function: [奶瓶沥干支架] drying rack (奶瓶沥干支架) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "drying rack", "drying racks"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["drying rack", "drying racks"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_knee_pillow(text: str) -> tuple[bool, float]:
    """Label Function: [夹腿枕] knee pillow (夹腿枕) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "knee pillow", "knee pillows"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["knee pillow", "knee pillows"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_baby_stroller(text: str) -> tuple[bool, float]:
    """Label Function: [推车] baby stroller (推车) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "baby stroller", "baby strollers"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["baby stroller", "baby strollers"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_air_purifier(text: str) -> tuple[bool, float]:
    """Label Function: [桌面空气净化器] air purifier (桌面空气净化器) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "air purifier", "air purifiers"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["air purifier", "air purifiers"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_play_mat(text: str) -> tuple[bool, float]:
    """Label Function: [应急垫] play mat (应急垫) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "play mat", "play mats"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["play mat", "play mats"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_baby_food(text: str) -> tuple[bool, float]:
    """Label Function: [辅食机] baby food (辅食机) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "baby food", "baby foods"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["baby food", "baby foods"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_breast_pump(text: str) -> tuple[bool, float]:
    """Label Function: [吸奶器] breast pump (吸奶器) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "breast pump", "breast pumps"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["breast pump", "breast pumps"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_postpartum_belly(text: str) -> tuple[bool, float]:
    """Label Function: [收腹带] postpartum belly (收腹带) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "postpartum belly", "postpartum bellys"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["postpartum belly", "postpartum bellys"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_wooden_high(text: str) -> tuple[bool, float]:
    """Label Function: [成长型餐椅] wooden high (成长型餐椅) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "wooden high", "wooden highs"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["wooden high", "wooden highs"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_sleeping_bag(text: str) -> tuple[bool, float]:
    """Label Function: [婴儿无袖睡袋] sleeping bag (婴儿无袖睡袋) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "sleeping bag", "sleeping bags"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["sleeping bag", "sleeping bags"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_food_grade(text: str) -> tuple[bool, float]:
    """Label Function: [成长型餐椅] food grade (成长型餐椅) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "food grade", "food grades"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["food grade", "food grades"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_grade_tray(text: str) -> tuple[bool, float]:
    """Label Function: [成长型餐椅] grade tray (成长型餐椅) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "grade tray", "grade trays"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["grade tray", "grade trays"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_adjustable_ergonomic(text: str) -> tuple[bool, float]:
    """Label Function: [成长型餐椅] adjustable ergonomic (成长型餐椅) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "adjustable ergonomic", "adjustable ergonomics"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["adjustable ergonomic", "adjustable ergonomics"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_nursing_pillow(text: str) -> tuple[bool, float]:
    """Label Function: [U型哺乳枕] nursing pillow (U型哺乳枕) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "nursing pillow", "nursing pillows"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["nursing pillow", "nursing pillows"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_nursing_pads(text: str) -> tuple[bool, float]:
    """Label Function: [一次性防溢乳垫] nursing pads (一次性防溢乳垫) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "nursing pads", "nursing pad"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["nursing pads", "nursing pad"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_extendable_upf(text: str) -> tuple[bool, float]:
    """Label Function: [推车] extendable upf (推车) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "extendable upf", "extendable upfs"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["extendable upf", "extendable upfs"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_upf_canopy(text: str) -> tuple[bool, float]:
    """Label Function: [推车] upf canopy (推车) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "upf canopy", "upf canopys"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["upf canopy", "upf canopys"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_stroller_convertible(text: str) -> tuple[bool, float]:
    """Label Function: [推车] stroller convertible (推车) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "stroller convertible", "stroller convertibles"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["stroller convertible", "stroller convertibles"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_bath_bomb(text: str) -> tuple[bool, float]:
    """Label Function: [沐浴炸弹] bath bomb (沐浴炸弹) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "bath bomb", "bath bombs"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["bath bomb", "bath bombs"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_para_beb(text: str) -> tuple[bool, float]:
    """Label Function: [磨甲器] para beb (磨甲器) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "para beb", "para bebs"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["para beb", "para bebs"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_electric_nasal(text: str) -> tuple[bool, float]:
    """Label Function: [手持式吸鼻器] electric nasal (手持式吸鼻器) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "electric nasal", "electric nasals"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["electric nasal", "electric nasals"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_baby_nail(text: str) -> tuple[bool, float]:
    """Label Function: [磨甲器] baby nail (磨甲器) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "baby nail", "baby nails"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["baby nail", "baby nails"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_breast_milk(text: str) -> tuple[bool, float]:
    """Label Function: [储奶壶] breast milk (储奶壶) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "breast milk", "breast milks"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["breast milk", "breast milks"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_vielen_dank(text: str) -> tuple[bool, float]:
    """Label Function: [记忆棉哺乳枕] vielen dank (记忆棉哺乳枕) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "vielen dank", "vielen danks"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["vielen dank", "vielen danks"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_important_max(text: str) -> tuple[bool, float]:
    """Label Function: [手持式吸鼻器] important max (手持式吸鼻器) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "important max", "important maxs"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["important max", "important maxs"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_memory_foam(text: str) -> tuple[bool, float]:
    """Label Function: [记忆棉哺乳枕] memory foam (记忆棉哺乳枕) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "memory foam", "memory foams"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["memory foam", "memory foams"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_belly_band(text: str) -> tuple[bool, float]:
    """Label Function: [托腹带] belly band (托腹带) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "belly band", "belly bands"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["belly band", "belly bands"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_baby_wipe(text: str) -> tuple[bool, float]:
    """Label Function: [湿巾加热盒] baby wipe (湿巾加热盒) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "baby wipe", "baby wipes"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["baby wipe", "baby wipes"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_any_questions(text: str) -> tuple[bool, float]:
    """Label Function: [手持式吸鼻器] any questions (手持式吸鼻器) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "any questions", "any question"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["any questions", "any question"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_belly_wrap(text: str) -> tuple[bool, float]:
    """Label Function: [收腹带] belly wrap (收腹带) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "belly wrap", "belly wraps"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["belly wrap", "belly wraps"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_food_maker(text: str) -> tuple[bool, float]:
    """Label Function: [辅食机] food maker (辅食机) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "food maker", "food makers"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["food maker", "food makers"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_air_ultra(text: str) -> tuple[bool, float]:
    """Label Function: [吸奶器] air ultra (吸奶器) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "air ultra", "air ultras"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["air ultra", "air ultras"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_ultra_slim(text: str) -> tuple[bool, float]:
    """Label Function: [吸奶器] ultra slim (吸奶器) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "ultra slim", "ultra slims"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["ultra slim", "ultra slims"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_tank_top(text: str) -> tuple[bool, float]:
    """Label Function: [哺乳背心] tank top (哺乳背心) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "tank top", "tank tops"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["tank top", "tank tops"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_baby_carrier(text: str) -> tuple[bool, float]:
    """Label Function: [腰凳背带] baby carrier (腰凳背带) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "baby carrier", "baby carriers"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["baby carrier", "baby carriers"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_non_slip(text: str) -> tuple[bool, float]:
    """Label Function: [应急垫] non slip (应急垫) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "non slip", "non slips"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["non slip", "non slips"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_car_seat(text: str) -> tuple[bool, float]:
    """Label Function: [ST01多功能婴儿标准车配件] car seat (ST01多功能婴儿标准车配件) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "car seat", "car seats"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["car seat", "car seats"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_der_rucksack(text: str) -> tuple[bool, float]:
    """Label Function: [妈咪包] der rucksack (妈咪包) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "der rucksack", "der rucksacks"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["der rucksack", "der rucksacks"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_conversation_user(text: str) -> tuple[bool, float]:
    """Label Function: [电动摇椅] conversation user (电动摇椅) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "conversation user", "conversation users"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["conversation user", "conversation users"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_slim_breast(text: str) -> tuple[bool, float]:
    """Label Function: [吸奶器] slim breast (吸奶器) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "slim breast", "slim breasts"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["slim breast", "slim breasts"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_google_maps(text: str) -> tuple[bool, float]:
    """Label Function: [记忆棉哺乳枕] google maps (记忆棉哺乳枕) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "google maps", "google map"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["google maps", "google map"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_maps_search(text: str) -> tuple[bool, float]:
    """Label Function: [记忆棉哺乳枕] maps search (记忆棉哺乳枕) -> I/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "maps search", "maps searchs"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["maps search", "maps searchs"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_baby_play(text: str) -> tuple[bool, float]:
    """Label Function: [应急垫] baby play (应急垫) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "baby play", "baby plays"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["baby play", "baby plays"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_sacco_nanna(text: str) -> tuple[bool, float]:
    """Label Function: [婴儿无袖睡袋] sacco nanna (婴儿无袖睡袋) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "sacco nanna", "sacco nannas"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["sacco nanna", "sacco nannas"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_hip_seat(text: str) -> tuple[bool, float]:
    """Label Function: [腰凳背带] hip seat (腰凳背带) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "hip seat", "hip seats"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["hip seat", "hip seats"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_individually_wrapped(text: str) -> tuple[bool, float]:
    """Label Function: [沐浴片] individually wrapped (沐浴片) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "individually wrapped", "individually wrappeds"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["individually wrapped", "individually wrappeds"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_mit_freundlichen(text: str) -> tuple[bool, float]:
    """Label Function: [背带] mit freundlichen (背带) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "mit freundlichen", "mit freundlichens"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["mit freundlichen", "mit freundlichens"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_pro_baby(text: str) -> tuple[bool, float]:
    """Label Function: [消毒器] pro baby (消毒器) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "pro baby", "pro babys"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["pro baby", "pro babys"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_stroller_organizer(text: str) -> tuple[bool, float]:
    """Label Function: [车挂包] stroller organizer (车挂包) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "stroller organizer", "stroller organizers"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["stroller organizer", "stroller organizers"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_breast_pads(text: str) -> tuple[bool, float]:
    """Label Function: [可水洗防溢乳垫] breast pads (可水洗防溢乳垫) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "breast pads", "breast pad"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["breast pads", "breast pad"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_baby_thermometer(text: str) -> tuple[bool, float]:
    """Label Function: [儿童温度计] baby thermometer (儿童温度计) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "baby thermometer", "baby thermometers"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["baby thermometer", "baby thermometers"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_nursing_tank(text: str) -> tuple[bool, float]:
    """Label Function: [哺乳背心] nursing tank (哺乳背心) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "nursing tank", "nursing tanks"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["nursing tank", "nursing tanks"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_este_servicio(text: str) -> tuple[bool, float]:
    """Label Function: [磨甲器] este servicio (磨甲器) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "este servicio", "este servicios"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["este servicio", "este servicios"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_sac_langer(text: str) -> tuple[bool, float]:
    """Label Function: [妈咪包] sac langer (妈咪包) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "sac langer", "sac langers"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["sac langer", "sac langers"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_ergowrap_postpartum(text: str) -> tuple[bool, float]:
    """Label Function: [收腹带] ergowrap postpartum (收腹带) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "ergowrap postpartum", "ergowrap postpartums"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["ergowrap postpartum", "ergowrap postpartums"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_pregnancy_pillow(text: str) -> tuple[bool, float]:
    """Label Function: [W型孕妇枕] pregnancy pillow (W型孕妇枕) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "pregnancy pillow", "pregnancy pillows"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["pregnancy pillow", "pregnancy pillows"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_lactation_massager(text: str) -> tuple[bool, float]:
    """Label Function: [乳房按摩仪] lactation massager (乳房按摩仪) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "lactation massager", "lactation massagers"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["lactation massager", "lactation massagers"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_follow_previous(text: str) -> tuple[bool, float]:
    """Label Function: [消毒器] follow previous (消毒器) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "follow previous", "follow previou"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["follow previous", "follow previou"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_diaper_bag(text: str) -> tuple[bool, float]:
    """Label Function: [妈咪包] diaper bag (妈咪包) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "diaper bag", "diaper bags"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["diaper bag", "diaper bags"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_hot_cold(text: str) -> tuple[bool, float]:
    """Label Function: [乳房冷热敷垫] hot cold (乳房冷热敷垫) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "hot cold", "hot colds"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["hot cold", "hot colds"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_tire_lait(text: str) -> tuple[bool, float]:
    """Label Function: [吸奶器背心] tire lait (吸奶器背心) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "tire lait", "tire laits"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["tire lait", "tire laits"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_der_schlafsack(text: str) -> tuple[bool, float]:
    """Label Function: [婴儿无袖睡袋] der schlafsack (婴儿无袖睡袋) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "der schlafsack", "der schlafsacks"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["der schlafsack", "der schlafsacks"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_die_creme(text: str) -> tuple[bool, float]:
    """Label Function: [电动乳头按摩霜] die creme (电动乳头按摩霜) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "die creme", "die cremes"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["die creme", "die cremes"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_pouch_holder(text: str) -> tuple[bool, float]:
    """Label Function: [防挤压支架] pouch holder (防挤压支架) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "pouch holder", "pouch holders"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["pouch holder", "pouch holders"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_sent_iphone(text: str) -> tuple[bool, float]:
    """Label Function: [U型双边孕妇枕] sent iphone (U型双边孕妇枕) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "sent iphone", "sent iphones"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["sent iphone", "sent iphones"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_therapy_packs(text: str) -> tuple[bool, float]:
    """Label Function: [乳房冷热敷垫] therapy packs (乳房冷热敷垫) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "therapy packs", "therapy pack"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["therapy packs", "therapy pack"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_table_tbody(text: str) -> tuple[bool, float]:
    """Label Function: [圆环背巾] table tbody (圆环背巾) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "table tbody", "table tbodys"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["table tbody", "table tbodys"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_entry_gmail(text: str) -> tuple[bool, float]:
    """Label Function: [吸奶器内衣] entry gmail (吸奶器内衣) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "entry gmail", "entry gmails"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["entry gmail", "entry gmails"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_gmail_source(text: str) -> tuple[bool, float]:
    """Label Function: [吸奶器内衣] gmail source (吸奶器内衣) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "gmail source", "gmail sources"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["gmail source", "gmail sources"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_soutien_gorge(text: str) -> tuple[bool, float]:
    """Label Function: [吸奶器背心] soutien gorge (吸奶器背心) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "soutien gorge", "soutien gorges"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["soutien gorge", "soutien gorges"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)


def lf_baby_swing(text: str) -> tuple[bool, float]:
    """Label Function: [电动摇椅] baby swing (电动摇椅) -> L1/neutral

    触发条件: 文本包含以下关键词之一
    关键词: "baby swing", "baby swings"
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = ["baby swing", "baby swings"]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {'nobody', 'couldnt', 'cant', 'werent', 'not', 'shouldnt', 'none', 'didnt', 'nothing', "n't", 'dont', 'never', 'no', 'arent', 'hadnt', 'wouldnt', 'havent', 'wasnt', 'hasnt', 'doesnt', 'isnt', 'wont'}):
            return False, 0.0
    

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)

# ── 注册表 ────────────────────────────────────────────────────────

LABEL_FUNCTION_REGISTRY = {
    "bath_bombs": lf_bath_bombs,  # [沐浴炸弹] bath bombs
    "wipe_warmer": lf_wipe_warmer,  # [湿巾加热盒] wipe warmer
    "nasal_aspirator": lf_nasal_aspirator,  # [手持式吸鼻器] nasal aspirator
    "drying_rack": lf_drying_rack,  # [奶瓶沥干支架] drying rack
    "knee_pillow": lf_knee_pillow,  # [夹腿枕] knee pillow
    "baby_stroller": lf_baby_stroller,  # [推车] baby stroller
    "air_purifier": lf_air_purifier,  # [桌面空气净化器] air purifier
    "play_mat": lf_play_mat,  # [应急垫] play mat
    "baby_food": lf_baby_food,  # [辅食机] baby food
    "breast_pump": lf_breast_pump,  # [吸奶器] breast pump
    "postpartum_belly": lf_postpartum_belly,  # [收腹带] postpartum belly
    "wooden_high": lf_wooden_high,  # [成长型餐椅] wooden high
    "sleeping_bag": lf_sleeping_bag,  # [婴儿无袖睡袋] sleeping bag
    "food_grade": lf_food_grade,  # [成长型餐椅] food grade
    "grade_tray": lf_grade_tray,  # [成长型餐椅] grade tray
    "adjustable_ergonomic": lf_adjustable_ergonomic,  # [成长型餐椅] adjustable ergonomic
    "nursing_pillow": lf_nursing_pillow,  # [U型哺乳枕] nursing pillow
    "nursing_pads": lf_nursing_pads,  # [一次性防溢乳垫] nursing pads
    "extendable_upf": lf_extendable_upf,  # [推车] extendable upf
    "upf_canopy": lf_upf_canopy,  # [推车] upf canopy
    "stroller_convertible": lf_stroller_convertible,  # [推车] stroller convertible
    "bath_bomb": lf_bath_bomb,  # [沐浴炸弹] bath bomb
    "para_beb": lf_para_beb,  # [磨甲器] para beb
    "electric_nasal": lf_electric_nasal,  # [手持式吸鼻器] electric nasal
    "baby_nail": lf_baby_nail,  # [磨甲器] baby nail
    "breast_milk": lf_breast_milk,  # [储奶壶] breast milk
    "vielen_dank": lf_vielen_dank,  # [记忆棉哺乳枕] vielen dank
    "important_max": lf_important_max,  # [手持式吸鼻器] important max
    "memory_foam": lf_memory_foam,  # [记忆棉哺乳枕] memory foam
    "belly_band": lf_belly_band,  # [托腹带] belly band
    "baby_wipe": lf_baby_wipe,  # [湿巾加热盒] baby wipe
    "any_questions": lf_any_questions,  # [手持式吸鼻器] any questions
    "belly_wrap": lf_belly_wrap,  # [收腹带] belly wrap
    "food_maker": lf_food_maker,  # [辅食机] food maker
    "air_ultra": lf_air_ultra,  # [吸奶器] air ultra
    "ultra_slim": lf_ultra_slim,  # [吸奶器] ultra slim
    "tank_top": lf_tank_top,  # [哺乳背心] tank top
    "baby_carrier": lf_baby_carrier,  # [腰凳背带] baby carrier
    "non_slip": lf_non_slip,  # [应急垫] non slip
    "car_seat": lf_car_seat,  # [ST01多功能婴儿标准车配件] car seat
    "der_rucksack": lf_der_rucksack,  # [妈咪包] der rucksack
    "conversation_user": lf_conversation_user,  # [电动摇椅] conversation user
    "slim_breast": lf_slim_breast,  # [吸奶器] slim breast
    "google_maps": lf_google_maps,  # [记忆棉哺乳枕] google maps
    "maps_search": lf_maps_search,  # [记忆棉哺乳枕] maps search
    "baby_play": lf_baby_play,  # [应急垫] baby play
    "sacco_nanna": lf_sacco_nanna,  # [婴儿无袖睡袋] sacco nanna
    "hip_seat": lf_hip_seat,  # [腰凳背带] hip seat
    "individually_wrapped": lf_individually_wrapped,  # [沐浴片] individually wrapped
    "mit_freundlichen": lf_mit_freundlichen,  # [背带] mit freundlichen
    "pro_baby": lf_pro_baby,  # [消毒器] pro baby
    "stroller_organizer": lf_stroller_organizer,  # [车挂包] stroller organizer
    "breast_pads": lf_breast_pads,  # [可水洗防溢乳垫] breast pads
    "baby_thermometer": lf_baby_thermometer,  # [儿童温度计] baby thermometer
    "nursing_tank": lf_nursing_tank,  # [哺乳背心] nursing tank
    "este_servicio": lf_este_servicio,  # [磨甲器] este servicio
    "sac_langer": lf_sac_langer,  # [妈咪包] sac langer
    "ergowrap_postpartum": lf_ergowrap_postpartum,  # [收腹带] ergowrap postpartum
    "pregnancy_pillow": lf_pregnancy_pillow,  # [W型孕妇枕] pregnancy pillow
    "lactation_massager": lf_lactation_massager,  # [乳房按摩仪] lactation massager
    "follow_previous": lf_follow_previous,  # [消毒器] follow previous
    "diaper_bag": lf_diaper_bag,  # [妈咪包] diaper bag
    "hot_cold": lf_hot_cold,  # [乳房冷热敷垫] hot cold
    "tire_lait": lf_tire_lait,  # [吸奶器背心] tire lait
    "der_schlafsack": lf_der_schlafsack,  # [婴儿无袖睡袋] der schlafsack
    "die_creme": lf_die_creme,  # [电动乳头按摩霜] die creme
    "pouch_holder": lf_pouch_holder,  # [防挤压支架] pouch holder
    "sent_iphone": lf_sent_iphone,  # [U型双边孕妇枕] sent iphone
    "therapy_packs": lf_therapy_packs,  # [乳房冷热敷垫] therapy packs
    "table_tbody": lf_table_tbody,  # [圆环背巾] table tbody
    "entry_gmail": lf_entry_gmail,  # [吸奶器内衣] entry gmail
    "gmail_source": lf_gmail_source,  # [吸奶器内衣] gmail source
    "soutien_gorge": lf_soutien_gorge,  # [吸奶器背心] soutien gorge
    "baby_swing": lf_baby_swing,  # [电动摇椅] baby swing
}


def apply_all(text: str) -> dict[str, tuple[bool, float]]:
    """对单条文本应用全部 label functions"""
    results = {}
    for tag_name, lf in LABEL_FUNCTION_REGISTRY.items():
        matched, conf = lf(text)
        if matched:
            results[tag_name] = (matched, conf)
    return results
