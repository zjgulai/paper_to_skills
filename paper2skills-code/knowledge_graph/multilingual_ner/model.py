"""
Multilingual NER — Universal NER v2 inspired implementation
Skill: Skill-Multilingual-NER-Universal-v2.md
"""

import re
from collections import defaultdict


class MultilingualNER:
    """多语言实体识别器（规则+词典版）"""

    def __init__(self):
        self.entity_dict = {
            'BRAND': {
                'en': ['Momcozy', 'Medela', 'Philips Avent', 'Spectra'],
                'de': ['Momcozy', 'Medela', 'Nuk'],
                'ja': ['Momcozy', 'Medela', 'Pigeon'],
                'zh': ['Momcozy', '美德乐', '新安怡', '贝亲']
            },
            'PRODUCT': {
                'en': ['breast pump', 'nursing bra', 'baby monitor'],
                'de': ['Milchpumpe', 'Still-BH', 'Babyphone'],
                'ja': ['搾乳器', '授乳ブラ', 'ベビーモニター'],
                'zh': ['吸奶器', '哺乳内衣', '婴儿监视器']
            }
        }

    def detect_language(self, text):
        if re.search(r'[一-鿿]', text):
            return 'zh'
        elif re.search(r'[぀-ゟ゠-ヿ]', text):
            return 'ja'
        elif re.search(r'[äöüß]', text):
            return 'de'
        return 'en'

    def extract_entities(self, text, lang=None):
        if lang is None:
            lang = self.detect_language(text)
        text_lower = text.lower()
        entities = []

        for entity_type, lang_dict in self.entity_dict.items():
            terms = lang_dict.get(lang, lang_dict.get('en', []))
            for term in terms:
                for m in re.finditer(re.escape(term.lower()), text_lower):
                    s, e = m.span()
                    entities.append({'text': text[s:e], 'type': entity_type, 'lang': lang})

        # 去重保留最长匹配
        entities.sort(key=lambda x: (text_lower.index(x['text'].lower()), -len(x['text'])))
        result = []
        for e in entities:
            if not result or text_lower.index(e['text'].lower()) >= text_lower.index(result[-1]['text'].lower()) + len(result[-1]['text']):
                result.append(e)
        return result


if __name__ == '__main__':
    ner = MultilingualNER()
    texts = [
        ("en", "Momcozy breast pump is great."),
        ("zh", "Momcozy吸奶器很好用。"),
    ]
    for lang, text in texts:
        print(f"[{lang}] {text}")
        for e in ner.extract_entities(text, lang):
            print(f"  {e['text']} ({e['type']})")
