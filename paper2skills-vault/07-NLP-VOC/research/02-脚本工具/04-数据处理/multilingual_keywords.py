"""多语言关键词对照字典

基于 momcozy 数据 other 文本挖掘，补充英/德/法/意/西/葡/荷/波兰等
欧洲及拉美市场多语言关键词。

结构：品线 → {语言代码: [关键词列表]}
语言代码: en(英) de(德) fr(法) es(西) it(意) pt(葡) nl(荷) pl(波兰)
"""

from typing import Optional


# ── 多语言关键词库 ─────────────────────────────────────────────────

MULTILINGUAL_KEYWORDS: dict[str, dict[str, list[str]]] = {
    # ═══════════════════════════════════════════════════════════════
    # breast_pump 吸奶器
    # ═══════════════════════════════════════════════════════════════
    "breast_pump": {
        "en": ["breast pump", "breastpump", "pump", "pumping", "milking", "express"],
        "de": ["milchpumpe", "milchpumpen", "absaugpumpe", "absaugen", "milch abpumpen", "brustpumpe"],
        "fr": ["tire lait", "tire-lait", "pompe", "tirer lait", "expression", "tirer son lait", "pompe allaitement"],
        "es": ["sacaleches", "saca leches", "extractor", "extractor leche", "bomba leche", "sacar leche"],
        "it": ["tiralatte", "tira latte", "pompa", "estrarre latte", "pompa seno", "tiralatte elettrico"],
        "pt": ["bomba tira leite", "extrator leite", "tirar leite", "bomba leite", "ordenhadeira"],
        "nl": ["borstkolf", "kolf", "melk afkolven", "kolven", "borstpomp"],
        "pl": ["laktator", "pompa", "odciąganie mleka", "laktator elektryczny"],
    },

    # ═══════════════════════════════════════════════════════════════
    # wearable_breast_pump 穿戴式吸奶器
    # ═══════════════════════════════════════════════════════════════
    "wearable_breast_pump": {
        "en": ["wearable pump", "hands free pump", "hands-free pump", "in bra pump", "wireless pump", "portable pump"],
        "de": ["tragbare milchpumpe", "milchpumpe bh", "drahtlose milchpumpe", "handfrei pumpe", "mobil milchpumpe"],
        "fr": ["tire lait portable", "tire lait mains libres", "tire lait sans fil", "tire lait discret", "tire-lait portable"],
        "es": ["sacaleches portátil", "sacaleches manos libres", "sacaleches inalámbrico", "extractor portátil"],
        "it": ["tiralatte portatile", "tiralatte senza fili", "tiralatte indossabile", "tiralatte mani libere"],
        "pt": ["bomba tira leite portátil", "bomba sem fio", "bomba mãos livres"],
        "nl": ["draagbare borstkolf", "draadloze borstkolf", "handsfree borstkolf"],
    },

    # ═══════════════════════════════════════════════════════════════
    # bottle_warmer 暖奶器
    # ═══════════════════════════════════════════════════════════════
    "bottle_warmer": {
        "en": ["bottle warmer", "milk warmer", "formula warmer", "baby food warmer"],
        "de": ["flaschenwärmer", "milchwärmer", "babykostwärmer", "fläschchenwärmer", "babyflaschen wärmer"],
        "fr": ["chauffe biberon", "chauffe-biberon", "réchauffeur biberon", "chauffe lait", "réchauffe biberon"],
        "es": ["calienta biberones", "calienta biberón", "calentador biberones", "calienta leche", "esterilizador calienta"],
        "it": ["scaldabiberon", "scalda biberon", "riscaldatore biberon", "scaldalatte"],
        "pt": ["aquecedor mamadeiras", "aquecedor biberões", "esquentador biberões"],
        "nl": ["flessenverwarmer", "flessen warmer", "babyfles verwarmer"],
        "pl": ["podgrzewacz butelek", "podgrzewacz do butelek", "podgrzewacz mleka"],
    },

    # ═══════════════════════════════════════════════════════════════
    # sterilizer 消毒器
    # ═══════════════════════════════════════════════════════════════
    "sterilizer": {
        "en": ["sterilizer", "steriliser", "sanitize", "disinfect", "uv sterilizer", "steam sterilizer"],
        "de": ["sterilisator", "desinfizieren", "uv sterilisator", "dampfsterilisator", "flaschen sterilisator", "vaporisator"],
        "fr": ["stérilisateur", "stériliser", "désinfecter", "stérilisateur uv", "stérilisateur vapeur", "stérilisateur biberon"],
        "es": ["esterilizador", "esterilizar", "desinfectar", "esterilizador uv", "esterilizador vapor", "esterilizador biberones"],
        "it": ["sterilizzatore", "sterilizzare", "disinfettare", "sterilizzatore uv", "sterilizzatore vapore"],
        "pt": ["esterilizador", "esterilizar", "desinfetar", "esterilizador uv", "esterilizador vapor"],
        "nl": ["sterilisator", "steriliseren", "ontsmetten", "flessensterilisator", "uv sterilisator"],
    },

    # ═══════════════════════════════════════════════════════════════
    # pregnancy_pillow 孕妇枕
    # ═══════════════════════════════════════════════════════════════
    "pregnancy_pillow": {
        "en": ["pregnancy pillow", "maternity pillow", "body pillow", "wedge pillow", "boppy"],
        "de": ["schwangerschaftskissen", "stillkissen", "seitenschläferkissen", "lagenkissen", "bauch kissen"],
        "fr": ["coussin grossesse", "coussin allaitement", "oreiller grossesse", "coussin maternité", "oreiller corps"],
        "es": ["almohada embarazo", "cojín embarazo", "almohada lactancia", "cojín lactancia", "almohada maternidad"],
        "it": ["cuscino gravidanza", "cuscino allattamento", "cuscino maternità", "cuscino corpo"],
        "pt": ["almofada gravidez", "travesseiro gravidez", "almofada amamentação"],
        "nl": ["zwangerschapskussen", "voedingskussen", "body kussen"],
    },

    # ═══════════════════════════════════════════════════════════════
    # nursing_bra 哺乳内衣
    # ═══════════════════════════════════════════════════════════════
    "nursing_bra": {
        "en": ["nursing bra", "breastfeeding bra", "pumping bra", "maternity bra", "hands free bra"],
        "de": ["still bh", "still-bh", "schwangerschafts bh", "umstands bh", "pump bh", "trage bh"],
        "fr": ["soutien gorge allaitement", "soutien-gorge allaitement", "brassière allaitement", "soutien gorge grossesse"],
        "es": ["sujetador lactancia", "sujetador amamantar", "top lactancia", "sujetador embarazo", "brasier lactancia"],
        "it": ["reggiseno allattamento", "reggiseno gravidanza", "top allattamento", "reggiseno maternità"],
        "pt": ["soutien amamentação", "sutiã amamentação", "top amamentação", "soutien gravidez"],
        "nl": ["voedingsbh", "zwangerschapsbh", "borstvoeding bh", "draagbh"],
    },

    # ═══════════════════════════════════════════════════════════════
    # baby_monitor 婴儿监视器
    # ═══════════════════════════════════════════════════════════════
    "baby_monitor": {
        "en": ["baby monitor", "video monitor", "baby camera", "wifi monitor", "breathing monitor"],
        "de": ["babyphone", "baby monitor", "babykamera", "babyfon", "baby überwachung", "baby phone"],
        "fr": ["babyphone", "baby phone", "moniteur bébé", "caméra bébé", "écoute bébé"],
        "es": ["vigilabebés", "vigila bebés", "monitor bebé", "cámara bebé", "intercomunicador bebé"],
        "it": ["baby monitor", "baby phone", "babyphone", "monitor neonato", "telecamera neonato"],
        "pt": ["babá eletrônica", "baba eletronica", "monitor bebé", "câmara bebé"],
        "nl": ["babyfoon", "baby monitor", "baby camera", "baby telefoon"],
    },

    # ═══════════════════════════════════════════════════════════════
    # sound_machine 白噪音机
    # ═══════════════════════════════════════════════════════════════
    "sound_machine": {
        "en": ["sound machine", "white noise", "white noise machine", "sleep machine", "noise machine"],
        "de": ["einschlafhilfe", "weißes rauschen", "weißrauschen", "schlafgerät", "geräuschmaschine", "baby einschlafhilfe"],
        "fr": ["bruit blanc", "machine bruit blanc", "aide sommeil", "veilleuse bruit blanc"],
        "es": ["ruido blanco", "máquina ruido blanco", "máquina sonidos", "ayuda dormir"],
        "it": ["rumore bianco", "macchina rumore bianco", "aiuto sonno", "suoni rilassanti"],
        "pt": ["ruído branco", "máquina ruído branco", "máquina sons", "ajuda dormir"],
        "nl": ["witte ruis", "witte ruis machine", "slaaphulp", "geluidsmachine"],
    },

    # ═══════════════════════════════════════════════════════════════
    # humidifier 加湿器
    # ═══════════════════════════════════════════════════════════════
    "humidifier": {
        "en": ["humidifier", "vaporizer", "cool mist", "warm mist", "ultrasonic humidifier"],
        "de": ["luftbefeuchter", "ultraschall luftbefeuchter", "vernebler", "raumbefeuchter", "luftbefeuchtung"],
        "fr": ["humidificateur", "humidificateur ultrasonique", "brumisateur", "humidificateur chambre"],
        "es": ["humidificador", "humidificador ultrasónico", "vaporizador", "humidificador bebé"],
        "it": ["umidificatore", "umidificatore ultrasuoni", "vaporizzatore", "umidificatore neonato"],
        "pt": ["humidificador", "humidificador ultrasónico", "vaporizador", "humidificador bebé"],
        "nl": ["luchtbevochtiger", "ultrasone luchtbevochtiger", "vernevelaar"],
    },

    # ═══════════════════════════════════════════════════════════════
    # air_purifier 空气净化器
    # ═══════════════════════════════════════════════════════════════
    "air_purifier": {
        "en": ["air purifier", "air cleaner", "hepa filter", "air filtration"],
        "de": ["luftreiniger", "luftfilter", "luft reiniger", "luftreinigung", "hepa filter"],
        "fr": ["purificateur air", "purificateur d'air", "filtre hepa", "assainisseur air"],
        "es": ["purificador aire", "purificador de aire", "filtro hepa", "limpiador aire"],
        "it": ["purificatore aria", "purificatore d'aria", "filtro hepa", "depuratore aria"],
        "pt": ["purificador ar", "purificador de ar", "filtro hepa", "limpeza ar"],
        "nl": ["luchtreiniger", "luchtfilter", "hepa filter", "lucht zuiveraar"],
    },

    # ═══════════════════════════════════════════════════════════════
    # baby_bottle 奶瓶
    # ═══════════════════════════════════════════════════════════════
    "baby_bottle": {
        "en": ["baby bottle", "feeding bottle", "sippy cup", "straw cup", "training cup", "nipple", "teat"],
        "de": ["babyflasche", "flasche", "schnullerflasche", "trinklernbecher", "trinkbecher", "sauger"],
        "fr": ["biberon", "biberons", "tasse d'apprentissage", "gobelet", "tétine", "biberon bébé"],
        "es": ["biberón", "biberones", "mamadera", "taza aprendizaje", "vaso entrenamiento", "chupete"],
        "it": ["biberon", "biberon", "tazza apprendimento", "bicchiere", "ciuccio"],
        "pt": ["biberão", "biberões", "mamadeira", "copo aprendizagem", "chupeta"],
        "nl": ["babyfles", "fles", "tuitbeker", "drinkbeker", "speen"],
        "pl": ["butelka", "buteleczka", "kubek", "smoczek"],
    },

    # ═══════════════════════════════════════════════════════════════
    # baby_carrier 背带
    # ═══════════════════════════════════════════════════════════════
    "baby_carrier": {
        "en": ["baby carrier", "wrap carrier", "ring sling", "front carrier", "hip carrier"],
        "de": ["babytrage", "trage", "tragetuch", "ringsling", "hüftsitz", "bauchtrage", "rückentrage"],
        "fr": ["porte bébé", "porte-bébé", "écharpe", "écharpe portage", "sling", "portage"],
        "es": ["mochila portabebés", "portabebés", "fular", "fular portabebés", "porteo"],
        "it": ["marsupio", "fascia portabebè", "porta bebè", "fascia", "zaino porta bimbo"],
        "pt": ["canguru", "canguru bebé", "sling", "wrap", "transportador bebé"],
        "nl": ["draagzak", "draagdoek", "draagzak baby", "ringsling"],
    },

    # ═══════════════════════════════════════════════════════════════
    # breast_milk_storage 储奶
    # ═══════════════════════════════════════════════════════════════
    "breast_milk_storage": {
        "en": ["milk storage", "milk bag", "storage bag", "milk container", "freeze milk"],
        "de": ["milch aufbewahrung", "milchbeutel", "aufbewahrungsbeutel", "milchbehälter", "muttermilch beutel"],
        "fr": ["conservation lait", "sac lait", "poche lait", "contenant lait", "congeler lait", "lait maternel"],
        "es": ["almacenamiento leche", "bolsa leche", "bolsa almacenamiento", "recipiente leche", "congelar leche"],
        "it": ["conservazione latte", "sacchetta latte", "contenitore latte", "congelare latte"],
        "pt": ["armazenamento leite", "saco leite", "recipiente leite", "congelar leite"],
        "nl": ["melk opslag", "melkzakje", "melk bewaarzak", "melkcontainer", "melk invriezen"],
    },

    # ═══════════════════════════════════════════════════════════════
    # postpartum_recovery 产后恢复
    # ═══════════════════════════════════════════════════════════════
    "postpartum_recovery": {
        "en": ["postpartum", "postpartum recovery", "shapewear", "waist trainer", "compression", "faja", "tummy control"],
        "de": ["nach der geburt", "postpartum", "bauchgurt", "kompressionsbandage", "mieder", "body shaper", "taillentrainer"],
        "fr": ["post-partum", "post partum", "gainante", "ceinture post partum", "compression", "ventre plat"],
        "es": ["posparto", "postparto", "faja", "cinturilla", "compresión", "cinturón postparto"],
        "it": ["post partum", "post-partum", "contenitiva", "cintura post partum", "compressione"],
        "pt": ["pós-parto", "cinta", "modelador", "compressão", "cinta pós parto"],
        "nl": ["postpartum", "na de bevalling", "buikband", "compressie", "shapewear"],
    },

    # ═══════════════════════════════════════════════════════════════
    # nipple_cream 乳头霜
    # ═══════════════════════════════════════════════════════════════
    "nipple_cream": {
        "en": ["nipple cream", "lanolin", "nipple balm", "nipple butter", "nipple ointment"],
        "de": ["brustwarzensalbe", "brustwarzen creme", "lanolin", "warzensalbe", "still creme"],
        "fr": ["crème mamelon", "baume mamelon", "lanoline", "soin mamelon", "crème allaitement"],
        "es": ["crema pezón", "bálsamo pezón", "lanolina", "ungüento pezón", "crema lactancia"],
        "it": ["crema capezzolo", "balsamo capezzolo", "lanolina", "unguento capezzolo"],
        "pt": ["creme mamilo", "bálsamo mamilo", "lanolina", "pomada mamilo"],
        "nl": ["tepelcrème", "tepel zalf", "lanoline", "tepelbalsem"],
    },

    # ═══════════════════════════════════════════════════════════════
    # breast_pad 防溢乳垫
    # ═══════════════════════════════════════════════════════════════
    "breast_pad": {
        "en": ["nursing pad", "breast pad", "bra pad", "leak pad", "disposable pad"],
        "de": ["stilleinlagen", "still einlagen", "absorber", "stilleinlage", "brust pads"],
        "fr": ["coussinet d'allaitement", "coussinet allaitement", "tétine d'allaitement", "compresses"],
        "es": ["discos lactancia", "discos desechables", "compresas lactancia", "almohadillas lactancia"],
        "it": ["coppette allattamento", "assorbenti allattamento", "dischetti allattamento"],
        "pt": ["discos amamentação", "discos amamentar", "absorventes amamentação"],
        "nl": ["zoogcompressen", "borstcompressen", "zoogkompres", "borst pads"],
    },

    # ═══════════════════════════════════════════════════════════════
    # baby_wipe 湿巾
    # ═══════════════════════════════════════════════════════════════
    "baby_wipe": {
        "en": ["baby wipe", "wet wipe", "diaper wipe"],
        "de": ["baby feuchttücher", "feuchttücher", "babypflegetücher", "feuchtes tuch"],
        "fr": ["lingettes bébé", "lingette bébé", "lingettes", "lingettes humides"],
        "es": ["toallitas bebé", "toallitas húmedas", "toallitas", "pañitos"],
        "it": ["salviettine bambino", "salviette neonato", "salviettine umidificate"],
        "pt": ["toalhitas bebé", "toalhitas", "toalhitas húmidas", "lenços umedecidos"],
        "nl": ["vochtige doekjes", "babydoekjes", "billendoekjes", "vochtige babydoekjes"],
    },

    # ═══════════════════════════════════════════════════════════════
    # stroller 推车
    # ═══════════════════════════════════════════════════════════════
    "stroller": {
        "en": ["stroller", "pram", "pushchair", "buggy", "baby stroller"],
        "de": ["kinderwagen", "buggy", "sportwagen", "babywagen", "kinderbuggy"],
        "fr": ["poussette", "landau", "poussette canne", "poussette trio"],
        "es": ["cochecito", "silla paseo", "carrito", "carro bebé", "coche bebé"],
        "it": ["passeggino", "carrozzina", "passeggino trio", "passeggino leggero"],
        "pt": ["carrinho", "carrinho bebé", "carrinho passeio", "coche bebé"],
        "nl": ["kinderwagen", "buggy", "wandelwagen", "baby wagen"],
    },

    # ═══════════════════════════════════════════════════════════════
    # car_seat 安全座椅
    # ═══════════════════════════════════════════════════════════════
    "car_seat": {
        "en": ["car seat", "infant car seat", "booster seat", "convertible car seat"],
        "de": ["autokindersitz", "kindersitz", "babyschale", "autositz", "autokindersitz"],
        "fr": ["siège auto", "siège-auto", "coque", "réhausseur", "siège bébé"],
        "es": ["silla coche", "silla auto", "portabebé", "elevador", "silla contramarcha"],
        "it": ["seggiolino auto", "seggiolino", "ovetto", "rialzo auto", "seggiolino bimbo"],
        "pt": ["cadeira auto", "cadeirinha", "ovo", "assento elevação", "cadeira bebé"],
        "nl": ["autostoel", "kinderzitje", "autostoeltje", "baby autostoel", "verhoger"],
    },

    # ═══════════════════════════════════════════════════════════════
    # diaper_bag 妈咪包
    # ═══════════════════════════════════════════════════════════════
    "diaper_bag": {
        "en": ["diaper bag", "nappy bag", "changing bag", "baby bag"],
        "de": ["wickeltasche", "windeltasche", "wickelrucksack", "baby tasche"],
        "fr": ["sac à langer", "sac langer", "sac à couches", "sac maternité"],
        "es": ["bolso cambiador", "bolsa pañales", "bolso maternal", "mochila pañales"],
        "it": ["borsa fasciatoio", "borsa cambio", "zaino fasciatoio", "borsa mamma"],
        "pt": ["mala maternidade", "mochila fraldas", "saco mudanças", "bolsa fraldas"],
        "nl": ["luiertas", "verzorgingstas", "luierrugzak", "baby tas"],
    },

    # ═══════════════════════════════════════════════════════════════
    # baby_clothing 婴儿服装
    # ═══════════════════════════════════════════════════════════════
    "baby_clothing": {
        "en": ["baby clothes", "onesie", "footie", "newborn clothes", "baby outfit"],
        "de": ["babykleidung", "baby kleider", "strampler", "body", "neugeborenen kleidung"],
        "fr": ["vetement bébé", "body bébé", "gigoteuse", "pyjama bébé", "tenue bébé"],
        "es": ["ropa bebé", "ropa bebé", "body", "pelele", "pijama bebé"],
        "it": ["vestiti neonato", "tutina", "pagliaccetto", "body", "pigiama neonato"],
        "pt": ["roupa bebé", "body", "macacão", "pijama bebé", "conjunto bebé"],
        "nl": ["babykleding", "rompertje", "pakje", "newborn kleding", "baby outfit"],
    },

    # ═══════════════════════════════════════════════════════════════
    # red_light_therapy 红光理疗
    # ═══════════════════════════════════════════════════════════════
    "red_light_therapy": {
        "en": ["red light therapy", "light therapy", "infrared therapy", "led therapy", "phototherapy"],
        "de": ["lichttherapie", "rotlicht therapie", "infrarottherapie", "led therapie"],
        "fr": ["thérapie lumière", "luminothérapie", "photothérapie", "thérapie lumière rouge"],
        "es": ["terapia luz", "fototerapia", "terapia luz roja", "luminoterapia"],
        "it": ["terapia luce", "fototerapia", "terapia luce rossa", "luminoterapia"],
        "pt": ["terapia luz", "fototerapia", "terapia luz vermelha"],
        "nl": ["lichttherapie", "rode licht therapie", "fototherapie"],
    },

    # ═══════════════════════════════════════════════════════════════
    # crib_playard 婴儿床/摇椅
    # ═══════════════════════════════════════════════════════════════
    "crib_playard": {
        "en": ["crib", "playard", "playpen", "bassinet", "travel crib"],
        "de": ["babybett", "laufstall", "reisebett", "wiege", "beistellbett"],
        "fr": ["lit bébé", "parc", "parc bébé", "berceau", "lit parapluie"],
        "es": ["cuna", "parque", "parque bebé", "moisés", "cuna viaje"],
        "it": ["lettino", "box", "parco giochi", "culletta", "lettino da viaggio"],
        "pt": ["berço", "parque", "moisés", "cama viagem", "berço portátil"],
        "nl": ["babybedje", "box", "reisbed", "wieg", "co-sleeper"],
    },

    # ═══════════════════════════════════════════════════════════════
    # customer_service 客服/物流（兜底分流）
    # ═══════════════════════════════════════════════════════════════
    "customer_service": {
        "en": ["shipping", "delivery", "return", "refund", "customer service", "exchange"],
        "de": ["lieferung", "versand", "rückgabe", "rücksendung", "kundenservice", "umtausch", "zustellung"],
        "fr": ["livraison", "expédition", "retour", "remboursement", "service client", "échange"],
        "es": ["envío", "entrega", "devolución", "reembolso", "servicio al cliente", "cambio"],
        "it": ["spedizione", "consegna", "reso", "rimborso", "servizio clienti", "cambio"],
        "pt": ["envio", "entrega", "devolução", "reembolso", "serviço cliente", "troca"],
        "nl": ["verzending", "levering", "retour", "terugbetaling", "klantenservice", "ruilen"],
    },
}


# ── 便捷函数 ──────────────────────────────────────────────────────

def get_all_keywords(product_line: str) -> list[str]:
    """获取某品线的所有语言关键词（扁平列表）"""
    lang_dict = MULTILINGUAL_KEYWORDS.get(product_line, {})
    result = []
    for kws in lang_dict.values():
        result.extend(kws)
    return result


def get_keywords_by_lang(product_line: str, lang: str) -> list[str]:
    """获取某品线某语言的关键词"""
    return MULTILINGUAL_KEYWORDS.get(product_line, {}).get(lang, [])


def flatten_all_keywords() -> dict[str, list[str]]:
    """将所有多语言关键词按品线扁平化为单一列表（用于规则引擎）"""
    return {
        line: get_all_keywords(line)
        for line in MULTILINGUAL_KEYWORDS.keys()
    }


# ── 统计 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("多语言关键词库统计")
    print("=" * 50)
    for line, lang_dict in MULTILINGUAL_KEYWORDS.items():
        total = sum(len(kws) for kws in lang_dict.values())
        langs = ", ".join(f"{l}({len(kws)})" for l, kws in lang_dict.items())
        print(f"{line:30s}: {total:3d} 词 | {langs}")
