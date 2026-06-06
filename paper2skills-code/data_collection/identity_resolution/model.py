"""
Privacy-Safe Identity Resolution - Sherpa.ai + Cross-Domain SID + CAMP
arXiv: 2604.19219 / 2606.01396 / 2604.16521
"""

import hashlib, re, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class UserIdentity:
    platform: str
    raw_id: str
    pseudonym: str = ""
    behavior_vector: List[float] = field(default_factory=list)
    semantic_token: int = -1

    def __post_init__(self):
        if not self.pseudonym:
            self.pseudonym = hashlib.sha256(f"{self.platform}:{self.raw_id}".encode()).hexdigest()[:16]


@dataclass
class CrossPlatformUser:
    canonical_id: str
    identities: List[UserIdentity] = field(default_factory=list)
    persona_token: int = -1

    def add_identity(self, i: UserIdentity): self.identities.append(i)

    @property
    def platforms(self): return [i.platform for i in self.identities]


class PSUProtocol:
    def _h(self, i: UserIdentity) -> str:
        return hashlib.sha256(i.raw_id.encode()).hexdigest()

    def find_matches(self, a: List[UserIdentity], b: List[UserIdentity]) -> List[Tuple[str, str]]:
        ha = {self._h(i): i.pseudonym for i in a}
        hb = {self._h(i): i.pseudonym for i in b}
        return [(ha[h], hb[h]) for h in ha if h in hb]

    def align_users(self, party_a: List[UserIdentity], party_b: List[UserIdentity]) -> List[CrossPlatformUser]:
        matches = self.find_matches(party_a, party_b)
        ap = {i.pseudonym: i for i in party_a}
        bp = {i.pseudonym: i for i in party_b}
        merged = []
        for pa, pb in matches:
            canon = hashlib.md5(f"{pa}{pb}".encode()).hexdigest()[:12]
            u = CrossPlatformUser(canonical_id=canon)
            u.add_identity(ap[pa]); u.add_identity(bp[pb])
            merged.append(u)
        return merged


class SemanticIDEncoder:
    N_TOKENS = 64

    def encode_behavior(self, vec: List[float]) -> int:
        if not vec or sum(vec) == 0: return 0
        weighted = sum(v * (i + 1) for i, v in enumerate(vec))
        return int(weighted / sum(vec) * 10) % self.N_TOKENS

    def similarity(self, a: int, b: int) -> float:
        return 1.0 - abs(a - b) / self.N_TOKENS


class PIIProtector:
    EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')
    PHONE_RE = re.compile(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b')
    NAME_RE  = re.compile(r'(?:my name is|I am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.I)

    def __init__(self, max_pii: int = 3):
        self.max_pii = max_pii
        self.log: Dict[str, List[str]] = defaultdict(list)

    def detect(self, text: str) -> List[Tuple[str, str]]:
        found = [(t, m.group(0)) for t, r in [("email", self.EMAIL_RE), ("phone", self.PHONE_RE)]
                 for m in r.finditer(text)]
        for m in self.NAME_RE.finditer(text):
            found.append(("name", m.group(1)))
        return found

    def track_and_protect(self, session_id: str, text: str) -> Tuple[str, float]:
        found = self.detect(text)
        self.log[session_id].extend([t for t, _ in found])
        risk = min(1.0, len(self.log[session_id]) / self.max_pii)
        protected = text
        if risk >= 0.5:
            for pii_type, value in found:
                protected = protected.replace(value, f"[{pii_type.upper()}_REDACTED]")
        return protected, risk


def test_psu_alignment():
    p = PSUProtocol()
    amazon = [UserIdentity("amazon", "u001"), UserIdentity("amazon", "u002")]
    tiktok = [UserIdentity("tiktok", "u001"), UserIdentity("tiktok", "u003")]
    merged = p.align_users(amazon, tiktok)
    assert len(merged) == 1 and len(merged[0].identities) == 2
    assert "amazon" in merged[0].platforms and "tiktok" in merged[0].platforms
    print(f"[PASS] psu_alignment: {len(merged)} merged, platforms={merged[0].platforms}")


def test_semantic_id():
    enc = SemanticIDEncoder()
    buyer = [0.8, 0.2, 0.1, 0.9]
    browser = [0.1, 0.1, 0.9, 0.1]
    similar_buyer = [v * 1.05 for v in buyer]
    t1, t2, t3 = enc.encode_behavior(buyer), enc.encode_behavior(browser), enc.encode_behavior(similar_buyer)
    assert enc.similarity(t1, t3) >= enc.similarity(t1, t2)
    print(f"[PASS] semantic_id: buyer={t1}, browser={t2}, similar={t3}")


def test_pii_protection():
    p = PIIProtector(max_pii=2)
    _, r1 = p.track_and_protect("s1", "Hello, my name is Alice Smith")
    protected, r2 = p.track_and_protect("s1", "Email: alice@example.com")
    assert r2 >= r1
    assert "REDACTED" in protected or "alice@example.com" not in protected
    print(f"[PASS] pii_protect: risk1={r1:.2f}, risk2={r2:.2f}, redacted={'REDACTED' in protected}")


def test_cross_platform_full():
    p = PSUProtocol(); enc = SemanticIDEncoder()
    a = [UserIdentity("amazon", f"u{i}", behavior_vector=[float(i%3)*0.3, 0.5]) for i in range(5)]
    b = [UserIdentity("tiktok", f"u{i}", behavior_vector=[float(i%3)*0.3+0.1, 0.4]) for i in range(3, 8)]
    merged = p.align_users(a, b)
    for u in merged:
        vec = [v for id_ in u.identities for v in id_.behavior_vector]
        u.persona_token = enc.encode_behavior(vec)
    assert len(merged) >= 1 and all(u.persona_token >= 0 for u in merged)
    print(f"[PASS] cross_platform: {len(merged)} users, tokens={[u.persona_token for u in merged]}")


if __name__ == "__main__":
    test_psu_alignment(); test_semantic_id(); test_pii_protection(); test_cross_platform_full()
    print("\n✅ All tests passed")
