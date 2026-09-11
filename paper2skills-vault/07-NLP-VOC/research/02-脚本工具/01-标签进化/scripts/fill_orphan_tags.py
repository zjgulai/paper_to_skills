"""Fill 88 orphan tags into dim_tag.

These tag_ids exist in voc_label but not in dim_tag (v3.x legacy IDs +
ALCHEmist weak-supervision tags). voc_label already has tag_cn / tag_en /
aipl_node / sentiment_preset for them. dept_owner is the only missing field,
filled via DeepSeek LLM with closed-set 8 departments.

Inserted rows are marked audit_status='auto_from_label' so they can be
distinguished from v4.5 dictionary curated rows.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

sys.path.insert(0, str(Path("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎").resolve()))
from llm_client import LLMClient

SEED_PATH = Path("/tmp/orphan_seed.jsonl")
PG_CFG = json.loads(Path("~/.paper2skills/voc_bi_pg.json").expanduser().read_text())

DEPT_VOCAB = ['全球客服中心','产品中心','仓储物流部','品牌市场中心','电商运营部','品质管理中心','法务合规部','用户运营部']
DEPT_PROMPT = (
    f"VOC 标签需指派主责部门。可选范围（机器可读）: {' / '.join(DEPT_VOCAB)}。"
    "根据 tag_cn、tag_en、AIPL、情感极性，选出最贴合的一个。只返回部门名字符串。\n"
)

def build_user_prompt(rec: dict) -> str:
    parts = [
        f"tag_id: {rec['tag_id']}",
        f"tag_cn: {rec['tag_cn']}",
        f"tag_en: {rec.get('tag_en') or ''}",
        f"AIPL: {rec.get('aipl_node') or ''}",
        f"polarity: {rec.get('polarity') or rec.get('sentiment_preset') or ''}",
    ]
    if rec.get('product_line'):
        parts.append(f"产品品线: {rec['product_line']}")
    return "\n".join(parts)

async def fill_dept_via_llm(records: list[dict], concurrency: int = 10) -> list[dict]:
    client = LLMClient()
    sem = asyncio.Semaphore(concurrency)

    async def one(rec):
        async with sem:
            try:
                resp = await client.chat_async(
                    vendor="deepseek",
                    messages=[
                        {"role": "system", "content": DEPT_PROMPT},
                        {"role": "user", "content": build_user_prompt(rec)},
                    ],
                    model="deepseek-chat",
                    temperature=0.1,
                    max_tokens=50,
                )
                raw = resp.content.strip()
                rec['dept_owner'] = raw if raw in DEPT_VOCAB else fallback_dept(rec, raw)
            except Exception as e:
                rec['dept_owner'] = fallback_dept(rec, None)
                rec['_llm_error'] = str(e)
            return rec

    return await asyncio.gather(*[one(r) for r in records])

def fallback_dept(rec: dict, raw: str | None) -> str:
    """Rule-based fallback when LLM returns a value outside closed vocab."""
    if raw:
        for d in DEPT_VOCAB:
            if d in raw:
                return d
    cn = (rec.get('tag_cn') or '').lower()
    aipl = (rec.get('aipl_node') or '').upper()
    if aipl == 'A' or '推荐' in cn or '种草' in cn or '品牌' in cn:
        return '品牌市场中心'
    if '客服' in cn or '咨询' in cn:
        return '全球客服中心'
    if '物流' in cn or '配送' in cn or '发货' in cn:
        return '仓储物流部'
    if '价格' in cn or '促销' in cn:
        return '电商运营部'
    return '产品中心'

def main():
    records = [json.loads(l) for l in SEED_PATH.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(records)} orphan records")

    t0 = time.time()
    enriched = asyncio.run(fill_dept_via_llm(records, concurrency=12))
    print(f"LLM dept_owner filled in {time.time()-t0:.2f}s")
    errors = [r for r in enriched if r.get('_llm_error')]
    print(f"  llm errors: {len(errors)}")

    rows = []
    for r in enriched:
        rows.append((
            r['tag_id'],
            r['tag_cn'],
            r.get('tag_en'),
            r.get('aipl_node'),
            r.get('polarity'),
            r['dept_owner'],
            None,
            None,
            False,
            'auto_from_label',
            r.get('product_line'),
            None,
        ))

    conn = psycopg2.connect(**{k: PG_CFG[k] for k in ['host','port','database','user','password']})
    with conn.cursor() as cur:
        cur.execute(
            """SELECT COUNT(*) FROM dim_tag WHERE tag_id = ANY(%s)""",
            ([r[0] for r in rows],),
        )
        pre_exists = cur.fetchone()[0]
        print(f"\nPre-insert: {pre_exists} of these tag_ids already in dim_tag (will be skipped or replaced)")

        if pre_exists > 0:
            cur.execute(
                """DELETE FROM dim_tag WHERE tag_id = ANY(%s) AND audit_status = 'auto_from_label'""",
                ([r[0] for r in rows],),
            )
            print(f"  cleared {cur.rowcount} previous 'auto_from_label' rows")

        execute_values(
            cur,
            """INSERT INTO dim_tag
                 (tag_id, tag_cn, tag_en, aipl_node, polarity, dept_owner,
                  biz_action, strategy_pkg, is_general, audit_status, product_line, collab_dept)
               VALUES %s
               ON CONFLICT (tag_id) DO NOTHING""",
            rows, page_size=200,
        )
        print(f"\nINSERT done. affected rows: {cur.rowcount}")
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM dim_tag")
        print(f"\ndim_tag total after fix: {cur.fetchone()[0]}")
        cur.execute("""
            SELECT COUNT(DISTINCT l.tag_id)
            FROM voc_label l LEFT JOIN dim_tag d ON l.tag_id = d.tag_id
            WHERE d.tag_id IS NULL
        """)
        print(f"voc_label orphan remaining: {cur.fetchone()[0]}")
        cur.execute("""
            SELECT dept_owner, COUNT(*) AS n
            FROM dim_tag WHERE audit_status='auto_from_label'
            GROUP BY dept_owner ORDER BY n DESC
        """)
        print("\nauto_from_label rows by dept:")
        for d, n in cur.fetchall(): print(f"  {n:>3d} {d}")
    conn.close()

if __name__ == "__main__":
    main()
