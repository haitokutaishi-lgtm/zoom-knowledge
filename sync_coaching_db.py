#!/usr/bin/env python3
"""
ナレッジフォルダの新しいMDファイルをcoaching DBに自動反映するスクリプト。
launchdで定期実行（例：30分ごと）する。
"""
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import asyncpg

# ─── 設定 ──────────────────────────────────────────────────────────
KNOWLEDGE_DIR = Path("/Users/yahashitakuro/Downloads/claude作業フォルダ/ナレッジ/Zoomセッション記録")
PROCESSED_FILE = Path(__file__).parent / "processed_coaching_md.json"

# .envから DATABASE_URL を読み込む
def _load_database_url() -> str:
    env_path = Path(__file__).parent / ".env"
    for line in env_path.read_text().splitlines():
        if line.startswith("COACHING_DATABASE_URL="):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("COACHING_DATABASE_URL not found in .env")

CLIENT_KEYWORDS: dict[int, list[str]] = {
    1:  ["ゆゆ"],
    2:  ["saori"],
    3:  ["やまし"],
    4:  ["NAOKI", "naoki"],
    5:  ["あずき"],
    7:  ["kizuku", "Kizuku", "kizku"],
    8:  ["ケンテイ"],
    9:  ["ゆうき", "yuuki", "ユウキ"],
    10: ["ひなの"],
    11: ["なむなむ"],
    12: ["佐藤"],
    13: ["hiromi3588"],
    14: ["YAMA", "CAMP YAMA"],
    15: ["Haruka Makino", "はるか", "牧野はるか", "CAMP はるか"],
    16: ["erika", "アカサカエリカ", "赤坂エリカ"],
    17: ["りょう"],
    18: ["KS"],
    19: ["Sato"],
    20: ["新昌靜", "しょうせい"],
    21: ["りんたろう", "rintaro", "秋山", "倫太郎"],
    22: ["Kobayashi Takahiro", "TAKAHIRO KOBAYASHI", "小林", "たか"],
    23: ["山田", "yamada", "国毅"],
    24: ["定國洋子", "yoko"],
    25: ["hitomi", "hitomix", "尾崎仁美", "UKさん"],
    26: ["Kai", "kai", "カイ"],
    27: ["YUKI", "yuki", "ゆき"],
}

# グループイベント・スキップ対象キーワード
SKIP_KEYWORDS = ["CAMP イベント", "CAMP EVENT", "CAMP YAMA", "CAMP はるか", "CAMP NAOKI"]


def _match_client_id(filename: str, content: str) -> int | None:
    """最長一致でクライアントIDを返す"""
    target = filename + " " + content[:500]
    target_lower = target.lower()
    best: tuple[int, int] | None = None
    for client_id, keywords in CLIENT_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in target_lower:
                if best is None or len(kw) > best[0]:
                    best = (len(kw), client_id)
    return best[1] if best else None


def _should_skip(filename: str) -> bool:
    """グループイベント等スキップ対象かどうか"""
    for kw in SKIP_KEYWORDS:
        if kw in filename:
            return True
    # 1on1セッションでなさそうなもの
    if "イベント" in filename or "EVENT" in filename.upper():
        return True
    return False


def _extract_date(filename: str) -> str | None:
    """ファイル名から YYYY-MM-DD を抽出"""
    m = re.match(r"(\d{4}-\d{2}-\d{2})", filename)
    return m.group(1) if m else None


def _extract_report_url(content: str) -> str | None:
    """MDからsurge.shのURLを抽出"""
    m = re.search(r"https://1on1-[\w\-]+\.surge\.sh", content)
    return m.group(0) if m else None


def _extract_summary(content: str) -> str:
    """## 📋 ミーティング概要 セクションを抽出"""
    m = re.search(r"##\s*📋\s*ミーティング概要\s*\n(.*?)(?=\n##|\Z)", content, re.DOTALL)
    if m:
        text = m.group(1).strip()
        # 先頭の空行・マークダウン記号を除去
        text = re.sub(r"\n{2,}", " ", text).strip()
        return text[:500]
    return ""


def _extract_actions(content: str) -> list[str]:
    """## ✅ アクションアイテム セクションから箇条書きを抽出"""
    m = re.search(r"##\s*✅\s*アクションアイテム\s*\n(.*?)(?=\n##|\Z)", content, re.DOTALL)
    if not m:
        return []
    actions = []
    for line in m.group(1).splitlines():
        line = line.strip()
        # - [ ] または - [x] のチェックボックス行
        item_m = re.match(r"-\s*\[[ xX]\]\s*(.+)", line)
        if item_m:
            text = item_m.group(1).strip()
            # 「（コンサルタント）」のような自社タスクを除外
            if "コンサルタント" not in text and "担当者" not in text:
                # 「（エリカさん）」のような括弧注釈を削除
                text = re.sub(r"[（(][^）)]*[）)]", "", text).strip()
                if text:
                    actions.append(text)
    return actions


def _load_processed() -> set[str]:
    if PROCESSED_FILE.exists():
        return set(json.loads(PROCESSED_FILE.read_text()))
    return set()


def _save_processed(processed: set[str]) -> None:
    PROCESSED_FILE.write_text(json.dumps(sorted(processed), ensure_ascii=False, indent=2))


async def sync():
    database_url = _load_database_url()
    conn = await asyncpg.connect(database_url)
    processed = _load_processed()
    new_count = 0
    skip_count = 0
    unknown_count = 0

    md_files = sorted(KNOWLEDGE_DIR.glob("*.md"))

    for md_path in md_files:
        filename = md_path.name

        if filename in processed:
            continue

        if _should_skip(filename):
            processed.add(filename)
            skip_count += 1
            continue

        date_str = _extract_date(filename)
        if not date_str:
            print(f"[SKIP] 日付不明: {filename}")
            processed.add(filename)
            continue

        content = md_path.read_text(encoding="utf-8")
        client_id = _match_client_id(filename, content)

        if client_id is None:
            print(f"[UNKNOWN] クライアント不明: {filename}")
            unknown_count += 1
            # 不明ファイルは次回も再チェックするため processed に追加しない
            continue

        # DB に既存チェック
        from datetime import date as date_type
        date_obj = date_type.fromisoformat(date_str)
        existing = await conn.fetchval(
            "SELECT id FROM sessions WHERE client_id=$1 AND date=$2",
            client_id, date_obj
        )
        if existing:
            processed.add(filename)
            continue

        report_url = _extract_report_url(content)
        summary = _extract_summary(content)
        actions = _extract_actions(content)

        session_id = await conn.fetchval(
            """INSERT INTO sessions (client_id, date, summary, report_url)
               VALUES ($1, $2, $3, $4) RETURNING id""",
            client_id, date_obj, summary, report_url
        )

        for action in actions:
            await conn.execute(
                "INSERT INTO action_items (session_id, content, status) VALUES ($1, $2, 'pending')",
                session_id, action
            )

        print(f"[OK] client_id={client_id} date={date_str} actions={len(actions)}: {filename}")
        processed.add(filename)
        new_count += 1

    await conn.close()
    _save_processed(processed)
    print(f"\n完了: 追加={new_count} スキップ={skip_count} 不明={unknown_count}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(sync())
