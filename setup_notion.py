"""
NotionデータベースをAPIで自動作成するセットアップスクリプト
実行: python setup_notion.py
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

NOTION_API_KEY     = os.getenv("NOTION_API_KEY", "")
NOTION_PARENT_PAGE = os.getenv("NOTION_PARENT_PAGE_ID", "")  # 作成先ページID


def create_knowledge_database():
    """ナレッジDBをNotionに作成"""
    db_schema = {
        "parent": {"type": "page_id", "page_id": NOTION_PARENT_PAGE},
        "title": [{"text": {"content": "🎙️ Zoomミーティングナレッジ"}}],
        "properties": {
            "タイトル":  {"title": {}},
            "日付":     {"date": {}},
            "ホスト":   {"rich_text": {}},
            "時間(分)": {"number": {"format": "number"}},
            "ステータス": {
                "select": {
                    "options": [
                        {"name": "完了",     "color": "green"},
                        {"name": "処理中",   "color": "yellow"},
                        {"name": "要確認",   "color": "red"},
                    ]
                }
            },
            "カテゴリ": {
                "multi_select": {
                    "options": [
                        {"name": "顧客ヒアリング", "color": "blue"},
                        {"name": "チームMTG",     "color": "purple"},
                        {"name": "商品開発",       "color": "orange"},
                        {"name": "営業",           "color": "pink"},
                    ]
                }
            },
        }
    }

    r = httpx.post(
        "https://api.notion.com/v1/databases",
        headers={
            "Authorization": f"Bearer {NOTION_API_KEY}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        },
        json=db_schema,
    )
    r.raise_for_status()
    db = r.json()
    print(f"✅ Notionデータベース作成完了！")
    print(f"   Database ID: {db['id']}")
    print(f"   URL: {db['url']}")
    print(f"\n.envに追記してください:")
    print(f"NOTION_DATABASE_ID={db['id']}")
    return db["id"]


if __name__ == "__main__":
    if not NOTION_PARENT_PAGE:
        print("❌ .envに NOTION_PARENT_PAGE_ID を設定してください")
        print("   NotionページのURLの末尾32文字がIDです")
        exit(1)
    create_knowledge_database()
