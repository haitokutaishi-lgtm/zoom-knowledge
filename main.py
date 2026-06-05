"""
Zoom自動ナレッジ化システム
Zoom録画 → Whisper文字起こし → Claude構造化 → Notion保存
"""

import os
import json
import hashlib
import hmac
import tempfile
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Zoom Knowledge Auto-Generator")

# ─── 環境変数 ────────────────────────────────────────────────
ZOOM_WEBHOOK_SECRET   = os.getenv("ZOOM_WEBHOOK_SECRET", "")
ZOOM_ACCOUNT_ID       = os.getenv("ZOOM_ACCOUNT_ID", "")
ZOOM_CLIENT_ID        = os.getenv("ZOOM_CLIENT_ID", "")
ZOOM_CLIENT_SECRET    = os.getenv("ZOOM_CLIENT_SECRET", "")
OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY     = os.getenv("ANTHROPIC_API_KEY", "")
NOTION_API_KEY        = os.getenv("NOTION_API_KEY", "")
NOTION_DATABASE_ID    = os.getenv("NOTION_DATABASE_ID", "")
DISCORD_WEBHOOK_URL   = os.getenv("DISCORD_WEBHOOK_URL", "")
SURGE_TOKEN           = os.getenv("SURGE_TOKEN", "")
LOCAL_KNOWLEDGE_DIR   = os.getenv(
    "LOCAL_KNOWLEDGE_DIR",
    str(Path.home() / "Downloads" / "claude作業フォルダ" / "ナレッジ")
)


# ─── Zoom Webhook受信 ─────────────────────────────────────────
@app.post("/webhook/zoom")
async def zoom_webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    headers = request.headers

    # Zoom URL検証リクエスト（最初の1回だけ）
    data = json.loads(body)
    if data.get("event") == "endpoint.url_validation":
        plain_token = data["payload"]["plainToken"]
        encrypted = hmac.new(
            ZOOM_WEBHOOK_SECRET.encode(),
            plain_token.encode(),
            hashlib.sha256
        ).hexdigest()
        return JSONResponse({"plainToken": plain_token, "encryptedToken": encrypted})

    # 署名検証
    if not _verify_zoom_signature(headers, body):
        raise HTTPException(status_code=401, detail="Invalid signature")

    event = data.get("event")
    logger.info(f"Zoom event: {event}")

    if event == "recording.completed":
        background_tasks.add_task(process_recording, data)
        return JSONResponse({"message": "accepted"})

    return JSONResponse({"message": "ignored"})


def _verify_zoom_signature(headers, body: bytes) -> bool:
    """Zoom Webhook署名を検証"""
    signature = headers.get("x-zm-signature", "")
    timestamp  = headers.get("x-zm-request-timestamp", "")
    message    = f"v0:{timestamp}:{body.decode()}"
    expected   = "v0=" + hmac.new(
        ZOOM_WEBHOOK_SECRET.encode(), message.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)


# ─── メイン処理パイプライン ────────────────────────────────────
async def process_recording(data: dict):
    """録画完了イベントを受け取りナレッジ化する"""
    try:
        payload  = data["payload"]["object"]
        topic    = payload.get("topic", "無題ミーティング")
        host     = payload.get("host_email", "不明")
        duration = payload.get("duration", 0)
        start_at = payload.get("start_time", "")
        files    = payload.get("recording_files", [])

        logger.info(f"処理開始: {topic} ({duration}分)")

        # M4A or MP4音声ファイルを選択（M4AはAudio Only）
        audio_file = _select_audio_file(files)
        if not audio_file:
            logger.warning("音声ファイルが見つかりません")
            return

        # Step1: 録画ダウンロード
        audio_url = audio_file["download_url"]
        token = await _get_zoom_access_token()
        audio_path = await download_recording(audio_url, token)

        # Step2: Whisperで文字起こし
        transcript = await transcribe_audio(audio_path)

        # Step3: Claudeでナレッジ化
        knowledge = await generate_knowledge(transcript, topic, duration)

        # Step4: Notionに保存
        notion_url = await save_to_notion(
            topic=topic,
            host=host,
            start_at=start_at,
            duration=duration,
            transcript=transcript,
            knowledge=knowledge
        )

        # Step5: HTMLレポート生成
        date_str = start_at[:10] if start_at else datetime.now().strftime("%Y-%m-%d")
        html = await generate_html_report(
            transcript=transcript,
            topic=topic,
            host=host,
            date=date_str,
            duration=duration
        )

        # Step6: surge.shにデプロイ
        surge_url = ""
        if SURGE_TOKEN and html:
            surge_url = await deploy_to_surge(html, topic, date_str)

        # Step7: Discordに通知
        await send_to_discord(
            topic=topic,
            date=date_str,
            surge_url=surge_url or "(デプロイ未設定)",
            notion_url=notion_url
        )

        # Step8: ローカルナレッジフォルダに保存（ローカル実行時のみ有効）
        save_to_local_knowledge(
            topic=topic, date=date_str, host=host, duration=duration,
            transcript=transcript, knowledge=knowledge, surge_url=surge_url
        )

        logger.info(f"全処理完了: surge={surge_url} notion={notion_url}")

    except Exception as e:
        logger.error(f"処理エラー: {e}", exc_info=True)
    finally:
        # 一時ファイル削除
        if "audio_path" in locals():
            Path(audio_path).unlink(missing_ok=True)


def _select_audio_file(files: list) -> Optional[dict]:
    """音声ファイルを優先して選択"""
    for ft in ["M4A", "TRANSCRIPT", "MP4"]:
        for f in files:
            if f.get("file_type") == ft and f.get("status") == "completed":
                return f
    return None


# ─── Step1: 録画ダウンロード ───────────────────────────────────
async def _get_zoom_access_token() -> str:
    """Server-to-Server OAuthでアクセストークン取得"""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://zoom.us/oauth/token",
            params={"grant_type": "account_credentials", "account_id": ZOOM_ACCOUNT_ID},
            auth=(ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET),
        )
        r.raise_for_status()
        return r.json()["access_token"]


async def download_recording(url: str, token: str) -> str:
    """Zoom録画をダウンロードして一時ファイルに保存"""
    logger.info("録画ダウンロード中...")
    async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
        r = await client.get(url, headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        suffix = ".m4a" if "m4a" in r.headers.get("content-type", "") else ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(r.content)
        tmp.close()
        logger.info(f"ダウンロード完了: {tmp.name} ({len(r.content)//1024}KB)")
        return tmp.name


# ─── Step2: Whisper文字起こし ──────────────────────────────────
async def transcribe_audio(audio_path: str) -> str:
    """OpenAI Whisper APIで文字起こし"""
    logger.info("文字起こし中...")
    async with httpx.AsyncClient(timeout=600) as client:
        with open(audio_path, "rb") as f:
            r = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                files={"file": (Path(audio_path).name, f, "audio/mpeg")},
                data={"model": "whisper-1", "language": "ja", "response_format": "text"},
            )
            r.raise_for_status()
            transcript = r.text
            logger.info(f"文字起こし完了: {len(transcript)}文字")
            return transcript


# ─── Step3: Claudeでナレッジ化 ────────────────────────────────
KNOWLEDGE_PROMPT = """
あなたはビジネスコンサルタントです。
以下のミーティング書き起こしを読み、商品・サービス開発に活用できるナレッジとして構造化してください。

# ミーティング情報
タイトル: {topic}
時間: {duration}分

# 書き起こし
{transcript}

---

以下の形式で日本語でまとめてください：

## 📋 ミーティング概要
（誰と何を話したか、3〜5行）

## 😤 顧客の課題・ペイン
（顧客が困っていること、不満点を箇条書き）

## 💡 拾えるニーズ・アイデア
（提供できる価値、改善のヒント、新商品アイデアを箇条書き）

## ✅ アクションアイテム
（次にやるべきことを [ ] チェックボックス形式で）

## 🔑 キーワード
（重要なキーワードを5〜10個、カンマ区切り）
"""


async def generate_knowledge(transcript: str, topic: str, duration: int) -> str:
    """Claude APIでナレッジ構造化"""
    logger.info("ナレッジ生成中...")
    prompt = KNOWLEDGE_PROMPT.format(
        topic=topic,
        duration=duration,
        transcript=transcript[:8000]  # 長すぎる場合は先頭8000文字
    )
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-opus-4-5",
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        r.raise_for_status()
        knowledge = r.json()["content"][0]["text"]
        logger.info("ナレッジ生成完了")
        return knowledge


# ─── Step4: Notionに保存 ───────────────────────────────────────
async def save_to_notion(
    topic: str, host: str, start_at: str,
    duration: int, transcript: str, knowledge: str
) -> str:
    """Notion DBにページを作成"""
    logger.info("Notionに保存中...")

    date_str = start_at[:10] if start_at else datetime.now().strftime("%Y-%m-%d")

    # ページプロパティ
    properties = {
        "タイトル": {"title": [{"text": {"content": f"{date_str} {topic}"}}]},
        "日付":    {"date": {"start": date_str}},
        "ホスト":  {"rich_text": [{"text": {"content": host}}]},
        "時間(分)": {"number": duration},
        "ステータス": {"select": {"name": "完了"}},
    }

    # ページ本文（ナレッジ + 書き起こし折りたたみ）
    content_blocks = _text_to_notion_blocks(knowledge)
    content_blocks += [
        {"object": "block", "type": "divider", "divider": {}},
        {
            "object": "block", "type": "toggle",
            "toggle": {
                "rich_text": [{"text": {"content": "📝 全文書き起こし"}}],
                "children": _text_to_notion_blocks(transcript[:5000])
            }
        }
    ]

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.notion.com/v1/pages",
            headers={
                "Authorization": f"Bearer {NOTION_API_KEY}",
                "Notion-Version": "2022-06-28",
                "Content-Type": "application/json",
            },
            json={
                "parent": {"database_id": NOTION_DATABASE_ID},
                "properties": properties,
                "children": content_blocks[:100],  # Notionは100ブロック制限
            },
        )
        r.raise_for_status()
        page_id  = r.json()["id"]
        page_url = r.json()["url"]
        logger.info(f"Notion保存完了: {page_url}")
        return page_url


def _text_to_notion_blocks(text: str) -> list:
    """テキストをNotionブロックに変換（見出し・箇条書き対応）"""
    blocks = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        if line.startswith("## "):
            blocks.append({
                "object": "block", "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content": line[3:]}}]}
            })
        elif line.startswith("# "):
            blocks.append({
                "object": "block", "type": "heading_1",
                "heading_1": {"rich_text": [{"text": {"content": line[2:]}}]}
            })
        elif line.startswith("- ") or line.startswith("• "):
            blocks.append({
                "object": "block", "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"text": {"content": line[2:]}}]}
            })
        elif line.startswith("- [ ] ") or line.startswith("- [x] "):
            checked = line.startswith("- [x]")
            blocks.append({
                "object": "block", "type": "to_do",
                "to_do": {
                    "rich_text": [{"text": {"content": line[6:]}}],
                    "checked": checked
                }
            })
        else:
            blocks.append({
                "object": "block", "type": "paragraph",
                "paragraph": {"rich_text": [{"text": {"content": line}}]}
            })
    return blocks


# ─── HTMLレポート生成プロンプト ────────────────────────────────
HTML_REPORT_PROMPT = """
あなたはコーチングのプロフェッショナルです。
以下のミーティング書き起こしを読み、受講生向けの1on1レポートHTMLを生成してください。

# ミーティング情報
タイトル: {topic}
参加者: {host}
日時: {date}
時間: {duration}分

# 書き起こし
{transcript}

---

以下の完全なHTMLを出力してください（```htmlなどのマークダウン不要、HTMLのみ）：

<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>1on1レポート - {topic}</title>
<style>
  body {{ font-family: 'Hiragino Sans', 'Yu Gothic', sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #333; }}
  .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 24px; }}
  .header h1 {{ margin: 0 0 8px; font-size: 1.6em; }}
  .header p {{ margin: 0; opacity: 0.9; font-size: 0.95em; }}
  .card {{ background: white; border-radius: 12px; padding: 24px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .card h2 {{ margin: 0 0 16px; font-size: 1.1em; color: #5a67d8; border-bottom: 2px solid #e8eaf6; padding-bottom: 8px; }}
  ul {{ margin: 0; padding-left: 20px; }}
  li {{ margin-bottom: 8px; line-height: 1.6; }}
  .action-item {{ display: flex; align-items: flex-start; gap: 12px; padding: 10px; border-radius: 8px; margin-bottom: 8px; }}
  .action-high {{ background: #fff5f5; border-left: 4px solid #fc8181; }}
  .action-mid  {{ background: #fffaf0; border-left: 4px solid #f6ad55; }}
  .badge {{ font-size: 0.75em; padding: 2px 8px; border-radius: 12px; font-weight: bold; white-space: nowrap; }}
  .badge-high {{ background: #fed7d7; color: #c53030; }}
  .badge-mid  {{ background: #feebc8; color: #c05621; }}
  .message {{ background: linear-gradient(135deg, #f0fff4, #e6fffa); border-left: 4px solid #48bb78; padding: 20px; border-radius: 8px; line-height: 1.8; }}
  .footer {{ text-align: center; color: #999; font-size: 0.8em; margin-top: 24px; }}
</style>
</head>
<body>

<div class="header">
  <h1>📋 1on1レポート</h1>
  <p>{topic} ｜ {date}</p>
</div>

<!-- 以下に実際の内容を書き起こしから生成してください -->
<!-- 各セクション: 会議の目的、主な議論内容、決定事項、ネクストアクション（優先度高・中）、コーチメッセージ -->
<!-- ネクストアクションは action-high / action-mid クラスを使って優先度を視覚化 -->

<div class="footer">
  <p>このレポートはAIが自動生成しました ｜ {date}</p>
</div>
</body>
</html>
"""


async def generate_html_report(transcript: str, topic: str, host: str, date: str, duration: int) -> str:
    """ClaudeでHTMLレポートを生成"""
    logger.info("HTMLレポート生成中...")
    prompt = HTML_REPORT_PROMPT.format(
        topic=topic, host=host, date=date, duration=duration,
        transcript=transcript[:8000]
    )
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-opus-4-5",
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        r.raise_for_status()
        html = r.json()["content"][0]["text"].strip()
        # マークダウンコードブロックが含まれていたら除去
        if html.startswith("```"):
            html = "\n".join(html.split("\n")[1:])
        if html.endswith("```"):
            html = "\n".join(html.split("\n")[:-1])
        logger.info("HTMLレポート生成完了")
        return html


# ─── surge.shデプロイ ──────────────────────────────────────────
async def deploy_to_surge(html: str, topic: str, date: str) -> str:
    """HTMLをsurge.shに自動デプロイしてURLを返す"""
    import subprocess, re

    # URLスラッグ生成: 1on1-20260605-topic-name
    slug_topic = re.sub(r'[^a-zA-Z0-9぀-鿿]', '-', topic)[:30].strip('-')
    domain = f"1on1-{date}-{slug_topic}.surge.sh".lower().replace(' ', '-')

    # 一時ディレクトリにindex.htmlを作成
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "index.html"
        html_path.write_text(html, encoding="utf-8")

        # surge deploy
        env = {**os.environ, "SURGE_TOKEN": SURGE_TOKEN}
        result = subprocess.run(
            ["surge", tmpdir, domain, "--token", SURGE_TOKEN],
            capture_output=True, text=True, env=env, timeout=60
        )
        if result.returncode != 0:
            logger.error(f"Surgeデプロイ失敗: {result.stderr}")
            return ""
        url = f"https://{domain}"
        logger.info(f"Surgeデプロイ完了: {url}")
        return url


# ─── Discord通知 ───────────────────────────────────────────────
async def send_to_discord(topic: str, date: str, surge_url: str, notion_url: str):
    """DiscordのWebhookにミーティングレポートを送信"""
    if not DISCORD_WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URL未設定、スキップ")
        return

    embed = {
        "title": f"📋 1on1レポート完成 — {topic}",
        "description": f"**{date}** のミーティングレポートが自動生成されました。\n内容確認後、受講生へ送付してください。",
        "color": 0x5a67d8,
        "fields": [
            {"name": "🌐 クライアント向けレポート", "value": f"[レポートを開く]({surge_url})", "inline": False},
            {"name": "📝 Notionナレッジ", "value": f"[Notionを開く]({notion_url})", "inline": False},
        ],
        "footer": {"text": "Zoom Knowledge Auto-Generator"},
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            DISCORD_WEBHOOK_URL,
            json={"embeds": [embed]},
        )
        r.raise_for_status()
        logger.info("Discord通知送信完了")


# ─── ローカルナレッジフォルダ保存 ─────────────────────────────
def save_to_local_knowledge(
    topic: str, date: str, host: str, duration: int,
    transcript: str, knowledge: str, surge_url: str
) -> str:
    """ローカルの「ナレッジ」フォルダにMarkdownで保存"""
    knowledge_dir = Path(LOCAL_KNOWLEDGE_DIR)
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    # ファイル名: 2026-06-05_ミーティング名.md
    safe_topic = "".join(c for c in topic if c not in r'\/:*?"<>|').strip()[:40]
    filename = f"{date}_{safe_topic}.md"
    filepath = knowledge_dir / filename

    content = f"""# {topic}

**日時：** {date}　**ホスト：** {host}　**時間：** {duration}分
{f'**クライアント向けレポート：** {surge_url}' if surge_url else ''}

---

{knowledge}

---

## 📝 全文書き起こし

<details>
<summary>クリックで展開</summary>

{transcript}

</details>

---
*自動生成: {datetime.now().strftime("%Y-%m-%d %H:%M")}*
"""

    filepath.write_text(content, encoding="utf-8")
    logger.info(f"ローカル保存完了: {filepath}")
    return str(filepath)


# ─── ヘルスチェック ────────────────────────────────────────────
@app.get("/")
async def health():
    return {"status": "ok", "service": "zoom-knowledge-auto-generator"}
