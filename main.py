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
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
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
    str(Path.home() / "Downloads" / "claude作業フォルダ" / "ナレッジ" / "Zoomセッション記録")
)


# ─── 重複処理防止（録画あり/なし の二重処理を防ぐ） ───────────
_processed_meetings: dict = {}  # meeting_id → datetime

def _try_claim_meeting(meeting_id: str) -> bool:
    """このミーティングの処理権を先着1件だけ取得する。
    recording.completed と meeting.ended が両方来ても1回だけ処理される。"""
    if meeting_id in _processed_meetings:
        logger.info(f"スキップ（処理済み）: {meeting_id}")
        return False
    _processed_meetings[meeting_id] = datetime.utcnow()
    return True


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

    if event == "meeting.ended":
        # 録画なしの場合の自動ナレッジ化（8分後に判定）
        background_tasks.add_task(process_meeting_ended_deferred, data)
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
        payload      = data["payload"]["object"]
        meeting_id   = str(payload.get("id", ""))
        topic        = payload.get("topic", "無題ミーティング")

        # 重複処理防止（meeting.ended側が先に処理した場合はスキップ）
        if meeting_id and not _try_claim_meeting(meeting_id):
            return
        host         = payload.get("host_email", "不明")
        duration     = payload.get("duration", 0)
        start_at     = payload.get("start_time", "")
        files        = payload.get("recording_files", [])
        # 参加者名: topicに「○○さん」「○○ with ○○」などが含まれることが多い
        # Zoom APIではhost以外の参加者はWebhookに含まれないためtopicから取得
        participant_name = _extract_participant_name(topic)

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
            surge_url = await deploy_to_surge(html, topic, date_str, participant_name)

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


async def process_meeting_ended_deferred(data: dict):
    """meeting.ended イベント受信 → 8分待って録画なしと判断したらナレッジ化"""
    payload    = data["payload"]["object"]
    meeting_id = str(payload.get("id", ""))
    topic      = payload.get("topic", "無題ミーティング")
    duration   = payload.get("duration", 0)
    start_time = payload.get("start_time", "")
    date_str   = start_time[:10] if start_time else datetime.now().strftime("%Y-%m-%d")

    logger.info(f"meeting.ended 受信: {topic} ({meeting_id}) — 8分後に録画有無を確認します")

    # 8分待つ（録画がある場合はこの間に recording.completed → _try_claim_meeting が先に処理）
    await asyncio.sleep(480)

    # 既に processing.completed 側で処理済みならスキップ
    if not _try_claim_meeting(meeting_id):
        logger.info(f"録画あり → 既に処理済みのためスキップ: {topic}")
        return

    logger.info(f"録画なしと判断 → トランスクリプト取得開始: {topic}")
    token = await _get_zoom_access_token()

    # ① Zoom API でトランスクリプト（VTTファイル）を試みる
    transcript = await _fetch_zoom_transcript(meeting_id, token)

    # ② なければ AI Companion サマリーを試みる
    if not transcript:
        transcript = await _fetch_ai_companion_summary(meeting_id, token)

    if not transcript:
        logger.warning(f"トランスクリプト取得できず（録画もなし）: {topic} — スキップ")
        return

    logger.info(f"トランスクリプト取得成功 ({len(transcript)}文字) → パイプライン開始")
    await process_manual_input(topic, transcript, date_str, duration)


async def _fetch_zoom_transcript(meeting_id: str, token: str) -> str:
    """Zoom録画APIからトランスクリプト（VTT）を取得してプレーンテキストに変換"""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(
                f"https://api.zoom.us/v2/meetings/{meeting_id}/recordings",
                headers={"Authorization": f"Bearer {token}"},
            )
            if not r.is_success:
                logger.info(f"録画API: {r.status_code} — トランスクリプトなし")
                return ""
            files = r.json().get("recording_files", [])
            for f in files:
                if f.get("file_type") == "TRANSCRIPT" and f.get("status") == "completed":
                    vtt_url = f["download_url"]
                    vr = await client.get(
                        vtt_url,
                        headers={"Authorization": f"Bearer {token}"},
                        follow_redirects=True,
                    )
                    if vr.is_success:
                        text = parse_vtt(vr.text)
                        logger.info(f"ZoomトランスクリプトVTT取得成功: {len(text)}文字")
                        return text
    except Exception as e:
        logger.warning(f"_fetch_zoom_transcript エラー: {e}")
    return ""


async def _fetch_ai_companion_summary(meeting_id: str, token: str) -> str:
    """Zoom AI Companion のミーティングサマリーを取得"""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(
                f"https://api.zoom.us/v2/meetings/{meeting_id}/meeting_summary",
                headers={"Authorization": f"Bearer {token}"},
            )
            if not r.is_success:
                logger.info(f"AI Companion API: {r.status_code} — サマリーなし（未対応プランの可能性）")
                return ""
            d = r.json()
            parts = []
            if d.get("summary_overview"):
                parts.append(d["summary_overview"])
            for item in d.get("summary_details", []):
                label   = item.get("summary_title", "")
                summary = item.get("summary", "")
                if label and summary:
                    parts.append(f"【{label}】\n{summary}")
                elif summary:
                    parts.append(summary)
            result = "\n\n".join(parts)
            if result:
                logger.info(f"AI Companion サマリー取得成功: {len(result)}文字")
            return result
    except Exception as e:
        logger.warning(f"_fetch_ai_companion_summary エラー: {e}")
    return ""


def _extract_participant_name(topic: str) -> str:
    """Zoomのミーティングタイトルから参加者名を抽出してローマ字化
    例: '1on1 牧野はるか' → 'makino-haruka'
        '矢橋×田中 MTG' → 'tanaka'（ホスト以外）
        'Haruka Makino 面談' → 'haruka-makino'
    ASCII名はそのまま使用、日本語は文字コードベースのスラッグに変換
    """
    import re
    # ASCII部分だけ抽出してスラッグ化
    ascii_parts = re.findall(r'[a-zA-Z]+', topic)
    if ascii_parts:
        slug = '-'.join(ascii_parts).lower()
        # 不要な一般単語を除去
        stop = {'mtg', 'meeting', 'on', 'with', 'and', 'the', 'zoom', 'call', 'session'}
        slug_parts = [p for p in slug.split('-') if p not in stop and len(p) > 1]
        if slug_parts:
            return '-'.join(slug_parts)[:30]
    # 日本語の場合: 文字をそのまま使う（deploy_to_surge側でASCII変換）
    return topic


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
        if not r.is_success:
            logger.error(f"Notionエラー詳細: {r.text}")
        r.raise_for_status()
        page_id  = r.json()["id"]
        page_url = r.json()["url"]
        logger.info(f"Notion保存完了: {page_url}")
        return page_url


def _safe_text(text: str) -> list:
    """Notionの2000文字制限に対応してrich_textを分割"""
    chunks = [text[i:i+1999] for i in range(0, len(text), 1999)]
    return [{"text": {"content": c}} for c in chunks] if chunks else [{"text": {"content": ""}}]


def _text_to_notion_blocks(text: str) -> list:
    """テキストをNotionブロックに変換（見出し・箇条書き・2000文字制限対応）"""
    blocks = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        if line.startswith("## "):
            blocks.append({
                "object": "block", "type": "heading_2",
                "heading_2": {"rich_text": _safe_text(line[3:])}
            })
        elif line.startswith("# "):
            blocks.append({
                "object": "block", "type": "heading_1",
                "heading_1": {"rich_text": _safe_text(line[2:])}
            })
        elif line.startswith("- ") or line.startswith("• "):
            blocks.append({
                "object": "block", "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": _safe_text(line[2:])}
            })
        elif line.startswith("- [ ] ") or line.startswith("- [x] "):
            checked = line.startswith("- [x]")
            blocks.append({
                "object": "block", "type": "to_do",
                "to_do": {
                    "rich_text": _safe_text(line[6:]),
                    "checked": checked
                }
            })
        else:
            blocks.append({
                "object": "block", "type": "paragraph",
                "paragraph": {"rich_text": _safe_text(line)}
            })
    return blocks


# ─── HTMLレポート生成プロンプト ────────────────────────────────
HTML_REPORT_PROMPT = """
あなたは「タイシ」です。USCPA（米国公認会計士）試験の指導者として、受講生に直接送付できる1on1レポートHTMLを生成してください。

# 重要：言葉遣いのルール
- 受講生（クライアント）への丁寧な敬語で書く（「〜されています」「〜いただけます」など）
- 上から目線・命令形・断定調は使わない
- 励ましと感謝の気持ちを込める
- 箇条書きも「・〜していただく」「・〜されることをおすすめします」のような丁寧な表現で

# ミーティング情報
タイトル: {topic}
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
<title>1on1レポート｜{topic}｜{date}</title>
<style>
  body {{ font-family: 'Hiragino Sans', 'Yu Gothic', sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #333; line-height: 1.7; }}
  .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 24px; }}
  .header h1 {{ margin: 0 0 6px; font-size: 1.5em; }}
  .header .meta {{ margin: 0; opacity: 0.9; font-size: 0.9em; }}
  .card {{ background: white; border-radius: 12px; padding: 24px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .card h2 {{ margin: 0 0 16px; font-size: 1.05em; color: #5a67d8; border-bottom: 2px solid #e8eaf6; padding-bottom: 8px; }}
  ul {{ margin: 0; padding-left: 20px; }}
  li {{ margin-bottom: 10px; }}
  .action-item {{ display: flex; align-items: flex-start; gap: 12px; padding: 12px; border-radius: 8px; margin-bottom: 10px; }}
  .action-high {{ background: #fff5f5; border-left: 4px solid #fc8181; }}
  .action-mid  {{ background: #fffaf0; border-left: 4px solid #f6ad55; }}
  .badge {{ font-size: 0.72em; padding: 2px 8px; border-radius: 12px; font-weight: bold; white-space: nowrap; margin-top: 2px; }}
  .badge-high {{ background: #fed7d7; color: #c53030; }}
  .badge-mid  {{ background: #feebc8; color: #c05621; }}
  .message {{ background: linear-gradient(135deg, #f0fff4, #e6fffa); border-left: 4px solid #48bb78; padding: 20px 24px; border-radius: 8px; line-height: 1.9; }}
  .footer {{ text-align: center; color: #aaa; font-size: 0.78em; margin-top: 28px; padding-top: 16px; border-top: 1px solid #eee; }}
</style>
</head>
<body>

<div class="header">
  <h1>📋 1on1セッションレポート</h1>
  <p class="meta">{topic}　｜　{date}</p>
</div>

<!-- 書き起こしをもとに以下のセクションをすべて生成してください -->
<!-- ① 今回のセッションについて（概要・目的、2〜3文） -->
<!-- ② 本日ご確認いただいた内容（主な議論、箇条書き、丁寧語） -->
<!-- ③ 決定事項・方針（合意した内容） -->
<!-- ④ 次回までのアクション（action-high/action-midクラスで優先度を視覚化、丁寧語） -->
<!-- ⑤ タイシからのメッセージ（messageクラス、温かく励ます文章、敬語、「タイシより」で締める） -->

<div class="footer">
  <p>本レポートはセッション内容をもとに作成しております。ご不明な点がございましたらお気軽にご連絡ください。</p>
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
async def deploy_to_surge(html: str, topic: str, date: str, participant_name: str = "") -> str:
    """HTMLをsurge.shに自動デプロイしてURLを返す"""
    import subprocess, re

    # URLスラッグ: 参加者名があればそれを使い、なければtopicのASCII部分を使用
    # 例: 1on1-2026-06-05-haruka-makino
    name_part = participant_name or topic
    slug = re.sub(r'[^a-zA-Z0-9]', '-', name_part)
    slug = re.sub(r'-+', '-', slug)[:30].strip('-').lower()
    if not slug:
        slug = "meeting"
    # 日付はYYYY-MM-DD形式
    domain = f"1on1-{date}-{slug}.surge.sh"

    # 一時ディレクトリにindex.htmlを作成
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "index.html"
        html_path.write_text(html, encoding="utf-8")

        # surge deploy（パスを明示的に指定）
        import shutil
        surge_bin = shutil.which("surge") or "/usr/local/bin/surge"
        env = {**os.environ, "SURGE_TOKEN": SURGE_TOKEN}
        result = subprocess.run(
            [surge_bin, tmpdir, domain, "--token", SURGE_TOKEN],
            capture_output=True, text=True, env=env, timeout=60
        )
        logger.info(f"Surge returncode: {result.returncode}")
        if result.returncode != 0:
            logger.error(f"Surgeデプロイ失敗: {result.stderr}")
            return ""
        url = f"https://{domain}"
        logger.info(f"Surgeデプロイ完了: {url}")
        return url


# ─── 転送用メッセージ生成 ──────────────────────────────────────
def _build_forwarding_message(topic: str, surge_url: str) -> str:
    """受講生にそのまま転送できるテキストメッセージを生成"""
    # topicから受講生名を抽出（日本語・英語どちらも対応）
    import re
    # "1on1 牧野はるか" / "1on1 Haruka Makino" / "面談 田中太郎" などに対応
    name = re.sub(r'(?i)(1on1|面談|ミーティング|mtg|meeting|session|zoom|\s*[-×x&]\s*)', '', topic).strip()
    # 自分のメールアドレスやアカウント名が残ったら除去
    name = re.sub(r'haitokutaishi|taishi|背徳タイシ', '', name, flags=re.IGNORECASE).strip()
    name = name if name else "受講生"

    return (
        f"{name}さん\n"
        f"\n"
        f"本日もセッションにご参加いただきありがとうございました。\n"
        f"本日のセッションレポートをお送りします。\n"
        f"ぜひご確認ください。\n"
        f"\n"
        f"▼本日のレポート\n"
        f"{surge_url}\n"
        f"\n"
        f"ご不明な点があればいつでもご連絡ください。\n"
        f"引き続きよろしくお願いいたします。\n"
        f"\n"
        f"タイシ"
    )


# ─── Discord通知 ───────────────────────────────────────────────
async def send_to_discord(topic: str, date: str, surge_url: str, notion_url: str):
    """DiscordのWebhookにミーティングレポートを送信"""
    if not DISCORD_WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URL未設定、スキップ")
        return

    # ① 内部確認用embed
    embed = {
        "title": f"📋 1on1レポート完成 — {topic}",
        "description": f"**{date}** のセッションレポートが自動生成されました。\n下のメッセージを長押しコピーして転送してください。",
        "color": 0x5a67d8,
        "fields": [
            {"name": "🌐 クライアント向けレポート", "value": surge_url, "inline": False},
            {"name": "📝 Notionナレッジ", "value": notion_url, "inline": False},
        ],
        "footer": {"text": "Zoom Knowledge Auto-Generator"},
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # ② 転送用テキスト（そのままコピーして送付できる）
    forwarding_text = _build_forwarding_message(topic, surge_url)

    async with httpx.AsyncClient(timeout=30) as client:
        # 1通目：内部確認embed＋「次のメッセージを転送して」の案内
        r = await client.post(
            DISCORD_WEBHOOK_URL,
            json={
                "embeds": [embed],
                "content": "📤 **↓ 次のメッセージを長押しコピーして転送してください**",
            },
        )
        r.raise_for_status()

        # 2通目：転送用テキストのみ（余計な文字一切なし）
        r2 = await client.post(
            DISCORD_WEBHOOK_URL,
            json={"content": forwarding_text},
        )
        r2.raise_for_status()
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


# ─── ZoomのVTT文字起こしパーサー ──────────────────────────────
def parse_vtt(content: str) -> str:
    """ZoomのVTT形式文字起こしをプレーンテキストに変換"""
    import re
    result = []
    for line in content.split("\n"):
        line = line.strip()
        # WEBVTT ヘッダー・空行・シーケンス番号・タイムコード行をスキップ
        if not line or line == "WEBVTT" or line.isdigit():
            continue
        if re.match(r"^\d{2}:\d{2}:\d{2}", line):  # タイムコード
            continue
        # 話者タグ除去: <v タイシ>テキスト</v>
        line = re.sub(r"<v [^>]+>", "", line)
        line = re.sub(r"</v>", "", line)
        # その他HTMLタグ除去
        line = re.sub(r"<[^>]+>", "", line)
        if line:
            result.append(line)
    return "\n".join(result)


# ─── 手動入力フォーム（録画なし対応） ─────────────────────────
MANUAL_FORM_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Zoomナレッジ 手動入力</title>
<style>
  body {{ font-family: 'Hiragino Sans', 'Yu Gothic', sans-serif; max-width: 700px; margin: 40px auto; padding: 20px; background: #f8f9fa; color: #333; }}
  h1 {{ color: #5a67d8; margin-bottom: 6px; }}
  p.sub {{ color: #666; margin-top: 0; font-size: 0.9em; }}
  label {{ display: block; margin-top: 20px; font-weight: bold; color: #444; font-size: 0.9em; }}
  input[type=text], input[type=date], input[type=number], textarea {{
    width: 100%; padding: 10px 12px; border: 1px solid #d1d5db; border-radius: 8px;
    margin-top: 6px; font-size: 14px; box-sizing: border-box; background: white;
  }}
  textarea {{ height: 320px; font-family: monospace; font-size: 13px; resize: vertical; }}
  .hint {{ font-size: 11px; color: #9ca3af; margin-top: 5px; }}
  .row {{ display: flex; gap: 16px; }}
  .row > div {{ flex: 1; }}
  button {{
    margin-top: 24px; padding: 14px 36px; background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border: none; border-radius: 10px; font-size: 15px; cursor: pointer; width: 100%;
  }}
  button:hover {{ opacity: 0.9; }}
  button:disabled {{ opacity: 0.6; cursor: not-allowed; }}
  #result {{ margin-top: 20px; padding: 16px 20px; border-radius: 10px; display: none; font-size: 14px; }}
  .success {{ background: #f0fff4; border: 1px solid #9ae6b4; color: #276749; }}
  .error   {{ background: #fff5f5; border: 1px solid #fc8181; color: #c53030; }}
</style>
</head>
<body>
<h1>📋 Zoomナレッジ 手動入力</h1>
<p class="sub">録画なしのミーティングでも、Zoomの文字起こしを貼り付けるだけでナレッジ化できます</p>

<form id="form">
  <label>ミーティングタイトル</label>
  <input type="text" name="topic" placeholder="例: 1on1 Haruka Makino" required>
  <div class="hint">命名規則: 「1on1 ＋ 受講生名（英語）」→ surge URLと転送メッセージの宛名に使われます</div>

  <div class="row">
    <div>
      <label>日付</label>
      <input type="date" name="date" id="date-input">
    </div>
    <div>
      <label>時間（分）</label>
      <input type="number" name="duration" value="60" min="1" max="480">
    </div>
  </div>

  <label>Zoomの文字起こし（VTT形式・プレーンテキスト どちらでも可）</label>
  <textarea name="transcript" placeholder="ここにZoomの文字起こしを貼り付けてください&#10;&#10;Zoom → 録画 → 文字起こしを表示 → 全選択 → コピー" required></textarea>
  <div class="hint">VTT形式（タイムコード付き）でも通常テキストでも自動判定します</div>

  <button type="submit">ナレッジ化して Discord に通知</button>
</form>
<div id="result"></div>

<script>
  document.getElementById('date-input').value = new Date().toISOString().slice(0, 10);

  document.getElementById('form').addEventListener('submit', async (e) => {{
    e.preventDefault();
    const btn = e.target.querySelector('button');
    btn.textContent = '⏳ 処理中... しばらくお待ちください';
    btn.disabled = true;
    const res = document.getElementById('result');
    res.style.display = 'none';

    try {{
      const r = await fetch('/manual', {{ method: 'POST', body: new FormData(e.target) }});
      const json = await r.json();
      res.className = 'success';
      res.innerHTML = '<strong>✅ 受付完了！</strong><br>' +
        json.topic + '（' + json.date + '）のナレッジ化を開始しました。<br>2〜3分後にDiscordをご確認ください。';
    }} catch (err) {{
      res.className = 'error';
      res.innerHTML = '<strong>❌ エラー</strong><br>' + err.message;
    }}
    res.style.display = 'block';
    btn.textContent = 'ナレッジ化して Discord に通知';
    btn.disabled = false;
  }});
</script>
</body>
</html>"""


@app.get("/manual", response_class=HTMLResponse)
async def manual_form():
    """録画なし手動入力フォームを表示"""
    return MANUAL_FORM_HTML


@app.post("/manual")
async def manual_input(
    background_tasks: BackgroundTasks,
    topic: str = Form(...),
    transcript: str = Form(...),
    date: str = Form(None),
    duration: int = Form(60),
):
    """Zoomの文字起こしを直接受け取ってナレッジ化（録画なし対応）"""
    date_str = date or datetime.now().strftime("%Y-%m-%d")

    # VTT形式かどうか自動判定して変換
    if "WEBVTT" in transcript or " --> " in transcript:
        transcript = parse_vtt(transcript)
        logger.info("VTT形式を検出 → プレーンテキストに変換しました")

    if not transcript.strip():
        raise HTTPException(status_code=400, detail="文字起こしテキストが空です")

    background_tasks.add_task(process_manual_input, topic, transcript, date_str, duration)
    logger.info(f"手動入力受付: {topic} ({date_str})")
    return JSONResponse({"message": "受付しました", "topic": topic, "date": date_str})


async def process_manual_input(topic: str, transcript: str, date_str: str, duration: int):
    """Whisperをスキップしてナレッジ化パイプラインを実行"""
    try:
        participant_name = _extract_participant_name(topic)
        logger.info(f"手動処理開始: {topic} ({duration}分) [{date_str}]")

        # Step1: Claudeでナレッジ化
        knowledge = await generate_knowledge(transcript, topic, duration)

        # Step2: Notionに保存
        notion_url = await save_to_notion(
            topic=topic, host="(手動入力)",
            start_at=date_str, duration=duration,
            transcript=transcript, knowledge=knowledge
        )

        # Step3: HTMLレポート生成
        html = await generate_html_report(
            transcript=transcript, topic=topic,
            host="(手動入力)", date=date_str, duration=duration
        )

        # Step4: surge.shにデプロイ
        surge_url = ""
        if SURGE_TOKEN and html:
            surge_url = await deploy_to_surge(html, topic, date_str, participant_name)

        # Step5: Discordに通知
        await send_to_discord(
            topic=topic, date=date_str,
            surge_url=surge_url or "(デプロイ未設定)",
            notion_url=notion_url
        )

        # Step6: ローカル保存
        save_to_local_knowledge(
            topic=topic, date=date_str, host="(手動入力)", duration=duration,
            transcript=transcript, knowledge=knowledge, surge_url=surge_url
        )

        logger.info(f"手動処理完了: surge={surge_url} notion={notion_url}")

    except Exception as e:
        logger.error(f"手動処理エラー: {e}", exc_info=True)


# ─── ヘルスチェック ────────────────────────────────────────────
@app.get("/")
async def health():
    return {"status": "ok", "service": "zoom-knowledge-auto-generator"}
