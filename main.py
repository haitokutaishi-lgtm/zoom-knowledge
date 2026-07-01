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
CRON_SECRET           = os.getenv("CRON_SECRET", "")   # 外部cronサービス認証用
COACHING_DATABASE_URL = os.getenv("COACHING_DATABASE_URL", "")  # coaching-tool（Neon）への同期用
LOCAL_KNOWLEDGE_DIR   = os.getenv(
    "LOCAL_KNOWLEDGE_DIR",
    str(Path.home() / "Downloads" / "claude作業フォルダ" / "ナレッジ" / "Zoomセッション記録")
)

# ─── ローカルLLM（無償運用モード） ──────────────────────────────
USE_LOCAL_LLM      = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")

_whisper_model = None

def _get_whisper_model():
    """faster-whisperモデルを遅延ロード（初回のみ）"""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        logger.info(f"faster-whisperモデル読み込み中... ({WHISPER_MODEL_SIZE})")
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


async def _ollama_generate(prompt: str, max_tokens: int = 2048) -> str:
    """OllamaでローカルLLM推論"""
    async with httpx.AsyncClient(timeout=900) as client:
        r = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                # num_ctxを明示しないとデフォルト4096になり、長い書き起こし入力だけで
                # コンテキストを使い切って出力が薄くなるため拡大しておく
                "options": {"num_predict": max_tokens, "num_ctx": 16384},
            },
        )
        r.raise_for_status()
        return r.json()["response"].strip()


# ─── 処理対象外トピック ───────────────────────────────────────
SKIP_TOPIC_KEYWORDS = ["背徳タイシ"]

def _is_skip_topic(topic: str) -> bool:
    return any(kw in topic for kw in SKIP_TOPIC_KEYWORDS)


# ─── 重複処理防止（録画あり/なし の二重処理を防ぐ） ───────────
# local_runner.pyはlaunchdから15分おきに毎回新規プロセスとして起動されるため、
# メモリ上の辞書だけではプロセスを跨いだ重複防止にならない。
# ファイルに永続化し、surge.shの一時的なエラーで誤って再処理されることを防ぐ。
PROCESSED_LOG_PATH = Path(__file__).parent / "processed_meetings.json"

def _load_processed_meetings() -> dict:
    if PROCESSED_LOG_PATH.exists():
        try:
            return json.loads(PROCESSED_LOG_PATH.read_text())
        except Exception:
            return {}
    return {}

_processed_meetings: dict = _load_processed_meetings()  # meeting_id → datetime(ISO文字列)

def _try_claim_meeting(meeting_id: str) -> bool:
    """このミーティングの処理権を先着1件だけ取得する。
    recording.completed と meeting.ended が両方来ても1回だけ処理される。"""
    if meeting_id in _processed_meetings:
        logger.info(f"スキップ（処理済み）: {meeting_id}")
        return False
    _processed_meetings[meeting_id] = datetime.utcnow().isoformat()
    try:
        PROCESSED_LOG_PATH.write_text(
            json.dumps(_processed_meetings, ensure_ascii=False, indent=2)
        )
    except Exception as e:
        logger.warning(f"処理済み記録の保存に失敗: {e}")
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

        # 対象外トピックはスキップ
        if _is_skip_topic(topic):
            logger.info(f"スキップ（対象外トピック）: {topic}")
            return

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

        # Step9: coaching-tool（Neon DB）に同期（クライアント特定できた場合のみ）
        await sync_to_coaching_db(topic=topic, date=date_str, knowledge=knowledge, report_url=surge_url)

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
        'CAMP YAMAさん：1on1セッション' → 'yama'
        'CAMP はるかさん' → 'はるか'
    CAMPは受講コミュニティ名であり個人名ではないため、まず除去してから名前を抽出する。
    ASCII名はそのまま使用、日本語は文字コードベースのスラッグに変換
    """
    import re
    cleaned = re.sub(r'\bCAMP\b', '', topic, flags=re.IGNORECASE).strip()
    # ASCII部分だけ抽出してスラッグ化
    ascii_parts = re.findall(r'[a-zA-Z]+', cleaned)
    if ascii_parts:
        slug = '-'.join(ascii_parts).lower()
        # 不要な一般単語を除去
        stop = {'mtg', 'meeting', 'on', 'with', 'and', 'the', 'zoom', 'call', 'session'}
        slug_parts = [p for p in slug.split('-') if p not in stop and len(p) > 1]
        if slug_parts:
            return '-'.join(slug_parts)[:30]
    # ASCIIで名前が取れない場合、「○○さん」の直前を日本語名として使う（姓名間の全角スペースは名前に含める）
    m = re.search(r'([^：:、,]+)さん', cleaned)
    if m:
        return m.group(1).strip()
    # 日本語の場合: 文字をそのまま使う（deploy_to_surge側でASCII変換）
    return cleaned or topic


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
    """音声を文字起こし（USE_LOCAL_LLM=trueならfaster-whisperでローカル実行）"""
    if USE_LOCAL_LLM:
        return await _transcribe_local(audio_path)
    return await _transcribe_openai(audio_path)


async def _transcribe_local(audio_path: str) -> str:
    """faster-whisperでローカル文字起こし（無償）"""
    logger.info("文字起こし中（ローカルWhisper）...")

    def _run():
        model = _get_whisper_model()
        segments, _info = model.transcribe(audio_path, language="ja", beam_size=5)
        return "".join(seg.text for seg in segments)

    loop = asyncio.get_event_loop()
    transcript = await loop.run_in_executor(None, _run)
    logger.info(f"文字起こし完了: {len(transcript)}文字")
    return transcript


async def _transcribe_openai(audio_path: str) -> str:
    """OpenAI Whisper APIで文字起こし（25MB超は自動ffmpeg圧縮）"""
    import subprocess
    logger.info("文字起こし中...")
    send_path = audio_path
    compressed_tmp = None
    size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
    if size_mb > 24:
        logger.info(f"ファイルサイズ {size_mb:.1f}MB → ffmpegで圧縮中...")
        compressed_tmp = audio_path.rsplit(".", 1)[0] + "_compressed.mp3"
        result = subprocess.run(
            ["ffmpeg", "-i", audio_path, "-vn", "-ar", "16000", "-ac", "1", "-b:a", "32k", compressed_tmp, "-y"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.warning(f"ffmpeg圧縮失敗: {result.stderr[:200]}")
            compressed_tmp = None
        else:
            compressed_size = Path(compressed_tmp).stat().st_size / (1024 * 1024)
            logger.info(f"圧縮完了: {size_mb:.1f}MB → {compressed_size:.1f}MB")
            send_path = compressed_tmp
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            with open(send_path, "rb") as f:
                r = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    files={"file": (Path(send_path).name, f, "audio/mpeg")},
                    data={"model": "whisper-1", "language": "ja", "response_format": "text"},
                )
                r.raise_for_status()
                transcript = r.text
                logger.info(f"文字起こし完了: {len(transcript)}文字")
                return transcript
    finally:
        if compressed_tmp:
            Path(compressed_tmp).unlink(missing_ok=True)


# ─── Step3: Claudeでナレッジ化 ────────────────────────────────
KNOWLEDGE_PROMPT = """
あなたはビジネスコンサルタントです。
以下のミーティング書き起こし全体を読み、商品・サービス開発に活用できるナレッジとして構造化してください。

このナレッジは、後で本人やチームが「会議で何を話したか」を振り返るための記録です。
要約しすぎて情報を削るのではなく、書き起こしに出てきた発言・具体例・数字・固有名詞をできるだけ漏れなく反映し、
読み返したときに会議の流れと内容が具体的に再現できるレベルの網羅性で書いてください。
箇条書きは1〜2行の短文で済ませず、誰が・何を・なぜ言ったのかが分かる具体的な記述にしてください。
誰の発言かは「〇〇さんが」のように自然な日本語の文章で示し、[名前]のようなタグ表記は使わないでください。
書き起こしの中のカタカナ語・横文字は、声に出して読んでも日本語としても英語の専門用語としても意味が取れないもの
（音声認識の誤変換と思われるもの。例：「eダイルテット」のような実在しない語）は、そのまま転記せず省略するか、
前後の文脈から確実に推測できる場合のみ正しい言葉に直してください。意味が不確かなまま無理に断定しないこと。
書き起こしに書かれていない出来事・数字・地名・予定を、推測や創作で書き加えることは禁止します。書き起こしに
根拠がない内容は書かないでください。
「## 📋 ミーティング概要」「## 🗣️ 話した内容の詳細」など各見出しで同じエピソードを繰り返し使い回さず、
それぞれ異なる切り口（概要は要約、詳細は具体的内容、課題は困りごと、ニーズはアイデア）で書いてください。

# ミーティング情報
タイトル: {topic}
時間: {duration}分

# 書き起こし（全文）
{transcript}

---

以下の形式で日本語でまとめてください：

## 📋 ミーティング概要
（誰と何を話したか、話の流れに沿って5〜8行で具体的に）

## 🗣️ 話した内容の詳細
（書き起こしに出てきた論点・トピックごとに、発言内容や具体例を漏れなく箇条書き。
　話題が複数あれば見出しを分けて、それぞれ3〜6項目程度の具体的な記述にする）

## 😤 顧客の課題・ペイン
（顧客が困っていること、不満点を、発言の背景や理由も含めて箇条書き）

## 💡 拾えるニーズ・アイデア
（提供できる価値、改善のヒント、新商品アイデアを、根拠となった発言と合わせて箇条書き）

## ✅ アクションアイテム
（次にやるべきことを [ ] チェックボックス形式で、担当や期限が話に出ていればそれも含めて）

## 🔑 キーワード
（重要なキーワードを5〜10個、カンマ区切り）
"""


async def generate_knowledge(transcript: str, topic: str, duration: int) -> str:
    """ナレッジ構造化（USE_LOCAL_LLM=trueならOllamaでローカル実行）"""
    logger.info("ナレッジ生成中...")
    prompt = KNOWLEDGE_PROMPT.format(
        topic=topic,
        duration=duration,
        transcript=transcript[:12000]  # 長すぎる場合は先頭12000文字
    )
    if USE_LOCAL_LLM:
        knowledge = await _ollama_generate(prompt, max_tokens=3072)
        # LLMがtopicの「CAMP YAMAさん」のような表記をそのまま使うことがあるため、
        # CAMPは受講コミュニティ名であり個人名ではないため除去する
        import re
        slug_name = _extract_participant_name(topic)
        display_name = slug_name.split("-")[0].capitalize() if slug_name and re.match(r"^[a-z-]+$", slug_name) else slug_name
        if display_name:
            knowledge = re.sub(
                rf"CAMP\s*{re.escape(display_name)}", display_name, knowledge, flags=re.IGNORECASE,
            )
        logger.info("ナレッジ生成完了（ローカルLLM）")
        return knowledge
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
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
- 受講生を指すときに「あなた」という表記は禁止。必ず名前（「〇〇さん」）を使うか、主語を省略する
- 名前は日本語の自然な呼び方（例：「Harukaさん」）で統一し、「haruka-makino」のようなローマ字をハイフンで繋いだ表記は使わない
- 「！」は文末（句点の代わり）にのみ使い、文や段落の先頭に「！」を単独で置かない
- 書き起こしの中のカタカナ語・横文字は、声に出して読んでも日本語としても英語の専門用語としても意味が取れないもの（音声認識の誤変換と思われるもの。例：「eダイルテット」のような実在しない語）は、そのまま転記せず省略するか、前後の文脈から確実に推測できる場合のみ正しい言葉に直すこと。意味が不確かなまま無理に断定しない
- 書き起こしに書かれていない出来事・数字・地名・予定を、推測や創作で書き加えることは禁止。書き起こしに根拠がない内容は書かないこと

# 重要：内容の密度
- このレポートは受講生が後で読み返して「あの日何を話したか」を振り返るための記録であり、お客様にも見せる完成品です。要約しすぎず、書き起こしに出てきた発言・具体例・数字・固有名詞を漏れなく反映してください。
- 「②本日ご確認いただいた内容」は1〜2行の箇条書きで済ませず、話題ごとに見出しを分け、各話題で3〜6項目程度、背景や理由も含めた具体的な記述にしてください。
- 書き起こしに出てきた具体的なアドバイス・方法論・推奨手順・試験傾向・数値目標は一語一句省略せず、見出しを立てて詳細に起こしてください。例えば「どの論点がどのくらい出るか」「なぜその方法がよいか」「具体的な手順の順序」などは、ポイントを絞って削るのではなく、ほぼそのまま文章化してください。
- 「④次回までのアクション」も、話の中で出た具体的な理由・期限・背景を添えてください。
- ①〜⑤の各セクションで同じエピソード・同じ発言を繰り返し使い回してはいけない。それぞれ違う角度・違う内容を担当させること（①は概要、②は詳細な論点、③は決定事項、④は今後の行動、⑤は称賛とエールに役割分担する）

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
  .signature {{ text-align: right; margin: 12px 0 0; font-weight: bold; color: #2f855a; }}
  .footer {{ text-align: center; color: #aaa; font-size: 0.78em; margin-top: 28px; padding-top: 16px; border-top: 1px solid #eee; }}
</style>
</head>
<body>

<div class="header">
  <h1>📋 1on1セッションレポート</h1>
  <p class="meta">{topic}　｜　{date}</p>
</div>

<!-- 書き起こしをもとに以下のセクションをすべて生成してください -->
<!-- ① 今回のセッションについて（概要・目的、3〜5文で具体的に） -->
<!-- ② 本日ご確認いただいた内容（話題ごとに見出しを分け、各話題3〜6項目の具体的な箇条書き、丁寧語。
     書き起こしに出てきたすべての話題を扱うこと。特に以下は必ず独立した見出しで詳細に記載すること：
     ・具体的なアドバイスや方法論（なぜそうするのかの理由も含む）
     ・試験傾向や頻出論点の情報（テストレット別・論点別など）
     ・数値目標・スケジュール・期限
     ・今後の学習方針の変更点や合意事項
     要約しすぎず書き起こしの発言・数字・固有名詞を漏れなく反映し、後で読み返して内容が再現できる密度にする。1見出しあたり最低3項目は書くこと。 -->
<!-- ③ 決定事項・方針（合意した内容、なぜそう決めたかの背景も） -->
<!-- ④ 次回までのアクション（丁寧語、理由や期限も添える）。
     各アクションは必ず次の構造で出力すること（liの入れ子は禁止、action-itemは div のみに使う）：
     <div class="action-item action-high"><span class="badge badge-high">優先度高</span><p>具体的なアクション内容</p></div>
     <div class="action-item action-mid"><span class="badge badge-mid">優先度中</span><p>具体的なアクション内容</p></div> -->
<!-- ⑤ タイシからのメッセージ（messageクラス）
     文字数の下限は厳守すること：5つの<p>タグそれぞれを100〜150文字程度で書く（合計500〜750文字、10行以上になる）。
     1つの<p>が2〜3文に満たない場合は不合格なので書き直すこと。短く済ませてはいけない。
     1段落目（100字以上）：今回の話の中で本人が話した努力・工夫・成長・苦労を具体的に2つ引用し、その状況を描写する
     2段落目（100字以上）：1段落目の内容それぞれに対して全力で称賛する（「素晴らしいです」「感動しました」程度の
       控えめな表現では不十分。「本当にすごいことです」「その粘り強さこそが合格者の共通点です」
       「ここまでやり切った〇〇さんを誇りに思います」のレベルの熱量で、読んだ瞬間に気持ちが上がるトーンにする。
       「！」は文末（句点の代わり）にのみ使い、文の先頭に「！」を置かないこと）
     3段落目（100字以上）：今回の課題や難所について、具体的な状況に触れながらタイシとしての視点や励ましを伝える
     4段落目（100字以上）：次回に向けた期待を、今回の話の続きを踏まえて熱く具体的に伝える
     5段落目（100字以上）：今回の話に即した一言の激励で締める
     最後に <p class="signature">タイシより</p> を必ず別途置く（badgeクラスなど他の用途のクラスを「タイシより」に付けないこと）。 -->

<div class="footer">
  <p>本レポートはセッション内容をもとに作成しております。ご不明な点がございましたらお気軽にご連絡ください。</p>
</div>
</body>
</html>
"""


def _fix_priority_badges(html: str) -> str:
    """優先度バッジの表示テキストは固定文言のため、LLMが何を生成しても強制的に上書きする。
    （小型ローカルモデルだとbadge-high/badge-midのクラス名から英単語priorityを連想し、
    「優 priority中」のように日英混在の文字列を生成することがあるため）"""
    import re
    html = re.sub(r'(<span class="[^"]*badge-high[^"]*">)[^<]*(</span>)', r"\g<1>優先度高\g<2>", html)
    html = re.sub(r'(<span class="[^"]*badge-mid[^"]*">)[^<]*(</span>)', r"\g<1>優先度中\g<2>", html)
    return html


async def generate_html_report(transcript: str, topic: str, host: str, date: str, duration: int) -> str:
    """HTMLレポートを生成（USE_LOCAL_LLM=trueならOllamaでローカル実行）"""
    logger.info("HTMLレポート生成中...")
    prompt = HTML_REPORT_PROMPT.format(
        topic=topic, host=host, date=date, duration=duration,
        transcript=transcript[:12000]
    )
    if USE_LOCAL_LLM:
        html = await _ollama_generate(prompt, max_tokens=8192)
        if html.startswith("```"):
            html = html.split("```")[1]
            if html.startswith("html"):
                html = html[4:]
        # _extract_participant_nameはURL用スラッグ（例: "haruka-makino"）を返すため、
        # 表示用には先頭の名前部分だけを取り出して人名らしい形に直す
        import re
        slug_name = _extract_participant_name(topic)
        display_name = ""
        if slug_name and re.match(r"^[a-z-]+$", slug_name):
            display_name = slug_name.split("-")[0].capitalize()
        elif slug_name:
            display_name = slug_name
        # プロンプト指示だけでは「あなた」表記が残ることがあるため確実に名前へ置換する
        if display_name:
            html = html.replace("あなた", f"{display_name}さん")
            # LLMが「haruka-makino」のようなスラッグ表記をそのまま生成することがあるため統一する
            if slug_name and re.match(r"^[a-z-]+$", slug_name):
                html = re.sub(
                    rf"\b{re.escape(slug_name)}(さん)?\b",
                    f"{display_name}さん", html, flags=re.IGNORECASE,
                )
            # LLMがtopicの「CAMP YAMAさん」のような表記をそのまま使うことがあるため、
            # CAMPは受講コミュニティ名であり個人名ではないため除去する
            html = re.sub(
                rf"CAMP\s*{re.escape(display_name)}", display_name, html, flags=re.IGNORECASE,
            )
        # 「！」を熱量表現に使う指示の副作用で文頭に意味のない「！」が付くことがあるため除去する
        html = re.sub(r"(<p[^>]*>)\s*！\s*", r"\1", html)
        html = _fix_priority_badges(html)
        logger.info("HTMLレポート生成完了（ローカルLLM）")
        return html.strip()
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
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
        html = _fix_priority_badges(html)
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
        surge_bin = (
            shutil.which("surge")
            or str(Path(__file__).parent / "node_modules" / ".bin" / "surge")
            or "/usr/local/bin/surge"
        )
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


# ─── coaching-tool（Neon DB）への同期 ─────────────────────────
# トピック文字列に含まれるキーワードでクライアントを特定する。
# 複数の表記ゆれ（漢字・ローマ字・discord名）があるため、キーワードは1人につき複数登録できる。
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
    16: ["erika"],
    17: ["りょう"],
    18: ["KS"],
    19: ["Sato"],
    20: ["新昌靜", "しょうせい"],
    21: ["りんたろう", "rintaro"],
    22: ["Kobayashi Takahiro", "TAKAHIRO KOBAYASHI", "小林", "たか"],
    23: ["山田", "yamada", "国毅"],
    24: ["定國洋子", "yoko"],
    25: ["hitomi", "hitomix", "尾崎仁美", "UKさん"],
}


def _match_client_id(topic: str) -> Optional[int]:
    """ミーティングタイトルからcoaching-toolのclient_idを特定する。
    マッチしない場合（合格者インタビュー等、既存クライアント以外の録音）はNoneを返す。
    例えば"YAMA"が"Yamada"の部分文字列になるなど、短いキーワードが別人の長いキーワードに
    誤って包含されるケースがあるため、最長一致のキーワードを優先する。"""
    topic_lower = topic.lower()
    best: tuple[int, int] | None = None  # (keyword_len, client_id)
    for client_id, keywords in CLIENT_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in topic_lower:
                if best is None or len(kw) > best[0]:
                    best = (len(kw), client_id)
    return best[1] if best else None


COACHING_SUMMARY_PROMPT = """
以下はコーチングの1on1面談を構造化したナレッジ記録です。
これを「コーチング管理ツール」に保存するための面談サマリーとアクションアイテムに変換してください。

# ナレッジ記録
{knowledge}

---

以下のJSON形式のみを出力してください（説明文・コードブロック記号など他の文字は一切出力しないこと）：

{{
  "summary": "面談で何を確認し、何が課題で、何を決めたかを3〜6文の具体的な日本語で。",
  "action_items": ["クライアント本人が次回の面談までに実際に行う学習行動を3〜5個。"]
}}

action_itemsには、コーチ側の商品開発タスクや「○○ツールを作る」「○○資料を作成する」のような
ビジネス改善アイデアは絶対に含めないこと。クライアントが自分の学習のために行う具体的な行動のみを書くこと。
本文に次回の予約に関する言及があれば、それも1項目として含めてよい。
"""


async def generate_coaching_summary(knowledge: str) -> Optional[dict]:
    """ナレッジ記録からクライアント向けサマリー＋アクションアイテムをJSONで抽出する"""
    if not ANTHROPIC_API_KEY:
        return None
    try:
        prompt = COACHING_SUMMARY_PROMPT.format(knowledge=knowledge[:8000])
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            r.raise_for_status()
            text = r.json()["content"][0]["text"].strip()
            # コードブロックで返ってきた場合に備えて剥がす
            if text.startswith("```"):
                text = text.strip("`").lstrip("json").strip()
            return json.loads(text)
    except Exception as e:
        logger.warning(f"generate_coaching_summary エラー: {e}")
        return None


async def sync_to_coaching_db(topic: str, date: str, knowledge: str, report_url: str) -> None:
    """クライアントを特定できた場合のみ、coaching-toolのNeon DBにsession・action_itemsを書き込む"""
    if not COACHING_DATABASE_URL:
        return
    client_id = _match_client_id(topic)
    if client_id is None:
        logger.info(f"coaching DB同期: クライアント特定できず（スキップ）: {topic}")
        return

    import asyncpg
    date_obj = datetime.strptime(date, "%Y-%m-%d").date()
    conn = None
    try:
        conn = await asyncpg.connect(COACHING_DATABASE_URL)
        existing = await conn.fetchval(
            "SELECT id FROM sessions WHERE client_id = $1 AND date = $2", client_id, date_obj
        )
        if existing:
            logger.info(f"coaching DB同期: 既存セッションのためスキップ (client_id={client_id}, date={date})")
            return

        coaching_data = await generate_coaching_summary(knowledge)
        if not coaching_data:
            logger.warning(f"coaching DB同期: サマリー生成失敗 (client_id={client_id})")
            return

        session_id = await conn.fetchval(
            """INSERT INTO sessions (client_id, date, summary, report_url)
               VALUES ($1, $2, $3, $4) RETURNING id""",
            client_id, date_obj, coaching_data.get("summary", ""), report_url or None,
        )
        for content in coaching_data.get("action_items", []):
            await conn.execute(
                "INSERT INTO action_items (session_id, content, status) VALUES ($1, $2, 'pending')",
                session_id, content,
            )
        logger.info(
            f"coaching DB同期完了: client_id={client_id}, session_id={session_id}, "
            f"actions={len(coaching_data.get('action_items', []))}"
        )
    except Exception as e:
        logger.error(f"coaching DB同期エラー: {e}", exc_info=True)
    finally:
        if conn:
            await conn.close()


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

        # Step7: coaching-tool（Neon DB）に同期（クライアント特定できた場合のみ）
        await sync_to_coaching_db(topic=topic, date=date_str, knowledge=knowledge, report_url=surge_url)

        logger.info(f"手動処理完了: surge={surge_url} notion={notion_url}")

    except Exception as e:
        logger.error(f"手動処理エラー: {e}", exc_info=True)


# ─── 定期ポーリング（Webhook 未着時の自動バックアップ） ────────
@app.on_event("startup")
async def start_periodic_poller():
    """サーバー起動時にバックグラウンドポーラーを開始"""
    asyncio.create_task(_recording_poller_loop())


async def _recording_poller_loop():
    """1時間おきに未処理録画をチェックして自動処理"""
    await asyncio.sleep(60)  # 起動直後は1分待つ
    while True:
        try:
            logger.info("⏰ 定期チェック開始")
            await _check_and_process_pending()
        except Exception as e:
            logger.error(f"定期チェックエラー: {e}", exc_info=True)
        await asyncio.sleep(3600)  # 1時間ごと


async def _check_and_process_pending():
    """過去7日間の録画で未処理のものを検出して処理"""
    import re
    from datetime import timedelta

    today    = datetime.utcnow()
    from_dt  = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    to_dt    = today.strftime("%Y-%m-%d")

    token = await _get_zoom_access_token()
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(
            "https://api.zoom.us/v2/users/me/recordings",
            headers={"Authorization": f"Bearer {token}"},
            params={"from": from_dt, "to": to_dt, "page_size": 30},
        )
        if not r.is_success:
            logger.warning(f"録画一覧取得失敗: {r.status_code}")
            return

        meetings = r.json().get("meetings", [])
        logger.info(f"録画件数: {len(meetings)}件")

        for meeting in meetings:
            topic      = meeting.get("topic", "")
            meeting_id = str(meeting.get("id", ""))
            start_time = meeting.get("start_time", "")
            date_str   = start_time[:10]
            duration   = meeting.get("duration", 0)
            files      = meeting.get("recording_files", [])

            # 対象外トピックはスキップ
            if _is_skip_topic(topic):
                logger.info(f"スキップ（対象外トピック）: {topic}")
                continue

            # M4A完了ファイルがない録画はスキップ
            m4a = next((f for f in files if f.get("file_type") == "M4A"
                        and f.get("status") == "completed"), None)
            if not m4a:
                continue

            # surge URL が既に存在する → 処理済みとみなしスキップ
            name = _extract_participant_name(topic)
            slug = re.sub(r'[^a-zA-Z0-9]', '-', name)
            slug = re.sub(r'-+', '-', slug)[:30].strip('-').lower() or "meeting"
            expected_url = f"https://1on1-{date_str}-{slug}.surge.sh"

            try:
                async with httpx.AsyncClient(timeout=8) as hc:
                    hr = await hc.head(expected_url, follow_redirects=True)
                    if hr.status_code == 200:
                        logger.info(f"処理済みスキップ: {topic}")
                        continue
            except Exception:
                pass  # チェック失敗なら処理を試みる

            # 処理権取得（重複防止）
            if not _try_claim_meeting(meeting_id):
                continue

            logger.info(f"未処理録画を発見 → 処理開始: {topic} ({date_str})")
            # ローカルLLM使用時はCPU/メモリ競合を避けるため1件ずつ順番に処理する
            if USE_LOCAL_LLM:
                await _process_from_api(meeting, token)
            else:
                asyncio.create_task(_process_from_api(meeting, token))


async def _process_from_api(meeting: dict, token: str):
    """Zoom APIから録画をダウンロードしてパイプライン実行"""
    topic      = meeting.get("topic", "")
    duration   = meeting.get("duration", 0)
    start_time = meeting.get("start_time", "")
    date_str   = start_time[:10]
    files      = meeting.get("recording_files", [])
    name       = _extract_participant_name(topic)

    audio_path = None
    try:
        m4a = next((f for f in files if f.get("file_type") == "M4A"
                    and f.get("status") == "completed"), None)
        if not m4a:
            return

        # 音声ダウンロード（トークンは1時間で失効するため直前に再取得）
        logger.info(f"音声ダウンロード中: {topic}")
        fresh_token = await _get_zoom_access_token()
        async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
            dl = await client.get(
                m4a["download_url"] + f"?access_token={fresh_token}"
            )
            if not dl.is_success:
                raise RuntimeError(f"音声ダウンロード失敗: HTTP {dl.status_code}")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".m4a")
            tmp.write(dl.content)
            tmp.close()
            audio_path = tmp.name

        transcript = await transcribe_audio(audio_path)
        knowledge  = await generate_knowledge(transcript, topic, duration)
        notion_url = await save_to_notion(
            topic=topic, host="", start_at=start_time,
            duration=duration, transcript=transcript, knowledge=knowledge,
        )
        html = await generate_html_report(
            transcript=transcript, topic=topic,
            host="", date=date_str, duration=duration,
        )
        surge_url = await deploy_to_surge(html, topic, date_str, name) if SURGE_TOKEN else ""
        await send_to_discord(
            topic=topic, date=date_str,
            surge_url=surge_url or "(デプロイ未設定)",
            notion_url=notion_url,
        )
        save_to_local_knowledge(
            topic=topic, date=date_str, host="", duration=duration,
            transcript=transcript, knowledge=knowledge, surge_url=surge_url,
        )
        logger.info(f"自動処理完了: {topic} → {surge_url}")

    except Exception as e:
        logger.error(f"_process_from_api エラー: {e}", exc_info=True)
    finally:
        if audio_path:
            Path(audio_path).unlink(missing_ok=True)


@app.post("/process-pending")
async def process_pending_endpoint(
    background_tasks: BackgroundTasks,
    request: Request,
):
    """未処理録画を今すぐチェック（外部cronまたは手動トリガー）"""
    # CRON_SECRETが設定されている場合は認証チェック
    if CRON_SECRET:
        auth = request.headers.get("x-cron-secret", "")
        if auth != CRON_SECRET:
            raise HTTPException(status_code=401, detail="Unauthorized")
    background_tasks.add_task(_check_and_process_pending)
    return JSONResponse({"message": "チェック開始。2〜5分後にDiscordをご確認ください。"})


# ─── ヘルスチェック ────────────────────────────────────────────
@app.get("/")
async def health():
    return {"status": "ok", "service": "zoom-knowledge-auto-generator"}
