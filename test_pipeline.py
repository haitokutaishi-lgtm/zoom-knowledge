"""
ローカルテスト用スクリプト
音声ファイルを直接渡してパイプラインを確認する
実行: python3 test_pipeline.py path/to/audio.mp4
"""

import asyncio
import sys
from datetime import datetime
from main import (
    transcribe_audio, generate_knowledge, save_to_notion,
    generate_html_report, deploy_to_surge, send_to_discord,
    save_to_local_knowledge
)

TOPIC    = "テストミーティング"
HOST     = "haitokutaishi@gmail.com"
DATE_STR = datetime.now().strftime("%Y-%m-%d")
DURATION = 30


async def test_with_file(audio_path: str):
    print(f"🎙️  テスト開始: {audio_path}\n")

    print("[1/6] Whisper文字起こし...")
    transcript = await transcribe_audio(audio_path)
    print(f"✅  {len(transcript)}文字")
    print(transcript[:200] + "...\n")

    print("[2/6] Claudeナレッジ化...")
    knowledge = await generate_knowledge(transcript, topic=TOPIC, duration=DURATION)
    print("✅  ナレッジ生成完了")
    print(knowledge[:300] + "...\n")

    print("[3/6] Notion保存...")
    notion_url = await save_to_notion(
        topic=TOPIC, host=HOST, start_at=DATE_STR + "T10:00:00Z",
        duration=DURATION, transcript=transcript, knowledge=knowledge,
    )
    print(f"✅  {notion_url}\n")

    print("[4/6] HTMLレポート生成...")
    html = await generate_html_report(
        transcript=transcript, topic=TOPIC, host=HOST,
        date=DATE_STR, duration=DURATION
    )
    print(f"✅  {len(html)}文字のHTML生成完了\n")

    print("[5/6] surge.shデプロイ...")
    surge_url = await deploy_to_surge(html, TOPIC, DATE_STR)
    print(f"✅  {surge_url}\n")

    print("[6/6] Discord通知...")
    await send_to_discord(
        topic=TOPIC, date=DATE_STR,
        surge_url=surge_url, notion_url=notion_url
    )
    print("✅  Discord通知送信完了\n")

    print("[+] ローカルナレッジ保存...")
    path = save_to_local_knowledge(
        topic=TOPIC, date=DATE_STR, host=HOST, duration=DURATION,
        transcript=transcript, knowledge=knowledge, surge_url=surge_url
    )
    print(f"✅  {path}\n")

    print("=" * 50)
    print(f"🎉 全処理完了！")
    print(f"   📋 Notion:  {notion_url}")
    print(f"   🌐 レポート: {surge_url}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python3 test_pipeline.py path/to/audio.mp4")
        sys.exit(1)
    asyncio.run(test_with_file(sys.argv[1]))
