"""
指定したトピック名に一致する録画だけを処理する単発スクリプト。
全件チェックでリソース競合を起こさず、特定の1件だけを今すぐ処理したい場合に使う。
使い方: python3 process_one.py "CAMP"
"""
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main import _get_zoom_access_token, _process_from_api, httpx  # noqa: E402


async def main(topic_query: str):
    today = datetime.utcnow()
    from_dt = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    to_dt = today.strftime("%Y-%m-%d")

    token = await _get_zoom_access_token()
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(
            "https://api.zoom.us/v2/users/me/recordings",
            headers={"Authorization": f"Bearer {token}"},
            params={"from": from_dt, "to": to_dt, "page_size": 30},
        )
        r.raise_for_status()
        meetings = r.json().get("meetings", [])

    matches = [m for m in meetings if m.get("topic", "") == topic_query]
    if not matches:
        print(f"トピック '{topic_query}' に一致する録画が見つかりません")
        print("候補:", [m.get("topic") for m in meetings])
        return

    meeting = matches[0]
    print(f"処理開始: {meeting.get('topic')} ({meeting.get('start_time')})")
    await _process_from_api(meeting, token)
    print("完了")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
