"""
ローカル自動実行スクリプト（無償運用モード）
launchdから定期的に呼ばれ、未処理のZoom録画をローカルLLMで処理する。
処理が15分(StartInterval)を超えて前回分がまだ動いている場合、
多重起動によるCPU/メモリ競合を防ぐためロックファイルで排他制御する。
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main import _check_and_process_pending  # noqa: E402

LOCK_PATH = Path(__file__).parent / "local_runner.lock"


def _acquire_lock() -> bool:
    if LOCK_PATH.exists():
        try:
            pid = int(LOCK_PATH.read_text().strip())
            os.kill(pid, 0)  # プロセスが生きているか確認（例外なければ生存）
            return False
        except (ValueError, ProcessLookupError, PermissionError):
            pass  # PIDが死んでいる/不正 → 古いロックとして上書き
    LOCK_PATH.write_text(str(os.getpid()))
    return True


def _release_lock():
    LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    if not _acquire_lock():
        print("前回の処理がまだ実行中のためスキップします")
        sys.exit(0)
    try:
        asyncio.run(_check_and_process_pending())
    finally:
        _release_lock()
