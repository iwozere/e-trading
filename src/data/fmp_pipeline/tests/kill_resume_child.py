"""
Child process for the kill-and-resume test (spec §13): runs a slow download against its own mock server and
appends every sunk ``task_id`` to a file, until the parent kills it.

Usage: ``python -m src.data.fmp_pipeline.tests.kill_resume_child <manifest.sqlite> <sink.txt>``
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fmp_pipeline.client.fmp_client import FetchResult
from src.data.fmp_pipeline.manifest.manifest import Manifest, Task
from src.data.fmp_pipeline.runner import Downloader
from src.data.fmp_pipeline.tests.helpers import FakeSleep, make_client
from src.data.fmp_pipeline.tests.mock_server import Fault, MockFmpServer

DUMP_ID = "kill-test"


def file_sink(path: Path):
    """Return a sink that appends ``task_id`` lines and flushes them to disk immediately."""

    def sink(task: Task, result: FetchResult) -> None:
        del result
        with path.open("a", encoding="utf-8") as fh:
            fh.write(task.task_id + "\n")
            fh.flush()

    return sink


async def _main(manifest_path: Path, sink_path: Path) -> None:
    with MockFmpServer() as fmp:
        fmp.add_fault(Fault(delay_s=0.05, times=None))  # slow enough for the parent to kill mid-run
        with Manifest(manifest_path, DUMP_ID) as manifest:
            async with make_client(fmp, FakeSleep()) as client:
                await Downloader(manifest, client, sink=file_sink(sink_path), workers=4).run()


if __name__ == "__main__":
    asyncio.run(_main(Path(sys.argv[1]), Path(sys.argv[2])))
