"""
Child process for the kill-and-resume test (spec §13): runs a slow download against its own mock server and
appends every sunk ``task_id`` to a file, until the parent kills it. With a third argument, responses also go to a
``RawArchive`` under that directory first (the production sink).

Usage: ``python -m src.data.fmp_pipeline.tests.kill_resume_child <manifest.sqlite> <sink.txt> [<raw_root>]``
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fmp_pipeline.archive.archive import RawArchive
from src.data.fmp_pipeline.client.fmp_client import FetchResult
from src.data.fmp_pipeline.manifest.manifest import Manifest, Task
from src.data.fmp_pipeline.runner import Downloader
from src.data.fmp_pipeline.tests.helpers import FakeSleep, make_client
from src.data.fmp_pipeline.tests.mock_server import Fault, MockFmpServer

DUMP_ID = "kill-test"


def file_sink(path: Path, archive: Optional[RawArchive] = None):
    """
    Return a sink that appends ``task_id`` lines and flushes them to disk immediately.

    Args:
        path: The ``task_id`` log.
        archive: If given, each response is archived first and the archive's sha256 is returned.
    """

    def sink(task: Task, result: FetchResult) -> Optional[str]:
        sha256 = archive(task, result) if archive is not None else None
        with path.open("a", encoding="utf-8") as fh:
            fh.write(task.task_id + "\n")
            fh.flush()
        return sha256

    return sink


async def _main(manifest_path: Path, sink_path: Path, raw_root: Optional[Path]) -> None:
    with MockFmpServer() as fmp:
        fmp.add_fault(Fault(delay_s=0.05, times=None))  # slow enough for the parent to kill mid-run
        with Manifest(manifest_path, DUMP_ID) as manifest:
            async with make_client(fmp, FakeSleep()) as client:
                archive = RawArchive(raw_root, DUMP_ID) if raw_root is not None else None
                await Downloader(manifest, client, sink=file_sink(sink_path, archive), workers=4).run()


if __name__ == "__main__":
    asyncio.run(_main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None))
