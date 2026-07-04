#!/usr/bin/env python3
"""
Shim for backward compatibility.
Delegates to src/ml/pipeline/shared/trf_downloader.py
"""

import sys
from pathlib import Path

# Add project root to path
# Paths are: src/ml/pipeline/p06_emps2/trf_downloader.py (this file)
# parents[0] = src/ml/pipeline/p06_emps2
# parents[1] = src/ml/pipeline
# parents[2] = src/ml
# parents[3] = src
# parents[4] = <root>
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import and run the shared implementation
from src.ml.pipeline.shared.trf_downloader import main

if __name__ == "__main__":
    main()
