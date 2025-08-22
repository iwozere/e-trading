import os
from pathlib import Path
def find_null_bytes_in_py_files(root_dir):
    for path in Path(root_dir).rglob('*.py'):
        try:
            with path.open('rb') as f:
                for i, line in enumerate(f, 1):
                    if b'\x00' in line:
                        print(f"Null byte found in {path} at line {i}")
        except Exception as e:
            print(f"Could not read {path}: {e}")

if __name__ == "__main__":
    find_null_bytes_in_py_files(".")
