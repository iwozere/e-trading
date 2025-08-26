"""Removes all lines containing only whitespace from all .py files in the project."""
from pathlib import Path
for path in Path('.').rglob('*.py'):
    if "\\." in str(path):
        continue
    print(path)
    try:
        try:
            lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
        except UnicodeDecodeError:
            lines = path.read_text(encoding='latin-1').splitlines(keepends=True)
    except Exception as read_err:
        print(f"[WARN] Could not read {path}: {read_err}")
        continue
    try:
        with path.open('w', encoding='utf-8') as f:
            for line in lines:
                if line.strip() != '':
                    f.write(line)
                else:
                    f.write("\n")
    except Exception as write_err:
        print(f"[WARN] Could not write {path}: {write_err}")
        continue
