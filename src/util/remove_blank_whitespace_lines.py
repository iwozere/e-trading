"""Removes all lines containing only whitespace from all .py files in the project."""
import os
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            if "\\." in path:
                continue
            print(path)
            try:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                except UnicodeDecodeError:
                    with open(path, 'r', encoding='latin-1') as f:
                        lines = f.readlines()
            except Exception as read_err:
                print(f"[WARN] Could not read {path}: {read_err}")
                continue
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    for line in lines:
                        if line.strip() != '':
                            f.write(line)
                        else:
                            f.write("\n")
            except Exception as write_err:
                print(f"[WARN] Could not write {path}: {write_err}")
                continue
