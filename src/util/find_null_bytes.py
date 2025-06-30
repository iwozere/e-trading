import os
def find_null_bytes_in_py_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'rb') as f:
                        for i, line in enumerate(f, 1):
                            if b'\x00' in line:
                                print(f"Null byte found in {file_path} at line {i}")
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")

if __name__ == "__main__":
    find_null_bytes_in_py_files(".")