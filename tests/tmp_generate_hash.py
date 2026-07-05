# tests/tmp_generate_hash.py
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import bcrypt
import getpass
import argparse

def generate_hash(password: str) -> str:
    """Generate a bcrypt hash for the password."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def main():
    parser = argparse.ArgumentParser(description="Generate bcrypt hash for a password.")
    parser.add_argument("-p", "--password", help="Plain text password (insecure: will be stored in shell history)")
    args = parser.parse_args()
    
    password = args.password
    if not password:
        password = getpass.getpass("Enter password to hash: ")
        if not password:
            print("Error: Password cannot be empty.")
            sys.exit(1)
            
    hashed = generate_hash(password)
    print("\nGenerated Bcrypt Hash:")
    print(hashed)

if __name__ == "__main__":
    main()
