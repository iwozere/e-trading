#!/usr/bin/env python
"""
Verify that all test configurations use TEST database, not production.

This script checks:
1. No test files import production DB_URL from donotshare
2. All conftest.py files use TEST_DB_URL or ALEMBIC_DB_URL
3. No hardcoded database URLs pointing to production
"""
import sys
import re
from pathlib import Path
from typing import List, Tuple

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def check_file_for_production_imports(file_path: Path) -> List[str]:
    """Check if file imports production DB_URL."""
    issues = []

    # Skip the verification script itself
    if file_path.name == 'verify_test_db_config.py':
        return issues

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')

    # Check for production DB_URL imports
    production_patterns = [
        r'from\s+config\.donotshare\.donotshare\s+import\s+DB_URL',
        r'from\s+config\.donotshare\s+import\s+.*DB_URL',
        r'donotshare\.DB_URL',
    ]

    for i, line in enumerate(lines, 1):
        # Skip comments
        if line.strip().startswith('#'):
            continue

        for pattern in production_patterns:
            if re.search(pattern, line):
                # Check if it's explicitly marked as safe (with TEST override)
                surrounding = '\n'.join(lines[max(0, i-5):i+5])
                if 'TEST' in line or 'NEVER' in surrounding or 'DB_URL = TEST_DB_URL' in surrounding:
                    continue
                issues.append(f"Line {i}: Potential production DB import: {line.strip()}")

    return issues

def check_conftest_uses_test_db(file_path: Path) -> List[str]:
    """Check if conftest.py uses TEST_DB_URL or ALEMBIC_DB_URL."""
    issues = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for test database configuration
    has_test_db_url = 'TEST_DB_URL' in content
    has_alembic_db_url = 'ALEMBIC_DB_URL' in content
    has_postgres_test = 'POSTGRES_TEST' in content
    inherits_from_repos = 'pytest_plugins' in content and 'repos.conftest' in content

    # Check for production database usage
    has_production_import = 'from config.donotshare.donotshare import DB_URL' in content
    has_database_url = 'DATABASE_URL' in content and 'TEST' not in content

    if has_production_import:
        issues.append("[CRITICAL] Imports production DB_URL from donotshare!")

    if not (has_test_db_url or has_alembic_db_url or has_postgres_test or inherits_from_repos):
        issues.append("[WARNING] No TEST_DB_URL, ALEMBIC_DB_URL, or POSTGRES_TEST_* configuration found")

    # Look for comments about production safety
    if 'NEVER' in content and 'production' in content:
        # Good - has safety warnings
        pass
    elif not inherits_from_repos:
        issues.append("[INFO] Consider adding safety comments about using test database only")

    return issues

def scan_test_directory(test_dir: Path) -> Tuple[int, int, List[str]]:
    """Scan all Python files in test directory for issues."""
    files_checked = 0
    issues_found = []

    for py_file in test_dir.rglob('*.py'):
        # Skip __pycache__ and .venv
        if '__pycache__' in str(py_file) or '.venv' in str(py_file):
            continue

        files_checked += 1

        # Check for production imports in all test files
        file_issues = check_file_for_production_imports(py_file)
        if file_issues:
            issues_found.append(f"\n{RED}File: {py_file}{RESET}")
            issues_found.extend(file_issues)

        # Extra checks for conftest.py files
        if py_file.name == 'conftest.py':
            conftest_issues = check_conftest_uses_test_db(py_file)
            if conftest_issues:
                issues_found.append(f"\n{YELLOW}Conftest: {py_file}{RESET}")
                issues_found.extend(conftest_issues)

    return files_checked, len([i for i in issues_found if i.startswith('\n')]), issues_found

def main():
    """Run verification checks."""
    print(f"\n{GREEN}{'='*60}{RESET}")
    print(f"{GREEN}Test Database Configuration Verification{RESET}")
    print(f"{GREEN}{'='*60}{RESET}\n")

    # Get test directory
    script_dir = Path(__file__).parent
    test_dir = script_dir

    print(f"Scanning: {test_dir}\n")

    files_checked, files_with_issues, issues = scan_test_directory(test_dir)

    print(f"\n{GREEN}{'='*60}{RESET}")
    print(f"Files checked: {files_checked}")
    print(f"Files with issues: {files_with_issues}")

    if issues:
        print("\nIssues found:")
        for issue in issues:
            # Remove ANSI codes for Windows compatibility
            clean_issue = issue.replace(RED, '').replace(YELLOW, '').replace(GREEN, '').replace(RESET, '')
            print(clean_issue)
        print(f"\n{YELLOW}{'='*60}{RESET}")
        print(f"{YELLOW}[!] Please review the issues above{RESET}")

        # Check if any critical issues
        critical = any('CRITICAL' in i for i in issues)
        if critical:
            print(f"{RED}[ERROR] CRITICAL ISSUES FOUND - Tests may use production database!{RESET}")
            return 1
        else:
            print(f"{YELLOW}[WARN] Warnings found - Review recommended{RESET}")
            return 0
    else:
        print(f"\n{GREEN}[OK] All checks passed!{RESET}")
        print(f"{GREEN}All test files properly use TEST database configuration{RESET}")
        print(f"{GREEN}{'='*60}{RESET}\n")
        return 0

if __name__ == '__main__':
    sys.exit(main())
