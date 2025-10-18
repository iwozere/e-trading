#!/usr/bin/env python3
"""
Documentation Code Example Validation Script
Validates Python code examples in HLA documentation for syntax correctness
"""

import re
import ast
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

class CodeValidator:
    def __init__(self, docs_path: str = "docs/HLA"):
        self.docs_path = Path(docs_path)
        self.errors = []
        self.warnings = []
        self.total_blocks = 0
        self.valid_blocks = 0

    def extract_code_blocks(self, file_path: Path) -> List[Tuple[int, str]]:
        """Extract Python code blocks from markdown files with line numbers."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"Error reading {file_path}: {e}")
            return []

        # Find Python code blocks with line numbers
        code_blocks = []
        lines = content.split('\n')
        in_python_block = False
        current_block = []
        block_start_line = 0

        for i, line in enumerate(lines, 1):
            if line.strip() == '```python':
                in_python_block = True
                block_start_line = i + 1
                current_block = []
            elif line.strip() == '```' and in_python_block:
                if current_block:
                    code_blocks.append((block_start_line, '\n'.join(current_block)))
                in_python_block = False
                current_block = []
            elif in_python_block:
                current_block.append(line)

        return code_blocks

    def validate_syntax(self, code_block: str) -> Tuple[bool, str]:
        """Validate Python syntax of code block."""
        try:
            # Try to parse the code
            ast.parse(code_block)
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax Error: {e.msg} (line {e.lineno})"
        except Exception as e:
            return False, f"Parse Error: {str(e)}"

    def should_skip_block(self, code_block: str) -> Tuple[bool, str]:
        """Determine if a code block should be skipped from validation."""
        stripped = code_block.strip()

        # Skip empty blocks
        if not stripped:
            return True, "empty block"

        # Skip comment-only blocks
        lines = [line.strip() for line in stripped.split('\n') if line.strip()]
        if all(line.startswith('#') for line in lines):
            return True, "comment-only block"

        # Skip import-only blocks (less than 3 lines with only imports)
        if len(lines) < 3 and all(
            line.startswith('import ') or
            line.startswith('from ') or
            line.startswith('#') or
            not line
            for line in lines
        ):
            return True, "import-only block"

        # Skip configuration examples (YAML/JSON-like content)
        if any(char in stripped for char in ['{', '}', ':', '- ']):
            yaml_like_lines = sum(1 for line in lines if ':' in line or line.startswith('- '))
            if yaml_like_lines > len(lines) * 0.5:  # More than 50% YAML-like
                return True, "configuration example"

        # Skip shell command examples
        if stripped.startswith('$') or stripped.startswith('./'):
            return True, "shell command example"

        return False, ""

    def validate_file(self, file_path: Path) -> Dict:
        """Validate all code examples in a single file."""
        file_results = {
            'file': str(file_path),
            'total_blocks': 0,
            'valid_blocks': 0,
            'skipped_blocks': 0,
            'errors': [],
            'warnings': []
        }

        code_blocks = self.extract_code_blocks(file_path)

        for line_num, code_block in code_blocks:
            file_results['total_blocks'] += 1
            self.total_blocks += 1

            # Check if block should be skipped
            should_skip, skip_reason = self.should_skip_block(code_block)
            if should_skip:
                file_results['skipped_blocks'] += 1
                file_results['warnings'].append(
                    f"Line {line_num}: Skipped ({skip_reason})"
                )
                continue

            # Validate syntax
            valid, error = self.validate_syntax(code_block)
            if valid:
                file_results['valid_blocks'] += 1
                self.valid_blocks += 1
            else:
                error_msg = f"Line {line_num}: {error}"
                file_results['errors'].append(error_msg)
                self.errors.append(f"{file_path}:{error_msg}")

        return file_results

    def validate_all(self, verbose: bool = False) -> bool:
        """Validate all code examples in documentation."""
        print("🔍 Validating Python code examples in documentation...")
        print("=" * 60)

        results = []

        # Find all markdown files
        md_files = list(self.docs_path.rglob('*.md'))
        if not md_files:
            print(f"❌ No markdown files found in {self.docs_path}")
            return False

        print(f"📄 Found {len(md_files)} markdown files")
        print()

        # Validate each file
        for md_file in sorted(md_files):
            file_results = self.validate_file(md_file)
            results.append(file_results)

            if verbose or file_results['errors']:
                self.print_file_results(file_results)

        # Print summary
        self.print_summary(results)

        return len(self.errors) == 0

    def print_file_results(self, results: Dict):
        """Print results for a single file."""
        file_path = results['file']
        total = results['total_blocks']
        valid = results['valid_blocks']
        skipped = results['skipped_blocks']
        errors = results['errors']

        if total == 0:
            return  # Skip files with no code blocks

        status = "✅" if not errors else "❌"
        print(f"{status} {file_path}")
        print(f"   Total blocks: {total}, Valid: {valid}, Skipped: {skipped}, Errors: {len(errors)}")

        if errors:
            for error in errors:
                print(f"   ❌ {error}")

        if results['warnings'] and len(results['warnings']) > 0:
            print(f"   ⚠️  {len(results['warnings'])} blocks skipped")

        print()

    def print_summary(self, results: List[Dict]):
        """Print validation summary."""
        total_files = len([r for r in results if r['total_blocks'] > 0])
        files_with_errors = len([r for r in results if r['errors']])
        total_errors = sum(len(r['errors']) for r in results)
        total_skipped = sum(r['skipped_blocks'] for r in results)

        print("=" * 60)
        print("📊 Validation Summary")
        print("=" * 60)
        print(f"Files processed: {total_files}")
        print(f"Total code blocks: {self.total_blocks}")
        print(f"Valid blocks: {self.valid_blocks}")
        print(f"Skipped blocks: {total_skipped}")
        print(f"Error blocks: {total_errors}")
        print(f"Files with errors: {files_with_errors}")
        print()

        if total_errors == 0:
            print("✅ All code examples are syntactically valid!")
            print()
            print("🎉 Documentation code quality: EXCELLENT")
        else:
            print(f"❌ Found {total_errors} syntax errors in {files_with_errors} files")
            print()
            print("🔧 To fix code validation errors:")
            print("1. Review the syntax errors listed above")
            print("2. Fix the Python code in the documentation")
            print("3. Run this script again to verify fixes")
            print("4. Consider adding # noqa comments for intentional examples")

def main():
    parser = argparse.ArgumentParser(
        description="Validate Python code examples in HLA documentation"
    )
    parser.add_argument(
        "--docs-path",
        default="docs/HLA",
        help="Path to documentation directory (default: docs/HLA)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed results for all files"
    )
    parser.add_argument(
        "--file",
        help="Validate specific file only"
    )

    args = parser.parse_args()

    validator = CodeValidator(args.docs_path)

    if args.file:
        # Validate single file
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            sys.exit(1)

        results = validator.validate_file(file_path)
        validator.print_file_results(results)

        if results['errors']:
            sys.exit(1)
    else:
        # Validate all files
        success = validator.validate_all(args.verbose)
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()