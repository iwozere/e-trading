#!/usr/bin/env python3
"""
Implementation Status Update Script
Automatically updates implementation status indicators in HLA documentation
based on actual codebase analysis
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

class ImplementationStatusUpdater:
    def __init__(self, src_path: str = "src", docs_path: str = "docs/HLA"):
        self.src_path = Path(src_path)
        self.docs_path = Path(docs_path)
        self.module_mapping = {
            "Data Management": "src/data",
            "Trading Engine": "src/trading",
            "ML & Analytics": "src/ml",
            "Communication": ["src/telegram", "src/web_ui", "src/notification"],
            "Infrastructure": ["src/data/db", "src/scheduler", "src/error_handling"],
            "Configuration": "src/config",
            "Security & Auth": "src/web_ui/backend/auth"
        }

    def analyze_module_implementation(self, module_paths: List[str]) -> str:
        """Analyze module implementation status based on code presence and completeness."""
        if isinstance(module_paths, str):
            module_paths = [module_paths]

        total_files = 0
        total_lines = 0
        has_tests = False
        has_docs = False

        for module_path in module_paths:
            path = Path(module_path)
            if not path.exists():
                continue

            # Count Python files and lines
            py_files = list(path.rglob("*.py"))
            total_files += len(py_files)

            for py_file in py_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # Count non-empty, non-comment lines
                        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
                        total_lines += len(code_lines)
                except:
                    continue

            # Check for tests
            if (path / "tests").exists() or any("test_" in f.name for f in py_files):
                has_tests = True

            # Check for documentation
            if (path / "README.md").exists() or (path / "docs").exists():
                has_docs = True

        # Determine status based on analysis
        if total_files == 0:
            return "📋"  # Planned - no implementation
        elif total_files < 3 or total_lines < 100:
            return "🔄"  # In Progress - minimal implementation
        elif total_files >= 5 and total_lines >= 500:
            return "✅"  # Complete - substantial implementation
        else:
            return "🔄"  # In Progress - moderate implementation

    def get_current_status_from_doc(self, doc_path: Path, module_name: str) -> str:
        """Extract current status from documentation."""
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Look for module status in table format
            pattern = rf'\*\*\[{re.escape(module_name)}\].*?\|\s*([✅🔄📋])'
            match = re.search(pattern, content)
            if match:
                return match.group(1)

            # Look for status in other formats
            pattern = rf'{re.escape(module_name)}.*?([✅🔄📋])'
            match = re.search(pattern, content)
            if match:
                return match.group(1)

        except Exception as e:
            print(f"Warning: Could not read {doc_path}: {e}")

        return "❓"  # Unknown status

    def update_status_in_file(self, file_path: Path, old_status: str, new_status: str, module_name: str) -> bool:
        """Update status in a specific file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Update in table format
            pattern = rf'(\*\*\[{re.escape(module_name)}\].*?\|\s*){re.escape(old_status)}'
            new_content = re.sub(pattern, rf'\g<1>{new_status}', content)

            # Update in other formats
            pattern = rf'({re.escape(module_name)}.*?){re.escape(old_status)}'
            new_content = re.sub(pattern, rf'\g<1>{new_status}', new_content)

            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True

        except Exception as e:
            print(f"Error updating {file_path}: {e}")

        return False

    def update_last_updated_date(self, file_path: Path):
        """Update the last updated date in documentation."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            current_date = datetime.now().strftime("%B %d, %Y")
            pattern = r'Last Updated: .*'
            new_content = re.sub(pattern, f'Last Updated: {current_date}', content)

            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True

        except Exception as e:
            print(f"Error updating date in {file_path}: {e}")

        return False

    def update_all_status_indicators(self) -> Dict[str, Dict]:
        """Update all implementation status indicators."""
        print("🔄 Analyzing implementation status...")
        print("=" * 50)

        results = {}

        for module_name, module_paths in self.module_mapping.items():
            print(f"📊 Analyzing {module_name}...")

            # Analyze current implementation
            new_status = self.analyze_module_implementation(module_paths)

            # Get current status from main README
            readme_path = self.docs_path / "README.md"
            current_status = self.get_current_status_from_doc(readme_path, module_name)

            results[module_name] = {
                'current_status': current_status,
                'new_status': new_status,
                'changed': current_status != new_status,
                'module_paths': module_paths
            }

            status_text = {
                "✅": "Complete",
                "🔄": "In Progress",
                "📋": "Planned",
                "❓": "Unknown"
            }

            print(f"   Current: {current_status} ({status_text.get(current_status, 'Unknown')})")
            print(f"   Analyzed: {new_status} ({status_text.get(new_status, 'Unknown')})")

            if current_status != new_status:
                print(f"   🔄 Status change detected!")
            else:
                print(f"   ✅ Status unchanged")
            print()

        return results

    def apply_status_updates(self, results: Dict[str, Dict]) -> int:
        """Apply status updates to documentation files."""
        print("📝 Applying status updates...")
        print("=" * 50)

        updates_made = 0
        files_to_update = [
            self.docs_path / "README.md",
            self.docs_path / "INDEX.md"
        ]

        # Add module-specific documentation files
        module_files = list((self.docs_path / "modules").glob("*.md"))
        files_to_update.extend(module_files)

        for module_name, result in results.items():
            if not result['changed']:
                continue

            old_status = result['current_status']
            new_status = result['new_status']

            print(f"🔄 Updating {module_name}: {old_status} → {new_status}")

            for file_path in files_to_update:
                if not file_path.exists():
                    continue

                if self.update_status_in_file(file_path, old_status, new_status, module_name):
                    print(f"   ✅ Updated {file_path.name}")
                    updates_made += 1

                    # Update last modified date
                    self.update_last_updated_date(file_path)

        return updates_made

    def generate_status_report(self, results: Dict[str, Dict]):
        """Generate a status report."""
        print("=" * 50)
        print("📊 Implementation Status Report")
        print("=" * 50)

        status_counts = {"✅": 0, "🔄": 0, "📋": 0}
        changed_modules = []

        for module_name, result in results.items():
            status = result['new_status']
            if status in status_counts:
                status_counts[status] += 1

            if result['changed']:
                changed_modules.append(module_name)

        print(f"Module Status Distribution:")
        print(f"  ✅ Complete: {status_counts['✅']} modules")
        print(f"  🔄 In Progress: {status_counts['🔄']} modules")
        print(f"  📋 Planned: {status_counts['📋']} modules")
        print()

        if changed_modules:
            print(f"Status Changes Detected: {len(changed_modules)} modules")
            for module in changed_modules:
                result = results[module]
                print(f"  • {module}: {result['current_status']} → {result['new_status']}")
        else:
            print("No status changes detected - all modules are up to date")

        print()

        # Calculate completion percentage
        total_modules = len(results)
        complete_modules = status_counts['✅']
        completion_percentage = (complete_modules / total_modules) * 100 if total_modules > 0 else 0

        print(f"Overall Completion: {completion_percentage:.1f}% ({complete_modules}/{total_modules} modules)")

        if completion_percentage >= 80:
            print("🎉 System is highly mature with most modules complete!")
        elif completion_percentage >= 60:
            print("🚀 System is well-developed with good module coverage")
        elif completion_percentage >= 40:
            print("🔄 System is actively developing with moderate coverage")
        else:
            print("🏗️ System is in early development phase")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Update implementation status indicators in HLA documentation"
    )
    parser.add_argument(
        "--src-path",
        default="src",
        help="Path to source code directory (default: src)"
    )
    parser.add_argument(
        "--docs-path",
        default="docs/HLA",
        help="Path to documentation directory (default: docs/HLA)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze status but don't update files"
    )

    args = parser.parse_args()

    # Verify paths exist
    if not Path(args.src_path).exists():
        print(f"❌ Source path not found: {args.src_path}")
        sys.exit(1)

    if not Path(args.docs_path).exists():
        print(f"❌ Documentation path not found: {args.docs_path}")
        sys.exit(1)

    updater = ImplementationStatusUpdater(args.src_path, args.docs_path)

    # Analyze current status
    results = updater.update_all_status_indicators()

    # Apply updates unless dry run
    if not args.dry_run:
        updates_made = updater.apply_status_updates(results)
        print(f"📝 Made {updates_made} status updates")
        print()
    else:
        print("🔍 Dry run mode - no files were modified")
        print()

    # Generate report
    updater.generate_status_report(results)

    # Suggest next steps
    changed_count = sum(1 for r in results.values() if r['changed'])
    if changed_count > 0 and not args.dry_run:
        print()
        print("🔧 Next steps:")
        print("1. Review the changes: git diff docs/HLA/")
        print("2. Commit the changes: git add docs/HLA/ && git commit -m 'docs: update implementation status'")
        print("3. Validate documentation: ./scripts/docs/validate_links.sh")

if __name__ == "__main__":
    main()