# Documentation Maintenance Scripts

This directory contains automated scripts for maintaining the High-Level Architecture documentation.

## Scripts Overview

### Core Validation Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| **validate_all.sh** | Master validation script - runs all checks | `bash validate_all.sh` |
| **validate_links.sh** | Validates all internal documentation links | `bash validate_links.sh` |
| **validate_code_examples.py** | Validates Python code examples for syntax | `python validate_code_examples.py` |
| **validate_structure.sh** | Validates documentation file structure | `bash validate_structure.sh` |

### Maintenance Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| **update_doc_versions.sh** | Updates version numbers across all docs | `bash update_doc_versions.sh 1.4.0` |
| **update_implementation_status.py** | Updates module implementation status | `python update_implementation_status.py` |

## Quick Start

### Run All Validations
```bash
# From project root directory
bash scripts/docs/validate_all.sh
```

### Update Documentation Version
```bash
# Update to version 1.4.0
bash scripts/docs/update_doc_versions.sh 1.4.0
```

### Validate Specific Aspects
```bash
# Check links only
bash scripts/docs/validate_links.sh

# Check code examples only
python scripts/docs/validate_code_examples.py

# Check structure only
bash scripts/docs/validate_structure.sh

# Update implementation status
python scripts/docs/update_implementation_status.py
```

## Script Details

### validate_all.sh
**Purpose**: Comprehensive validation of all documentation aspects
**Features**:
- Link validation
- Code example syntax checking
- Implementation status analysis
- Documentation structure validation
- Optional markdown linting and spell checking

**Exit Codes**:
- `0`: All validations passed
- `1`: One or more validations failed

### validate_links.sh
**Purpose**: Validates all internal markdown links and diagram references
**Features**:
- Checks internal `.md` file links
- Validates diagram `.mmd` file references
- Checks section anchor links
- Reports broken links with file and line numbers

**Output**: Detailed report of broken links with suggestions for fixes

### validate_code_examples.py
**Purpose**: Validates Python code examples in documentation
**Features**:
- Syntax validation of Python code blocks
- Intelligent skipping of comments, imports, and configuration examples
- Detailed error reporting with line numbers
- Support for single file or directory validation

**Options**:
```bash
python validate_code_examples.py --help
python validate_code_examples.py --verbose
python validate_code_examples.py --file docs/HLA/modules/data-management.md
```

### validate_structure.sh
**Purpose**: Validates documentation file and directory structure
**Features**:
- Checks for required files and directories
- Validates section presence in key documents
- Reports missing components
- Provides statistics on documentation coverage

### update_doc_versions.sh
**Purpose**: Updates version numbers across all documentation
**Features**:
- Updates version numbers in all markdown files
- Updates "Last Updated" dates
- Creates backup files during update
- Provides git commit suggestions

**Usage**:
```bash
bash update_doc_versions.sh 1.4.0
```

### update_implementation_status.py
**Purpose**: Analyzes codebase and updates implementation status indicators
**Features**:
- Analyzes actual code implementation
- Updates status indicators (✅🔄📋) in documentation
- Provides detailed analysis report
- Supports dry-run mode for analysis only

**Options**:
```bash
python update_implementation_status.py --help
python update_implementation_status.py --dry-run
python update_implementation_status.py --src-path src --docs-path docs/HLA
```

## Windows Usage

Since these scripts are designed for Unix-like systems, Windows users should:

### Option 1: Use Git Bash or WSL
```bash
# In Git Bash or WSL
bash scripts/docs/validate_all.sh
```

### Option 2: Use PowerShell for Python Scripts
```powershell
# In PowerShell
python scripts/docs/validate_code_examples.py
python scripts/docs/update_implementation_status.py
```

### Option 3: Manual Execution
For shell scripts on Windows, you can execute the commands manually or use the Windows Subsystem for Linux (WSL).

## Integration with Development Workflow

### Pre-commit Validation
Add to your pre-commit workflow:
```bash
# Validate documentation before committing
bash scripts/docs/validate_all.sh
```

### CI/CD Integration
Add to your CI pipeline:
```yaml
# Example GitHub Actions step
- name: Validate Documentation
  run: |
    bash scripts/docs/validate_all.sh
```

### Release Process
Include in release checklist:
1. Update version: `bash scripts/docs/update_doc_versions.sh <version>`
2. Update status: `python scripts/docs/update_implementation_status.py`
3. Validate all: `bash scripts/docs/validate_all.sh`
4. Commit changes: `git add docs/ && git commit -m "docs: update for release <version>"`

## Troubleshooting

### Common Issues

**Permission Denied (Unix/Linux)**:
```bash
chmod +x scripts/docs/*.sh
```

**Python Module Not Found**:
```bash
# Ensure you're in the project root and virtual environment is activated
cd /path/to/project
source .venv/bin/activate  # or activate.bat on Windows
```

**Bash Not Found (Windows)**:
- Install Git Bash or use WSL
- Alternatively, use PowerShell for Python scripts only

**Link Validation False Positives**:
- Check file paths are correct (case-sensitive on Unix)
- Ensure relative paths are properly formatted
- Verify target files exist

### Getting Help

**Script Help**:
```bash
python scripts/docs/validate_code_examples.py --help
python scripts/docs/update_implementation_status.py --help
```

**Verbose Output**:
```bash
python scripts/docs/validate_code_examples.py --verbose
```

**Debug Mode**:
```bash
bash -x scripts/docs/validate_links.sh  # Debug shell script
```

## Maintenance Schedule

### Automated (CI/CD)
- **Every commit**: Link validation
- **Every PR**: Full validation suite
- **Every release**: Version updates and status updates

### Manual
- **Weekly**: Run full validation suite
- **Monthly**: Update implementation status
- **Quarterly**: Review and improve scripts
- **Per release**: Update versions and validate all

## Contributing

### Adding New Validation Scripts
1. Follow the naming convention: `validate_<aspect>.sh` or `validate_<aspect>.py`
2. Include help text and error handling
3. Add to `validate_all.sh` master script
4. Update this README with script documentation
5. Test on both Unix and Windows environments

### Script Requirements
- **Error Handling**: Proper exit codes and error messages
- **Help Text**: Usage instructions and examples
- **Logging**: Clear output with status indicators (✅❌⚠️)
- **Cross-platform**: Consider Windows compatibility where possible

---

**Scripts Version**: 1.0.0  
**Last Updated**: January 18, 2025  
**Maintained By**: Documentation Team