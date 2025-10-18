#!/bin/bash
# Master Documentation Validation Script
# Runs all documentation validation checks

echo "🔍 Running comprehensive documentation validation..."
echo "=================================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

VALIDATION_PASSED=true
TOTAL_CHECKS=0
PASSED_CHECKS=0

# Function to run a validation check
run_check() {
    local check_name="$1"
    local check_command="$2"
    local required="$3"  # "required" or "optional"
    
    echo "🔍 $check_name"
    echo "----------------------------------------"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if eval "$check_command"; then
        echo "✅ $check_name: PASSED"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo "❌ $check_name: FAILED"
        if [ "$required" = "required" ]; then
            VALIDATION_PASSED=false
        else
            echo "⚠️  This is an optional check - continuing..."
        fi
    fi
    
    echo ""
}

# Check if documentation directory exists
if [ ! -d "docs/HLA" ]; then
    echo "❌ Documentation directory 'docs/HLA' not found!"
    echo "Please run this script from the project root directory."
    exit 1
fi

echo "📁 Project root: $PROJECT_ROOT"
echo "📚 Documentation path: docs/HLA"
echo ""

# 1. Link Validation
run_check "Link Validation" \
    "bash '$SCRIPT_DIR/validate_links.sh'" \
    "required"

# 2. Code Example Validation
run_check "Code Example Validation" \
    "python3 '$SCRIPT_DIR/validate_code_examples.py'" \
    "required"

# 3. Implementation Status Check
run_check "Implementation Status Analysis" \
    "python3 '$SCRIPT_DIR/update_implementation_status.py' --dry-run" \
    "optional"

# 4. File Structure Validation
run_check "Documentation Structure Validation" \
    "bash '$SCRIPT_DIR/validate_structure.sh'" \
    "required"

# 5. Markdown Syntax Validation (if markdownlint is available)
if command -v markdownlint >/dev/null 2>&1; then
    run_check "Markdown Syntax Validation" \
        "markdownlint docs/HLA --config '$SCRIPT_DIR/.markdownlint.json'" \
        "optional"
else
    echo "ℹ️  Markdown Syntax Validation: SKIPPED (markdownlint not installed)"
    echo ""
fi

# 6. Spelling Check (if aspell is available)
if command -v aspell >/dev/null 2>&1; then
    run_check "Spelling Check" \
        "bash '$SCRIPT_DIR/spell_check.sh'" \
        "optional"
else
    echo "ℹ️  Spelling Check: SKIPPED (aspell not installed)"
    echo ""
fi

# Summary
echo "=================================================="
echo "📊 Validation Summary"
echo "=================================================="
echo "Total checks run: $TOTAL_CHECKS"
echo "Checks passed: $PASSED_CHECKS"
echo "Checks failed: $((TOTAL_CHECKS - PASSED_CHECKS))"
echo ""

if [ "$VALIDATION_PASSED" = true ]; then
    echo "✅ All required validations PASSED!"
    echo ""
    echo "🎉 Documentation quality: EXCELLENT"
    echo ""
    echo "📋 Optional improvements:"
    echo "• Install markdownlint for markdown syntax checking"
    echo "• Install aspell for spell checking"
    echo "• Run implementation status updates if needed"
    
    exit 0
else
    echo "❌ Some required validations FAILED!"
    echo ""
    echo "🔧 To fix validation issues:"
    echo "1. Review the failed checks above"
    echo "2. Fix the identified issues"
    echo "3. Run this script again to verify fixes"
    echo "4. Consider running individual validation scripts for detailed output"
    echo ""
    echo "📋 Individual validation scripts:"
    echo "• Link validation: ./scripts/docs/validate_links.sh"
    echo "• Code validation: ./scripts/docs/validate_code_examples.py"
    echo "• Status update: ./scripts/docs/update_implementation_status.py"
    
    exit 1
fi