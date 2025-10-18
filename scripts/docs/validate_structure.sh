#!/bin/bash
# Documentation Structure Validation Script
# Validates that all required documentation files and structure exist

echo "🏗️ Validating documentation structure..."

DOCS_PATH="docs/HLA"
STRUCTURE_VALID=true

# Function to check if file exists
check_file() {
    local file_path="$1"
    local description="$2"
    local required="$3"  # "required" or "optional"
    
    if [ -f "$file_path" ]; then
        echo "✅ $description: $file_path"
        return 0
    else
        if [ "$required" = "required" ]; then
            echo "❌ MISSING REQUIRED: $description: $file_path"
            STRUCTURE_VALID=false
        else
            echo "⚠️  MISSING OPTIONAL: $description: $file_path"
        fi
        return 1
    fi
}

# Function to check if directory exists
check_directory() {
    local dir_path="$1"
    local description="$2"
    local required="$3"  # "required" or "optional"
    
    if [ -d "$dir_path" ]; then
        echo "✅ $description: $dir_path/"
        return 0
    else
        if [ "$required" = "required" ]; then
            echo "❌ MISSING REQUIRED: $description: $dir_path/"
            STRUCTURE_VALID=false
        else
            echo "⚠️  MISSING OPTIONAL: $description: $dir_path/"
        fi
        return 1
    fi
}

echo "📁 Checking core documentation structure..."
echo "----------------------------------------"

# Core documentation files
check_file "$DOCS_PATH/README.md" "Main architecture overview" "required"
check_file "$DOCS_PATH/INDEX.md" "Documentation index" "required"
check_file "$DOCS_PATH/TEMPLATE.md" "Documentation template" "optional"
check_file "$DOCS_PATH/VALIDATION_REPORT.md" "Validation report" "optional"
check_file "$DOCS_PATH/MAINTENANCE_PROCEDURES.md" "Maintenance procedures" "optional"

echo ""
echo "📊 Checking diagram directory..."
echo "----------------------------------------"

# Diagrams directory and files
check_directory "$DOCS_PATH/diagrams" "Diagrams directory" "required"
if [ -d "$DOCS_PATH/diagrams" ]; then
    check_file "$DOCS_PATH/diagrams/system-overview.mmd" "System overview diagram" "required"
    check_file "$DOCS_PATH/diagrams/data-flow.mmd" "Data flow diagram" "required"
    check_file "$DOCS_PATH/diagrams/module-interactions.mmd" "Module interactions diagram" "required"
    check_file "$DOCS_PATH/diagrams/service-communication.mmd" "Service communication diagram" "optional"
    check_file "$DOCS_PATH/diagrams/api-endpoints.mmd" "API endpoints diagram" "optional"
    check_file "$DOCS_PATH/diagrams/user-scenarios.mmd" "User scenarios diagram" "optional"
fi

echo ""
echo "🏗️ Checking module documentation..."
echo "----------------------------------------"

# Module documentation directory
check_directory "$DOCS_PATH/modules" "Module documentation directory" "required"

if [ -d "$DOCS_PATH/modules" ]; then
    # Required module documentation files
    check_file "$DOCS_PATH/modules/data-management.md" "Data Management module" "required"
    check_file "$DOCS_PATH/modules/trading-engine.md" "Trading Engine module" "required"
    check_file "$DOCS_PATH/modules/ml-analytics.md" "ML & Analytics module" "required"
    check_file "$DOCS_PATH/modules/communication.md" "Communication module" "required"
    check_file "$DOCS_PATH/modules/infrastructure.md" "Infrastructure module" "required"
    check_file "$DOCS_PATH/modules/configuration.md" "Configuration module" "required"
    check_file "$DOCS_PATH/modules/security-auth.md" "Security & Auth module" "required"
fi

echo ""
echo "📋 Checking supporting documentation..."
echo "----------------------------------------"

# Supporting documentation files
check_file "$DOCS_PATH/database-architecture.md" "Database architecture" "required"
check_file "$DOCS_PATH/data-providers-sources.md" "Data providers and sources" "required"
check_file "$DOCS_PATH/background-services.md" "Background services" "optional"
check_file "$DOCS_PATH/notification-services.md" "Notification services" "optional"
check_file "$DOCS_PATH/migration-evolution.md" "Migration and evolution" "optional"
check_file "$DOCS_PATH/documentation-procedures.md" "Documentation procedures" "optional"

echo ""
echo "🔍 Checking file content requirements..."
echo "----------------------------------------"

# Check that key files have required sections
check_file_sections() {
    local file_path="$1"
    local required_sections="$2"
    local file_description="$3"
    
    if [ ! -f "$file_path" ]; then
        return 1
    fi
    
    echo "📄 Checking sections in $file_description..."
    
    IFS=',' read -ra SECTIONS <<< "$required_sections"
    local missing_sections=()
    
    for section in "${SECTIONS[@]}"; do
        section=$(echo "$section" | xargs)  # trim whitespace
        if grep -q "^#.*$section" "$file_path"; then
            echo "   ✅ Section found: $section"
        else
            echo "   ❌ Missing section: $section"
            missing_sections+=("$section")
        fi
    done
    
    if [ ${#missing_sections[@]} -eq 0 ]; then
        echo "   ✅ All required sections present"
        return 0
    else
        echo "   ❌ Missing ${#missing_sections[@]} required sections"
        STRUCTURE_VALID=false
        return 1
    fi
}

# Check main README sections
if [ -f "$DOCS_PATH/README.md" ]; then
    check_file_sections "$DOCS_PATH/README.md" \
        "System Overview,Module Architecture,Quick Navigation" \
        "Main README"
fi

# Check module documentation sections
if [ -f "$DOCS_PATH/modules/data-management.md" ]; then
    check_file_sections "$DOCS_PATH/modules/data-management.md" \
        "Purpose & Responsibilities,Key Components,Integration Points" \
        "Data Management module"
fi

echo ""
echo "📊 Structure validation summary..."
echo "----------------------------------------"

if [ "$STRUCTURE_VALID" = true ]; then
    echo "✅ Documentation structure is VALID"
    echo ""
    echo "🎉 All required files and directories are present"
    echo "📋 Structure follows HLA documentation standards"
    
    # Count files for statistics
    total_md_files=$(find "$DOCS_PATH" -name "*.md" | wc -l)
    total_diagrams=$(find "$DOCS_PATH/diagrams" -name "*.mmd" 2>/dev/null | wc -l)
    
    echo ""
    echo "📊 Documentation statistics:"
    echo "• Total markdown files: $total_md_files"
    echo "• Total diagram files: $total_diagrams"
    echo "• Module documentation files: $(find "$DOCS_PATH/modules" -name "*.md" 2>/dev/null | wc -l)"
    
    exit 0
else
    echo "❌ Documentation structure has ISSUES"
    echo ""
    echo "🔧 To fix structure issues:"
    echo "1. Create missing required files and directories"
    echo "2. Add missing sections to existing files"
    echo "3. Use the documentation templates for new files"
    echo "4. Run this script again to verify fixes"
    echo ""
    echo "📋 Templates available:"
    echo "• Module template: docs/HLA/TEMPLATE.md"
    echo "• Maintenance procedures: docs/HLA/MAINTENANCE_PROCEDURES.md"
    
    exit 1
fi