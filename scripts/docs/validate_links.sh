#!/bin/bash
# Documentation Link Validation Script
# Validates all internal links in HLA documentation

echo "🔍 Validating documentation links..."
echo "=================================="

BROKEN_LINKS=0
TOTAL_LINKS=0

# Function to check if a file exists
check_file_exists() {
    local file_path=$1
    if [ -f "$file_path" ]; then
        return 0
    else
        return 1
    fi
}

# Function to resolve relative path
resolve_path() {
    local base_dir=$1
    local link_path=$2
    
    if [[ $link_path == /* ]]; then
        # Absolute path
        echo "$link_path"
    elif [[ $link_path == ../* ]]; then
        # Parent directory reference
        echo "$(dirname "$base_dir")/$link_path"
    else
        # Same directory or subdirectory
        echo "$base_dir/$link_path"
    fi
}

# Check internal markdown links
echo "📄 Checking internal markdown links..."
find docs/HLA -name "*.md" -type f | while read file; do
    base_dir=$(dirname "$file")
    
    # Extract markdown links [text](link.md)
    grep -n "\[.*\](.*\.md)" "$file" | while IFS=: read line_num link_line; do
        # Extract the link path
        link_path=$(echo "$link_line" | sed -n 's/.*\[.*\](\([^)]*\.md\)).*/\1/p')
        
        if [ -n "$link_path" ]; then
            TOTAL_LINKS=$((TOTAL_LINKS + 1))
            
            # Resolve the full path
            if [[ $link_path == ../* ]]; then
                # Handle relative paths going up
                resolved_path=$(resolve_path "$base_dir" "$link_path")
            else
                # Handle same directory or subdirectory paths
                resolved_path="$base_dir/$link_path"
            fi
            
            # Normalize path (remove ./ and resolve ../)
            resolved_path=$(realpath -m "$resolved_path" 2>/dev/null || echo "$resolved_path")
            
            if ! check_file_exists "$resolved_path"; then
                echo "❌ BROKEN LINK in $file:$line_num"
                echo "   Link: $link_path"
                echo "   Resolved to: $resolved_path"
                echo "   Context: $(echo "$link_line" | sed 's/^[[:space:]]*//')"
                echo ""
                BROKEN_LINKS=$((BROKEN_LINKS + 1))
            fi
        fi
    done
done

# Check diagram references
echo "📊 Checking diagram references..."
find docs/HLA -name "*.md" -type f | while read file; do
    base_dir=$(dirname "$file")
    
    # Extract diagram references [text](diagrams/diagram.mmd)
    grep -n "\[.*\](.*\.mmd)" "$file" | while IFS=: read line_num link_line; do
        link_path=$(echo "$link_line" | sed -n 's/.*\[.*\](\([^)]*\.mmd\)).*/\1/p')
        
        if [ -n "$link_path" ]; then
            TOTAL_LINKS=$((TOTAL_LINKS + 1))
            
            # Resolve the full path
            resolved_path=$(resolve_path "$base_dir" "$link_path")
            resolved_path=$(realpath -m "$resolved_path" 2>/dev/null || echo "$resolved_path")
            
            if ! check_file_exists "$resolved_path"; then
                echo "❌ BROKEN DIAGRAM LINK in $file:$line_num"
                echo "   Link: $link_path"
                echo "   Resolved to: $resolved_path"
                echo ""
                BROKEN_LINKS=$((BROKEN_LINKS + 1))
            fi
        fi
    done
done

# Check section references within files
echo "🔗 Checking section references..."
find docs/HLA -name "*.md" -type f | while read file; do
    # Extract section links [text](#section)
    grep -n "\[.*\](#[^)]*)" "$file" | while IFS=: read line_num link_line; do
        section_link=$(echo "$link_line" | sed -n 's/.*\[.*\](#\([^)]*\)).*/\1/p')
        
        if [ -n "$section_link" ]; then
            TOTAL_LINKS=$((TOTAL_LINKS + 1))
            
            # Convert section link to header format (lowercase, spaces to dashes)
            expected_header=$(echo "$section_link" | tr '[:upper:]' '[:lower:]' | sed 's/-/ /g')
            
            # Check if header exists in file (flexible matching)
            if ! grep -qi "^#.*$expected_header" "$file" && ! grep -qi "^#.*$(echo "$section_link" | sed 's/-/ /g')" "$file"; then
                echo "⚠️  POTENTIAL BROKEN SECTION LINK in $file:$line_num"
                echo "   Section: #$section_link"
                echo "   Expected header pattern: $expected_header"
                echo ""
            fi
        fi
    done
done

# Summary
echo "=================================="
echo "📊 Link Validation Summary"
echo "=================================="
echo "Total links checked: $TOTAL_LINKS"
echo "Broken links found: $BROKEN_LINKS"

if [ $BROKEN_LINKS -eq 0 ]; then
    echo "✅ All links are valid!"
    exit 0
else
    echo "❌ Found $BROKEN_LINKS broken links"
    echo ""
    echo "🔧 To fix broken links:"
    echo "1. Check if the target files exist"
    echo "2. Verify the relative path is correct"
    echo "3. Ensure file names match exactly (case-sensitive)"
    echo "4. Run this script again after fixes"
    exit 1
fi