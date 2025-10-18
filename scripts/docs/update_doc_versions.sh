#!/bin/bash
# Documentation Version Update Script
# Updates version information across all HLA documentation

NEW_VERSION=$1
if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <new_version>"
    echo "Example: $0 1.4.0"
    exit 1
fi

echo "Updating documentation versions to $NEW_VERSION..."

# Update version in main README
if [ -f "docs/HLA/README.md" ]; then
    sed -i.bak "s/Current Version: [0-9]\+\.[0-9]\+\.[0-9]\+/Current Version: $NEW_VERSION/g" docs/HLA/README.md
    sed -i.bak "s/Version: [0-9]\+\.[0-9]\+\.[0-9]\+/Version: $NEW_VERSION/g" docs/HLA/README.md
    echo "✅ Updated main README version"
fi

# Update version in all module documents
find docs/HLA/modules -name "*.md" -exec sed -i.bak "s/Version: [0-9]\+\.[0-9]\+\.[0-9]\+/Version: $NEW_VERSION/g" {} \;
echo "✅ Updated module documentation versions"

# Update version in supporting documents
find docs/HLA -name "*.md" -not -path "*/modules/*" -not -name "README.md" -exec sed -i.bak "s/Version: [0-9]\+\.[0-9]\+\.[0-9]\+/Version: $NEW_VERSION/g" {} \;
echo "✅ Updated supporting document versions"

# Update last updated date
CURRENT_DATE=$(date +"%B %d, %Y")
find docs/HLA -name "*.md" -exec sed -i.bak "s/Last Updated: .*/Last Updated: $CURRENT_DATE/g" {} \;
echo "✅ Updated last modified dates"

# Clean up backup files
find docs/HLA -name "*.bak" -delete
echo "✅ Cleaned up backup files"

# Update INDEX.md version
if [ -f "docs/HLA/INDEX.md" ]; then
    sed -i.bak "s/Index Version: [0-9]\+\.[0-9]\+\.[0-9]\+/Index Version: $NEW_VERSION/g" docs/HLA/INDEX.md
    rm -f docs/HLA/INDEX.md.bak
    echo "✅ Updated INDEX version"
fi

echo ""
echo "🎉 Documentation versions successfully updated to $NEW_VERSION"
echo "📅 Last updated date set to: $CURRENT_DATE"
echo ""
echo "Next steps:"
echo "1. Review the changes: git diff docs/HLA/"
echo "2. Commit the changes: git add docs/HLA/ && git commit -m 'docs: update version to $NEW_VERSION'"
echo "3. Validate links: ./scripts/docs/validate_links.sh"