#!/bin/bash
set -euo pipefail

# Pinned luna commit. Bump this and tools/luna.patch in sync.
LUNA_SHA="1ddb4c9502b196e8db1a23dd3e7d8f766703782c"
LUNA_SHORT_SHA="${LUNA_SHA:0:8}"

script_dir="$(cd "$(dirname "$0")" && pwd)"
luna_dir="$script_dir/luna-$LUNA_SHORT_SHA"
patch_file="$script_dir/luna.patch"
sentinel="$luna_dir/.patched"

if [ -f "$sentinel" ]; then
    echo "luna-$LUNA_SHORT_SHA already cloned and patched"
    exit 0
fi

if [ ! -d "$luna_dir/.git" ]; then
    echo "Cloning luna at $LUNA_SHA into $luna_dir"
    git clone https://github.com/ai-ar-research/luna.git "$luna_dir"
    git -C "$luna_dir" checkout "$LUNA_SHA"
fi

echo "Applying $patch_file"
git -C "$luna_dir" apply "$patch_file"

touch "$sentinel"
echo "luna ready at $luna_dir"
