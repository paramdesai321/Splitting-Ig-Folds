#!/usr/bin/env bash
#
# check_empty.sh — recursively list all zero-byte files in a directory
#
# Usage:
#   ./check_empty.sh [TARGET_DIR]
# Examples:
#   ./check_empty.sh            # scan current directory
#   ./check_empty.sh /path/to/dir

set -euo pipefail

# If no argument given, default to current directory
TARGET_DIR="${1:-.}"

# Verify target exists and is a directory
if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: '$TARGET_DIR' is not a directory." >&2
  exit 1
fi

echo "Scanning '$TARGET_DIR' for empty files..."
echo

# Method A: using -empty (BSD & GNU)
#find "$TARGET_DIR" -type f -empty -print
find "$TARGET_DIR" -type f -size -200c -print

# (you can comment out Method A and uncomment Method B below if your find
#  doesn’t support -empty)

#: <<'METHOD_B'
# Method B: match by exact size (portable)
# find "$TARGET_DIR" -type f -size 0c -print
#: METHOD_B

echo
echo "Done."

