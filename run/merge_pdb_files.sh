#!/bin/bash

# Check for argument
if [ -z "$1" ]; then
  echo "Usage: $0 <filename_prefix>"
  exit 1
fi

# Run another script with the argument
bash pdb-parse.sh "$1"

# Set file variables properly (no spaces around =, use proper string concatenation)
PDB_File1="${1}.pdb"
PDB_File2="Plane_${1}.pdb"
OutputFile="${1}withPlane.pdb"

# Combine the files
cat "$PDB_File1" "$PDB_File2" > "$OutputFile"

