#!/usr/bin/bash

echo Enter Protein Identification Number: 
read PIN


 curl -O "https://files.rcsb.org/download/$PIN.pdb"

cat <<EOF
--------------------------
ATTENTION: $PIN.pdb is now dowloaded 
-------------------------
EOF
python3 parsing.py "$PIN"

cat <<EOF
----------------------------------------------------------------
ATTENTION: open ATOMlines$PIN.txt to access the ATOM lines from for $PIN.pdb 
-----------------------------------------------------------------
EOF
