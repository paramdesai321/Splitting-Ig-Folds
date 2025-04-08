#!/bin/bash
echo Enter the PIN
read PIN

cat << EOF
------------------
ATTENTION: $PIN.pdb is now dowloaded 
------------------
EOF
python3 parsing_BCEF.py "$PIN"

cat << EOF
-----------------
Extracting Backbone Atoms 
ATTENTION: Backbone File is in the directory Backbone
EOF
python3 CA_C_N_parsing.py "$PIN"
cat << EOF
----------------
Labelling Strands 
ATTENTION: The modified version with labels of B,C,E,F Strands is in directory Labeled_Strands
---------------
EOF
python3 steve_parser.py "$PIN"

cat << EOF
----------------
Creating Class Labels
ATTENTION: The Labeled files is in Class_Labels directory
----------------
EOF
python3 labels.py "$PIN"

cat << EOF
---------------
Running SVM
--------------
EOF

#python3 svm_from_scratch.py "$PIN"
python3 sklearn_svm.py "$PIN"


