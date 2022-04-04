The Cranfield text collection has been used as an input for tokenizing and stemming.
Only the contents of the <TEXT> tags have been used as instructed.

All other details of the program 


Following packages have been used:
-----------------------------------
glob
re
string
time
sys
os
operator
xml.dom.minidom
nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
import ast


Steps needed to run on CS1 Grads server(UTD Unix Machine)
=========================================================

Installation of dependencies:
------------------------------
pip3.6 install --user nltk

Code Execution:
--------------
python3.6 indexer.py /people/cs/s/sanda/cs6322/resourcesIR



