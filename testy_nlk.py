
import nltk
sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""
import os
os.chdir('E:\Wojtek\_DSCN_\Python_libraries')
tokens = nltk.word_tokenize(sentence)
os.chdir('E:\Wojtek\_DSCN_\Python_libraries')
tagged = nltk.pos_tag(tokens)
