'''

@author: liuxing
'''
from nltk.corpus import gutenberg
from nltk import FreqDist
fd=FreqDist()
for word in gutenberg.words('austen-persuation.txt')