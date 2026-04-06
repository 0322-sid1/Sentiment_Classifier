"""
utils.py — Text preprocessing utilities for Sentiment Classifier
(No external NLP libraries required — pure Python + regex)
"""

import re
import string

STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","ma","just"
}

_SUFFIXES = [
    ("ies","y"),("ied","y"),("ves","f"),("ness",""),("ment",""),
    ("ing",""),("tion",""),("edly",""),("ed",""),("er",""),
    ("est",""),("ly",""),("s",""),
]

def _lemmatize(word):
    if len(word) <= 3:
        return word
    for suffix, rep in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[:len(word)-len(suffix)] + rep
    return word

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [_lemmatize(t) for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)
