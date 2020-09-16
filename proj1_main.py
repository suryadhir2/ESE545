#Surya Dhir, Ryan Galvankar, Ben Jacob
#ESE 545 Project 1

import numpy as np
import pandas as pd
import scipy.sparse as sci
import matplotlib.pyplot as plt
import matplotlib as mpl
import binascii
import csv
from proj1_functions import *

### Problem 1 ###
#Part 1
df=pd.read_json('amazonReviews.json', lines=True)
#Part 2
df = df.drop(columns=['asin','reviewerName', 'helpful', 'overall', 'summary', 'reviewTime', 'unixReviewTime'])
#Part 3
stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
df['reviewText_without_stopwords'] = df['reviewText'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
df = df.drop(columns=['reviewText'])
df['reviewNoPunc'] = df['reviewText_without_stopwords'].str.replace('[^\w\s]','')
df['reviewTextFinal'] = df['reviewNoPunc'].str.lower()
df = df.drop(columns=['reviewText_without_stopwords','reviewNoPunc'])

### Problem 2 ###
nrow, ncol = df.shape
k = 4
reviews = []
for text in df['reviewTextFinal']:
    k_shingles = create_k_shingles(text, k)
    reviews.append(k_shingles)
n = len(reviews)
listOfOnes = []
for i in range(0, n):
    for text in reviews[i]:
        num = shingle_index(text)
        listOfOnes.append((i,num))
sparsematrix = sci.lil_matrix((37**4, n))
n1 = len(listOfOnes)
i = 0
for tup in listOfOnes:
    i = i+1
    y = tup[0]
    x = tup[1]
    sparsematrix[x,y] = 1
    
### Problem 3 ###
n = len(reviews)
jaccards = np.zeros(10_000)
for i in range(10_000):
    pair0, pair1 = np.random.choice(n, 2, replace=False)
    k_shingles0 = reviews[pair0]
    k_shingles1 = reviews[pair1]
    jaccards[i] = len(k_shingles0.intersection(k_shingles1)) / len(k_shingles0.union(k_shingles1))
    jaccards[i] = 1 - jaccards[i]   
print(jaccards.min(), jaccards.max())
plt.hist(jaccards, bins='auto')
plt.show()

### Problem 4 ###
class mut_key(object):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def __repr__(self):
        return "'"+self.name+"'"
reviews_dict = dict()
for record in df.itertuples():
    id = record.reviewerID
    text = record.reviewTextFinal
    reviews_dict[mut_key(id)] = create_k_shingles(text, k)
    universal_set = set()
for id, shingle_set in reviews_dict.items():
    universal_set.update(shingle_set)
n_documents = len(reviews_dict)
bm = dict()
for id, shingle_set in reviews_dict.items():
    index4one = set()
    for s in shingle_set:
        h = binascii.crc32(s.encode('utf8'))
        index4one.add(h)
    bm[id] = index4one
    
### Problem 5 ###
R = n_documents
m = 24
hash_functions = [hash_r(R) for _ in range(m)]
M = dict()
for c, index4one in bm.items():
    index4one = sorted(list(index4one))
    M[c] = np.full((m,), 2**32-1, dtype=np.uint32)
    for r in index4one:
        for i in range(m):
            hf = hash_functions[i]
            M[c][i] = min(hf(r), M[c][i])
b = 4
r = 6
buckets = []
vhashes = []
for bi in range(b):
    vhashes.append(vhash(r, n_documents))
for bi in range(b):
    vh = vhashes[bi]
    bucket = dict()
    for c, sig in M.items():
        if bucket.get(vh(sig[bi*r:bi*r+r])):
            bucket[vh(sig[bi*r:bi*r+r])].add(c) 
        else:
            bucket[vh(sig[bi*r:bi*r+r])] = {c}
    buckets.append(bucket)  
pairs = set()
for bucket in buckets:
    for pair_set in bucket.values():
        if len(pair_set) > 1:
            pairs.update(combinations(pair_set, 2))
t = tuple(pairs)
l_n = []
for i in range(len(pairs)):
    l_p = []
    for x in t[i]:
        l_p.append(x)
        l_p.append(reviews_dict[x])
    s1 = l_p[1]
    s2 = l_p[3]
    s_2 = len(s1.union(s2))
    s_1 = len(s1.intersection(s2))
    if s_2 != 0:
        similarity = s_1 / s_2
        distance = 1 - similarity
    else:
        distance = 1
    l_p.append(distance)
    l_n.append(l_p) 
with open("similars.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(pairs)
    
### Part 6 ####

### Set givenId = reviewID
#givenId = 'AAD2HTJR5QKYI'
shingle_set = create_k_shingles(givenId, k)
index4one = set()
for s in shingle_set:
    h = binascii.crc32(s.encode('utf8'))
    index4one.add(h)
index4one = sorted(list(index4one))
sig = np.full((m,), 2**32-1, dtype=np.uint32)
for rr in index4one:
    for i in range(m):
        hf = hash_functions[i]
        sig[i] = min(hf(rr), sig[i])
neighbors = []     
for bi in range(b):
    vh = vhashes[bi]
    bucket = buckets[bi]
    if bucket.get(vh(sig[bi*r:bi*r+r])):
        neighbors.append(bucket.get(vh(sig[bi*r:bi*r+r])))
dist_check = []
for i in range(len(neighbors)):
    sim = len(shingle_set.intersection(neighbors[i])) / len(shingle_set.union(neighbors[i]))
    dist = 1 - sim
    dist_check.append(dist)
x = min(float(s) for s in dist_check)
neighbor_index = dist_check.index(x)
resultId = neighbors[neighbor_index]
print(resultId)