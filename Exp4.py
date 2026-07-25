import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

docs = []

# Input documents
n = int(input("Enter number of documents: "))

for i in range(n):
    docs.append(input("Enter document: "))

# Input query
query = input("\nEnter search query: ")

# ---------------- TF-IDF ----------------

vectorizer = TfidfVectorizer()

# Convert documents into TF-IDF vectors
X = vectorizer.fit_transform(docs)

# Convert query into TF-IDF vector
query_vec = vectorizer.transform([query])

# Calculate cosine similarity
scores = cosine_similarity(query_vec, X)

print("\nTF-IDF Similarity Scores:")

for i, s in enumerate(scores[0]):
    print("Document", i+1, ":", round(s, 3))


# ---------------- LSA ----------------

# Reduce dimensions using SVD
svd = TruncatedSVD(n_components=2)

X_lsa = svd.fit_transform(X)

query_lsa = svd.transform(query_vec)

# Calculate similarity after LSA
lsa_scores = cosine_similarity(query_lsa, X_lsa)

print("\nLSA Similarity Scores:")

for i, s in enumerate(lsa_scores[0]):
    print("Document", i+1, ":", round(s, 3))


# Find best matching document
best = np.argmax(lsa_scores)

print("\nMost Relevant Document:")
print(docs[best])