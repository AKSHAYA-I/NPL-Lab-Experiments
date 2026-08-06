import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

reviews = []

n = int(input("Enter number of reviews: "))

for i in range(n):
    reviews.append(input("Enter review: "))

# Convert text into document-term matrix
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews)

# Apply LDA
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

words = vectorizer.get_feature_names_out()

print("\nTopics:")
for i, topic in enumerate(lda.components_):
    print("\nTopic", i + 1)
    top_words = topic.argsort()[-5:]
    for j in top_words:
        print(words[j])

# Convert sparse matrix to dense
X_dense = X.toarray()

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=2)
X_tsne = tsne.fit_transform(X_dense)

print("\nt-SNE Coordinates:")
for i, point in enumerate(X_tsne):
    print("Review", i + 1, ":", point)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

for i in range(len(reviews)):
    plt.text(X_tsne[i, 0], X_tsne[i, 1], "R" + str(i + 1))

plt.title("t-SNE Visualization of Customer Reviews")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()