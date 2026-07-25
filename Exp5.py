import nltk
from nltk import word_tokenize, pos_tag

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Input legal text
text = input("Enter legal text: ")

# Tokenization
tokens = word_tokenize(text)

# POS tagging
tags = pos_tag(tokens)

print("\nDetected Named Entities:")

count = 0

# Detect proper nouns
for word, tag in tags:
    if tag == "NNP":
        print(word, "-> ENTITY")
        count += 1


# Get actual number of entities
actual = int(input("\nEnter actual number of entities: "))

# Calculate accuracy
accuracy = (min(count, actual) / max(count, actual)) * 100

print("\nPredicted Entities:", count)
print("NER Accuracy:", round(accuracy, 2), "%")