from transformers import pipeline
import numpy as np

# Define the list of words to match against
word_list = ["puppy", "banana", "love", "pee"]

# Define the sentence to match
sentence = "I like to eat fruit for breakfast"

# Load a pre-trained language model
model = pipeline("feature-extraction", model="bert-base-uncased")

# Encode the sentence and the words in the list
encoded_sentence = np.mean(model(sentence)[0], axis=0)
encoded_words = [np.mean(model(word)[0], axis=0) for word in word_list]

# Compute the similarity between the sentence and each word in the list
similarities = [np.dot(encoded_sentence, encoded_word) / (np.linalg.norm(encoded_sentence) * np.linalg.norm(encoded_word)) for encoded_word in encoded_words]

# Find the word in the list with the highest similarity to the sentence
closest_word = word_list[similarities.index(max(similarities))]

print("The word closest to the sentence is:", similarities)
print("The word closest to the sentence is:", closest_word)
