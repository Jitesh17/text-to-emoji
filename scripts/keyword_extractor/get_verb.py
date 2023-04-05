import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

def get_verb(sentence):
    # Tokenize the sentence
    tokens = word_tokenize(sentence)
    
    # Tag the tokens with their part of speech
    tagged_tokens = pos_tag(tokens)
    
    # Find the first verb in the tagged tokens
    for token in tagged_tokens:
        if token[1].startswith('VB'):
            return token[0]
    
    # If no verb is found, return None
    return None

# Example usage
sentence = "The cat sat on the mat."
verb = get_verb(sentence)
print(verb) # Output: sat

from textblob import TextBlob

sentence = "I feel so happy that I got the job!"

# Use TextBlob's sentiment_assessments property to get emotion scores
blob = TextBlob(sentence)
emotion_scores = blob.sentiment_assessments
print(emotion_scores)
# Classify the emotion based on the highest score
emotion = max(emotion_scores, key=emotion_scores.get)

print("The sentence is classified as:", emotion)
