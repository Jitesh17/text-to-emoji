import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# Load pre-trained transformer model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define function to classify sentence into an emotion category
def classify_sentiment(sentence):
    inputs = tokenizer.encode_plus(sentence, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1)
    label_map = {0: "negative", 1: "positive"}
    return label_map[predicted_class.item()]

# Define function to extract keywords using TextRank algorithm
def textrank_summarize(sentence, emotion="negative"):
    # Classify sentence into emotion category
    sentiment = classify_sentiment(sentence)
    if sentiment != emotion:
        return None
    
    # Extract keywords using TextRank algorithm
    sentences = sent_tokenize(sentence)
    words = defaultdict(int)
    for sentence in sentences:
        # Preprocess sentence
        sentence = sentence.lower()
        sentence = re.sub(r'[^\w\s]','',sentence)
        tokens = word_tokenize(sentence)
        tokens = [token for token in tokens if token not in stopwords.words('english')]
        tokens = [WordNetLemmatizer().lemmatize(token, pos='v') for token in tokens if len(token) > 2]
        # Compute word frequency
        for token in tokens:
            words[token] += 1
    # Compute TextRank score for each word
    ranks = defaultdict(int)
    for _ in range(5):
        for word in words:
            rank = 0
            for other_word in words:
                if other_word != word:
                    sim = nltk.jaccard_distance(set(word), set(other_word))
                    rank += (sim * words[other_word]) / sum(words.values())
            ranks[word] = 0.15 + 0.85 * rank
        words = ranks.copy()
    # Sort words by score and return top keyword
    keywords = sorted(words.items(), key=lambda x: x[1], reverse=True)
    if keywords:
        return keywords[0][0]
    else:
        return None

# Example usage
sentence = "I am happy and lonely today."
keyword = textrank_summarize(sentence, emotion="positive")
print(keyword)
