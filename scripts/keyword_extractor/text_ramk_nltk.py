import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
nltk.download('wordnet')
def textrank_summarize(sentence):
    # Extract the emotion word from the sentence using regular expressions
    regex = r'\b(happy|sad|angry|fearful)\b'
    match = re.search(regex, sentence.lower())
    if match:
        return match.group(1)

    # If there is no emotion word, extract the keyword using TextRank algorithm
    words = word_tokenize(sentence.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, pos='v') for word in words]
    co_occurrences = defaultdict(int)
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            key = (words[i], words[j])
            co_occurrences[key] += 1
    scores = defaultdict(float)
    damping_factor = 0.85
    iterations = 100
    for i in range(iterations):
        for key in co_occurrences:
            score = 0.0
            for other_word in co_occurrences:
                if key[0] == other_word[1]:
                    score += (co_occurrences[other_word] / sum(co_occurrences[(other_word[0], w)] for w in words))
            scores[key[1]] = (1 - damping_factor) + damping_factor * score
    keywords = [word for word in scores.keys() if nltk.pos_tag([word])[0][1].startswith('VB')]
    if keywords:
        return max(keywords, key=lambda word: scores[word])

    # If there are no keywords or emotion words, return None
    return None

sentence = "I am feeling bad and lonely today."
emotion_word = textrank_summarize(sentence)
print(emotion_word)
