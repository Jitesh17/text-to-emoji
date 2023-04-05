# %%

import numpy as np
import pandas as pd
import json
import spacy
from IPython.core.display import display, HTML

def load_emojis():
    rows = []
    with open('../data/emojis.json') as f:
        for emoji in json.loads(f.read()):
            rows.append([emoji['name'], emoji['unicode'], ' '.join(emoji['keywords']), emoji['definition']])    
    return np.array(rows)
    
emojis = load_emojis()

# %%
df = pd.DataFrame(emojis, columns=['name', 'unicode', 'keywords', 'definition'])
df.head()
# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the EmojiNet dataset
# df = pd.read_csv('emojinet.csv')

# Count the frequency of each emoji
freq = df['name'].value_counts()
print(freq)

# Plot a bar chart of emoji frequency
plt.bar(freq.index, freq.values)
plt.xlabel('Emoji')
plt.ylabel('Frequency')
plt.show()

# Compute the similarity matrix between emojis
desc = df['definition'].str.lower().values
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(desc)
similarity = cosine_similarity(tfidf)

# Visualize the similarity matrix as a heatmap
plt.imshow(similarity, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(df)), df['name'], rotation=90)
plt.yticks(range(len(df)), df['name'])
plt.show()

# # Load the Emoji Sentiment Ranking dataset
# sentiment = pd.read_csv('emoji_sentiment.csv')

# # Join the EmojiNet dataset with the sentiment dataset
# merged = df.merge(sentiment, on='unicode')

# # Compute the mean sentiment score for each emoji category
# mean_sentiment = merged.groupby('category')['sentiment_score'].mean()

# # Plot a bar chart of mean sentiment scores
# plt.bar(mean_sentiment.index, mean_sentiment.values)
# plt.xlabel('Category')
# plt.ylabel('Mean Sentiment Score')
# plt.show()

# %%
