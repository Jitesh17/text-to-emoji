from sklearn.feature_extraction.text import TfidfVectorizer

sentence = "This is a sample sentence for TF-IDF implementation"
sentence = 'I feel sad to hear that you lost to a 5 years old kid'
vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit_transform([sentence])
print(vectorizer.get_feature_names_out())

