import nltk
from rake_nltk import Rake
nltk.download('stopwords')
sentence = "This is a sample sentence for RAKE implementation"
sentence = 'I feel sad to hear that you lost to a 5 years old kid'
r = Rake()
r.extract_keywords_from_text(sentence)
keywords = r.get_ranked_phrases()
print(keywords[:1])
