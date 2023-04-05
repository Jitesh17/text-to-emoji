# from gensim.summarization.summarizer import summarize

# sentence = "This is a sample sentence for TextRank implementation"
# summary = summarize(sentence, word_count=1)
# print(summary)
# from gensim.summarizer import summarize

# sentence = "This is a sample sentence for TextRank implementation"
# sentence = 'I feel sad to hear that you lost to a 5 years old kid'
# summary = summarize(sentence, word_count=1)
# print(summary)

from summa.summarizer import summarize

sentence = "This is a sample sentence for TextRank implementation"
sentence = 'I feel sad to hear that you lost to a 5 years old kid'
summary = summarize(sentence, words=1)
print(summary)
