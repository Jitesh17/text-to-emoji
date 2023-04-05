# %%

# Load the T5 model and tokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=512)
# %%
# Define a function to preprocess the input sentence
def preprocess(text):
#   text = f'extract emotion from:  {text} '
  text = "extract verb from the text:  " + text + " "
  input_ids = tokenizer.encode(text, return_tensors='pt')
  return input_ids

# Define a function to extract the verb or emotion keyword from the T5 model's output
def extract_keyword(output):
  output = tokenizer.decode(output[0], skip_special_tokens=True)
  return output.split()[-1]

# Call the T5 model on the preprocessed input sentence and extract the keyword from the output using the extraction function
def extract_verb_or_emotion(text):
  input_ids = preprocess(text)
#   output = model.generate(input_ids, min_length=1, length_penalty=2.0, num_beams=4, early_stopping=True, do_sample=True, max_new_tokens=20)
  output = model.generate(input_ids, min_length=1, num_beams=4, max_new_tokens=10)


  keyword = extract_keyword(output)
  return keyword


# Example usage
text = "I am  happy "
# text = "I am running to the moon"
keyword = extract_verb_or_emotion(text)
print(f'keyword={keyword}') # Output: "happy"

# %%
