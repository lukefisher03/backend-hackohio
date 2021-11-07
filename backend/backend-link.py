from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from starlette.routing import Host
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from fastapi import FastAPI
import nltk
from nltk.tokenize import word_tokenize
from pydantic import BaseModel
import uvicorn

glove_input_file = 'glove.6B.200d.txt'
word2vec_output_file = 'glove.6B.200d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)
filename = 'glove.6B.200d.txt.word2vec'
glove_model = KeyedVectors.load_word2vec_format(filename, binary=False)

def word_to_glove_vector(model, word):
    try:
        return glove_model.get_vector(word)
    except:
        return np.random.uniform(-1, 1, size=200)


nltk.download('punkt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset = pd.read_csv("modified_test_dataset.csv")
dataset.head()

import regex as re

contractions_dict = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
def expand_contractions(s, contractions_dict=contractions_dict):
     def replace(match):
        return contractions_dict[match.group(0)]
     return contractions_re.sub(replace, s)

def tokenize(sentence):
    return [t.lower() for t in word_tokenize(expand_contractions(sentence))]


def to_tensor(sentence):
    arr = [word_to_glove_vector(glove_model, t) for t in sentence]
    return torch.Tensor(arr) 

class InputSentence(BaseModel):
    input_sentence: str


app = FastAPI()

@app.post("/")
async def create_output(imp:InputSentence):
    print(imp)
    tokens = tokenize(imp.input_sentence)
    tenz = to_tensor(tokens) # pray sharply for the effect, with vigor
    print(tokens)
    print(tenz.shape)
    return {"body": {
        "tensor_shape":list(tenz.shape),
        "tokenized_string":tokens}
        }
uvicorn.run(app, host="localhost", port=5000, log_level="info")