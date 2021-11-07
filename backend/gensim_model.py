import gensim
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import numpy
import random
from gensim.test.utils import get_tmpfile

input_phrase = ["i like it when babies recover from sickness"]
df = pd.read_csv("modified_test_dataset.csv")
content = df.content

def tokenize(data):#function to tokenize our corpus
    tokens = []
    for line in data:
        tokens.append(gensim.utils.simple_preprocess(line))
    return tokens

def tokenize_tagged(data):
    tokens = []
    for i,line in enumerate(data):
        tokens.append(gensim.models.doc2vec.TaggedDocument(line, [i]))
    return tokens
train_corpus, test_corpus = train_test_split(content.tolist(), test_size=0.2)
train_corpus, test_corpus = tokenize_tagged(train_corpus), tokenize(input_phrase)

#[["word","word","word"],["word","word"]]


model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=2, epochs=400)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

fname = get_tmpfile("my_doc2vec_model")
model.save(fname)

# Pick a random document from the test corpus and infer a vector from the model
doc_id = random.randint(0, len(test_corpus) - 1)
inferred_vector = model.infer_vector(test_corpus[doc_id])
print(inferred_vector)
sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

# Compare and print the most/median/least similar documents from the train corpus
print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ''.join(train_corpus[sims[index][0]].words)))


# fact_repo = [
#     {
#         "link": "[insert link to article]",
#         "text": "text",
#         "embed": [insert embedding]
#     },
#     ...
# ]

# def rank_similarity(query, fact_repo, topn=-1):
#     vec = doc2vec(query)
#     sorted_repo = sorted(fact_repo, lambda a: np.linalg.norm(vec - np.array(a["embed"])))
#     return sorted_repo[:topn]