from gensim.models import KeyedVectors
import numpy as np
import pickle


# # Load the GloVe model from a file.

def load_glove_model(glove_file):
    with open(glove_file, 'rb') as fp:
        glove = pickle.load(fp)
    return glove
#     return KeyedVectors.load_word2vec_format(glove_file, binary=True, encoding="utf-8")



# Compute the dot product between the input query and a set of texts using a GloVe model.
def compute_similarity(query, texts, model):
    query_tokens = query.lower().split()
    query_vector = np.mean([model[token] for token in query_tokens if token in model], axis=0)
    print(query_vector)

    similarities = []

    for text in texts:
        text_tokens = text.lower().split()
        text_vector = np.mean([model[token] for token in text_tokens if token in model], axis=0)
        similarity = np.dot(query_vector, text_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(text_vector))
        similarities.append((text, similarity))

    # Sort by similarity in descending order and retrieve the top 10
    similarities = [sim for sim in similarities if not np.isnan(sim[1])]
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similarities = similarities[:10]
    return top_similarities

