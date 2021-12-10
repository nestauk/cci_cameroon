# %%
import torch.nn.functional as F
from sentence_transformers import util
from io import StringIO 
import torch

# %%
def fetch_data(url_list):
    """function fetches data to use from a remote source(s).
    parameter - url_list is a list of urls to data sources
    returns list of sentences. A version that returns a dataframe that holds the rumours and the labels would
    developed for use."""
    # we loop through each url and create our sentences data
    sentences = []
    for url in url_list:
        res = requests.get(url)
        # extract to dataframe
        data = pd.read_csv(
            StringIO(res.text), sep="\t", header=None, error_bad_lines=False
        )
        # add to columns 1 and 2 to sentences list
        sentences.extend(data[1].tolist())
        sentences.extend(data[2].tolist())
    return sentences


# %%
def similarity(embeddings_1, embeddings_2):
    """This function computes the similarity index between the vectors of two embeddings. It takes two embeddings
    and returns a tensor containing the pair-wise cosine similarity scores of the vectors in the embeddings."""
    normalized_embeddings_1 = F.normalize(embeddings_1, p=2)
    normalized_embeddings_2 = F.normalize(embeddings_2, p=2)
    return torch.matmul(
        normalized_embeddings_1, normalized_embeddings_2.transpose(0, 1)
    )


# %%
def compute_embedding_cosign_score(model_name, corpus1, corpus2, tensor=False):
    """performs embedding of text corpora and computes cosine similarity score between text vectors in the embedings.
    parameters
    model_name - the name of the model to be used for the embeddings
    corpus1 - first data corpus
    corpus2 - second data corpus
    tensor - boolean which specifies if the resulting embeddings should be a tensor object or not.
    returns a matrix of similarity scores
    """
    if tensor:
        emb1 = model_name.encode(corpus1, convert_to_tensor=True)
        emb2 = model_name.encode(corpus2, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2)
    emb1 = model_name.encode(corpus1, convert_to_tensor=True)
    emb2 = model_name.encode(corpus2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2)  # cosign score


# %%
