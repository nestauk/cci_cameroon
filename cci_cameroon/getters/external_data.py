# %%
import torch.nn.functional as F


# %%
def fetch_data(url_list):
    """function fetches data to use from a remote source(s).
    @param: list of urls
    return: list of sentences. A version that returns a dataframe that holds the rumours and the labels would
    developed for use."""
    # we loop through each url and create our sentences data
    sentences = []
    for url in urls:
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
