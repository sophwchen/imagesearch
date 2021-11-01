import re, string
from collections import Counter
import numpy as np

import gensim
from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import TruncatedSVD

import mygrad as mg
from mynn.layers.dense import dense
from mygrad.nnet.initializers import glorot_normal

def get_glove(path):
    """
    Returns the glove algorithm given a file path.

    Parameters
    ----------
    path : str
        may resemble r"./dat/glove.6B.200d.txt.w2v"
    
    Returns
    -------
    glove
    """
    glove = KeyedVectors.load_word2vec_format(path, binary=False)
    return glove

def strip_punc(corpus) -> str:
    """
    Removes all punctuation from corpus.

    Parameters
    ----------
    corpus : str
    
    Returns
    -------
    str
    """
    punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
    return punc_regex.sub('', corpus)

def embed_caption(caption):
    """ 
    Produces list of tokens for caption, removing all punctuation
    and making all the characters lower-cased.
    
    Parameters
    ----------
    caption : str
    
    Returns
    -------
    list
        array of every token from the caption
    """
    caption = strip_punc(caption.lower())
    tokens = caption.split()
    return tokens

def to_counter(doc) -> Counter:
    """
    Produce word-count of document, removing all punctuation
    and making all the characters lower-cased.
    
    Parameters
    ----------
    doc : str
    
    Returns
    -------
    collections.Counter
        lower-cased word -> count
    
    Examples
    --------
    >>> to_counter("I am a dog.")
    Counter({'a': 1, 'am': 1, 'dog': 1, 'i': 1})
    """
    return Counter(strip_punc(doc).lower().split())

def to_idf(vocab, counters) -> np.ndarray:
    """ 
    Given the vocabulary, and the word-counts for each document, computes
    the inverse document frequency (IDF) for each term in the vocabulary.
    
    Parameters
    ----------
    vocab : Sequence[str]
        Ordered list of words that we care about.

    counters : Iterable[collections.Counter]
        The word -> count mapping for each document.
    
    Returns
    -------
    numpy.ndarray
        An array whose entries correspond to those in `vocab`, storing
        the IDF for each term `t`: 
                           log10(N / nt)
        Where `N` is the number of documents, and `nt` is the number of 
        documents in which the term `t` occurs.
    """
    N = len(counters)
    nt = [sum(1 if t in counter else 0 for counter in counters) for t in vocab]
    nt = np.array(nt, dtype=float)
    return np.log10(N / nt)

def embed_idf_weighted_sum(captions, idf_dict, glove, glove_shape):
    """
    Parameters
    ----------
    captions: list[str], length = N
        List of string captions for the same image.

    glove_filepath: str, D-dimensional glove embedding
        Local file path for the glove algorithm, should resemble r".../glove.6B.200d.txt.w2v"
    
    Returns
    -------
    numpy.ndarray, shape=(N, D)
        An array of unit vectors for phrase embeddings of every caption in captions.
    """
    # gets the glove word embeddings
#     glove_shape = glove["word"].shape

    # vector for storing phrase embedding unit vectors for each caption in captions
    w = []

    # vector of Counters for every caption in captions
    counters = [to_counter(caption) for caption in captions]
    
    for caption in captions:
        # gets the tokenized caption ("Cat in box." -> ["cat","in","box"])
        tokens = embed_caption(caption)

        # creates the glove vector for the caption
        glove_phrase = np.array([
            glove[i] if i in glove else np.zeros(glove_shape) for i in tokens
        ])
        
        # calculates the IDF vector for the caption
#         idf_phrase = to_idf(tokens, counters)
        idf_phrase = [idf_dict[token] for token in tokens]
        
        # calculates IDF-weighted sum of glove embeddings to get a phrase embedding for the caption
        # print(glove_phrase.shape, idf_phrase.shape)
        w_phrase = mg.matmul(glove_phrase.T, idf_phrase)
        
        # turns the phrase embedding vector into a unit vector
        w_unit = w_phrase / mg.sqrt((w_phrase ** 2).sum(keepdims=True))

        # adds w_unit to w
        w.append(w_unit)
    
    return w

class Model:
    def __init__(self, dim_input, dim_output):
        self.W_embed = dense(
            dim_input, 
            dim_output, 
            weight_initializer=glorot_normal, 
            bias=False
        )

    def __call__(self, x):
        w = self.W_embed(x)
        return w / np.sqrt(np.einsum("nd, nd -> n", w, w)).reshape(-1, 1)

    def save_params(self, file_path):
        with open(file_path, mode="wb") as opened_file:
            for x in self.parameters():
                np.savez(opened_file, x.data)

    def load_params(self, file_path):
        with open(file_path,mode="rb") as opened_file:
            saved_params = np.load(opened_file)
            self.parameters.data = saved_params
    @property
    def parameters(self):
        return self.W_embed.parameters