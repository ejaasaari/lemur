# LEMUR: Learned Multi-Vector Retrieval

Official implementation of the method described in the paper [LEMUR: Learned Multi-Vector Retrieval](https://arxiv.org/pdf/2601.21853) (ICML '26). LEMUR speeds up multi-vector similarity search for late interaction models such as ColBERT by learning a lightweight, corpus-specific reduction to single-vector similarity search. 

## Installation

From the repo root:

```bash
pip install .
```

On macOS, it is recommended to use the Homebrew version of Clang as the compiler:

```bash
brew install llvm libomp
CC=/opt/homebrew/opt/llvm/bin/clang CXX=/opt/homebrew/opt/llvm/bin/clang++ pip install .
```

## Example usage

```python
import torch
import numpy as np
from lemur import Lemur
from lemur.maxsim import MaxSim

# train: torch.tensor float32, shape (num_corpus_token_embeddings, dim)
# train_counts: torch.tensor uint64, shape (num_corpus_documents, )
# test: torch.tensor float32, shape (num_query_token_embeddings, dim)
# test_counts: torch.tensor uint64, shape (num_query_documents, )
# train_counts/test_counts: array containing the number of token embeddings for each document

# Optional:
# Pass learn/learn_counts to fit() to improve performance by using a sample from the query
# distribution as a training set. Ideally, learn should contain at least 100 000 rows
# (token embeddings) and can also be e.g. the corpus documents encoded using the query encoder.

lemur = Lemur(index="lemur_index", device="cpu")  # or "cuda" or "mps"
lemur.fit(
    train=train,
    train_counts=train_counts,
    epochs=10,
    verbose=True,
)

# Set epochs = 0 to skip training the MLP
# This still works well but usually requires 2-4x more candidates to rerank

# 1) Compute features for test queries
feats = lemur.compute_features((test, test_counts))

# 2) Compute approximate maxsim scores for all corpus documents and select k' candidates
scores = feats @ lemur.W.T
k_candidates = 200
topk = torch.topk(scores, k_candidates, dim=1)
cand = topk.indices

# If the number of corpus documents is large (e.g. > 1 000 000), it is recommended to instead
# index the rows of lemur.W using an approximate nearest neighbor search library that supports
# maximum inner product search. The index can be queried using feats.

# 3) Rerank with MaxSim (note that this is done on CPU even if the index is built on GPU)
cand_np = np.ascontiguousarray(cand.cpu().numpy().astype(np.int32))

ms = MaxSim(train, train_counts)
k_final = 10
reranked = ms.rerank_subset(
    test,
    test_counts,
    k_final,
    cand_np,
)

print(reranked)

# Compute weights for new corpus documents
new_W = lemur.compute_weights(new_docs, new_docs_counts)
```

## Citation

If you use the library in an academic context, please consider citing the following paper:

> Jääsaari, E., Hyvönen, V., & Roos, T. (2026). LEMUR: Learned Multi-Vector Retrieval. arXiv preprint arXiv:2601.21853.

```
@article{jaasaari2026lemur,
  title={{LEMUR}: Learned Multi-Vector Retrieval},
  author={J{\"a}{\"a}saari, Elias and Hyv{\"o}nen, Ville and Roos, Teemu},
  journal={arXiv preprint arXiv:2601.21853},
  year={2026}
}
```

## License

LEMUR is available under the MIT License (see `LICENSE`).
