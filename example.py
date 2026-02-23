import torch
import numpy as np
from lemur import Lemur
from lemur.maxsim import MaxSim

# train: torch.tensor float32, shape (num_corpus_token_embeddings, dim)
# train_counts: torch.tensor uint64, shape (num_corpus_documents, )
# test: torch.tensor float32, shape (num_query_token_embeddings, dim)
# test_counts: torch.tensor uint64, shape (num_query_documents, )
# train_counts is an array containing the number of token embeddings for each corpus document

lemur = Lemur(index="lemur_index", device="cpu")  # or "cuda" or "mps"
lemur.fit(
    train=train,
    train_counts=train_counts,
    epochs=10,
    verbose=True,
)

# Set epochs = 0 to skip training the MLP
# This still works well but usually requires 2-4x more candidates to rerank

# 1) Compute features
feats = lemur.compute_features((test, test_counts))

# 2) Compute approximate maxsim scores for all corpus documents and select k' candidates
scores = feats @ lemur.W.T
k_candidates = 200
topk = torch.topk(scores, k_candidates, dim=1)
cand = topk.indices

# If the number of corpus documents is large (e.g. > 100 000), it is recommended to instead
# index the rows of lemur.W using an approximate nearest neighbor search library that supports
# maximum inner product search. The index can be queried using feats.

# 3) Rerank with MaxSim
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

# Compute weights for new points
new_W = lemur.compute_weights(new_docs, new_docs_counts)
