"""Quick embeddings playground for Lesson 0001.

Run: uv run python playground.py
"""

import numpy as np

from rag_pipeline import Embedder

embedder = Embedder()

# Three sentences. Two are paraphrases. One is unrelated.
sentences = [
    "How do I configure Neovim folding?",
    "What's the best way to set up code folding in Neovim?",
    "How do I cook a hard-boiled egg?",
    "How to fold laundry neatly into thirds?",
]

vecs = embedder.embed(sentences)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print("\nPairwise cosine similarity:\n")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        score = cos(vecs[i], vecs[j])
        print(f"  {score:.3f}  '{sentences[i][:40]}...'  vs  '{sentences[j][:40]}...'")

print(f"\nEach vector has {vecs.shape[1]} dimensions.")
print(f"Vector length (should be ~1.0): {np.linalg.norm(vecs[0]):.4f}")
print(f"First 5 numbers of sentence 0: {vecs[0][:5]}")
