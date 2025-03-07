import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mashup_embs = np.load("mashup_emb.npy")  # [num_samples, dim]
labels = np.load("labels.npy", allow_pickle=True)

tsne = TSNE(n_components=2, perplexity=3, random_state=42)
mashup_2d = tsne.fit_transform(mashup_embs)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(mashup_2d[:, 0], mashup_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title("t-SNE Visualization of Mashup Embeddings")
plt.savefig('tsne_visualization.png', bbox_inches='tight')
plt.close()