import numpy as np
import time
import random
import torch

class GraphEraser(object):
    def __init__(self,
                 tri_file=None,
                 unlearn_file=None,
                 weight_file=None,

                 ):
        self.tri_file = tri_file
        self.unlearn_file = unlearn_file
        self.model = torch.load(weight_file)
        self.entity_embeddings = self.model['ent_embeddings.weight'].cpu().numpy()
        self.num_triples, self.triples, self.removed_triples = self.load_data(file_path=self.tri_file,
                                                                              unlearn_file=self.unlearn_file)

    def load_data(self, file_path, unlearn_file):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        num_triples = int(lines[0].strip())
        triples = np.array([list(map(int, line.strip().split())) for line in lines[1:]])

        with open(unlearn_file, 'r') as f:
            lines = f.readlines()
        removed_triples = np.array([list(map(int, line.strip().split())) for line in lines[1:]])
        return num_triples, triples, removed_triples

    def BEKM(self, k, delta, T):
        n_embeddings = self.entity_embeddings.shape[0]
        centroids = self.entity_embeddings[random.sample(range(n_embeddings), k)]
        t = 0

        while True:
            F = []
            for i in range(n_embeddings):
                for j in range(k):
                    distance = np.linalg.norm(self.entity_embeddings[i] - centroids[j])
                    F.append((distance, i, j))
            F.sort()
            cluster_sizes = [0] * k
            assignments = [-1] * n_embeddings
            for distance, i, j in F:
                if assignments[i] == -1 and cluster_sizes[j] < delta:
                    assignments[i] = j
                    cluster_sizes[j] += 1

            new_centroids = np.zeros((k, self.entity_embeddings.shape[1]))
            for j in range(k):
                assigned_nodes = [i for i in range(n) if assignments[i] == j]
                if assigned_nodes:
                    new_centroids[j] = np.mean(self.entity_embeddings[assigned_nodes], axis=0)
                else:
                    new_centroids[j] = centroids[j]

            if t > T or np.allclose(new_centroids, centroids):
                break

            centroids = new_centroids
            t += 1

        clusters = [[] for _ in range(k)]
        for i in range(n_embeddings):
            clusters[assignments[i]].append(i)
        return clusters

