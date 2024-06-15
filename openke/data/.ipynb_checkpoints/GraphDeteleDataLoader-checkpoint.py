import numpy as np
import torch
import torch.nn as nn

class GraphDeteleDataLoader(object):
    def __init__(self, tri_file=None, unlearn_file=None, weight_file=None, device='cuda'):
        self.tri_file = tri_file
        self.unlearn_file = unlearn_file
        self.device = device
        self.num_triples, self.triples, self.removed_triples = self.load_data(file_path=self.tri_file, unlearn_file=self.unlearn_file)
        self.adj_matrix = self.create_adj_matrix(self.triples)
        self.model = torch.load(weight_file, map_location=self.device)
        self.entity_embeddings = self.model['ent_embeddings.weight'].to(self.device)
        self.relation_embeddings = self.model['rel_embeddings.weight'].to(self.device)

    def load_data(self, file_path, unlearn_file):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        num_triples = int(lines[0].strip())
        triples = np.array([list(map(int, line.strip().split())) for line in lines[1:]])

        with open(unlearn_file, 'r') as f:
            lines = f.readlines()
        removed_triples = np.array([list(map(int, line.strip().split())) for line in lines[1:]])
        return num_triples, triples, removed_triples

    def create_adj_matrix(self, triples):
        num_entities = max(np.max(triples[:, 0]), np.max(triples[:, 1])) + 1
        adj_matrix = np.zeros((num_entities, num_entities), dtype=int)
        for h, t, _ in triples:
            adj_matrix[h, t] = 1
            adj_matrix[t, h] = 1
        return adj_matrix

    def convert_to_batch_data(self, triples, device):
        batch_data = {
            'batch_h': torch.tensor([triple[0] for triple in triples], dtype=torch.long).to(device),
            'batch_t': torch.tensor([triple[1] for triple in triples], dtype=torch.long).to(device),
            'batch_r': torch.tensor([triple[2] for triple in triples], dtype=torch.long).to(device),
            'batch_y': torch.tensor([1 for _ in triples], dtype=torch.long).to(device),
            'mode': "normal"
        }
        return batch_data

    def mask_embeddings(self):
        removed_triples = self.removed_triples
        adj_matrix = self.adj_matrix
        head_group = torch.tensor(removed_triples[:, 0], device=self.device)
        tail_group = torch.tensor(removed_triples[:, 1], device=self.device)

        combined_group = torch.cat((head_group, tail_group))
        update_entities1 = torch.unique(combined_group)

        update_entities1 = update_entities1.cpu().numpy()  
        update_entities2 = set(update_entities1)

        for entity in update_entities1:
            first_hop_neighbors = np.where(adj_matrix[entity] > 0)[0]
            update_entities2.update(first_hop_neighbors)

        update_entities2 = torch.tensor(list(update_entities2), device=self.device)
        embedding_shape = self.entity_embeddings.shape[0]
        mask_matrix = torch.zeros(embedding_shape, dtype=torch.bool, device=self.device)
        mask_matrix[update_entities2] = 1

        deletion_weight = nn.Parameter(torch.ones(self.entity_embeddings.shape[1], self.entity_embeddings.shape[1]) / 1000).to(self.device)

        new_entity_embeddings = self.entity_embeddings.clone()
        new_entity_embeddings[mask_matrix] = torch.matmul(new_entity_embeddings[mask_matrix], deletion_weight)
        with torch.no_grad():
            for i in range(embedding_shape):
                if not mask_matrix[i]:
                    new_entity_embeddings[i] = self.entity_embeddings[i]

        self.model['ent_embeddings.weight'] = new_entity_embeddings

        # 冻结 rel_embeddings.weight
        with torch.no_grad():
            self.model['rel_embeddings.weight'] = self.relation_embeddings