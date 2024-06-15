import numpy as np
import time
import random
import torch
from sklearn.cluster import KMeans


class CosineSchemaDataLoader(object):
    def __init__(self,
                 n_clusters=10,
                 tri_file=None,
                 ent_file=None,
                 rel_file=None,
                 unlearn_file=None,
                 schema_file=None,
                 weight_file=None):
        self.tri_file = tri_file
        self.ent_file = ent_file
        self.rel_file = rel_file
        self.unlearn_file = unlearn_file
        self.schema_file = schema_file
        self.num_triples, self.triples, self.removed_triples = self.load_data(file_path=self.tri_file,
                                                                              unlearn_file=self.unlearn_file)
        self.schemas = self.load_schemas(file_path=schema_file)
        self.adj_matrix = self.create_adj_matrix(self.triples)
        self.model = torch.load(weight_file)
        self.n_clusters = n_clusters
        self.entity_embeddings = self.model['ent_embeddings.weight'].cpu().numpy()
        self.relation_embeddings = self.model['rel_embeddings.weight']
        self.labels = self.entity_cluster(entity_embeddings=self.entity_embeddings, n_clusters=self.n_clusters)

    def load_data(self, file_path, unlearn_file):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        num_triples = int(lines[0].strip())
        triples = np.array([list(map(int, line.strip().split())) for line in lines[1:]])

        with open(unlearn_file, 'r') as f:
            lines = f.readlines()
        removed_triples = np.array([list(map(int, line.strip().split())) for line in lines[1:]])
        return num_triples, triples, removed_triples

    def load_schemas(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        num_schemas = int(lines[0].strip())
        schemas = {}

        for line in lines[1:]:
            parts = list(map(int, line.strip().split()))
            schema_index = parts[0]
            node_indices = parts[1:]
            schemas[schema_index] = node_indices

        return schemas

    def create_adj_matrix(self, triples):
        adj_matrix = {}
        for h, t, r in triples:
            if h not in adj_matrix:
                adj_matrix[h] = {}
            if t not in adj_matrix:
                adj_matrix[t] = {}
            adj_matrix[h][t] = r
            adj_matrix[t][h] = r

        return adj_matrix

    def entity_cluster(self, entity_embeddings, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(entity_embeddings)
        labels = kmeans.labels_
        return labels

    def query_match(self, triples, adj_matrix, query_triple, entity_labels):

        query_head, query_tail, query_relation = query_triple
        match_head, match_tail = None, None
        for h, t, r in triples:
            if (r == query_relation and h != query_head and t != query_tail):
                match_head = h
                match_tail = t
                break

        query_head_neighbors = list(adj_matrix.get(query_head, {}).items())  # e1
        query_tail_neighbors = list(adj_matrix.get(query_tail, {}).items())  # e2
        random.shuffle(query_head_neighbors)
        random.shuffle(query_tail_neighbors)
        match_head_neighbors = None
        match_tail_neighbors = None
        if match_head and match_tail:
            match_head_neighbors = list(adj_matrix.get(match_head, {}).items())  # e3
            match_tail_neighbors = list(adj_matrix.get(match_tail, {}).items())  # e4
            random.shuffle(match_head_neighbors)
            random.shuffle(match_tail_neighbors)


        e1 = None
        e2 = None
        e3 = None
        e4 = None

        for neighbor, relation in query_head_neighbors:
            if neighbor != query_tail:
                e1 = neighbor
                break
        for neighbor, relation in query_tail_neighbors:
            if neighbor != query_head:
                e2 = neighbor
                break
        if match_head_neighbors and e1:
            for neighbor, relation in match_head_neighbors:
                if neighbor != match_tail and (entity_labels[neighbor] == entity_labels[e1]):
                    e3 = neighbor
                    break
        if match_tail_neighbors and e2:
            for neighbor, relation in match_tail_neighbors:
                if neighbor != match_head and (entity_labels[neighbor] == entity_labels[e2]):
                    e4 = neighbor
                    break

        return e1, e2, e3, e4


    def query_match_edges(self, triples, adj_matrix, query_triple, entity_labels):

        query_head, query_tail, query_relation = query_triple
        match_head, match_tail = None, None
        for h, t, r in triples:
            if (r != query_relation and h != query_head and t != query_tail and entity_labels[h] == entity_labels[query_head] and entity_labels[t] == entity_labels[query_tail]):
                match_head = h
                match_tail = t
                break

        query_head_neighbors = list(adj_matrix.get(query_head, {}).items())  # e1
        query_tail_neighbors = list(adj_matrix.get(query_tail, {}).items())  # e2
        random.shuffle(query_head_neighbors)
        random.shuffle(query_tail_neighbors)
        match_head_neighbors = None
        match_tail_neighbors = None
        if match_head and match_tail:
            return match_head, match_tail

        else:
            return match_head, match_tail

    def query_match_entity(self, triples, adj_matrix, query_triple, entity_labels):
        query_head, query_tail, query_relation = query_triple
        match_head, match_tail = None, None
        for h, t, r in triples:
            if (r == query_relation and h != query_head and t != query_tail):
                match_head = h
                match_tail = t
                break

        query_head_neighbors = list(adj_matrix.get(query_head, {}).items())  # e1
        query_tail_neighbors = list(adj_matrix.get(query_tail, {}).items())  # e2
        random.shuffle(query_head_neighbors)
        random.shuffle(query_tail_neighbors)
        match_head_neighbors = None
        match_tail_neighbors = None
        if match_head and match_tail:
            match_head_neighbors = list(adj_matrix.get(match_head, {}).items())  # e3
            match_tail_neighbors = list(adj_matrix.get(match_tail, {}).items())  # e4
            random.shuffle(match_head_neighbors)
            random.shuffle(match_tail_neighbors)


        e1 = None
        e2 = None
        e3 = None
        e4 = None

        for neighbor, relation in query_head_neighbors:
            if neighbor != query_tail:
                e1 = neighbor
                break
        for neighbor, relation in query_tail_neighbors:
            if neighbor != query_head:
                e2 = neighbor
                break
        if match_head_neighbors and e1:
            for neighbor, relation in match_head_neighbors:
                if neighbor != match_tail and (entity_labels[neighbor] == entity_labels[e1]):
                    e3 = neighbor
                    break
        if match_tail_neighbors and e2:
            for neighbor, relation in match_tail_neighbors:
                if neighbor != match_head and (entity_labels[neighbor] == entity_labels[e2]):
                    e4 = neighbor
                    break

        return e1, e2, e3, e4
            
    def convert_to_batch_data(self, triples, device):
        batch_data = {
            'batch_h': torch.tensor([triple[0] for triple in triples], dtype=torch.long).to(device),
            'batch_t': torch.tensor([triple[1] for triple in triples], dtype=torch.long).to(device),
            'batch_r': torch.tensor([triple[2] for triple in triples], dtype=torch.long).to(device),
            'batch_y': torch.tensor([1 for _ in triples], dtype=torch.long).to(device),
            'mode': "normal"
        }
        return batch_data

    def entity_embedding(self, node, device):
        e1_data = {
            'batch_h': torch.tensor(node, dtype=torch.long).to(device),
            'batch_t': torch.tensor(1, dtype=torch.long).to(device),
            'batch_r': torch.tensor(1, dtype=torch.long).to(device),
            'batch_y': torch.tensor(1, dtype=torch.long).to(device),
            'mode': "normal"
        }
        return self.model.ent_embeddings(e1_data['batch_h'])