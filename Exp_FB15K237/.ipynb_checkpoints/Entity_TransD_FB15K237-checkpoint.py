import openke
from openke.config import Trainer, Tester
from openke.module.model import TransD
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader, EmbeddingDataLoader
import torch
import torch.nn.functional as F
import random
import time
from tqdm import tqdm

def compute_embedding(model, e1, e2):
    e1 = torch.tensor(e1, dtype=torch.long).to(device)
    e2 = torch.tensor(e2, dtype=torch.long).to(device)
    embed_1 = model.ent_embeddings(e1)
    embed_2 = model.ent_embeddings(e2)
    embedding = embed_1 - embed_2
    return embedding

lr = 3e-5
n_cluster = 10
device = torch.device('cuda:1')

Cosine_dataloader = TrainDataLoader(
    in_path=None,
    tri_file='./benchmarks/FB15K237/train2id.txt',
    ent_file="./benchmarks/FB15K237/entity2id.txt",
    rel_file="./benchmarks/FB15K237/relation2id.txt",
    nbatches=100,
    threads=8,
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    neg_ent=25,
    neg_rel=0)
Cosine_model = TransD(
	ent_tot = Cosine_dataloader.get_ent_tot(),
	rel_tot = Cosine_dataloader.get_rel_tot(),
	dim_e = 200, 
	dim_r = 200, 
	p_norm = 1, 
	norm_flag = True)

Cosine_model.to(device)
Cosine_model.load_checkpoint('./checkpoint/FB15K237/FB15K237_TransD.ckpt')

tri_file = './benchmarks/FB15K237/train2id.txt'
unlearn_file = './benchmarks/FB15K237/deleted_node_unlearning.txt'
schema_file = './benchmarks/FB15K237/type_constrain.txt'
weight_file = './checkpoint/FB15K237/FB15K237_TransD.ckpt'

Schema_dataloader = EmbeddingDataLoader.CosineSchemaDataLoader(
    n_clusters=n_cluster,
    tri_file=tri_file,
    unlearn_file=unlearn_file,
    schema_file=schema_file,
    weight_file=weight_file)
Cosine_Sampling = NegativeSampling(
	model = Cosine_model, 
	loss = MarginLoss(margin = 5.0),
	batch_size = Cosine_dataloader.get_batch_size()
)

trainer = Trainer(model = Cosine_Sampling, 
                  data_loader = Cosine_dataloader, 
                  train_times = 1000, 
                  alpha = lr, 
                  use_gpu = True)
trainer.optimizer = torch.optim.Adam(
    trainer.model.parameters(),
    lr=trainer.alpha,
    weight_decay=trainer.weight_decay,
)

start_time = time.time()
total_loss = 0.0

with tqdm(total=len(Schema_dataloader.removed_triples), desc="Processing triples") as pbar:
    for idx, data in enumerate(Schema_dataloader.removed_triples):
        trainer.optimizer.zero_grad()
        max_similarity = float('-inf')
        all_iterations = 10
        loss_value = 0.0
        e1, e2, e3, e4 = None, None, None, None
        attempts = 0
        max_attempts = 10
        while (e1 is None or e2 is None or e3 is None or e4 is None) and attempts < max_attempts:
            e1, e2, e3, e4 = Schema_dataloader.query_match_entity(Schema_dataloader.triples, 
                                                           Schema_dataloader.adj_matrix, 
                                                           data,
                                                           Schema_dataloader.labels)
            attempts += 1
        if attempts >= max_attempts:
            pbar.update(1)
            continue
            
        Embed_Query = compute_embedding(Cosine_model, e1, e2)
        Embed_Match = compute_embedding(Cosine_model, e3, e4)
        cosine_similarity = torch.nn.functional.cosine_similarity(Embed_Query, Embed_Match, dim=0)
        if cosine_similarity > max_similarity:
            max_similarity = cosine_similarity
        loss_value = 1 - max_similarity
        loss_value.backward()
        total_loss += loss_value.item()
        trainer.optimizer.step()
        pbar.set_description(f"Processing triples (Loss: {total_loss / (idx + 1):.4f})")
        pbar.update(1)
print(f'Running Time: {time.time() - start_time}s')

Cosine_model.save_checkpoint(f'./checkpoint/FB15K237/Entity_Unlearning_TransD_FB15K237.ckpt')

test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")
tester = Tester(model = Cosine_model, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
