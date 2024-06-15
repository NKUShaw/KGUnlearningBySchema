import openke
import time
import torch
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader, GraphDeteleDataLoader

start_time = time.time()
dataloader = GraphDeteleDataLoader.GraphDeteleDataLoader(
                 tri_file='./benchmarks/FB15K237/train2id.txt',
                 unlearn_file='./benchmarks/FB15K237/deleted_node_unlearning.txt',
                 weight_file='./checkpoint/FB15K237/FB15K237_TransE.ckpt',
                 device='cuda:1')
dataloader.mask_embeddings()

train_dataloader = TrainDataLoader(
    in_path=None,
    tri_file='./benchmarks/FB15K237/remain_node_unlearning.txt',
    ent_file="./benchmarks/FB15K237/entity2id.txt",
    rel_file="./benchmarks/FB15K237/relation2id.txt",
    nbatches=100,
    threads=8,
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    neg_ent=25,
    neg_rel=0)
test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200,
	p_norm = 1, 
	norm_flag = True)
transe.load_state_dict(dataloader.model)
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 250, alpha = 1e-4, use_gpu = True)

trainer.optimizer = torch.optim.SGD(
				transe.parameters(),
				lr=trainer.alpha,
				weight_decay=trainer.weight_decay,
			)

trainer.run()
transe.save_checkpoint('./checkpoint/FB15K237/Delete_Nodes_TransE_FB15K237.ckpt')
end_time = time.time()
print(f'Running Times: {end_time - start_time}s')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)