import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
    in_path=None,
    tri_file='./benchmarks/FB15K237/deleted_node_unlearning.txt',
    ent_file="./benchmarks/FB15K237/entity2id.txt",
    rel_file="./benchmarks/FB15K237/relation2id.txt",
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)

transe.to('cuda:1')
# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
transe.load_checkpoint('./checkpoint/FB15K237/FB15K237_TransE.ckpt')
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 100, alpha = 1e-3, use_gpu = True)
trainer.run_grad_ascent()
transe.save_checkpoint('./checkpoint/FB15K237/GradAscent_Nodes_TransE_FB15K237.ckpt')

# test the model
transe.load_checkpoint('./checkpoint/FB15K237/GradAscent_Nodes_TransE_FB15K237.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)