import openke
from openke.config import Trainer, Tester
from openke.module.model import TransH
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
    in_path=None,
    tri_file='./benchmarks/YAGO3-10/deleted_node_unlearning.txt',
    ent_file="./benchmarks/YAGO3-10/entity2id.txt",
    rel_file="./benchmarks/YAGO3-10/relation2id.txt",
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/YAGO3-10/", "link")

# define the model
transh = TransH(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transh, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
transh.load_checkpoint('./checkpoint/YAGO/YAGO_TransH.ckpt')
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 100, alpha = 1e-4, use_gpu = True)
trainer.run_grad_ascent()
transh.save_checkpoint('./checkpoint/GradAscent_Edges_TransH_YAGO.ckpt')

# test the model
transh.load_checkpoint('./checkpoint/GradAscent_Edges_TransH_YAGO.ckpt')
tester = Tester(model = transh, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)