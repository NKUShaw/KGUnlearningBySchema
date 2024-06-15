import openke
import time
import torch
from openke.config import Trainer, Tester
from openke.module.model import TransD
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

def calculate_gradients(model, data):
    model.eval()  

    loss = model.model({
        'batch_h': torch.autograd.Variable(torch.from_numpy(data['batch_h']).cuda()),
        'batch_t': torch.autograd.Variable(torch.from_numpy(data['batch_t']).cuda()),
        'batch_r': torch.autograd.Variable(torch.from_numpy(data['batch_r']).cuda()),
        'batch_y': torch.autograd.Variable(torch.from_numpy(data['batch_y']).cuda()),
        'mode': data['mode']
    })
    loss_scalar = torch.mean(loss)
    params = filter(lambda p: p.requires_grad, model.parameters())
    params_to_update = [param for name, param in model.named_parameters() if name.endswith('.weight')]
    grads = torch.autograd.grad(loss_scalar, params_to_update, create_graph=True)
    del loss, loss_scalar
    torch.cuda.empty_cache()
    return grads

def hvps(grad_all, model_params, h_estimate):
    element_product = 0
    for grad_elem, v_elem in zip(grad_all, h_estimate):
        element_product += torch.sum(grad_elem * v_elem)
    return_grads = torch.autograd.grad(element_product, model_params, create_graph=True)
    del element_product
    torch.cuda.empty_cache()
    return return_grads
    
def GIF_unleanring(model, train_dataloader, test_dataloader, iteration=1, damp=0.0, scale=50):
    start_time = time.time()

    for data in train_dataloader:
        grad_full = calculate_gradients(model, data)

    for data in test_dataloader:
        grad_removed = calculate_gradients(model, data)

    grad1 = [g1 - g2 for g1, g2 in zip(grad_full, grad_removed)]
    grad2 = grad_removed
    res_tuple = (grad_full, grad1, grad2)

    v = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
    h_estimate = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
    
    for _ in range(iteration):
        model_params  = [p for p in model.parameters() if p.requires_grad]
        hv = hvps(res_tuple[0], model_params, h_estimate)
        with torch.no_grad():
            h_estimate = [ v1 + (1-damp)*h_estimate1 - hv1/scale for v1, h_estimate1, hv1 in zip(v, h_estimate, hv)]
            
    params_change = [h_est / scale for h_est in h_estimate]
    params_esti   = [p1 + p2 for p1, p2 in zip(params_change, model_params)]
    
    del grad_full, grad_removed, res_tuple, v, h_estimate, params_change
    torch.cuda.empty_cache()
    
    print(time.time() - start_time)
    
    return params_esti

def update_and_save_checkpoint(checkpoint_path, new_checkpoint_path, new_params):
    weights = torch.load(checkpoint_path)
    weights['ent_embeddings.weight'] = new_params[0]
    weights['rel_embeddings.weight'] = new_params[1]
    torch.save(weights, new_checkpoint_path)
    print(f"Updated checkpoint saved to {new_checkpoint_path}")


train_dataloader = TrainDataLoader(
	in_path = None, 
    tri_file = "./benchmarks/FB15K237/train2id.txt",
    ent_file = "./benchmarks/FB15K237/entity2id.txt",
    rel_file = "./benchmarks/FB15K237/relation2id.txt",
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)
print('----------------------------------------------------------------------')
retrain_dataloader = TrainDataLoader(
	in_path = None, 
    tri_file = './benchmarks/FB15K237/remain_node_unlearning.txt',
    ent_file = "./benchmarks/FB15K237/entity2id.txt",
    rel_file = "./benchmarks/FB15K237/relation2id.txt",
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

model = TransD(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim_e = 200, 
	dim_r = 200, 
	p_norm = 1, 
	norm_flag = True)
model = torch.nn.DataParallel(model)
model.to('cuda')
model.module.load_checkpoint("./checkpoint/FB15K237/FB15K237_TransD.ckpt")
model = NegativeSampling(
	model = model, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

params_esti = GIF_unleanring(model, train_dataloader, retrain_dataloader, iteration=300, damp=0.0, scale=50)
update_and_save_checkpoint(checkpoint_path="./checkpoint/FB15K237/FB15K237_TransD.ckpt", 
                           new_checkpoint_path="./checkpoint/FB15K237/GIF_Nodes_TransD_FB15K237.ckpt", 
                           new_params=params_esti)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

# define the model
unlearn_transe = TransD(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim_e = 200, 
	dim_r = 200, 
	p_norm = 1, 
	norm_flag = True)
# unlearn_transe = torch.nn.DataParallel(unlearn_transe)
# test the model
unlearn_transe.load_checkpoint("./checkpoint/FB15K237/GIF_Nodes_TransD_FB15K237.ckpt")
unlearn_tester = Tester(model = unlearn_transe, data_loader = test_dataloader, use_gpu = True)
unlearn_tester.run_link_prediction(type_constrain = False)