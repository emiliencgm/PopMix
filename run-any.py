import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Go GCLRec")
    parser.add_argument('--dataset', type=str, default='yelp2018', help="dataset")
    parser.add_argument('--method', type=str, default='LightPopMix', help="method: LightPopMix, LightPopMix_wo_p, LightPopMix_lambda, LightPopMix_tau, LightPopMix_grid")
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument('--device', type=int, default=0, help="device")
    parser.add_argument('--visual', type=int, default=0, help="visualization")
    parser.add_argument('--valid', type=int, default=0, help="validation")
    parser.add_argument('--c', type=str, default='noting to comment', help="comment in CMD after training")
    parser.add_argument('--temp_tau', type=float, default=0.1, help="temp_tau")
    parser.add_argument('--lambda1', type=float, default=0.1, help="lambda1")

    return parser.parse_args()
args = parse_args()
if args.valid == 1:
    project = 'TestCode_Valid'
else:
    project = 'TestCode_No_Valid'

#For all
lr = 0.001
weight_decay = 1e-4
num_layers = 3
latent_dim_rec = 64
batch_size =2048
init_method = 'Normal'

temp_tau = args.temp_tau
lambda1 = args.lambda1


if args.method == 'LightPopMix':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book']:
        centroid = 'eigenvector'
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag LightPopMix --group ours --job_type {args.dataset} --model LightGCN --loss Adaptive --augment No --lambda1 {lambda1} --temp_tau {temp_tau} --centroid_mode {centroid}\
                    --sampling uii --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 5 --seed {args.seed} --c {args.c}')
    
    if args.dataset in ['ifashion', 'last-fm']:
        centroid = 'pagerank'
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag LightPopMix --group ours --job_type {args.dataset} --model LightGCN --loss Adaptive --augment No --lambda1 {lambda1} --temp_tau {temp_tau} --centroid_mode {centroid}\
                    --sampling uii --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 5 --seed {args.seed} --c {args.c}')



elif args.method == 'LightGCN':
    #HyperParms: None
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion']:
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag {args.method} --group baseline --job_type {args.dataset} --model LightGCN --loss BPR --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 5 --seed {args.seed} --c {args.c}')

elif args.method == 'GTN':
    #HyperParms: None
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion']:
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag {args.method} --group baseline --job_type {args.dataset} --model GTN --loss BPR --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 5 --seed {args.seed} --c {args.c}')

elif args.method == 'SGL-ED':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion']:
        p_drop = 0.1
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag SGL --group baseline --job_type {args.dataset} --model SGL --loss BPR_Contrast --augment ED --lambda1 {lambda1} --p_drop {p_drop} --temp_tau {temp_tau} \
                    --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 1 --seed {args.seed} --c {args.c}')
        
elif args.method == 'SGL-RW':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion']:
        p_drop = 0.1
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag SGL --group baseline --job_type {args.dataset} --model SGL --loss BPR_Contrast --augment RW --lambda1 {lambda1} --p_drop {p_drop} --temp_tau {temp_tau} \
                    --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 1 --seed {args.seed} --c {args.c}')
        
elif args.method == 'SimGCL':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion']:
        eps_SimGCL = 0.1
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag {args.method} --group baseline --job_type {args.dataset} --model SimGCL --loss BPR_Contrast --augment No --lambda1 {lambda1} --eps_SimGCL {eps_SimGCL} --temp_tau {temp_tau} \
                    --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 1 --seed {args.seed} --c {args.c}')
        
elif args.method == 'PDA':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion']:
        pop_gamma = 0.02
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag {args.method} --group baseline --job_type {args.dataset} --model LightGCN --loss PDA --augment No --pop_gamma {pop_gamma} \
                    --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 4 --seed {args.seed} --c {args.c}')
        
elif args.method == 'BC':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion']:
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag {args.method} --group baseline --job_type {args.dataset} --model LightGCN --loss BC --augment No --lambda1 {lambda1} --temp_tau_pop {temp_tau} --temp_tau {temp_tau} \
                    --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 5 --seed {args.seed} --c {args.c}')
        
elif args.method == 'Adaloss':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book']:
        centroid = 'eigenvector'
        os.system(f'python main.py --project {project} --name {args.method} --notes Ablation_only_Adaloss --tag Ablation_loss --group ours --job_type {args.dataset} --model LightGCN --loss Adaptive --augment No --lambda1 {lambda1} --temp_tau {temp_tau} --centroid_mode {centroid}\
                    --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 5 --seed {args.seed} --c {args.c}')
    
    if args.dataset in ['ifashion', 'last-fm']:
        centroid = 'pagerank'
        os.system(f'python main.py --project {project} --name {args.method} --notes Ablation_only_Adaloss --tag Ablation_loss --group ours --job_type {args.dataset} --model LightGCN --loss Adaptive --augment No --lambda1 {lambda1} --temp_tau {temp_tau} --centroid_mode {centroid}\
                    --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 5 --seed {args.seed} --c {args.c}')
        
elif args.method == 'LightPopMix_wo_p':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book']:
        centroid = 'eigenvector'
        os.system(f'python main.py --project {project} --name {args.method} --notes Ablation_wo_projectors --tag Ablation_proj --group ours --job_type {args.dataset} --model LightGCN --loss Adaptive --augment No --lambda1 {lambda1} --temp_tau {temp_tau} --centroid_mode {centroid}\
                    --sampling uii --projector wo --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 5 --seed {args.seed} --c {args.c}')
    
    if args.dataset in ['ifashion', 'last-fm']:
        centroid = 'pagerank'
        os.system(f'python main.py --project {project} --name {args.method} --notes Ablation_wo_projectors --tag Ablation_proj --group ours --job_type {args.dataset} --model LightGCN --loss Adaptive --augment No --lambda1 {lambda1} --temp_tau {temp_tau} --centroid_mode {centroid}\
                    --sampling uii --projector wo --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 5 --seed {args.seed} --c {args.c}')

else:
    print('#=====================#')
    print(args.method)
    print('#=====================#')
