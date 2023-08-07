import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Go GCLRec")
    parser.add_argument('--dataset', type=str, default='yelp2018', help="dataset")
    parser.add_argument('--method', type=str, default='LightGCN', help="method: LightGCN, GTN, SGL-ED, SGL-RW, SimGCL, BC, PDA, Adaloss (, LightPopMix)")
    parser.add_argument('--device', type=int, default=0, help="device")
    return parser.parse_args()
args = parse_args()
#For all
lr = 0.001
weight_decay = 1e-4
num_layers = 3
latent_dim_rec = 64
batch_size =2048
init_method = 'Normal'
seed = 2023
valid = 0
project = 'TestCode_No_Valid'
visual = 0
c = '_'


if args.method == 'LightGCN':
    #HyperParms: None
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion', 'Tencent']:
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag {args.method} --group baseline --job_type {args.dataset} --model LightGCN --loss BPR --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {valid} --if_visual {visual} --visual_epoch 5 --seed {seed} --c {c}')

if args.method == 'GTN':
    #HyperParms: None
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion', 'Tencent']:
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag {args.method} --group baseline --job_type {args.dataset} --model GTN --loss BPR --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {valid} --if_visual {visual} --visual_epoch 5 --seed {seed} --c {c}')

if args.method == 'SGL-ED':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion', 'Tencent']:
        lambda1 = 0.1
        temp_tau = 0.2
        p_drop = 0.1
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag SGL --group baseline --job_type {args.dataset} --model SGL --loss BPR_Contrast --augment ED --lambda1 {lambda1} --p_drop {p_drop} --temp_tau {temp_tau} \
                    --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {valid} --if_visual {visual} --visual_epoch 1 --seed {seed} --c {c}')
        
if args.method == 'SGL-RW':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion', 'Tencent']:
        lambda1 = 0.1
        temp_tau = 0.2
        p_drop = 0.1
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag SGL --group baseline --job_type {args.dataset} --model SGL --loss BPR_Contrast --augment RW --lambda1 {lambda1} --p_drop {p_drop} --temp_tau {temp_tau} \
                    --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {valid} --if_visual {visual} --visual_epoch 1 --seed {seed} --c {c}')
        
if args.method == 'SimGCL':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion', 'Tencent']:
        lambda1 = 0.1
        temp_tau = 0.2
        eps_SimGCL = 0.1
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag {args.method} --group baseline --job_type {args.dataset} --model SimGCL --loss BPR_Contrast --augment No --lambda1 {lambda1} --eps_SimGCL {eps_SimGCL} --temp_tau {temp_tau} \
                    --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {valid} --if_visual {visual} --visual_epoch 1 --seed {seed} --c {c}')
        
if args.method == 'PDA':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion', 'Tencent']:
        pop_gamma = 0.02
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag {args.method} --group baseline --job_type {args.dataset} --model LightGCN --loss PDA --augment No --pop_gamma {pop_gamma} \
                    --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {valid} --if_visual {visual} --visual_epoch 4 --seed {seed} --c {c}')
        
if args.method == 'BC':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion', 'Tencent']:
        temp_tau = 0.1
        lambda1 = 0.1
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag {args.method} --group baseline --job_type {args.dataset} --model LightGCN --loss BC --augment No --lambda1 {lambda1} --temp_tau_pop {temp_tau} --temp_tau {temp_tau} \
                    --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {valid} --if_visual {visual} --visual_epoch 5 --seed {seed} --c {c}')
        
if args.method == 'Adaloss':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book']:
        temp_tau = 0.1
        lambda1 = 0.1
        centroid = 'eigenvector'
        os.system(f'python main.py --project {project} --name {args.method} --notes Ablation_only_Adaloss --tag Ablation_loss --group ours --job_type {args.dataset} --model LightGCN --loss Adaptive --augment No --lambda1 {lambda1} --temp_tau {temp_tau} --centroid_mode {centroid}\
                    --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {valid} --if_visual {visual} --visual_epoch 5 --seed {seed} --c {c}')
    
    if args.dataset in ['ifashion', 'last-fm', 'Tencent']:
        temp_tau = 0.1
        lambda1 = 0.1
        centroid = 'pagerank'
        os.system(f'python main.py --project {project} --name {args.method} --notes Ablation_only_Adaloss --tag Ablation_loss --group ours --job_type {args.dataset} --model LightGCN --loss Adaptive --augment No --lambda1 {lambda1} --temp_tau {temp_tau} --centroid_mode {centroid}\
                    --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {valid} --if_visual {visual} --visual_epoch 5 --seed {seed} --c {c}')