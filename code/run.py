import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Go GCLRec")
    parser.add_argument('--dataset', type=str, default='yelp2018', help="dataset")
    parser.add_argument('--method', type=str, default='LightPopMix', help="method: LightPopMix, LightPopMix_wo_p, LightPopMix_lambda, LightPopMix_tau, LightPopMix_grid")
    parser.add_argument('--device', type=int, default=0, help="device")
    return parser.parse_args()
args = parse_args()


if args.method == 'LightPopMix':
    if args.dataset in ['yelp2018', 'gowalla']:
        temp_tau = 0.1
        lambda1 = 0.1
        centroid = 'eigenvector'
    if args.dataset in ['amazon-book']:
        temp_tau = 0.09
        lambda1 = 0.1
        centroid = 'eigenvector'
        
    if args.dataset in ['ifashion']:
        temp_tau = 0.15
        lambda1 = 0.1
        centroid = 'pagerank'
    if args.dataset in ['last-fm']:
        temp_tau = 0.1
        lambda1 = 0.001
        centroid = 'pagerank'
    if args.dataset in ['Tencent']:
        temp_tau = 0.1
        lambda1 = 0.1
        centroid = 'pagerank'

    os.system(f'python main.py --project TestCode_Valid --name {args.method} --notes _ --tag LightPopMix --group ours --job_type {args.dataset} --model LightGCN --loss Adaptive --augment No --lambda1 {lambda1} --temp_tau {temp_tau} --centroid_mode {centroid}\
                --sampling uii --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid 0 --if_visual 0 --visual_epoch 5 --seed 2023 --c _')


else:
    print('#=====================#')
    print(args.method)
    print('#=====================#')
