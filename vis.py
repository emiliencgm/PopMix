#Yelp2018, iFashion 长尾分布

#Yelp2018上LightGCN，SimGCL的Ratings

#Yelp2018上LightGCN，SimGCL，BC loss， Adaloss的user、item分布

#Yelp2018上LightGCN和PopMix的double label

import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Go GCLRec")
    parser.add_argument('--device', type=int, default=0, help="device")
    parser.add_argument('--method', type=str, default='LightGCN', help="method")
    return parser.parse_args()
args = parse_args()

dataset = 'yelp2018'
device = args.device
method = args.method
seed = 2023
visual = 1
valid = 0
c = 'VISUAL'
temp_tau = 0.1
lambda1 = 0.1

os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')