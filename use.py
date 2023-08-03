import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Go GCLRec")
    parser.add_argument('--dataset', type=str, default='yelp2018', help="dataset")
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument('--device', type=int, default=0, help="device")
    return parser.parse_args()
args = parse_args()


dataset = None
method = None
seed = None
device = None
visual = None
valid = None
c = None
temp_tau = None
lambda1 = None


if True:
    os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')
    # python run-any.py --dataset yelp2018 --method BC --seed 2023 --device 0 --visual 0 --valid 0 --c NTS --temp_tau 0.1 --lambda1 0.1