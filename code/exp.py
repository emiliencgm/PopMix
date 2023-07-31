import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Go GCLRec")
    parser.add_argument('--dataset', type=str, default='yelp2018', help="dataset")
    parser.add_argument('--device', type=int, default=0, help="device")
    return parser.parse_args()
args = parse_args()

dataset = args.dataset
device = args.device

seed = 2023
visual = 0
valid = 0
c = 'NTS'

# for method in ['LightPopMix', 'LightPopMix_wo_p', 'LightPopMix_lambda', 'LightPopMix_tau']:
for method in ['LightPopMix_tau','LightPopMix_lambda']:
    os.system(f'python run.py --dataset {dataset} --method {method} --device {device} --seed {seed} --visual {visual} --valid {valid} --c {c}')
