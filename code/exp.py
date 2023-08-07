import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Go GCLRec")
    parser.add_argument('--dataset', type=str, default='yelp2018', help="dataset")
    parser.add_argument('--init', type=str, default='Normal', help="init")
    parser.add_argument('--device', type=int, default=0, help="device")
    return parser.parse_args()
args = parse_args()

dataset = args.dataset
device = args.device

seed = 2023
init = args.init
visual = 0
valid = 0
c = 'NTS'

if init == 'Normal':
    if dataset in ['yelp2018']:
        for (method, temp_tau, lambda1) in [('LightPopMix', 0.1, 0.08)]:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')

    if dataset in ['gowalla']:
        for (method, temp_tau, lambda1) in [('LightPopMix', 0.11, 0.08)]:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')

    if dataset in ['amazon-book']:
        for (method, temp_tau, lambda1) in []:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')


    if dataset in ['last-fm']:
        for (method, temp_tau, lambda1) in [('LightPopMix', 0.08, 0.001)]:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')


    if dataset in ['ifashion']:
        for (method, temp_tau, lambda1) in [('LightPopMix', 0.15, 0.05)]:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')




if init == 'Xavier':
    if dataset in ['yelp2018']:
        for (method, temp_tau, lambda1) in [('LightPopMix', 0.1, 0.1)]:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')

    if dataset in ['gowalla']:
        for (method, temp_tau, lambda1) in [('LightPopMix', 0.1, 0.1)]:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')

    if dataset in ['amazon-book']:
        for (method, temp_tau, lambda1) in [('LightPopMix', 0.09, 0.1)]:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')


    if dataset in ['last-fm']:
        for (method, temp_tau, lambda1) in [('LightPopMix', 0.1, 0.001),
                                            ('LightPopMix', 0.08, 0.001)]:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')


    if dataset in ['ifashion']:
        for (method, temp_tau, lambda1) in [('LightPopMix', 0.15, 0.05),
                                            ('LightPopMix', 0.1, 0.1),
                                            ('LightPopMix', 0.15, 0.1)]:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')

