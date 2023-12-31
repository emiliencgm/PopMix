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
        for (method, temp_tau, lambda1) in []:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')

    if dataset in ['gowalla']:
        for (method, temp_tau, lambda1) in [('BC', 0.11, 0.1)]:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')

    if dataset in ['amazon-book']:
        for (method, temp_tau, lambda1) in [('BC', 0.08, 0.1)]:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')


    if dataset in ['last-fm']:
        for (method, temp_tau, lambda1) in [('BC', 0.08, 0.001)]:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')


    if dataset in ['ifashion']:
        for (method, temp_tau, lambda1) in [('BC', 0.14, 0.1),
                                            ('SGL-ED', 0.5, 0.4),
                                            ('SGL-RW', 0.5, 0.4),
                                            ('SimGCL', 0.5, 0.4)]:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')




if init == 'Xavier':
    if dataset in ['yelp2018']:
        for (method, temp_tau, lambda1) in [
                                            ('SGL-ED', 0.2, 0.1),
                                            ('SGL-RW', 0.2, 0.1),
                                            ('LightGCN', 0.2, 0.1),
                                            ('GTN', 0.2, 0.1),
                                            ('PDA', 0.2, 0.1),
                                            ('SimGCL', 0.2, 0.1),
                                            ('BC', 0.1, 0.1)]:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')

    if dataset in ['gowalla']:
        for (method, temp_tau, lambda1) in [
                                            ('BC', 0.1, 0.1),
                                            ('SGL-ED', 0.2, 0.1),
                                            ('SGL-RW', 0.2, 0.1),
                                            ('LightGCN', 0.2, 0.1),
                                            ('GTN', 0.2, 0.1),
                                            ('PDA', 0.2, 0.1),
                                            ('SimGCL', 0.2, 0.1)]:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')

    if dataset in ['amazon-book']:
        for (method, temp_tau, lambda1) in [('BC', 0.08, 0.1),
                                            ('SGL-ED', 0.2, 0.1),
                                            ('SGL-RW', 0.2, 0.1),
                                            ('LightGCN', 0.2, 0.1),
                                            ('GTN', 0.2, 0.1),
                                            ('PDA', 0.2, 0.1),
                                            ('SimGCL', 0.2, 0.1),]:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')


    if dataset in ['last-fm']:
        for (method, temp_tau, lambda1) in [
                                            ('BC', 0.1, 0.001),
                                            ('BC', 0.08, 0.001),
                                            ('SGL-ED', 0.2, 0.1),
                                            ('SGL-RW', 0.2, 0.1),
                                            ('SGL-ED', 0.2, 0.01),
                                            ('SGL-RW', 0.2, 0.01),
                                            ('LightGCN', 0.2, 0.1),
                                            ('GTN', 0.2, 0.1),
                                            ('PDA', 0.2, 0.1),
                                            ('SimGCL', 0.2, 0.1),
                                            ('SimGCL', 0.2, 0.01)]:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')


    if dataset in ['ifashion']:
        for (method, temp_tau, lambda1) in [
                                            ('BC', 0.14, 0.1),
                                            ('SGL-ED', 0.5, 0.4),
                                            ('SGL-RW', 0.5, 0.4),
                                            ('SGL-ED', 0.2, 0.1),
                                            ('SGL-RW', 0.2, 0.1),
                                            ('LightGCN', 0.2, 0.1),
                                            ('GTN', 0.2, 0.1),
                                            ('PDA', 0.2, 0.1),
                                            ('SimGCL', 0.5, 0.4),
                                            ('SimGCL', 0.2, 0.1)]:
            os.system(f'python run-any.py --dataset {dataset} --method {method} --seed {seed} --init_method {init} --device {device} --visual {visual} --valid {valid} --c {c} --temp_tau {temp_tau} --lambda1 {lambda1}')

