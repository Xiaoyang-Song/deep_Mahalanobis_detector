import numpy as np
import os

# TO RUN, use the following: python jobs/SEEOOD_Baselines/summarize.py  > jobs/SEEOOD_Baselines/summary.txt

exp_name = ['mnist', 'fashionmnist', 'mnist-fashionmnist', 'svhn', 'cifar10-svhn']

# DEFINE THE INDEX OF THE LINE TO EXTRACT
# Index: [msp, odin, maha]
# If source code is changed, this index may need to be updated
index_dict = {
    'mnist': [236, 241, -3],
    'fashionmnist': [236, 241, -3],
    'mnist-fashionmnist': [191, 196, -3],
    'svhn': [326, 331, -3],
    'cifar10-svhn': [236, 241, -3]
}

method = ['MSP', 'ODIN', 'MAHA']

for i in range(3):
    print("-"*50)
    print(f"METHOD: {method[i]}")
    for exp in exp_name:

        log_path = os.path.join('jobs', 'SEEOOD_Baselines', 'out', f'{exp}.log')
        with open(log_path, 'r') as f:
            lines = f.readlines()

            lines = lines[index_dict[exp][i]]

            results = []
            for _ in range(6):
                results.append(lines[0:6].strip())
                lines = lines[6:].strip()
            # print(results)
            tpr95 = results[0]
            tpr99 = results[1]
            auroc = results[2] 
            
            print(f'Experiment: {exp}')
            print(f"TPR95: {tpr95}")
            print(f"TPR99: {tpr99}")
            print(f"AUROC: {auroc}\n")
    
    print("-"*50 + "\n\n\n")