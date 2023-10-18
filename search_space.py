import argparse
import numpy as np

BACKBONE = ['Inception', 'ConvNeXt', 'ResNeXt']


BLOCK_LEVEL = {
    'top_k': [1, 2, 3, 4, 5],
    'e_layers': [2, 3, 4, 5, 6],
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'embed': ['timeF', 'fixed', 'learned'],
}

BACKBONE_LEVEL = {
    'd_model': {
        'long_term_forecast': [16, 32, 64, 128, 256],
        'short_term_forecast': [8, 16, 32, 64],
        'imputation': [32, 64, 128],
        'classification': [16, 32, 64],
        'anomaly_detection': [16, 32, 64, 128]
    },
    'ff_ratio': [1., 1.25, 1.5, 1.75, 2.],
    'num_kernels': [2, 4, 6, 8, 10]
}


def get_initial_architecture(task_name):
    arch = {          
        'top_k': np.random.choice(BLOCK_LEVEL['top_k']),
        'e_layers': np.random.choice(BLOCK_LEVEL['e_layers']),
        'dropout': np.random.choice(BLOCK_LEVEL['dropout']),
        'embed': np.random.choice(BLOCK_LEVEL['embed']),

        'num_kernels': np.random.choice(BACKBONE_LEVEL['num_kernels'])
    }
    arch['d_model'] = np.random.choice(BACKBONE_LEVEL['d_model'][task_name])
    arch['d_ff'] = min(int(arch['d_model'] * np.random.choice(BACKBONE_LEVEL['ff_ratio'])), 512)
    
    return arch


def get_random_architectures_block(N=1):
    arch_list = []
    
    for _ in range(N):
        arch = {          
            'top_k': np.random.choice(BLOCK_LEVEL['top_k']),
            'e_layers': np.random.choice(BLOCK_LEVEL['e_layers']),
            'dropout': np.random.choice(BLOCK_LEVEL['dropout']),
            'embed': np.random.choice(BLOCK_LEVEL['embed']),
            
        }
        arch['d_model'] = 64
        arch['d_ff'] = 64
        arch['num_kernels'] = 6
        
        arch_list.append(arch)
    
    return arch_list


def get_random_architectures_backbone(task_name, N=1):
    arch_list = []
    
    for _ in range(N):
        arch = {          
            'top_k': 3,
            'e_layers': 3,
            'dropout': 0.1,
            'embed': 'timeF',
            
            'num_kernels': np.random.choice(BACKBONE_LEVEL['num_kernels'])
        }
        arch['d_model'] = np.random.choice(BACKBONE_LEVEL['d_model'][task_name])
        arch['d_ff'] = min(int(arch['d_model'] * np.random.choice(BACKBONE_LEVEL['ff_ratio'])), 512)
        arch_list.append(arch)
    
    return arch_list


def get_random_architectures(task_name, N=1):
    arch_list = []
    
    for _ in range(N):
        arch = {          
            'top_k': np.random.choice(BLOCK_LEVEL['top_k']),
            'e_layers': np.random.choice(BLOCK_LEVEL['e_layers']),
            'dropout': np.random.choice(BLOCK_LEVEL['dropout']),
            'embed': np.random.choice(BLOCK_LEVEL['embed']),
            
            'num_kernels': np.random.choice(BACKBONE_LEVEL['num_kernels'])
        }
        arch['d_model'] = np.random.choice(BACKBONE_LEVEL['d_model'][task_name])
        arch['d_ff'] = min(int(arch['d_model'] * np.random.choice(BACKBONE_LEVEL['ff_ratio'])), 512)
        arch_list.append(arch)
    
    return arch_list