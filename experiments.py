# original training configs
TRAINING_CONFIGS = {
    'long_term_forecast': {
        'lr': 0.0001,
        'batch_size': 32,
        'train_epochs': 10
    }, 
    'short_term_forecast': {
        'lr': 0.001,
        'batch_size': 16,
        'train_epochs': 10,
        'loss': 'SMAPE'
    },
    'imputation': {
        'lr': 0.001,
        'batch_size': 16,
        'train_epochs': 10
    }, 
    'classification': {
        'lr': 0.001,
        'batch_size': 16,
        'train_epochs': 30,
        'patience': 10
    }, 
    'anomaly_detection': {
        'lr': 0.0001,
        'batch_size': 128,
        'train_epochs': 10
    }
}

DATA_CONFIGS = {
    'ETTh1': {'data': 'ETTh1', 'enc_in': 7, 'dec_in': 7, 'c_out': 7, 'root_path': './dataset/ETT-small/', 'data_path': 'ETTh1.csv'}, 
    'ETTh2': {'data': 'ETTh2', 'enc_in': 7, 'dec_in': 7, 'c_out': 7, 'root_path': './dataset/ETT-small/', 'data_path': 'ETTh2.csv'}, 
    'ETTm1': {'data': 'ETTm1', 'enc_in': 7, 'dec_in': 7, 'c_out': 7, 'root_path': './dataset/ETT-small/', 'data_path': 'ETTm1.csv'}, 
    'ETTm2': {'data': 'ETTm2', 'enc_in': 7, 'dec_in': 7, 'c_out': 7, 'root_path': './dataset/ETT-small/', 'data_path': 'ETTm2.csv'},
    'ECL': {'data': 'custom', 'enc_in': 321, 'dec_in': 321, 'c_out': 321, 'root_path': './dataset/electricity/', 'data_path': 'electricity.csv'},
    'weather': {'data': 'custom', 'enc_in': 21, 'dec_in': 21, 'c_out': 21, 'root_path': './dataset/weather/', 'data_path': 'weather.csv'},
    'traffic': {'data': 'custom', 'enc_in': 862, 'dec_in': 862, 'c_out': 862, 'root_path': './dataset/traffic/', 'data_path': 'traffic.csv'},
    'Exchange': {'data': 'custom', 'enc_in': 8, 'dec_in': 8, 'c_out': 8, 'root_path': './dataset/exchange_rate/', 'data_path': 'exchange_rate.csv'},
    'MSL': {'data': 'MSL', 'enc_in': 55, 'c_out': 55, 'root_path': './dataset/MSL/', 'anomaly_ratio': 1.0},
    'PSM': {'data': 'PSM', 'enc_in': 25, 'c_out': 25, 'root_path': './dataset/PSM/', 'anomaly_ratio': 1.0},
    'SMAP': {'data': 'SMAP', 'enc_in': 25, 'c_out': 25, 'root_path': './dataset/SMAP/', 'anomaly_ratio': 1.0},
    'SMD': {'data': 'SMD', 'enc_in': 38, 'c_out': 38, 'root_path': './dataset/SMD/', 'anomaly_ratio': 0.5},
    'SWaT': {'data': 'SWAT', 'enc_in': 51, 'c_out': 51, 'root_path': './dataset/SWaT/', 'anomaly_ratio': 1.0},
}