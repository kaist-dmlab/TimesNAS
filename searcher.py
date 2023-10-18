import argparse, os, logging, warnings, json, uuid, random

logging.disable(logging.WARNING)
warnings.filterwarnings('ignore')

import time
import torch
import search_space
import numpy as np
import experiments as exp_setup

from torch import nn
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification

from zc_proxies.TimeZC.main_proxy import compute_zc_score

from statsmodels.tsa.seasonal import STL
from scipy.fft import fft, dct
from pywt import dwt

from data_provider.m4 import M4Meta
from data_provider.data_factory import data_provider

method_name = f'TimesNAS'
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

def main(Exp, args):
    
    exp = Exp(args)
    if args.task_name == 'classification':
        _, train_loader = exp._get_data(flag='TRAIN')        
    else:
        _, train_loader = exp._get_data(flag='train')

    trainbatches = []
    td_infos, dft_infos, dwt_infos = [], [], []
    for i, batch in enumerate(train_loader):
        batch_td, batch_dft, batch_dwt = [], [], []
        if i == args.maxbatch:
            break

        # idx = np.random.choice(args.batch_size)

        if args.task_name == 'long_term_forecast':
            batch_x = batch[0]

            for idx in range(args.batch_size):
                x_resid = []
                for v in range(batch_x[idx].shape[1]):
                    stl_result = STL(batch_x[idx][:, v], period=3).fit()
                    x_resid.append(stl_result.resid)
                batch_td.append(np.array(x_resid).transpose())
                batch_dft.append(torch.fft.fft(batch_x[idx]).abs().numpy())
                batch_dwt.append(np.concatenate(dwt(batch_x[idx].numpy(), 'haar')))

        elif args.task_name == 'short_term_forecast':
            batch_x = batch[0]

            for idx in range(args.batch_size):
                x_resid = []
                for v in range(batch_x[idx].shape[1]):
                    stl_result = STL(batch_x[idx][:, v], period=3).fit()
                    x_resid.append(stl_result.resid)
                batch_td.append(np.array(x_resid).transpose())
                batch_dft.append(torch.fft.fft(batch_x[idx]).abs().numpy())
                batch_dwt.append(np.concatenate(dwt(batch_x[idx].numpy(), 'haar')))

        elif args.task_name == 'classification':
            batch_x = batch[0]
            
            for idx in range(args.batch_size):
                x_resid = []
                for v in range(batch_x[idx].shape[1]):
                    stl_result = STL(batch_x[idx][:, v], period=3).fit()
                    x_resid.append(stl_result.resid)
                batch_td.append(np.array(x_resid).transpose())
                batch_dft.append(torch.fft.fft(batch_x[idx]).abs().numpy())
                batch_dwt.append(np.concatenate(dwt(batch_x[idx].numpy(), 'haar')))

        elif args.task_name == 'anomaly_detection':
            batch_x = batch[0]

            for idx in range(args.batch_size):
                x_resid = []
                for v in range(batch_x[idx].shape[1]):
                    stl_result = STL(batch_x[idx][:, v], period=3).fit()
                    x_resid.append(stl_result.resid)            
                batch_td.append(np.array(x_resid).transpose())
                batch_dft.append(torch.fft.fft(batch_x[idx]).abs().numpy())
                batch_dwt.append(np.concatenate(dwt(batch_x[idx].numpy(), 'haar')))

        elif args.task_name == 'imputation':
            batch_x = batch[0]

            for idx in range(args.batch_size):
                x_resid = []
                for v in range(batch_x[idx].shape[1]):
                    stl_result = STL(batch_x[idx][:, v], period=3).fit()
                    x_resid.append(stl_result.resid)
                batch_td.append(np.array(x_resid).transpose())
                batch_dft.append(torch.fft.fft(batch_x[idx]).abs().numpy())
                batch_dwt.append(np.concatenate(dwt(batch_x[idx].numpy(), 'haar')))

        td_infos.append(torch.Tensor(batch_td))
        dft_infos.append(torch.Tensor(batch_dft))
        dwt_infos.append(torch.Tensor(batch_dwt))

        trainbatches.append(batch)
    
    for subset_size in [int(args.population_size)]:
        popu_structure_list = []
        popu_zero_shot_score_list = []

        candidate_archs = np.load(f'arch_results/{args.task_name}_{subset_size}.npy', allow_pickle=True)
        
        start_timer = time.time()
        for arch_configs in candidate_archs:
            args.d_model = arch_configs['d_model']
            args.d_ff = arch_configs['d_ff']
            args.num_kernels = arch_configs['num_kernels']
            args.top_k = arch_configs['top_k']
            args.e_layers = arch_configs['e_layers']
            args.dropout = arch_configs['dropout']
            args.embed = arch_configs['embed']
            
            args.td_size = td_infos[0].shape[1]
            args.dft_size = dft_infos[0].shape[1]
            args.dwt_size = dwt_infos[0].shape[1]
            
            print(args.model_id)
            print(arch_configs)
            exp = Exp(args)  # set experiments

            the_nas_score = compute_zc_score(exp, args, trainbatches, td_infos=td_infos, dft_infos=dft_infos, dwt_infos=dwt_infos)

            popu_structure_list.append(arch_configs)
            popu_zero_shot_score_list.append(the_nas_score)
        end_time = time.time() - start_timer

        # export best structure
        best_score = max(popu_zero_shot_score_list)
        best_idx = popu_zero_shot_score_list.index(best_score)
        best_structure_str = popu_structure_list[best_idx]
        # summary architecture:score pairs
        arch_scores = {
            'best': {
                'arch_configs': best_structure_str,
                'score': best_score,
                'idx': best_idx
            },
            'arch_scores': {
                'candidate_archs': popu_structure_list,
                'cadidate_scores': popu_zero_shot_score_list
            },
            'search_time': end_time
        }
        np.save(f'zc_results/{args.task_name}/{args.data_name}.npy', arch_scores, allow_pickle=True)
        print(f'Saved! >>> zc_results/{args.task_name}/{args.data_name}.npy')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimesNAS')
    # ZiCo
    parser.add_argument('--population_size', type=int, default=1000, help='population size of evolution.')
    parser.add_argument('--save_dir', type=str, default=None, help='output directory')
    parser.add_argument('--maxbatch', type=int, default=2, help='N in Eq. (15)')
    # TimesNAS
    parser.add_argument('--dft_size', type=int, default=None, help='size of auxiliary task infomartion')
    parser.add_argument('--dwt_size', type=int, default=None, help='size of auxiliary frequency-domain decomposition infomartion')
    parser.add_argument('--td_size', type=int, default=None, help='size of auxiliary time-domain decomposition infomartion')
    parser.add_argument('--method_name', type=str, default='TimesNAS', help='searcher method')    
    parser.add_argument('--data_name', type=str, help='semantic data name')
    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast', help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='Autoformer', help='model name, options: [Autoformer, Transformer, TimesNet]') 
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    # data loader
    parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')
    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    # optimization
    parser.add_argument('--num_workers', type=int, default=40, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)    

    args = parser.parse_args()
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    gpu = args.gpu
    if gpu is not None:
        torch.cuda.set_device('cuda:{}'.format(gpu))
        torch.backends.cudnn.benchmark = True    
    
    data_name = args.data_name
    args.is_training = 1
    args.model = 'TimesNAS'
    args.des = 'Exp'
    args.itr = 1

    if args.task_name == 'long_term_forecast':        
        args.features = 'M'
        args.seq_len = 96
        args.label_len = 48

        args.train_epochs = exp_setup.TRAINING_CONFIGS[args.task_name]['train_epochs']
        args.learning_rate = exp_setup.TRAINING_CONFIGS[args.task_name]['lr']
        
        Exp = Exp_Long_Term_Forecast

        args.data = exp_setup.DATA_CONFIGS[data_name]['data']
        args.enc_in = exp_setup.DATA_CONFIGS[data_name]['enc_in']
        args.dec_in = exp_setup.DATA_CONFIGS[data_name]['dec_in']
        args.c_out = exp_setup.DATA_CONFIGS[data_name]['c_out']
        args.root_path = exp_setup.DATA_CONFIGS[data_name]['root_path']
        args.data_path = exp_setup.DATA_CONFIGS[data_name]['data_path']

        pred_lens = [96, 192, 336, 720]
        for pred_len in pred_lens:
            args.model_id = f'{data_name}_96_{pred_len}'
            args.pred_len = pred_len
            main(Exp, args)
        
    elif args.task_name == 'short_term_forecast':
        args.root_path = './dataset/m4'
        args.data = 'm4'
        args.features = 'M'
        args.loss = 'SMAPE'
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1
        
        args.batch_size = exp_setup.TRAINING_CONFIGS[args.task_name]['batch_size']
        args.train_epochs = exp_setup.TRAINING_CONFIGS[args.task_name]['train_epochs']
        args.learning_rate = exp_setup.TRAINING_CONFIGS[args.task_name]['lr']        
        
        Exp = Exp_Short_Term_Forecast
        
        for seasonal_patterns in M4Meta.seasonal_patterns:
            args.seasonal_patterns = seasonal_patterns
            args.model_id = f'm4_{seasonal_patterns}'
            main(Exp, args)
            
    elif args.task_name == 'classification':
        args.data = 'UEA'
        
        args.batch_size = exp_setup.TRAINING_CONFIGS[args.task_name]['batch_size']
        args.train_epochs = exp_setup.TRAINING_CONFIGS[args.task_name]['train_epochs']
        args.learning_rate = exp_setup.TRAINING_CONFIGS[args.task_name]['lr']
        args.patience = 10
        
        Exp = Exp_Classification

        args.root_path = f'./dataset/{data_name}/'
        args.model_id = data_name

        main(Exp, args)
        
    elif args.task_name == 'anomaly_detection':
        args.seq_len = 100 
        args.features = 'M'
        args.pred_len = 0

        args.batch_size = exp_setup.TRAINING_CONFIGS[args.task_name]['batch_size']
        args.train_epochs = exp_setup.TRAINING_CONFIGS[args.task_name]['train_epochs']
        args.learning_rate = exp_setup.TRAINING_CONFIGS[args.task_name]['lr']
        
        Exp = Exp_Anomaly_Detection

        args.root_path = f'./dataset/{data_name}'
        args.model_id = data_name.upper()
        args.data = exp_setup.DATA_CONFIGS[data_name]['data']
        args.enc_in = exp_setup.DATA_CONFIGS[data_name]['enc_in']
        args.c_out = exp_setup.DATA_CONFIGS[data_name]['c_out']
        args.anomaly_ratio = exp_setup.DATA_CONFIGS[data_name]['anomaly_ratio']

        main(Exp, args)
        
    elif args.task_name == 'imputation':
        args.features = 'M'
        args.seq_len = 96 
        args.pred_len = 0
        args.label_len = 0    

        args.batch_size = exp_setup.TRAINING_CONFIGS[args.task_name]['batch_size']
        args.train_epochs = exp_setup.TRAINING_CONFIGS[args.task_name]['train_epochs']
        args.learning_rate = exp_setup.TRAINING_CONFIGS[args.task_name]['lr']     
        
        Exp = Exp_Imputation
    
        args.data = exp_setup.DATA_CONFIGS[data_name]['data']
        args.root_path = exp_setup.DATA_CONFIGS[data_name]['root_path']
        args.data_path = exp_setup.DATA_CONFIGS[data_name]['data_path']
        args.enc_in = exp_setup.DATA_CONFIGS[data_name]['enc_in']
        args.c_out = exp_setup.DATA_CONFIGS[data_name]['c_out']

        ratios = [0.125, 0.25, 0.375, 0.5]
        for ratio in ratios:
            args.model_id = f'{data_name}_mask_{str(ratio)}'
            args.mask_rate = ratio            
            main(Exp, args)

    else:
        print('not implemented task!')
        exit()
