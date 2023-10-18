import argparse, os, logging, warnings, json, uuid, random

import torch
import numpy as np
import experiments as exp_setup

from data_provider.m4 import M4Meta
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNAS')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='TimesNAS',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    # TimesNAS
    parser.add_argument('--dft_size', type=int, default=None, help='size of auxiliary task infomartion')
    parser.add_argument('--dwt_size', type=int, default=None, help='size of auxiliary frequency-domain decomposition infomartion')
    parser.add_argument('--td_size', type=int, default=None, help='size of auxiliary time-domain decomposition infomartion')
    parser.add_argument('--method_name', type=str, default='TimesNAS', help='searcher method')    
    parser.add_argument('--data_name', type=str, help='semantic data name')

    # data loader
    parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
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
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=3, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    gpu = args.gpu
    if gpu is not None:
        torch.cuda.set_device('cuda:{}'.format(gpu))
        torch.backends.cudnn.benchmark = True        
    
    args.is_training = 1
    args.model = 'TimesNAS'
    args.des = 'Exp'
    args.itr = 3
    
    args.batch_size = exp_setup.TRAINING_CONFIGS[args.task_name]['batch_size']
    args.train_epochs = exp_setup.TRAINING_CONFIGS[args.task_name]['train_epochs']
    args.learning_rate = exp_setup.TRAINING_CONFIGS[args.task_name]['lr']      

    if args.task_name == 'long_term_forecast':
        args.features = 'M'
        args.seq_len = 96
        args.label_len = 48
        
        args.data = exp_setup.DATA_CONFIGS[args.data_name]['data']
        args.enc_in = exp_setup.DATA_CONFIGS[args.data_name]['enc_in']
        args.dec_in = exp_setup.DATA_CONFIGS[args.data_name]['dec_in']
        args.c_out = exp_setup.DATA_CONFIGS[args.data_name]['c_out']
        args.root_path = exp_setup.DATA_CONFIGS[args.data_name]['root_path']
        args.data_path = exp_setup.DATA_CONFIGS[args.data_name]['data_path'] 
        
        args.model_id = f'{args.data_name}_96_{args.pred_len}' 
        
        Exp = Exp_Long_Term_Forecast
        
    elif args.task_name == 'short_term_forecast':
        args.root_path = './dataset/m4'
        args.data = 'm4'
        args.features = 'M'
        args.loss = 'SMAPE'
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1        
        
        Exp = Exp_Short_Term_Forecast
        
    elif args.task_name == 'imputation':
        args.features = 'M'
        args.seq_len = 96 
        args.pred_len = 0
        args.label_len = 0

        args.model_id = f'{args.data_name}_mask_{args.mask_rate}'
        args.data = exp_setup.DATA_CONFIGS[args.data_name]['data']
        args.root_path = exp_setup.DATA_CONFIGS[args.data_name]['root_path']
        args.data_path = exp_setup.DATA_CONFIGS[args.data_name]['data_path']
        args.enc_in = exp_setup.DATA_CONFIGS[args.data_name]['enc_in']
        args.c_out = exp_setup.DATA_CONFIGS[args.data_name]['c_out']
          
        Exp = Exp_Imputation
        
    elif args.task_name == 'anomaly_detection':
        args.seq_len = 100 
        args.features = 'M'
        args.pred_len = 0

        args.root_path = f'./dataset/{args.data_name}'
        args.model_id = args.data_name.upper()
        args.data = exp_setup.DATA_CONFIGS[args.data_name]['data']
        args.enc_in = exp_setup.DATA_CONFIGS[args.data_name]['enc_in']
        args.c_out = exp_setup.DATA_CONFIGS[args.data_name]['c_out']
        args.anomaly_ratio = exp_setup.DATA_CONFIGS[args.data_name]['anomaly_ratio']

        Exp = Exp_Anomaly_Detection
        
    elif args.task_name == 'classification':
        args.data = 'UEA'
        args.patience = 10
        
        args.root_path = f'./dataset/{args.data_name}/'
        args.model_id = args.data_name
        
        Exp = Exp_Classification


    for ii in range(args.itr):
        # setting record of experiments
        if args.task_name == 'long_term_forecast':
            arch_configs = np.load(f'zc_results/{args.task_name}/{args.data_name}.npy', allow_pickle=True).item()
            args.d_model =  arch_configs['d_model']
            args.d_ff = arch_configs['d_ff']
            args.num_kernels = arch_configs['num_kernels']
            args.top_k = arch_configs['top_k']
            args.e_layers = arch_configs['e_layers']
            args.dropout = arch_configs['dropout']
            args.embed = arch_configs['embed']
            
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)            

            print('Args in experiment:')
            print(args)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            _, train_stats = exp.train(setting)                
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mae, mse, rmse, mape, mspe = exp.test(setting)
            torch.cuda.empty_cache()

            arch_perf = {
                'key': setting,
                'arch_configs': arch_configs,
                'train_stats': train_stats,
                'final_mae': mae,
                'final_mse': mse,
                'final_rmse': rmse,
                'final_mape': mape,
                'final_mspe': mspe,
            }
            np.save(f"arch_results/{args.task_name}/{args.method_name}_{setting}_Final.npy", arch_perf)
            print(f'SAVED {setting} !!!')

        elif args.task_name == 'short_term_forecast':
            key_id = str(uuid.uuid1())[:8]
            arch_perf = {
                'key': f'm4_overall_{key_id}',
                'arch_configs': {},
                'train_stats': {},
            }
            for idx, seasonal_patterns in enumerate(M4Meta.seasonal_patterns):
                args.seasonal_patterns = seasonal_patterns
                args.model_id = f'm4_{seasonal_patterns}'

                arch_configs = np.load(f'zc_results/{args.task_name}/{args.seasonal_patterns}.npy', allow_pickle=True).item()
                args.d_model =  arch_configs['d_model']
                args.d_ff = arch_configs['d_ff']
                args.num_kernels = arch_configs['num_kernels']
                args.top_k = arch_configs['top_k']
                args.e_layers = arch_configs['e_layers']
                args.dropout = arch_configs['dropout']
                args.embed = arch_configs['embed']
                
                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    args.task_name,
                    args.model_id,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des, ii)                

                print('Args in experiment:')
                print(args)

                exp = Exp(args)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                _, train_stats = exp.train(setting)
                arch_perf['train_stats'][seasonal_patterns] = train_stats
                arch_perf['arch_configs'][seasonal_patterns] = arch_configs

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                final_perf = exp.test(setting)
                if idx == len(M4Meta.seasonal_patterns) - 1 and final_perf != None:
                    smape_results, owa_results, mape, mase = final_perf
                    arch_perf['final_smape'] = smape_results
                    arch_perf['final_owa'] = owa_results
                    arch_perf['final_mape'] = mape
                    arch_perf['final_mase'] = mase                

                torch.cuda.empty_cache()
            np.save(f"arch_results/{args.task_name}/{args.method_name}_{setting}_Final.npy", arch_perf)
            print(f'SAVED {setting} !!!')
            
        elif args.task_name == 'classification':                        
            arch_configs = np.load(f'zc_results/{args.task_name}/{args.data_name}.npy', allow_pickle=True).item()
            args.d_model =  arch_configs['d_model']
            args.d_ff = arch_configs['d_ff']
            args.num_kernels = arch_configs['num_kernels']
            args.top_k = arch_configs['top_k']
            args.e_layers = arch_configs['e_layers']
            args.dropout = arch_configs['dropout']
            args.embed = arch_configs['embed']
            
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)            
            
            print('Args in experiment:')
            print(args)
            
            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            _, train_stats = exp.train(setting)            
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            final_acc = exp.test(setting)
            torch.cuda.empty_cache()
            
            arch_perf = {
                'key': setting,
                'arch_configs': arch_configs,
                'train_stats': train_stats,
                'final_acc': final_acc
            }            
            np.save(f"arch_results/{args.task_name}/{args.method_name}_{setting}_Final.npy", arch_perf)
            print(f'SAVED {setting} !!!')
            
        elif args.task_name == 'anomaly_detection':
            arch_configs = np.load(f'zc_results/{args.task_name}/{args.data_name}.npy', allow_pickle=True).item()
            args.d_model =  arch_configs['d_model']
            args.d_ff = arch_configs['d_ff']
            args.num_kernels = arch_configs['num_kernels']
            args.top_k = arch_configs['top_k']
            args.e_layers = arch_configs['e_layers']
            args.dropout = arch_configs['dropout']
            args.embed = arch_configs['embed']

            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)            

            print('Args in experiment:')
            print(args)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            _, train_stats = exp.train(setting)            
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            accuracy, precision, recall, f_score = exp.test(setting)
            torch.cuda.empty_cache()

            arch_perf = {
                'key': setting,
                'arch_configs': arch_configs,
                'train_stats': train_stats,
                'final_acc': accuracy,
                'final_precision': precision,
                'final_recall': recall,
                'final_f1': f_score
            }
            np.save(f"arch_results/{args.task_name}/{args.method_name}_{setting}_Final.npy", arch_perf)
            print(f'SAVED {setting} !!!')
            
        elif args.task_name == 'imputation' and ii > 0:
            arch_configs = np.load(f'zc_results/{args.task_name}/{args.data_name}.npy', allow_pickle=True).item()
            args.d_model =  arch_configs['d_model']
            args.d_ff = arch_configs['d_ff']
            args.num_kernels = arch_configs['num_kernels']
            args.top_k = arch_configs['top_k']
            args.e_layers = arch_configs['e_layers']
            args.dropout = arch_configs['dropout']
            args.embed = arch_configs['embed']
            
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)            

            print('Args in experiment:')
            print(args)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            _, train_stats = exp.train(setting)                
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mae, mse, rmse, mape, mspe = exp.test(setting)
            torch.cuda.empty_cache()
            
            arch_perf = {
                'key': setting,
                'arch_configs': arch_configs,
                'train_stats': train_stats,
                'final_mae': mae,
                'final_mse': mse,
                'final_rmse': rmse,
                'final_mape': mape,
                'final_mspe': mspe
            }

            np.save(f"arch_results/{args.task_name}/{args.method_name}_{setting}_Final.npy", arch_perf)
            print(f'SAVED {setting} !!!')            