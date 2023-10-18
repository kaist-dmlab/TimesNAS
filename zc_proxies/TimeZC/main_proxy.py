"""
Modified from https://github.com/SLDGroup/ZiCo/blob/main/ZeroShotProxy/compute_zico.py
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np

working_dir = os.path.dirname(os.path.abspath(__file__))



def getgrad(model:torch.nn.Module, grad_dict:dict, step_iter=0):
    if step_iter == 0:
        for name,mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear) or isinstance(mod, nn.Conv1d):
                # print(mod.weight.grad.data.size())
                # print(mod.weight.data.size())
                if mod.weight.grad != None:
                    grad_dict[name]=[mod.weight.grad.data.cpu().reshape(-1).numpy()]
    else:
        for name,mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear) or isinstance(mod, nn.Conv1d):
                if mod.weight.grad != None:
                    grad_dict[name].append(mod.weight.grad.data.cpu().reshape( -1).numpy())
    return grad_dict


def caculate_zico(grad_dict):
    allgrad_array=None
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname]= np.array(grad_dict[modname])
    nsr_mean_sum = 0
    nsr_mean_sum_abs = 0
    nsr_mean_avg = 0
    nsr_mean_avg_abs = 0
    for j, modname in enumerate(grad_dict.keys()):
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx])
        if tmpsum==0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
            nsr_mean_avg_abs += np.log(np.mean(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx]))
    return nsr_mean_sum_abs


def compute_zc_score(exp, args, train_loader, td_infos=None, dft_infos=None, dwt_infos=None):
    grad_dict = {}
    
    network = exp.model
    network.train()
    network.to(exp.device)
    
    if args.task_name == 'long_term_forecast':
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            network.zero_grad()
            batch_x, batch_y = batch_x.float().to(exp.device), batch_y.float().to(exp.device)
            batch_x_mark, batch_y_mark = batch_x_mark.float().to(exp.device), batch_y_mark.float().to(exp.device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)
            
            if td_infos == None and dft_infos == None and dwt_infos == None:
                outputs = network(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                td_input = td_infos[i].to(exp.device)
                dft_input = dft_infos[i].to(exp.device)
                dwt_input = dwt_infos[i].to(exp.device)
                outputs = network(batch_x, batch_x_mark, dec_inp, batch_y_mark, td_input=td_input, dft_input=dft_input, dwt_input=dwt_input)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].float().to(exp.device)
            
            loss = exp._select_criterion()(outputs, batch_y)            
            loss.backward()
                        
            grad_dict = getgrad(network, grad_dict, i)
                
    elif args.task_name == 'short_term_forecast':
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            network.zero_grad()
            batch_x, batch_y = batch_x.float().to(exp.device), batch_y.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :exp.args.label_len, :], dec_inp], dim=1).float().to(exp.device)
            
            if td_infos == None and dft_infos == None and dwt_infos == None:
                outputs = network(batch_x, None, dec_inp, None)
            else:
                td_input = td_infos[i].to(exp.device)
                dft_input = dft_infos[i].to(exp.device)
                dwt_input = dwt_infos[i].to(exp.device)
                outputs = outputs = network(batch_x, None, dec_inp, None, td_input=td_input, dft_input=dft_input, dwt_input=dwt_input)
            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -exp.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -exp.args.pred_len:, f_dim:].to(exp.device)
            batch_y_mark = batch_y_mark[:, -exp.args.pred_len:, f_dim:].to(exp.device)
            
            loss = exp._select_criterion(args.loss)(batch_x, exp.args.frequency_map, outputs, batch_y, batch_y_mark)
            loss.backward()
            
            grad_dict = getgrad(network, grad_dict, i)
            
    elif args.task_name == 'classification':
        for i, (batch_x, label, padding_mask) in enumerate(train_loader):
            network.zero_grad()
            batch_x = batch_x.float().to(exp.device)
            label = label.to(exp.device)
            padding_mask = padding_mask.float().to(exp.device)
            
            if td_infos == None and dft_infos == None and dwt_infos == None:
                outputs = network(batch_x, padding_mask, None, None)
            else:
                td_input = td_infos[i].to(exp.device)
                dft_input = dft_infos[i].to(exp.device)
                dwt_input = dwt_infos[i].to(exp.device)                
                outputs = network(batch_x, padding_mask, None, None, td_input=td_input, dft_input=dft_input, dwt_input=dwt_input)
            
            loss = exp._select_criterion()(outputs, label.long().squeeze(-1))
            loss.backward()
            
            grad_dict = getgrad(network, grad_dict, i)
        
    elif args.task_name == 'anomaly_detection':
        for i, (batch_x, batch_y) in enumerate(train_loader):
            network.zero_grad()
            batch_x = batch_x.float().to(exp.device)
            
            if td_infos == None and dft_infos == None and dwt_infos == None:
                outputs = network(batch_x, None, None, None)
            else:
                td_input = td_infos[i].to(exp.device)
                dft_input = dft_infos[i].to(exp.device)
                dwt_input = dwt_infos[i].to(exp.device)                
                outputs = network(batch_x, None, None, None, td_input=td_input, dft_input=dft_input, dwt_input=dwt_input)
            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            
            loss = exp._select_criterion()(outputs, batch_x)
            loss.backward()
                        
            grad_dict = getgrad(network, grad_dict, i)
            
    elif args.task_name == 'imputation':
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            network.zero_grad()
            batch_x = batch_x.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            
            B, T, N = batch_x.shape
            mask = torch.rand((B, T, N)).to(exp.device)
            mask[mask <= args.mask_rate] = 0  # masked
            mask[mask > args.mask_rate] = 1  # remained
            inp = batch_x.masked_fill(mask == 0, 0)
            
            if td_infos == None and dft_infos == None and dwt_infos == None:
                outputs = network(inp, batch_x_mark, None, None, mask)
            else:
                td_input = td_infos[i].to(exp.device)
                dft_input = dft_infos[i].to(exp.device)
                dwt_input = dwt_infos[i].to(exp.device)                
                outputs = network(inp, batch_x_mark, None, None, mask, td_input=td_input, dft_input=dft_input, dwt_input=dwt_input)
                        
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            
            loss = exp._select_criterion()(outputs[mask == 0], batch_x[mask == 0])
            loss.backward()
            
            grad_dict = getgrad(network, grad_dict, i)
            
    else:
        print('not implemented task!')
        return
    
    res = caculate_zico(grad_dict)
    
    del network
    del train_loader
    
    torch.cuda.empty_cache()
    
    return res