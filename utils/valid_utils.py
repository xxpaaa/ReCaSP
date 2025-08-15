#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-

import torch
import os
from utils.core_utils import _get_splits, _init_model, _init_loaders, _extract_survival_metadata, _init_loss_function, _summary


def _get_val_results(args,model,train_loader,val_loader,loss_fn):
    all_survival = _extract_survival_metadata(train_loader, val_loader)
    results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss\
        = _summary(args.dataset_factory, model, args.omics_format, val_loader, loss_fn, all_survival, device=args.device, bag_loss=args.bag_loss,
                   lambda_epochs=args.lambda_epochs, num_classes=args.n_classes, epoch=-1, reg_loss=args.reg_loss, reg_loss_alpha=args.reg_loss_alpha) #epoch=-1, 即best epoch

    print('Best Val c-index: {:.4f} | Val c-index2: {:.4f} | Val IBS: {:.4f} | Val iauc: {:.4f}'.format(
        val_cindex, val_cindex_ipcw, val_IBS, val_iauc))

    return results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss


def _val(datasets,cur,args):
    '''

        :param datasets: tuple
        :param cur: Int
        :param args: argspace.Namespace
        :param log_file: file
        :return:
        '''

    # ----> gets splits and summarize
    train_split, val_split = _get_splits(datasets, cur, args)

    # ----> initialize model
    model = _init_model(args)

    # ----> load params of model
    path = os.path.join(args.results_dir, "model_best_s{}.pth".format(cur))
    model.load_state_dict(torch.load(path), strict=True)
    print("Loaded model from {}".format(path))

    # ----> init loss function
    loss_fn = _init_loss_function(args)

    # ----> initialize loaders
    train_loader, val_loader = _init_loaders(args, train_split, val_split)

    # ----> val
    results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _get_val_results(args, model, train_loader, val_loader, loss_fn)
    best_epoch = -1
    return results_dict, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, best_epoch)