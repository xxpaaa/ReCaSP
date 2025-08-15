from ast import Lambda
import numpy as np
import pdb
import os
from custom_optims.radam import RAdam
from models.model_ReCaSP import ReCaSP
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv
from utils.file_utils import _save_pkl

from transformers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)


#----> pytorch imports
import torch
from torch.nn.utils.rnn import pad_sequence

from utils.general_utils import _get_split_loader, _print_network, _save_splits
from utils.loss_func import NLLSurvLoss, TrustedSurvLoss, DQNCOSLoss

import torch.optim as optim



def _get_splits(datasets, cur, args):
    r"""
    Summarize the train and val splits and return them individually
    
    Args:
        - datasets : tuple
        - cur : Int 
        - args: argspace.Namespace
    
    Return:
        - train_split : SurvivalDataset
        - val_split : SurvivalDataset
    
    """

    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val splits...', end=' ')
    train_split, val_split = datasets
    if not args.only_test:
        _save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    return train_split,val_split


def _init_loss_function(args):
    r"""
    Init the survival loss function
    
    Args:
        - args : argspace.Namespace 
    
    Returns:
        - loss_fn : NLLSurvLoss or NLLRankSurvLoss
    
    """
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'trust_nll_surv':
        loss_fn = TrustedSurvLoss(alpha=args.alpha_surv, beta=args.beta_surv, survLoss_type='nll_surv')
    else:
        raise NotImplementedError

    if args.reg_loss is not None:
        #"mse_loss", "l1_loss"
        if args.reg_loss == "mse_loss":
            loss_fn = [loss_fn, torch.nn.MSELoss()]
        elif args.reg_loss == "l1_loss":
            loss_fn = [loss_fn, torch.nn.L1Loss()]
        elif args.reg_loss == "dqn_align_loss":
            loss_fn = [loss_fn, DQNCOSLoss()]
        else:
            raise NotImplementedError

    print('Done!')
    return loss_fn

def _init_optim(args, model):
    r"""
    Init the optimizer 
    
    Args: 
        - args : argspace.Namespace 
        - model : torch model 
    
    Returns:
        - optimizer : torch optim 
    """
    print('\nInit optimizer ...', end=' ')

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.opt == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "radam":
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "lamb":
        optimizer = Lambda(model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError

    return optimizer

def _init_model(args):
    
    print('\nInit Model...', end=' ')
    
    if args.method == "ReCaSP":

        model_dict = {'omic_sizes': args.omic_sizes, "wsi_embedding_dim":args.encoding_dim, "dropout":args.encoder_dropout,
                      'num_classes': args.n_classes, "wsi_projection_dim": args.wsi_projection_dim, "bag_loss": args.bag_loss,
                      "reg_loss": args.reg_loss, "pooling_type": args.pooling_type, "decoder_number_layer": args.decoder_number_layer}

        model = ReCaSP(**model_dict)

    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        model = model.to(torch.device('cuda'))

    print('Done!')
    if not args.only_test:
        _print_network(args.results_dir, model)

    return model

def _init_loaders(args, train_split, val_split):
    r"""
    Init dataloaders for the train and val datasets 

    Args:
        - args : argspace.Namespace 
        - train_split : SurvivalDataset 
        - val_split : SurvivalDataset 
    
    Returns:
        - train_loader : Pytorch Dataloader 
        - val_loader : Pytorch Dataloader

    """

    print('\nInit Loaders...', end=' ')
    if train_split:
        train_loader = _get_split_loader(args, train_split, training=True, testing=False, weighted=args.weighted_sample, batch_size=args.batch_size)
    else:
        train_loader = None

    if val_split:
        val_loader = _get_split_loader(args, val_split,  testing=False, batch_size=1)
    else:
        val_loader = None
    print('Done!')

    return train_loader,val_loader

def _extract_survival_metadata(train_loader, val_loader):
    r"""
    Extract censorship and survival times from the train and val loader and combine to get numbers for the fold
    We need to do this for train and val combined because when evaulating survival metrics, the function needs to know the 
    distirbution of censorhsip and survival times for the trainig data
    
    Args:
        - train_loader : Pytorch Dataloader
        - val_loader : Pytorch Dataloader
    
    Returns:
        - all_survival : np.array
    
    """

    all_censorships = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.censorship_var].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.censorship_var].to_numpy()],
        axis=0)

    all_event_times = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.label_col].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.label_col].to_numpy()],
        axis=0)

    all_survival = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    return all_survival

def _unpack_data(omics_format, device, data):
    r"""
    Depending on the model type, unpack the data and put it on the correct device
    
    Args:
        - omics_format : String
        - device : torch.device 
        - data : tuple 
    
    Returns:
        - data_WSI : torch.Tensor
        - mask : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - data_omics : torch.Tensor
        - clinical_data_list : list
        - mask : torch.Tensor
    
    """
    data_WSI = data[0].to(device)
    if omics_format == "gene":
        mask = None
        data_omics = data[1].to(device)
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif omics_format == "groups":

        data_omic1 = data[1].type(torch.FloatTensor).to(device)
        data_omic2 = data[2].type(torch.FloatTensor).to(device)
        data_omic3 = data[3].type(torch.FloatTensor).to(device)
        data_omic4 = data[4].type(torch.FloatTensor).to(device)
        data_omic5 = data[5].type(torch.FloatTensor).to(device)
        data_omic6 = data[6].type(torch.FloatTensor).to(device)
        data_omics = [data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6]

        y_disc, event_time, censor, clinical_data_list = data[7], data[8], data[9], data[10]
        if data[11][0, 0] == 1:
            mask = None
        else:
            mask = data[11].to(device)

    elif omics_format == "pathways":

        data_omics = []
        for idx,item in enumerate(data[1]):
            for idy,omic in enumerate(item):
                omic = omic.to(device)
                omic = omic.unsqueeze(0)
                if idx == 0:
                    data_omics.append(omic)
                else:
                    data_omics[idy] = torch.cat((data_omics[idy],omic),dim=0)

        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)

        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
        
    else:
        raise ValueError('Unsupported omics type:', omics_format)
    
    y_disc, event_time, censor = y_disc.to(device), event_time.to(device), censor.to(device)

    return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list

def _process_data_and_forward(model, omics_format, device, data):
    r"""
    Depeding on the omics format, process the input data and do a forward pass on the model
    
    Args:
        - model : Pytorch model
        - omics_format : String
        - device : torch.device
        - data : tuple
    
    Returns:
        - out : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - clinical_data_list : List
    
    """
    data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list = _unpack_data(omics_format, device, data)

    input_args = {"x_path": data_WSI.to(device)}
    input_args["return_attn"] = False
    input_args["wsi_patch_mask"] = mask

    if omics_format == "gene":

        input_args["x_omics"] = data_omics.to(device)

        out = model(**input_args)

    elif omics_format in ["groups", "pathways"]:

        for i in range(len(data_omics)):
            input_args['x_omic%s' % str(i + 1)] = data_omics[i].type(torch.FloatTensor).to(device)

        out = model(**input_args)

    else:
        raise ValueError('Unsupported omics type:', omics_format)

    return out, y_disc, event_time, censor, clinical_data_list


def _calculate_risk(h):
    r"""
    Take the logits of the model and calculate the risk for the patient 
    
    Args: 
        - h : torch.Tensor 
    
    Returns:
        - risk : torch.Tensor 
    
    """
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()

def _update_arrays(all_risk_scores, all_censorships, all_event_times, all_clinical_data, event_time, censor, risk, clinical_data_list):
    r"""
    Update the arrays with new values 
    
    Args:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - risk : torch.Tensor
        - clinical_data_list : List
    
    Returns:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
    
    """
    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    all_clinical_data.append(clinical_data_list)
    return all_risk_scores, all_censorships, all_event_times, all_clinical_data

def get_lossValues(bag_loss, h, y_disc, event_time, censor, evidence, loss_fn, epoch, lambda_epochs, num_classes):
    if bag_loss == 'nll_surv':
        loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
        loss1, loss2 = loss, torch.tensor(0.)
    elif bag_loss == 'trust_nll_surv':
        loss, loss1, loss2 = loss_fn(h=h, evidence=evidence, y=y_disc, t=event_time, c=censor, class_num=num_classes, global_step=epoch, annealing_step=lambda_epochs)  # h, y, t, c, class_num, global_step, annealing_step
    else:
        raise NotImplementedError

    return loss, loss1, loss2

def _train_loop_survival(epoch, model, omics_format, loader, optimizer, scheduler, loss_fn, device='cuda', bag_loss='nll_surv', lambda_epochs=1, num_classes=4, reg_loss=None, reg_loss_alpha=1.):
    r"""
    Perform one epoch of training 

    Args:
        - epoch : Int
        - model : Pytorch model
        - omics_format : String
        - loader : Pytorch dataloader
        - optimizer : torch.optim
        - loss_fn : custom loss function class 
    
    Returns:
        - c_index : Float
        - total_loss : Float 
    
    """
    # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_loss = 0.

    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []

    # one epoch
    for batch_idx, data in enumerate(loader):
        
        optimizer.zero_grad()

        out, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, omics_format, device, data)

        h, evidence, w2o_cls, o2w_cls = out

        if len(h.shape) == 1:
            h = h.unsqueeze(0)

        # loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
        if isinstance(loss_fn, list):
            # survival loss + sim loss + sim loss
            loss, _, _ = get_lossValues(bag_loss, h, y_disc, event_time, censor, evidence, loss_fn[0], epoch, lambda_epochs, num_classes)  # sur_loss
            if reg_loss == "dqn_align_loss":
                ce_loss0 = reg_loss_alpha * loss_fn[1](w2o_cls)
                ce_loss1 = reg_loss_alpha * loss_fn[1](o2w_cls)
                loss3 = ce_loss0 + ce_loss1
                loss = loss + loss3
            else:
                raise NotImplementedError

        else:
            loss, _, _ = get_lossValues(bag_loss, h, y_disc, event_time, censor, evidence, loss_fn, epoch, lambda_epochs, num_classes)

        # 检查 loss 是否为 NaN
        if torch.isnan(loss).item():
            print("Loss is NaN, skipping this batch")
            continue

        loss_value = loss.item()
        loss = loss / y_disc.shape[0]
        
        risk, _ = _calculate_risk(h)

        all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)

        total_loss += loss_value

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (batch_idx % 2) == 0:
            print("batch: {}, loss: {:.4f}".format(batch_idx, loss.item()))
    
    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print("Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}".format(epoch, total_loss, c_index))

    return c_index, total_loss

def _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores):
    r"""
    Calculate various survival metrics 
    
    Args:
        - loader : Pytorch dataloader
        - dataset_factory : SurvivalDatasetFactory
        - survival_train : np.array
        - all_risk_scores : np.array
        - all_censorships : np.array
        - all_event_times : np.array
        - all_risk_by_bin_scores : np.array
        
    Returns:
        - c_index : Float
        - c_index_ipcw : Float
        - BS : np.array
        - IBS : Float
        - iauc : Float
    
    """
    
    data = loader.dataset.metadata["survival_months_dss"]
    bins_original = dataset_factory.bins
    which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])

    #---> delete the nans and corresponding elements from other arrays 
    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    #<---

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw, BS, IBS, iauc = 0., 0., 0., 0.

    # change the datatype of survival test to calculate metrics 
    try:
        survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    except:
        print("Problem converting survival test datatype, so all metrics 0.")
        return c_index, c_index_ipcw, BS, IBS, iauc
   
    # cindex2 (cindex_ipcw)
    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.
    
    # brier score 
    try:
        _, BS = brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing BS')
        BS = 0.
    
    # IBS
    try:
        IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        IBS = 0.

    # iauc
    try:
        _, iauc = cumulative_dynamic_auc(survival_train, survival_test, estimate=1-all_risk_by_bin_scores[:, 1:], times=which_times_to_eval_at[1:])
    except:
        print('An error occured while computing iauc')
        iauc = 0.
    
    return c_index, c_index_ipcw, BS, IBS, iauc

def _summary(dataset_factory, model, omics_format, loader, loss_fn, survival_train=None, device='cuda', bag_loss='nll_surv', lambda_epochs=1, num_classes=4, epoch=1, reg_loss=None, reg_loss_alpha=1.0):
    r"""
    Run a validation loop on the trained model 
    
    Args: 
        - dataset_factory : SurvivalDatasetFactory
        - model : Pytorch model
        - omics_format : String
        - loader : Pytorch loader
        - loss_fn : custom loss function clas
        - survival_train : np.array
    
    Returns:
        - patient_results : dictionary
        - c_index : Float
        - c_index_ipcw : Float
        - BS : List
        - IBS : Float
        - iauc : Float
        - total_loss : Float

    """
    # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.

    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    all_logits = []
    all_slide_ids = []

    slide_ids = loader.dataset.metadata['slide_id']
    count = 0
    with torch.no_grad():
        for data in loader:

            data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list = _unpack_data(omics_format, device, data)

            input_args = {"x_path": data_WSI.to(device)}
            input_args["return_attn"] = False
            input_args["wsi_patch_mask"] = mask

            if omics_format == "gene":

                input_args["x_omics"] = data_omics.to(device)

                out = model(**input_args)

            elif omics_format in ["groups", "pathways"]:

                for i in range(len(data_omics)):
                    input_args['x_omic%s' % str(i + 1)] = data_omics[i].type(torch.FloatTensor).to(device)

                out = model(**input_args)

            else:
                raise ValueError('Unsupported omics type:', omics_format)

            h, evidence, w2o_cls, o2w_cls = out

            if len(h.shape) == 1:
                h = h.unsqueeze(0)

            if isinstance(loss_fn, list):
                # survival loss + sim loss + sim loss
                loss, _, _ = get_lossValues(bag_loss, h, y_disc, event_time, censor, evidence, loss_fn[0], epoch, lambda_epochs, num_classes)  # sur_loss
                if reg_loss == "dqn_align_loss":
                    ce_loss0 = reg_loss_alpha * loss_fn[1](w2o_cls)
                    ce_loss1 = reg_loss_alpha * loss_fn[1](o2w_cls)
                    loss3 = ce_loss0 + ce_loss1
                    loss = loss + loss3
                else:
                    raise NotImplementedError
            else:
                loss, _, _ = get_lossValues(bag_loss, h, y_disc, event_time, censor, evidence, loss_fn, epoch, lambda_epochs, num_classes)

            loss_value = loss.item()
            loss = loss / y_disc.shape[0]

            risk, risk_by_bin = _calculate_risk(h)
            all_risk_by_bin_scores.append(risk_by_bin)

            all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)
            all_logits.append(h.detach().cpu().numpy())
            total_loss += loss_value
            all_slide_ids.append(slide_ids.values[count])
            count += 1

    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)

    all_clinical_data = np.concatenate(all_clinical_data, axis=0)

    patient_results = {}
    for i in range(len(all_slide_ids)):
        slide_id = slide_ids.values[i]
        case_id = slide_id[:12]
        patient_results[case_id] = {}
        patient_results[case_id]["time"] = all_event_times[i]
        patient_results[case_id]["risk"] = all_risk_scores[i]
        patient_results[case_id]["censorship"] = all_censorships[i]
        patient_results[case_id]["clinical"] = all_clinical_data[i]
        patient_results[case_id]["logits"] = all_logits[i]

    c_index, c_index2, BS, IBS, iauc = _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores)

    return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss


def _get_lr_scheduler(args, optimizer, dataloader):
    scheduler_name = args.lr_scheduler
    warmup_epochs = args.warmup_epochs
    epochs = args.max_epochs if hasattr(args, 'max_epochs') else args.epochs

    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * len(dataloader)
    else:
        warmup_steps = 0
    if scheduler_name=='constant':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_name=='cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    elif scheduler_name=='linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    return lr_scheduler

def _save_results(cur, results_dict, args):
    r"""
    Saves the results of the model.

    Args:
        - cur
        - results_dict
        - args: argspace.Namespace
    """
    filename = os.path.join(args.results_dir, "split_{}_results.pkl".format(cur))
    if os.path.exists(filename):
        os.remove(filename)
    # print("Saving results...")
    _save_pkl(filename, results_dict)

def _step(cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader):
    r"""
    Trains the model for the set number of epochs and validates it.
    
    Args:
        - cur
        - args
        - loss_fn
        - model
        - optimizer
        - lr scheduler 
        - train_loader
        - val_loader
        
    Returns:
        - results_dict : dictionary
        - val_cindex : Float
        - val_cindex_ipcw  : Float
        - val_BS : List
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """

    all_survival = _extract_survival_metadata(train_loader, val_loader)
    
    for epoch in range(1, args.max_epochs+1):
        _, total_loss = _train_loop_survival(epoch, model, args.omics_format, train_loader, optimizer, scheduler, loss_fn, device=args.device, bag_loss=args.bag_loss,
                                             lambda_epochs=args.lambda_epochs, num_classes=args.n_classes, reg_loss=args.reg_loss, reg_loss_alpha=args.reg_loss_alpha)
        if total_loss == 0.:
            print('Epoch:{} total_loss is 0, break'.format(epoch))
            break
        results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _summary(args.dataset_factory, model, args.omics_format, val_loader, loss_fn, all_survival,
                                                                                                    device=args.device, bag_loss=args.bag_loss, lambda_epochs=args.lambda_epochs,
                                                                                                    num_classes=args.n_classes, epoch=epoch, reg_loss=args.reg_loss, reg_loss_alpha=args.reg_loss_alpha)
        print('Epoch:{} c-index: {:.4f} | Val c-index2: {:.4f} | Val IBS: {:.4f} | Val iauc: {:.4f}'.format(
                epoch, val_cindex, val_cindex_ipcw, val_IBS, val_iauc))

        _save_results(cur, results_dict, args)
        if val_cindex >= args.max_cindex:
            args.max_cindex = val_cindex
            args.max_cindex_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.results_dir, "model_best_s{}.pth".format(cur)))


    # save the trained model
    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pth".format(cur)))
    
    results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _summary(args.dataset_factory, model, args.omics_format, val_loader, loss_fn, all_survival,
                                                                                                device=args.device, bag_loss=args.bag_loss, lambda_epochs=args.lambda_epochs,
                                                                                                num_classes=args.n_classes, epoch=args.max_epochs, reg_loss=args.reg_loss, reg_loss_alpha=args.reg_loss_alpha)

    print('Final Val c-index: {:.4f} | Val c-index2: {:.4f} | Val IBS: {:.4f} | Val iauc: {:.4f}'.format(
        val_cindex, val_cindex_ipcw, val_IBS, val_iauc))

    best_model = torch.load(os.path.join(args.results_dir, "model_best_s{}.pth".format(cur)))
    model.load_state_dict(best_model)
    results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _summary(args.dataset_factory, model, args.omics_format, val_loader, loss_fn, all_survival,
                                                                                     device=args.device, bag_loss=args.bag_loss, lambda_epochs=args.lambda_epochs,
                                                                                     num_classes=args.n_classes, epoch=args.max_cindex_epoch, reg_loss=args.reg_loss, reg_loss_alpha=args.reg_loss_alpha)
    print('Best Val c-index: {:.4f} | Val c-index2: {:.4f} | Val IBS: {:.4f} | Val iauc: {:.4f}'.format(
        val_cindex, val_cindex_ipcw, val_IBS, val_iauc))

    return results_dict, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, args.max_cindex_epoch)

def _train_val(datasets, cur, args):
    """   
    Performs train val test for the fold over number of epochs

    Args:
        - datasets : tuple
        - cur : Int 
        - args : argspace.Namespace 
    
    Returns:
        - results_dict : dict
        - val_cindex : Float
        - val_cindex2 : Float
        - val_BS : Float
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """

    #----> gets splits and summarize
    train_split, val_split = _get_splits(datasets, cur, args)
    
    #----> init loss function
    loss_fn = _init_loss_function(args)

    #----> init model
    model = _init_model(args)
    
    #---> init optimizer
    optimizer = _init_optim(args, model)

    #---> init loaders
    train_loader, val_loader = _init_loaders(args, train_split, val_split)

    # lr scheduler 
    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

    #---> do train val
    results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss, args.max_cindex_epoch) = _step(cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader)

    return results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss, args.max_cindex_epoch)
