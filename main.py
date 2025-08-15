import sys
#----> pytorch imports
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

#----> general imports
import pandas as pd
import numpy as np
import pdb
from timeit import default_timer as timer
from datasets.dataset_survival import SurvivalDatasetFactory
from utils.core_utils import _train_val
from utils.file_utils import _save_pkl
from utils.general_utils import _get_start_end, _prepare_for_experiment
from utils.valid_utils import _val
from utils.process_args import _process_args
import os, datetime
import argparse

import warnings
warnings.filterwarnings("ignore")

def main(args):
    # ----> Prep
    args = _prepare_for_experiment(args)

    # ----> create dataset factory
    args.dataset_factory = SurvivalDatasetFactory(
        study=args.study,
        label_file=args.label_file,
        omics_dir=args.omics_dir,
        csvData_root_dir=args.csvData_root_dir,
        seed=args.seed,
        print_info=True,
        n_bins=args.n_classes,
        label_col=args.label_col,
        eps=1e-6,
        num_patches=args.num_patches,
        is_mcat=True if args.omics_format == 'groups' else False,
        is_survpath=True if args.omics_format == 'pathways' else False,
        type_of_pathway=args.type_of_path)


    #----> prep for 5 fold cv study
    folds = _get_start_end(args)
    
    #----> storing the val and test cindex for 5 fold cv
    all_val_cindex = []
    all_val_cindex_ipcw = []
    all_val_BS = []
    all_val_IBS = []
    all_val_iauc = []
    all_val_loss = []

    for i in folds:
        
        datasets = args.dataset_factory.return_splits(
            args,
            csv_path='{}/splits_{}.csv'.format(args.split_dir, i),
            fold=i
        )
        
        print("Created train and val datasets for fold {}".format(i))

        args.max_cindex = 0.0
        args.max_cindex_epoch = 0

        if args.only_test:
            args.max_cindex_epoch =-1

            # ----> only testing
            results, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, max_cindex_epoch) = _val(datasets, i, args)
            # write results to pkl
            filename = os.path.join(args.results_dir, 'split_{}_best_results_test.pkl'.format(i))
            print("Saving results...")
            _save_pkl(filename, results)
        else:
            #----> train and val
            results, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, max_cindex_epoch) = _train_val(datasets, i, args)
            # write results to pkl
            filename = os.path.join(args.results_dir, 'split_{}_best_results.pkl'.format(i))
            print("Saving results...")
            _save_pkl(filename, results)

        all_val_cindex.append(val_cindex)
        all_val_cindex_ipcw.append(val_cindex_ipcw)
        all_val_BS.append(val_BS)
        all_val_IBS.append(val_IBS)
        all_val_iauc.append(val_iauc)
        all_val_loss.append(total_loss)

    final_df = pd.DataFrame({
        'folds': folds,
        'val_cindex': all_val_cindex,
        'val_cindex_ipcw': all_val_cindex_ipcw,
        'val_IBS': all_val_IBS,
        'val_iauc': all_val_iauc,
        'val_BS': all_val_BS,
        "val_loss": all_val_loss,
    })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(args.k_start, args.k_end) #start, end
    else:
        if args.only_test:
            save_name = 'summary_test.csv'
        else:
            save_name = 'summary.csv'
        
    final_df.to_csv(os.path.join(args.results_dir, save_name))


if __name__ == "__main__":
    dt = datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S')

    start = timer()

    #----> read the args
    args = _process_args()

    if args.dt is not None:
        dt = args.dt

    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if not os.path.isdir('./result'):
        os.makedirs('./result', exist_ok=True)


    print(args)

    args.dt = dt

    main(args)

    #---> stop timer and print
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))