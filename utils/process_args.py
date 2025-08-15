import argparse

def _process_args():
    r"""
    Function creates a namespace to read terminal-based arguments for running the experiment

    Args
        - None 

    Return:
        - args : argparse.Namespace

    """

    parser = argparse.ArgumentParser(description='Configurations for ReCaSP Survival Prediction Training')
    parser.add_argument('--method', type=str, default="ReCaSP", help='methd type')
    #---> study related
    parser.add_argument('--study', type=str, default='stad', help='study name, including coadread, brca, stad, blca, hnsc')
    parser.add_argument('--task', type=str, choices=['survival'], default='survival')
    parser.add_argument('--n_classes', type=int, default=4, help='number of classes (4 bins for survival)')
    parser.add_argument('--results_dir', default='./result/', help='results directory (default: ./results)')
    parser.add_argument("--type_of_path", type=str, default="combine", choices=["xena", "hallmarks", "combine"])

    #----> data related
    parser.add_argument('--data_root_dir', type=str, default="../../../../code_reproduction_survival/data/CTranPath/", help='data directory')
    parser.add_argument('--csvData_root_dir', type=str, default="../../../../code_reproduction_survival/data/PIBD/datasets_csv/", help='data directory')
    # parser.add_argument('--label_file', type=str, default="./datasets_csv/metadata/", help='Path to csv with labels')
    # parser.add_argument('--omics_dir', type=str, default="./datasets_csv/raw_rna_data/", help='Path to dir with omics csv for all modalities')
    parser.add_argument('--num_patches', type=int, default=4096, help='number of patches')
    parser.add_argument('--label_col', type=str, default="survival_months_dss", help='type of survival (OS, DSS, PFI)')
    parser.add_argument("--wsi_projection_dim", type=int, default=256)
    parser.add_argument("--encoding_layer_1_dim", type=int, default=8)
    parser.add_argument("--encoding_layer_2_dim", type=int, default=16)
    parser.add_argument("--encoder_dropout", type=float, default=0.25)

    #----> split related 
    parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--split_dir', type=str, default='splits', help='manually specify the set of splits to use, '
                    +'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--which_splits', type=str, default="5foldcv", help='where are splits')
        
    #----> training related 
    parser.add_argument('--max_epochs', type=int, default=30, help='maximum number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate (default: 0.005)')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--opt', type=str, default="radam", help="Optimizer")
    parser.add_argument('--reg_type', type=str, default="None", help="regularization type [None, L1, L2]")
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--bag_loss', type=str, choices=["nll_surv", "trust_nll_surv"], default='trust_nll_surv',
                        help='survival loss function (default: trust_nll_surv)')
    parser.add_argument('--alpha_surv', type=float, default=0.5, help='weight given to uncensored patients')
    parser.add_argument('--reg_loss', type=str, choices=[None, "dqn_align_loss"], default="dqn_align_loss",
                        help='modal constraint loss function (default: dqn_align_loss)')
    parser.add_argument('--reg_loss_alpha', type=float, default=1.0, help='weight given to modal constraint loss')
    parser.add_argument('--reg', type=float, default=0.001, help='weight decay / L2 (default: 1e-3)')
    parser.add_argument('--lr_scheduler', type=str, default='cosine')
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--max_cindex', type=float, default=0.0, help='maximum c-index')

    #---> model related
    parser.add_argument('--fusion', type=str, default=None, help='concat, bilinear, or None')
    parser.add_argument('--omics_format', type=str, default="pathways", choices=["gene","groups","pathways"], help='format of omics data')
    parser.add_argument('--encoding_dim', type=int, default=768, help='WSI encoding dim')

    # ---> gpu id
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--lambda_epochs', type=int, default=1, metavar='N', help='gradually increase the value of lambda from 0 to 1') #50, 10
    parser.add_argument('--beta_surv', type=float, default=0.5, help='weight given to trusted loss')
    parser.add_argument('--pooling_type', type=str, default='avg', help='avg, max, or attention')
    parser.add_argument("--decoder_number_layer", type=int, default=1)

    # ---> only test the model
    parser.add_argument('--only_test', action='store_true', default=False, help='only test')
    parser.add_argument('--dt', type=str, default=None, help='datetime to train the model')

    args = parser.parse_args()

    args.data_root_dir = '{}{}/tiles-l1-s224/feats-l1-s224-CTransPath-sampler/pt_files/'.format(args.data_root_dir, args.study)
    args.label_file = '{}metadata/tcga_{}.csv'.format(args.csvData_root_dir, args.study)
    args.omics_dir = '{}raw_rna_data/{}/{}/'.format(args.csvData_root_dir, args.type_of_path, args.study)
    # args.clin_file = "{}clinical_data/tcga_{}_clinical.csv".format(args.csvData_root_dir, args.study)
    args.results_dir = args.results_dir + args.method + '/file_save/'
    args.study = 'tcga_' + args.study

    if not (args.task == "survival"):
        print("Task and folder does not match")
        exit()

    return args