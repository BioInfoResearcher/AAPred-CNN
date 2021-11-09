import argparse


def get_default_parser():
    parser = argparse.ArgumentParser(description='my_parser')

    '''
    project setting
    '''
    # parser.add_argument('-learn_name', type=str, default='AAP_CV')
    # parser.add_argument('-learn_name', type=str, default='AAP_ensemble')
    # parser.add_argument('-learn_name', type=str, default='AAP_TE')
    # parser.add_argument('-learn_name', type=str, default='AAP_CNN')
    parser.add_argument('-learn_name', type=str, default='AAP_RNN')
    # parser.add_argument('-learn_name', type=str, default='ACP_TE')
    # parser.add_argument('-learn_name', type=str, default='ACP_CNN')
    # parser.add_argument('-learn_name', type=str, default='ACP_RNN')
    parser.add_argument('-train_mode', type=str, default='train', choices=['train', 'continue_train', 'test',
                                                                           'cross_validation'])
    # parser.add_argument('-train_mode', type=str, default='test',
    #                     choices=['train', 'continue_train', 'test', 'cross_validation'])
    # parser.add_argument('-train_mode', type=str, default='continue_train',
    #                     choices=['train', 'continue_train', 'test', 'cross_validation'])
    # parser.add_argument('-train_mode', type=str, default='cross_validation',
    #                     choices=['train', 'continue_train', 'test', 'cross_validation'])
    # parser.add_argument('-k_fold', type=int, default=5)
    parser.add_argument('-visualization', type=bool, default=True)

    '''
    data setting
    '''
    parser.add_argument('-num_class', type=int, default=2)
    # parser.add_argument('-batch_size', type=int, default=256)
    # parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-batch_size', type=int, default=64)
    # parser.add_argument('-batch_size', type=int, default=32)
    # parser.add_argument('-batch_size', type=int, default=16)
    # parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-max_len', type=int, default=68)
    # parser.add_argument('-max_len', type=int, default=15)
    parser.add_argument('-proportion', type=float, default=None)
    # parser.add_argument('-proportion', type=float, default=1)
    # parser.add_argument('-proportion', type=float, default=0.9)
    # parser.add_argument('-proportion', type=float, default=0.8)
    # parser.add_argument('-proportion', type=float, default=0.7)
    # parser.add_argument('-proportion', type=float, default=0.6)
    # parser.add_argument('-proportion', type=float, default=0.5)
    # parser.add_argument('-proportion', type=float, default=0.4)
    # parser.add_argument('-proportion', type=float, default=0.3)
    # parser.add_argument('-proportion', type=float, default=0.2)
    # parser.add_argument('-proportion', type=float, default=0.1)
    parser.add_argument('-path_train_data', type=str,
                        default='../data/task_data/Anti-angiogenic Peptide/train/main.tsv')
    parser.add_argument('-path_test_data', type=str,
                        default='../data/task_data/Anti-angiogenic Peptide/test/main.tsv')
    # parser.add_argument('-path_train_data', type=str,
    #                     default='../data/task_data/Anti-angiogenic Peptide/train/benchmark.tsv')
    # parser.add_argument('-path_test_data', type=str,
    #                     default='../data/task_data/Anti-angiogenic Peptide/test/benchmark.tsv')
    # parser.add_argument('-path_train_data', type=str,
    #                     default='../data/task_data/Anti-angiogenic Peptide/train/NT15.tsv')
    # parser.add_argument('-path_test_data', type=str,
    #                     default='../data/task_data/Anti-angiogenic Peptide/test/NT15.tsv')
    # parser.add_argument('-path_train_data', type=str,
    #                     default='../data/task_data/Finetune Dataset/Anti-cancer Peptide/train/train_main.tsv')
    # parser.add_argument('-path_test_data', type=str,
    #                     default='../data/task_data/Finetune Dataset/Anti-cancer Peptide/test/test_main.tsv')
    # parser.add_argument('-path_train_data', type=str,
    #                     default='../data/task_data/Finetune Dataset/Anti-cancer Peptide/train/train_alternate.tsv')
    # parser.add_argument('-path_test_data', type=str,
    #                     default='../data/task_data/Finetune Dataset/Anti-cancer Peptide/test/test_alternate.tsv')
    parser.add_argument('-path_tokenizer', type=str, default='../data/meta_data')

    '''
    engineering setting
    '''
    # parser.add_argument('-fast_dev_run', type=bool, default=True)
    parser.add_argument('-fast_dev_run', type=bool, default=False)
    parser.add_argument('-auto_scale_batch_size', type=str, default=None)
    # parser.add_argument('-auto_scale_batch_size', type=str, default='power')
    parser.add_argument('-auto_lr_find', type=str, default=None)
    # parser.add_argument('-auto_lr_find', type=str, default='lr')
    # parser.add_argument('-max_epochs', type=int, default=80)
    parser.add_argument('-max_epochs', type=int, default=150)
    parser.add_argument('-val_check_interval', type=float, default=0.2)
    parser.add_argument('-accumulate_grad_batches', type=int, default=1)
    # parser.add_argument('-precision', type=int, default=16)
    parser.add_argument('-precision', type=int, default=32)
    parser.add_argument('-progress_bar_refresh_rate', type=int, default=1)
    parser.add_argument('-num_sanity_val_steps', type=int, default=2)
    parser.add_argument('-cuda', type=bool, default=True)
    parser.add_argument('-gpus', type=str, default='0,1')
    # parser.add_argument('-gpus', type=str, default='0')
    parser.add_argument('-auto_select_gpus', type=bool, default=True)
    parser.add_argument('-log_gpu_memory', type=str, default='all')
    parser.add_argument('-accelerator', type=str, default='ddp')
    parser.add_argument('-benchmark', type=bool, default=True)
    parser.add_argument('-sync_batchnorm', type=bool, default=True)
    parser.add_argument('-stochastic_weight_avg', type=bool, default=True)
    parser.add_argument('-weights_summary', type=str, default='full')
    parser.add_argument('-num_workers', type=int, default=8)

    return parser
