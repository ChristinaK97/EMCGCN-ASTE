# coding utf-8

import argparse
import random
import sys
import numpy as np
import torch

from data import label2id
from train_model import ModelTraining
sys.path.insert(0, '../reproducibility/')
from save_results import SaveResults
# Don't remove : from prepare_vocab import VocabHelp
from prepare_vocab import VocabHelp


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default="../data/D1/",
                        help='dataset and embedding path prefix')
    parser.add_argument('--model_dir', type=str, default="savemodel/",
                        help='model path prefix')
    parser.add_argument('--task', type=str, default="triplet", choices=["triplet"],
                        help='option: pair, triplet')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help='option: train, test')
    parser.add_argument('--dataset', type=str, default="res14", choices=["res14", "lap14", "res15", "res16"],
                        help='dataset')
    parser.add_argument('--max_sequence_len', type=int, default=102,
                        help='max length of a sentence')
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')

    parser.add_argument('--bert_model_path', type=str,
                        default="bert-base-uncased",
                        help='pretrained bert model path')
    parser.add_argument('--bert_feature_dim', type=int, default=768,
                        help='dimension of pretrained bert feature')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='bathc size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='training epoch number')
    parser.add_argument('--class_num', type=int, default=len(label2id),
                        help='label number')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--bert_lr', default=2e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")

    parser.add_argument('--emb_dropout', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--pooling', default='avg', type=str, help='[max, avg, sum]')
    parser.add_argument('--gcn_dim', type=int, default=300, help='dimension of GCN')
    parser.add_argument('--relation_constraint', default=True, action='store_true')
    parser.add_argument('--symmetry_decoding', default=False, action='store_true')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_model(results):
    # torch.set_printoptions(precision=None, threshold=float("inf"), edgeitems=None, linewidth=None, profile=None)

    args = parse_arguments()

    if args.seed is not None:
        set_seed(args.seed)

    if args.task == 'triplet':
        args.class_num = len(label2id)

    # Οι μέθοδοι για την εκπαίδευση του μοντέλου μετακινήθηκαν σε ξεχωριστή κλάση
    model = ModelTraining(args, results)
    if args.mode == 'train':
        model.train()
        model.test()
    else:
        model.test()


def get_args():
    parsed_args = parse_arguments()
    """
    Args του μοντέλου. Η τιμή του καθενός είναι είτε
    - Μία τιμή σαν str
    - Ένα list τιμών str. Πολλαπλές τιμές για να τρέξουν πολλά μοντέλα
        πχ Με ορίσματα:
            '--dataset' = ['res14', 'res15']
            '--seed' = ['0', '1']
            Θα οριστούν οι συνδιασμοί (4 μοντέλα):
            [('res14', '0'), ('res14', '1'), ('res15', '0'), ('res15', '1')]
    @return:
    """
    prefixes = ['../data/D1/', '../data/D2/']
    datasets = ['res14', 'lap14', 'res15', 'res16']
    return [
        "--mode", str(parsed_args.mode),
        '--bert_model_path', str(parsed_args.bert_model_path),
        '--bert_feature_dim', str(parsed_args.bert_feature_dim),

        '--batch_size', str(parsed_args.batch_size),
        '--epochs', str(parsed_args.epochs),
        '--learning_rate', str(parsed_args.learning_rate),
        '--bert_lr', str(parsed_args.bert_lr),
        '--adam_epsilon', str(parsed_args.adam_epsilon),
        '--weight_decay', str(parsed_args.weight_decay),
        '--seed', str(parsed_args.seed),

        '--num_layers', str(parsed_args.num_layers),
        '--gcn_dim', str(parsed_args.gcn_dim),
        '--pooling', str(parsed_args.pooling),
        '--prefix', str(parsed_args.prefix),
        '--dataset', str(parsed_args.dataset)
    ]


if __name__ == '__main__':
    """
    1. Για κάθε συνδιασμό που ορίζουν τα args (κάθε μοντέλο)
        2. Αν δεν έχει αποθηκεύσει ήδη τα αποτελέσματα αυτού μοντέλου
            3. Για να διαβαστούν από την parse_arguments 
            4. Εκτέλεση mode μοντέλου
            5. Αποθήκευση των αποτελεσμάτων που μαζεύτηκαν για το μοντέλο σε ξεχωριστό json
    6. Merge τα json του κάθε μοντέλου σε ένα
    """
    merge_results_files = True

    results = SaveResults(get_args())
    for model_param in results:         # 1
        if model_param is not None:     # 2
            print(model_param[1:])
            sys.argv = model_param      # 3
            run_model(results)          # 4
            results.write_output()      # 5
    if merge_results_files:             # 6
        results.merge_outputs()
