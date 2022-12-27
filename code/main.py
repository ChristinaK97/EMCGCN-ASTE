# coding utf-8

import argparse
import random
import sys
import numpy as np
import torch

from data import label2id
from train_model import ModelTraining
from reproducibility.save_results import SaveResults
# Don't remove : from prepare_vocab import VocabHelp
from prepare_vocab import VocabHelp

from reproducibility.paper_experiments import run_datasets_with_multiple_seeds

def run_parser():
    """
    Κάθε argument θα είναι
    - Μία τιμή τύπου που ορίζει το όρισμα type ή
    - Ένα list τιμών αυτού του τύπου. Πολλαπλές τιμές για να τρέξουν πολλά μοντέλα
        πχ Με ορίσματα:
            --dataset  res14 res15 --seed 0 1
            Θα οριστούν οι συνδιασμοί (4 μοντέλα):
            [(res14, 0), (res14, 1), (res15, 0), (res15, 1)]
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--tag', type=str, default=" ",
                        help='A description given to the current experiment')

    parser.add_argument('--prefix', nargs='+', type=str, default="../data/D1/",
                        help='dataset and embedding path prefix')
    parser.add_argument('--model_dir', type=str, default="savemodel/",
                        help='model path prefix')
    parser.add_argument('--task', type=str, default="triplet", choices=["triplet"],
                        help='option: pair, triplet')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help='option: train, test')
    parser.add_argument('--dataset', nargs='+', type=str, default="res14",
                        help='dataset ["res14", "lap14", "res15", "res16"]')
    parser.add_argument('--max_sequence_len', type=int, default=102,
                        help='max length of a sentence')
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')

    parser.add_argument('--bert_model_path', type=str,
                        default="bert-base-uncased",
                        help='pretrained bert model path')
    parser.add_argument('--bert_feature_dim', type=int, default=768,
                        help='dimension of pretrained bert feature')

    parser.add_argument('--batch_size', nargs='+', type=int, default=16,
                        help='bathc size')
    parser.add_argument('--epochs', nargs='+', type=int, default=100,
                        help='training epoch number')
    parser.add_argument('--class_num', type=int, default=len(label2id),
                        help='label number')
    parser.add_argument('--seed', nargs='+', default=1000, type=int)
    parser.add_argument('--learning_rate', nargs='+', default=1e-3, type=float)
    parser.add_argument('--bert_lr', nargs='+', default=2e-5, type=float)
    parser.add_argument("--adam_epsilon", nargs='+', default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", nargs='+', default=0.0, type=float, help="Weight deay if we apply some.")

    parser.add_argument('--emb_dropout', nargs='+', type=float, default=0.5)
    parser.add_argument('--num_layers', nargs='+', type=int, default=1)
    parser.add_argument('--pooling', nargs='+', default='avg', type=str, help='[max, avg, sum]')
    parser.add_argument('--gcn_dim', nargs='+', type=int, default=300, help='dimension of GCN')
    parser.add_argument('--relation_constraint', default=True, action='store_true')
    parser.add_argument('--symmetry_decoding', default=False, action='store_true')

    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_model(args, results):
    # torch.set_printoptions(precision=None, threshold=float("inf"), edgeitems=None, linewidth=None, profile=None)

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


def main(args_as_list=None):
    """
    1. Parse τα args είτε από terminal είτε από ide
    2. Για κάθε συνδιασμό που ορίζουν τα args (κάθε μοντέλο)
        3. Αν δεν έχει αποθηκεύσει ήδη τα αποτελέσματα αυτού μοντέλου
            4. Εκτέλεση mode μοντέλου
            5. Αποθήκευση των αποτελεσμάτων που μαζεύτηκαν για το μοντέλο σε ξεχωριστό json
    6. Merge τα json του κάθε μοντέλου σε ένα
    """
    if args_as_list is not None:  # δε χρησιμοποιήθηκε terminal
        sys.argv.extend(args_as_list)

    parser = run_parser()              # 1
    merge_results_files = True

    results = SaveResults(parser)
    for model_param in results:         # 2
        if model_param is not None:     # 3
            print(model_param)
            run_model(model_param, results)          # 4
            results.write_output()      # 5
    if merge_results_files:             # 6
        results.merge_outputs()



if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'multiple_seeds':
        sys.argv.remove('multiple_seeds')
        args = run_datasets_with_multiple_seeds()
    else:
        args = None
    main(args)


