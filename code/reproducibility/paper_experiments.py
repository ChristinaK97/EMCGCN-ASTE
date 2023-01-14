import random


def get_args_as_list(tag, prefixes, datasets, seeds, batch_sizes, bert_lr='2e-5', freezing_points=None, use_refining=True):
    # prefixes '../data/D1/', '../data/D2/'
    # datasets 'res14', 'lap14', 'res15', 'res16'
    if freezing_points is None:
        freezing_points = ['-1']

    return [
        '--tag', tag,
        "--mode", "train",
        '--freezing_point', *arg_values_as_string(freezing_points),
        '--bert_model_path', 'bert-base-uncased',
        '--bert_feature_dim', '768',

        '--use_refining', '1' if use_refining else '0',

        '--batch_size', *arg_values_as_string(batch_sizes),
        '--epochs', '1',
        '--learning_rate', '1e-3',
        '--bert_lr', bert_lr,
        '--adam_epsilon', '1e-8',
        '--weight_decay', '0.0',
        '--seed', *arg_values_as_string(seeds),

        '--num_layers', '1',
        '--gcn_dim', '300',
        '--pooling', 'avg',
        '--prefix', *prefixes,
        '--dataset', *datasets
    ]


def arg_values_as_string(values):
    return [str(v) for v in values] if isinstance(values, list) else [str(values)]


def run_datasets_with_multiple_seeds():
    tag = "run datasets with multiple seeds"
    prefixes = ['../data/D1/', '../data/D2/']
    datasets = ['res14', 'lap14', 'res15', 'res16']
    seeds = [0, 42, 92, 153, 1000]
    batch_sizes = 16

    args = get_args_as_list(tag, prefixes, datasets, seeds, batch_sizes)
    return args


def bert_without_finetuning():
    tag = "bert without finetuning"
    prefixes = ['../data/D2/']
    datasets = ['res14', 'lap14', 'res15', 'res16']

    args = get_args_as_list(tag, prefixes, datasets, seeds=1000, batch_sizes=6, bert_lr='0')
    return args

def bert_unfreeze_layers():
    tag = 'bert unfreeze layers'

    prefixes = ['../data/D2/']
    datasets = ['res14', 'res15']
    freezing_points = [str(i) for i in range(10, -1, -1)]

    args = get_args_as_list(tag, prefixes, datasets, seeds=1000, batch_sizes=6, freezing_points=freezing_points)
    return args


def no_refining_strategy():
    tag = "no refining strategy"
    prefixes = ['../data/D2/']
    datasets = ['res15', 'res16']

    args = get_args_as_list(tag, prefixes, datasets, seeds=1000, batch_sizes=6, use_refining=False)
    return args