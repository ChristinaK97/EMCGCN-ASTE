import random

def get_args_as_list(tag, prefixes, datasets, seeds, batch_sizes):

    # prefixes '../data/D1/', '../data/D2/'
    # datasets 'res14', 'lap14', 'res15', 'res16'
    return [
        '--tag', tag,
        "--mode", "train",
        '--bert_model_path', 'bert-base-uncased',
        '--bert_feature_dim', '768',

        '--batch_size', *arg_values_as_string(batch_sizes),
        '--epochs', '100',
        '--learning_rate', '1e-3',
        '--bert_lr', '2e-5',
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
    seeds = [0, 42, 1000] + [random.randint(1, 1000) for _ in range(2)]
    batch_sizes = 6

    args = get_args_as_list(tag, prefixes, datasets, seeds, batch_sizes)
    return args











