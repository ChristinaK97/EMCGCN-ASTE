import json
from itertools import product

import pandas as pd

def dataset_triple_stats(prefix, dataset, split, labels):
    # print(sentence_packs[0]['triples'][0]['sentiment'])
    sentence_packs = json.load(open(prefix + dataset + f'/{split}.json'))
    triples_per_sentence = []
    for sentence in sentence_packs:
        s_triples = {sentiment : 0 for sentiment in labels}
        for triple in sentence['triples']:
            s_triples[triple['sentiment']] += 1
        triples_per_sentence.append(s_triples)

    df = pd.DataFrame(triples_per_sentence)

    df['# labels'] = df.astype(bool).sum(axis=1)
    df['triples'] = df[list(df.columns)].sum(axis=1)

    num_of_different_labels = df.groupby(['# labels'])['# labels'].size().reset_index(name='count')
    describe = df.describe()
    sum_per_label = df.sum()

    labels = ['triples'] + labels

    stats_ = {'dataset': f"{prefix[-3:-1]}-{dataset} {split}" ,
             '# sentences' : df.shape[0]}
    for sentiment in labels:
        stats_[f'# {sentiment}'] = sum_per_label[sentiment]
        stats_[f'{sentiment}_mean'] = describe[sentiment]['mean']

    missing_3 = True
    for _, row in num_of_different_labels.iterrows():
        stats_[f"# s {row['# labels']} l"] = row['count']
        if row['# labels'] == 3:
            missing_3 = False
    if missing_3:
        stats_[f"# s 3 l"] = 0
    return stats_

def print_latex_table_rows(df):
    print(len(df.columns))
    print(' & '.join(df.columns) + ' \\\\')
    for _, row in df.iterrows():
        st = ' & '.join([str(round(row[col],4)) if isinstance(row[col], float)
                         else str(row[col])
                         for col in df.columns
        ]) + ' \\\\'
        print(st)

if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    sentiment_labels = ['triples', 'positive', 'negative', 'neutral']
    prefixes = [f'../../data/D{D}/' for D in [1, 2]]
    datasets = [f'res{N}' for N in [14, 15, 16]] + ['lap14']
    splits = ['train', 'dev', 'test']

    stats = [dataset_triple_stats(prefix=x[0], dataset=x[1], split=x[2],
                                  labels=sentiment_labels[1:])
            for x in product(prefixes, datasets, splits)]

    stats = pd.DataFrame(stats)
    stats = stats[['dataset', '# sentences'] + [f'# {sent}' for sent in sentiment_labels] +
                [f'{sent}_mean' for sent in sentiment_labels] +
                # number of sentences in the dataset with x different sentiments
                [f"# s {x} l" for x in [1,2,3]]]
    print(stats)
    print_latex_table_rows(stats)
