import json
import os
import random
import re
import time

import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import AdamW

import utils
from data import load_data_instances, DataIterator
from model import EMCGCN
from prepare_vocab import VocabHelp


class ModelTraining:

    def __init__(self, args, results):
        """
        @param args: Έξοδος της main.parse_arguments
        @param results: Αντικείμενο της SaveResults όπου θα μαζευτούν τα αποτελέσματα
                        της εκπαίδευσης και του inference του μοντέλου
        """
        self.args = args
        self.LFtoUse = args.use_features
        self.results = results

    def get_bert_optimizer(self, model):
        # # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        diff_part = ["bert.embeddings", "bert.encoder"]
        optimizer_grouped_parameters = []

        if self.args.bert_lr != 0 :

            # print('Finetune bert')

            # 'bert.embeddings.{word_embeddings, position_embeddings, token_type_embeddings}.weight',
            #  FOR i = 0,..., # bert layers  :
            # 'bert.encoder.layer.i.attention.self.{query, key, value}.weight',
            # 'bert.encoder.layer.i.attention.output.dense.weight',
            # 'bert.encoder.layer.i.intermediate.dense.weight',
            # 'bert.encoder.layer.i.output.dense.weight'
            optimizer_grouped_parameters.append(
                {
                    "params": [p for n, p in model.named_parameters() if
                               not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)
                               and self.unfreeze(n, self.args.freezing_point)],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.bert_lr
                })

            # 'bert.embeddings.LayerNorm.{weight, bias}',
            #  FOR i = 0,..., # bert layers  :
            # 'bert.encoder.layer.i.attention.self.{query, key, value}.bias',
            # 'bert.encoder.layer.i.attention.output.dense.bias',
            # 'bert.encoder.layer.i.attention.output.LayerNorm.{weight, bias}',
            # 'bert.encoder.layer.i.intermediate.dense.bias',
            # 'bert.encoder.layer.i.output.dense.bias',
            # 'bert.encoder.layer.i.output.LayerNorm.{weight, bias}'
            optimizer_grouped_parameters.append(
                {
                    "params": [p for n, p in model.named_parameters() if
                               any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)
                               and self.unfreeze(n, self.args.freezing_point)],
                    "weight_decay": 0.0,
                    "lr": self.args.bert_lr
                })

        # Model parameters :
        optimizer_grouped_parameters.append(
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate
            })
        optimizer_grouped_parameters.append(
            {
                "params": [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate
            })
        optimizer = AdamW(optimizer_grouped_parameters, eps=self.args.adam_epsilon)

        return optimizer


    def unfreeze(self, named_param, freezing_point):
        # for example for freezing_point = 9, encoding layers 10 and 11 will be trainable
        # to train every layer, set freezing_point default value to -1 (since first enc layer=0)
        try:
            layer = re.findall(r'bert.encoder.layer.\d+', named_param)[0]
            layer = int(re.findall('\d+', layer)[0])
            # print('Unfreeze', layer, layer > freezing_point)
            return layer > freezing_point
        except Exception:
            # named parameter is not a bert encoder layer related parameter,
            # so it should be trainable
            return True



    def train(self):

        # load dataset
        train_sentence_packs = json.load(open(self.args.prefix + self.args.dataset + '/train.json'))
        random.shuffle(train_sentence_packs)
        dev_sentence_packs = json.load(open(self.args.prefix + self.args.dataset + '/dev.json'))

        """
        word_pair_position:  post_emb     relative position distance         rpd
        word_pair_deprel:    deprel_emb   syntactic dependency relations     dep
        word_pair_pos:       postag_emb   part of speech combination         psc
        word_pair_synpost:   synpost_emb  tree based distance                tbd
        """
        post_vocab = VocabHelp.load_vocab(self.args.prefix + self.args.dataset + '/vocab_post.vocab')
        deprel_vocab = VocabHelp.load_vocab(self.args.prefix + self.args.dataset + '/vocab_deprel.vocab')
        postag_vocab = VocabHelp.load_vocab(self.args.prefix + self.args.dataset + '/vocab_postag.vocab')
        synpost_vocab = VocabHelp.load_vocab(self.args.prefix + self.args.dataset + '/vocab_synpost.vocab')

        vocabs = {'post': post_vocab, 'deprel': deprel_vocab, 'postag': postag_vocab, 'synpost': synpost_vocab}
        print(self.LFtoUse)
        for feature in self.LFtoUse:
            setattr(self.args, f'{feature}_size', len(vocabs[feature]))

        instances_train = load_data_instances(train_sentence_packs, post_vocab, deprel_vocab, postag_vocab,
                                              synpost_vocab, self.args)
        instances_dev = load_data_instances(dev_sentence_packs, post_vocab, deprel_vocab, postag_vocab, synpost_vocab,
                                            self.args)
        random.shuffle(instances_train)
        trainset = DataIterator(instances_train, self.args)
        devset = DataIterator(instances_dev, self.args)

        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model = EMCGCN(self.args, self.LFtoUse).to(self.args.device)
        optimizer = self.get_bert_optimizer(model)

        # label = ['N', 'B-A', 'I-A', 'A', 'B-O', 'I-O', 'O', 'negative', 'neutral', 'positive']
        weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]).float().cuda()

        best_joint_f1 = 0
        best_joint_epoch = 0
        start = time.time()

        for i in range(self.args.epochs):
            print('Epoch:{}'.format(i))
            for j in trange(trainset.batch_count):
                _, sentences, tokens, lengths, masks, _, _, aspect_tags, tags, word_pair_position, \
                    word_pair_deprel, word_pair_pos, word_pair_synpost, tags_symmetry = trainset.get_batch(j)
                tags_flatten = tags.reshape([-1])
                tags_symmetry_flatten = tags_symmetry.reshape([-1])
                if self.args.relation_constraint:
                    word_pair_feat = {'post': word_pair_position, 'deprel': word_pair_deprel, 'postag': word_pair_pos,
                                      'synpost': word_pair_synpost}
                    predictions = model(tokens, masks, word_pair_feat)

                    biaffine_pred = predictions.pop('ba')
                    final_pred = predictions.pop('p')
                    feat_pred = {feature : prediction for feature, prediction in predictions.items()}

                    l_ba = 0.10 * F.cross_entropy(biaffine_pred.reshape([-1, biaffine_pred.shape[3]]),
                                                  tags_symmetry_flatten, ignore_index=-1)
                    l_feat = {
                        feature: 0.01 * F.cross_entropy(pred.reshape([-1, pred.shape[3]]), tags_symmetry_flatten, ignore_index=-1)
                        for feature, pred in feat_pred.items()
                    }

                    if self.args.symmetry_decoding:
                        l_p = F.cross_entropy(final_pred.reshape([-1, final_pred.shape[3]]), tags_symmetry_flatten,
                                              weight=weight, ignore_index=-1)
                    else:
                        l_p = F.cross_entropy(final_pred.reshape([-1, final_pred.shape[3]]), tags_flatten,
                                              weight=weight, ignore_index=-1)

                    loss = l_ba + l_p
                    for feature, l in l_feat.items():
                        loss += l

                else:
                    preds = \
                        model(tokens, masks, word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost)[-1]
                    preds_flatten = preds.reshape([-1, preds.shape[3]])
                    if self.args.symmetry_decoding:
                        loss = F.cross_entropy(preds_flatten, tags_symmetry_flatten, weight=weight, ignore_index=-1)
                    else:
                        loss = F.cross_entropy(preds_flatten, tags_flatten, weight=weight, ignore_index=-1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            joint_precision, joint_recall, joint_f1 = self.eval(model, devset)

            if joint_f1 > best_joint_f1:
                model_path = self.args.model_dir + 'bert' + self.args.task + '.pt'
                torch.save(model, model_path)
                best_joint_f1 = joint_f1
                best_joint_epoch = i

        print('best epoch: {}\tbest dev {} f1: {:.5f}\n\n'.format(best_joint_epoch, self.args.task, best_joint_f1))

        # MOD
        self.results.update('best_epoch', best_joint_epoch)
        self.results.update('best_epoch_f1', best_joint_f1)
        self.results.update('training_time', time.time() - start)

    def eval(self, model, dataset, FLAG=False, given_set='dev_set'):
        model.eval()
        with torch.no_grad():
            all_ids = []
            all_sentences = []
            all_preds = []
            all_labels = []
            all_lengths = []
            all_sens_lengths = []
            all_token_ranges = []
            for i in range(dataset.batch_count):
                sentence_ids, sentences, tokens, lengths, masks, sens_lens, token_ranges, aspect_tags, tags, \
                    word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost, tags_symmetry = dataset.get_batch(
                    i)
                word_pair_feat = {'post': word_pair_position, 'deprel': word_pair_deprel, 'postag': word_pair_pos,
                                  'synpost': word_pair_synpost}
                preds = model(tokens, masks, word_pair_feat)['p']  # p is the final prediction
                preds = F.softmax(preds, dim=-1)
                preds = torch.argmax(preds, dim=3)
                all_preds.append(preds)
                all_labels.append(tags)
                all_lengths.append(lengths)
                all_sens_lengths.extend(sens_lens)
                all_token_ranges.extend(token_ranges)
                all_ids.extend(sentence_ids)
                all_sentences.extend(sentences)

            all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
            all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
            all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()

            metric = utils.Metric(self.args, all_preds, all_labels, all_lengths, all_sens_lengths, all_token_ranges,
                                  ignore_index=-1)
            precision, recall, f1 = metric.score_uniontags()
            aspect_results = metric.score_aspect()
            opinion_results = metric.score_opinion()
            print('Aspect term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(aspect_results[0], aspect_results[1],
                                                                      aspect_results[2]))
            print('Opinion term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(opinion_results[0], opinion_results[1],
                                                                       opinion_results[2]))
            print(self.args.task + '\t\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

            scores_per_class = metric.tagReport() if FLAG else {}

        model.train()
        # MOD
        self.results.update(given_set, {**{'f1': f1, 'precision': precision, 'recall': recall}, **scores_per_class})
        return precision, recall, f1

    def test(self):
        print("Evaluation on testset:")
        model_path = self.args.model_dir + 'bert' + self.args.task + '.pt'
        model = torch.load(model_path).to(self.args.device)
        model.eval()

        sentence_packs = json.load(open(self.args.prefix + self.args.dataset + '/test.json'))
        post_vocab = VocabHelp.load_vocab(self.args.prefix + self.args.dataset + '/vocab_post.vocab')
        deprel_vocab = VocabHelp.load_vocab(self.args.prefix + self.args.dataset + '/vocab_deprel.vocab')
        postag_vocab = VocabHelp.load_vocab(self.args.prefix + self.args.dataset + '/vocab_postag.vocab')
        synpost_vocab = VocabHelp.load_vocab(self.args.prefix + self.args.dataset + '/vocab_synpost.vocab')
        instances = load_data_instances(sentence_packs, post_vocab, deprel_vocab, postag_vocab, synpost_vocab,
                                        self.args)
        testset = DataIterator(instances, self.args)
        self.eval(model, testset, not self.args.use_refining, given_set='test_set')
