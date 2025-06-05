from __future__ import absolute_import, division, print_function
import argparse
import json
import logging
import os
import random
import numpy as np
import torch
from transformers import (BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer, AutoTokenizer)
from torch.optim import AdamW
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from transformers import BertForTokenClassification, XLMRobertaForTokenClassification
from torch import nn
from seqeval.metrics import classification_report
import torch.nn.functional as F

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Ner(XLMRobertaForTokenClassification):
    def forward(self, input_ids, device, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None):
        sequence_output = self.roberta(input_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            # attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask


def readfile(filename):
    f = open(filename, encoding="utf8")
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence, label))

    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir, data_file):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, data_file)), "train")

    def get_dev_examples(self, data_dir, data_file):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, data_file)), "dev")

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_ids,
                          valid_ids=valid, label_mask=label_mask))
    return features


def main():
    output_dir = 'czech_procedure'
    data_file = 'cardioccc_cz_proc_bio_format.tsv'
    dev_file = 'cardioccc_cz_proc_bio_format.tsv'
    # label_list = ["O", "B-MEDICATION", "I-MEDICATION", "[CLS]", "[SEP]"]
    # label_list = ["O", "B-DISEASE", "I-DISEASE", "[CLS]", "[SEP]"]
    label_list = ["O", "B-PROCEDURE", "I-PROCEDURE", "[CLS]", "[SEP]"]

    # region initial parameters
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir", default='../data/CardioCCC_Dataset/Training/Czech',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--dev_data_dir",
                        default='../data/CardioCCC_Dataset/Testing/Czech',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model",
                        default='bert_models/xlm-ner-slavic',
                        type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name", default='ner', type=str, help="The name of the task to train.")
    parser.add_argument("--output_dir", default=output_dir, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train", default=True, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, action='store_true', help="Whether to run eval or not.")
    parser.add_argument("--eval_on", default="dev", help="Whether to run eval on the dev set or test set.")
    parser.add_argument("--do_lower_case", action='store_false',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=9e-6, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.15, type=float, help="attention_probs_dropout_prob")
    parser.add_argument("--hidden_dropout_prob", default=0.15, type=float, help="hidden_dropout_prob")
    parser.add_argument("--hidden_act", default='gelu', type=str, help="hidden_act")
    parser.add_argument("--num_attention_heads", default=16, type=int, help="num_attention_heads")
    parser.add_argument("--num_hidden_layers", default=16, type=float, help="num_hidden_layers")
    parser.add_argument("--num_train_epochs", default=10, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.000, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-7, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--no_cuda", action='store_false', help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()
    # endregion

    # region Preparation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    processor = NerProcessor()

    num_labels = 9
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
    config = RobertaConfig.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name)
    config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    config.hidden_dropout_prob = args.hidden_dropout_prob
    config.hidden_act = args.hidden_act
    config.num_attention_heads = args.num_attention_heads
    config.num_hidden_layers = args.num_hidden_layers
    model = Ner.from_pretrained(args.bert_model, from_tf=False, config=config)
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)]}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                      weight_decay=args.weight_decay)
    global_step = 0
    train_loss_list = []
    eval_loss_list = []
    # endregion

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        # region Train
        if args.do_train:
            train_examples = processor.get_train_examples(args.data_dir, data_file)
            train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
            logger.info("***** Running training *****")
            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
            all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
            all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                                       all_lmask_ids)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask = batch
                loss = model(input_ids, device, segment_ids, input_mask, label_ids, valid_ids, l_mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                optimizer.step()
                model.zero_grad()
                global_step += 1
            print('Train loss  = ', tr_loss / nb_tr_steps)
            train_loss_list.append(tr_loss / nb_tr_steps)

            # region Eval
            if args.do_eval:
                dev_examples = processor.get_dev_examples(args.dev_data_dir, dev_file)
                dev_features = convert_examples_to_features(dev_examples, label_list, args.max_seq_length, tokenizer)
                logger.info("***** Running evaluation on dev *****")
                all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)
                all_valid_ids = torch.tensor([f.valid_ids for f in dev_features], dtype=torch.long)
                all_lmask_ids = torch.tensor([f.label_mask for f in dev_features], dtype=torch.long)
                dev_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                                         all_lmask_ids)
                dev_sampler = SequentialSampler(dev_data)
                dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size)

                model.eval()
                y_true = []
                y_pred = []
                label_map = {i: label for i, label in enumerate(label_list, 1)}
                eval_loss = 0
                nb_eval_examples, nb_eval_steps = 0, 0
                for step, batch in enumerate(tqdm(dev_dataloader, desc="Evaluating")):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask = batch
                    with torch.no_grad():
                        loss = model(input_ids, device, segment_ids, input_mask, label_ids, valid_ids, l_mask)
                        logits = model(input_ids, device, segment_ids, input_mask, valid_ids=valid_ids,
                                       attention_mask_label=l_mask)
                    eval_loss += loss.item()
                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

                    logits = torch.argmax(F.log_softmax(logits[:, :, 1:len(label_list) - 1], dim=2), dim=2) + 1
                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()

                    for i, label in enumerate(label_ids):
                        temp_1 = []
                        temp_2 = []
                        for j, m in enumerate(label):
                            if j == 0:
                                continue
                            elif label_ids[i][j] == len(label_map):
                                y_true.append(temp_1)
                                y_pred.append(temp_2)
                                break
                            else:
                                temp_1.append(label_map[label_ids[i][j]])
                                temp_2.append(label_map[logits[i][j]])
                print('Eval loss  = ', eval_loss / nb_eval_steps)
                eval_loss_list.append(eval_loss / nb_eval_steps)

        # region save model
        if not os.path.exists(args.output_dir + '/checkpoint_' + str(epoch)):
            os.makedirs(args.output_dir + '/checkpoint_' + str(epoch))

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_to_save.save_pretrained(args.output_dir + '/checkpoint_' + str(epoch))
        tokenizer.save_pretrained(args.output_dir + '/checkpoint_' + str(epoch))
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                        "max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1,
                        "label_map": label_map}
        json.dump(model_config,
                  open(os.path.join(args.output_dir + '/checkpoint_' + str(epoch), "model_config.json"), "w"))

        ### Evaluation results
        report = classification_report(y_true, y_pred, digits=4, suffix=False)
        output_eval_file = os.path.join(args.output_dir + '/checkpoint_' + str(epoch), "eval_results_test.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results dev*****")
            logger.info("\n%s", report)
            writer.write(report)

            for index, (train_loss, eval_loss) in enumerate(zip(train_loss_list, eval_loss_list)):
                writer.write("\nEpoch " + str(index) + ": " + str(train_loss) + ", " + str(eval_loss))


if __name__ == "__main__":
    main()
