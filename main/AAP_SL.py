import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import re
import argparse
import torch
import numpy as np
import pytorch_lightning as pl
import torch.utils.data as Data

from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core import LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from model import TextCNN
from util import util_file, util_metric, util_data
from config import parser


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, attn_mask, labels):
        self.input_ids = input_ids
        self.attn_mask = attn_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_mask[idx], self.labels[idx]


class MyDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.batch_size = args.batch_size
        self.k = None

    def construct_dataset(self, input_ids, attn_mask, label):
        return MyDataSet(input_ids, attn_mask, label)

    def prepare_data(self):
        self.train_raw_data, self.train_label = util_file.read_tsv_data(self.args.path_train_data,
                                                                        skip_first=True)
        self.test_raw_data, self.test_label = util_file.read_tsv_data(self.args.path_test_data,
                                                                      skip_first=True)

        num_train, num_test = len(self.train_raw_data), len(self.test_raw_data)
        self.all_data = self.train_raw_data + self.test_raw_data
        print('self.train_raw_data', len(self.train_raw_data))
        print('self.test_raw_data', len(self.test_raw_data))
        self.all_input_ids = torch.tensor(util_data.tokenize(self.all_data, self.args.path_tokenizer)['input_ids'])
        self.all_attn_mask = torch.tensor(util_data.tokenize(self.all_data, self.args.path_tokenizer)['attention_mask'])
        self.train_ids, self.train_attn_mask = self.all_input_ids[:num_train], self.all_attn_mask[:num_train]
        self.test_ids, self.test_attn_mask = self.all_input_ids[num_train:], self.all_attn_mask[num_train:]
        print('self.train_ids', len(self.train_ids))
        print('self.test_ids', len(self.test_ids))

        if self.args.max_len < len(self.train_ids[0]):
            print('WARN:', 'self.args.max_len < max len of the dataset')
        self.args.max_len = len(self.train_ids[0])
        print('len(self.train_ids[0])', len(self.train_ids[0]))
        print('Reset max_len', self.args.max_len)

        if self.args.proportion is not None:
            self.train_ids = self.train_ids[:int(args.proportion * len(self.train_ids))]
            self.train_attn_mask = self.train_attn_mask[:int(args.proportion * len(self.train_attn_mask))]
            self.train_label = self.train_label[:int(args.proportion * len(self.train_label))]

        if self.args.train_mode == 'cross_validation':
            self.k = 0
            self.train_ids_CV = []
            self.valid_ids_CV = []
            self.train_attn_mask_CV = []
            self.valid_attn_mask_CV = []
            self.train_label_CV = []
            self.valid_label_CV = []
            for iter_k in range(self.args.k_fold):
                train_ids_k = [x for i, x in enumerate(self.train_ids) if
                               i % self.args.k_fold != iter_k]
                valid_ids_k = [x for i, x in enumerate(self.train_ids) if
                               i % self.args.k_fold == iter_k]
                train_attn_mask_k = [x for i, x in enumerate(self.train_attn_mask) if
                                     i % self.args.k_fold != iter_k]
                valid_attn_mask_k = [x for i, x in enumerate(self.train_attn_mask) if
                                     i % self.args.k_fold == iter_k]
                train_label_k = [x for i, x in enumerate(self.train_label) if i % self.args.k_fold != iter_k]
                valid_label_k = [x for i, x in enumerate(self.train_label) if i % self.args.k_fold == iter_k]
                self.train_ids_CV.append(train_ids_k)
                self.valid_ids_CV.append(valid_ids_k)
                self.train_attn_mask_CV.append(train_attn_mask_k)
                self.valid_attn_mask_CV.append(valid_attn_mask_k)
                self.train_label_CV.append(train_label_k)
                self.valid_label_CV.append(valid_label_k)

    def setup(self, stage):
        # 实现数据集的定义，每张GPU都会执行该函数, stage 用于标记是用于什么阶段
        if stage == 'fit' or stage is None:
            if self.args.train_mode == 'train':
                self.train_dataset = self.construct_dataset(self.train_ids, self.train_attn_mask, self.train_label)
                self.val_dataset = self.construct_dataset(self.test_ids, self.test_attn_mask, self.test_label)
            elif args.train_mode == 'cross_validation':
                self.train_dataset = self.construct_dataset(self.train_ids_CV[self.k], self.train_attn_mask_CV[self.k],
                                                            self.train_label_CV[self.k])
                self.val_dataset = self.construct_dataset(self.valid_ids_CV[self.k], self.valid_attn_mask_CV[self.k],
                                                          self.valid_label_CV[self.k])
                self.k += 1
            else:
                raise RuntimeError('No such args.train_mode')
        if stage == 'test' or stage is None:
            self.test_dataset = self.construct_dataset(self.test_ids, self.test_attn_mask, self.test_label)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)


class Lit_DeepAAPred(LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        if args.loss == 'CE':
            self.CE = torch.nn.CrossEntropyLoss()
        elif args.loss == 'FL':
            self.FL = Focal_Loss.FocalLoss(args.num_class, alpha=torch.tensor(args.alpha), gamma=args.gamma)
        else:
            raise RuntimeError('No such args.loss')

        self.model = None
        if args.model == 'TextCNN':
            self.model = TextCNN.TextCNN(args)
        elif args.model == 'TextRNN':
            self.model = TextRNN.TextRNN(args)
        elif args.model == 'BiLSTM_Attention':
            self.model = BiLSTM_Attention.BiLSTM_Attention(args)
        elif args.model == 'Transformer Encoder':
            # self.model = Transformer_Encoder.Transformer_Encoder(args)
            self.model = TE.TE(args)
        elif args.model == 'Ensemble':
            self.model = Ensemble.Ensemble(args)
        else:
            raise RuntimeError('No such args.model')

        self.save_hyperparameters()

    def forward(self, input_ids, attn_mask):
        logits = None
        if self.args.model == 'TextCNN' or self.args.model == 'TextRNN' or self.args.model == 'BiLSTM_Attention':
            logits, embedding = self.model(input_ids)
        elif self.args.model == 'Transformer Encoder' or self.args.model == 'Ensemble':
            logits, embedding = self.model(input_ids, attn_mask)
        else:
            raise RuntimeError('No such args.model')
        return logits

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.reg)

    def get_acc(self, logits, labels):
        return (torch.max(logits, 1)[1] == labels).sum() / labels.size(0)

    def get_loss(self, logits, labels):
        loss = None
        if self.args.loss == 'CE':
            loss = self.CE(logits, labels)
        elif self.args.loss == 'FL':
            loss = self.FL(logits, labels)
        return loss

    def training_step(self, batch, batch_idx):
        input_ids, attn_mask, labels = batch
        # print('training_step input_ids', input_ids.size())
        logits = self.forward(input_ids, attn_mask)
        loss = self.get_loss(logits, labels)
        acc = self.get_acc(logits, labels)
        return {'loss': loss, 'acc': acc}

    def training_step_end(self, batch_parts):
        loss = batch_parts['loss']
        acc = batch_parts['acc']
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

    def validation_step(self, batch, batch_idx):
        input_ids, attn_mask, labels = batch
        # print('validation_step input_ids', input_ids.size())
        logits = self.forward(input_ids, attn_mask)
        loss = self.get_loss(logits, labels)
        # acc = self.get_acc(logits, labels)

        pred_prob = logits.softmax(dim=1)[:, 1].cpu()
        label_pred = []
        for p in pred_prob:
            pred = 1 if p >= 0.5 else 0
            label_pred.append(pred)
        label_real = labels.cpu()

        # [ACC, Sensitivity, Specificity, AUC, MCC]
        metric, roc_data, prc_data = util_metric.caculate_metric(pred_prob, label_pred, label_real)
        return {'loss': loss, 'ACC': metric[0], 'SE': metric[1], 'SP': metric[2], 'AUC': metric[3], 'MCC': metric[4]}

    def validation_step_end(self, batch_parts):
        loss = batch_parts['loss']
        acc = batch_parts['ACC']
        sen = batch_parts['SE']
        sp = batch_parts['SP']
        AUC = batch_parts['AUC']
        MCC = batch_parts['MCC']

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_ACC', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_SE', sen, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_SP', sp, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_AUC', AUC, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_MCC', MCC, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        input_ids, attn_mask, labels = batch
        logits = self.forward(input_ids, attn_mask)
        loss = self.get_loss(logits, labels)
        # acc = self.get_acc(logits, labels)

        pred_prob = logits.softmax(dim=1)[:, 1].cpu()
        label_pred = []
        for p in pred_prob:
            pred = 1 if p >= 0.5 else 0
            label_pred.append(pred)
        label_real = labels.cpu()

        # [ACC, Sensitivity, Specificity, AUC, MCC]
        metric, roc_data, prc_data = util_metric.caculate_metric(pred_prob, label_pred, label_real)
        return {'loss': loss, 'ACC': metric[0], 'SE': metric[1], 'SP': metric[2], 'AUC': metric[3], 'MCC': metric[4]}

    def test_step_end(self, batch_parts):
        loss = batch_parts['loss']
        acc = batch_parts['ACC']
        sen = batch_parts['SE']
        sp = batch_parts['SP']
        AUC = batch_parts['AUC']
        MCC = batch_parts['MCC']

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_ACC', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_SE', sen, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_SP', sp, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_AUC', AUC, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_MCC', MCC, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument('-model', type=str, default='Ensemble',
        #                     choices=['Ensemble', 'TextCNN', 'TextRNN', 'BiLSTM_Attention', 'Transformer Encoder'])
        # parser.add_argument('-model', type=str, default='TextCNN',
        #                     choices=['TextCNN', 'TextRNN', 'BiLSTM_Attention', 'Transformer Encoder'])
        # parser.add_argument('-model', type=str, default='TextRNN',
        #                     choices=['TextCNN', 'TextRNN', 'BiLSTM_Attention', 'Transformer Encoder'])
        parser.add_argument('-model', type=str, default='BiLSTM_Attention',
                            choices=['TextCNN', 'TextRNN', 'BiLSTM_Attention', 'Transformer Encoder'])
        # parser.add_argument('-model', type=str, default='Transformer Encoder',
        #                     choices=['TextCNN', 'TextRNN', 'BiLSTM_Attention', 'Transformer Encoder'])

        # parser.add_argument('-pos_embed_type', type=str, default='pe',
        #                     choices=['pe', 'random'])
        parser.add_argument('-pos_embed_type', type=str, default='random',
                            choices=['pe', 'random'])

        parser.add_argument('-vocab_size', type=int, default=28)
        # parser.add_argument('-lr', type=float, default=0.003)
        # parser.add_argument('-lr', type=float, default=0.001)
        # parser.add_argument('-lr', type=float, default=0.0007)
        # parser.add_argument('-lr', type=float, default=0.0005)
        # parser.add_argument('-lr', type=float, default=0.0003)
        parser.add_argument('-lr', type=float, default=0.0001)
        # parser.add_argument('-lr', type=float, default=0.00005)
        # parser.add_argument('-lr', type=float, default=0.00001)

        parser.add_argument('-reg', type=float, default=0.000)
        # parser.add_argument('-reg', type=float, default=0.001)
        # parser.add_argument('-reg', type=float, default=0.01)
        # parser.add_argument('-loss', type=str, default='CE', choices=['CE', 'FL'])
        parser.add_argument('-loss', type=str, default='FL', choices=['CE', 'FL'])
        parser.add_argument('-alpha', type=list, default=[0.1, 0.9])
        parser.add_argument('-gamma', type=int, default=2)

        # Common
        parser.add_argument('-static', type=bool, default=False)
        parser.add_argument('-fine_tune', type=bool, default=False)
        parser.add_argument('-vectors', type=object, default=None)
        parser.add_argument('-dropout', type=float, default=0.5)

        # TextCNN
        parser.add_argument('-CNN_dim_embedding', type=int, default=128)
        parser.add_argument('-num_filter', type=int, default=128)
        # parser.add_argument('-filter_sizes', type=str, default='1,2,4,8,12,15')
        parser.add_argument('-filter_sizes', type=str, default='1,2,4,8,16,32,48,64')

        # TextRNN or BiLSTM + Attention
        parser.add_argument('-RNN_net', type=str, default='LSTM', choices=['LSTM', 'GRU', 'RNN'])
        parser.add_argument('-RNN_dim_embedding', type=int, default=64)
        parser.add_argument('-hidden_size', type=int, default=64)
        parser.add_argument('-RNN_num_layers', type=int, default=1)
        parser.add_argument('-bidirectional', type=bool, default=True)
        parser.add_argument('-attention_size', type=int, default=32)

        # Transformer Encoder
        parser.add_argument('-TE_num_layers', type=int, default=1)
        parser.add_argument('-n_head', type=int, default=16)
        parser.add_argument('-d_model', type=int, default=32)
        parser.add_argument('-dim_feedforward', type=int, default=2048)
        parser.add_argument('-activation', type=str, default='gelu')
        return parser


if __name__ == '__main__':
    pl.seed_everything(50)
    parser = parser.get_default_parser()
    # 设置超参数
    parser = MyDataModule.add_argparse_args(parser)  # 设置数据代码超参数
    parser = pl.Trainer.add_argparse_args(parser)  # 设置工程代码超参数
    parser = Lit_DeepAAPred.add_model_specific_args(parser)  # 设置研究代码超参数
    args = parser.parse_args()

    print('args', args)

    if args.train_mode == 'train':
        AAP_dm = MyDataModule(args)
        AAP_dm.prepare_data()

        checkpoint_callback = ModelCheckpoint(filename='AAP, {epoch:03d}, {val_ACC:.4f}',
                                              monitor='val_ACC', save_top_k=5, mode='max')
        logger = TensorBoardLogger(save_dir='../log', name=args.learn_name)
        trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=checkpoint_callback)
        AAP_dm.setup(stage='fit')
        model = Lit_DeepAAPred(args)
        print('=' * 100)
        print('model.hparams: \n', model.hparams)
        print('=' * 100)

        trainer.fit(model=model, datamodule=AAP_dm)
        test_result = trainer.test(ckpt_path="best", datamodule=AAP_dm)

        best_ACC = test_result[0]['test_ACC']
        best_SE = test_result[0]['test_SE']
        best_SP = test_result[0]['test_SP']
        best_AUC = test_result[0]['test_AUC']
        best_MCC = test_result[0]['test_MCC']

        log_text = '\n' + '=' * 20 + ' Independent Test Performance' + '=' * 20 \
                   + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            best_ACC, best_SE, best_SP, best_AUC, best_MCC) + '\n' + '=' * 60
        print('=' * 50)
        print(log_text)
    elif args.train_mode == 'continue_train':
        # TODO: TO be debug
        checkpoint_callback = ModelCheckpoint(filename='AAP, {epoch:03d}, {val_ACC:.4f}',
                                              monitor='val_ACC', save_top_k=5, mode='max')
        logger = TensorBoardLogger('log_dir', name='AAP_CNN')

        CKPT_PATH = 'log_dir/test_PL/version_10/checkpoints/mnist-epoch=000-val_acc=0.8959.ckpt'
        checkpoint = torch.load(CKPT_PATH)
        hparams = checkpoint["hyper_parameters"]
        loaded_args = hparams['args']

        AAP_dm = MyDataModule(loaded_args)
        AAP_dm.prepare_data()
        AAP_dm.setup(stage='fit')
        model = Lit_DeepAAPred.load_from_checkpoint(checkpoint_path=CKPT_PATH)
        trainer = pl.Trainer(resume_from_checkpoint=CKPT_PATH, logger=logger, callbacks=checkpoint_callback)
        trainer.fit(model=model, datamodule=AAP_dm)
    elif args.train_mode == 'cross_validation':
        AAP_dm = MyDataModule(args)
        AAP_dm.prepare_data()

        acc, se, sp, auc, mcc = [], [], [], [], []

        for k in range(args.k_fold):
            AAP_dm.setup(stage='fit')

            checkpoint_callback = ModelCheckpoint(filename='AAP_CV, {epoch:03d}, {val_ACC:.4f}',
                                                  monitor='val_ACC', save_top_k=5, mode='max')
            logger = TensorBoardLogger(save_dir='log_dir', name=args.learn_name)
            trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=checkpoint_callback)
            model = Lit_DeepAAPred(args)
            print('=' * 100)
            print('model.hparams: \n', model.hparams)
            print('=' * 100)

            trainer.fit(model=model, datamodule=AAP_dm)
            test_result = trainer.test(ckpt_path="best", datamodule=AAP_dm)
            print('test_result [{}]'.format(k), test_result)
            acc.append(test_result[0]['test_ACC'])
            se.append(test_result[0]['test_SE'])
            sp.append(test_result[0]['test_SP'])
            auc.append(test_result[0]['test_AUC'])
            mcc.append(test_result[0]['test_MCC'])

        avg_ACC = np.mean(acc)
        avg_SE = np.mean(se)
        avg_SP = np.mean(sp)
        avg_AUC = np.mean(auc)
        avg_MCC = np.mean(mcc)
        log_text = '\n' + '=' * 20 + ' Cross Validation Performance' + '=' * 20 \
                   + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            avg_ACC, avg_SE, avg_SP, avg_AUC, avg_MCC) + '\n' + '=' * 60
        print('=' * 50)
        print(log_text)
    elif args.train_mode == 'test':
        CKPT_PATH = 'log_dir/AAP_CNN/version_1/checkpoints/AAP, epoch=042, val_ACC=0.8036.ckpt'

        model = Lit_DeepAAPred.load_from_checkpoint(checkpoint_path=CKPT_PATH)
        checkpoint = torch.load(CKPT_PATH)
        hparams = checkpoint["hyper_parameters"]
        loaded_args = hparams['args']
        print('args', args)
        print('loaded_args', loaded_args)
        print('=' * 50)
        for key, value in loaded_args.__dict__.items():
            for key2, value2 in args.__dict__.items():
                if key == key2 and value != value2:
                    print('key, value', key, value)
                    print('key2, value2', key2, value2)
                    print('-' * 20)
        print('=' * 50)

        AAP_dm = MyDataModule(loaded_args)
        AAP_dm.prepare_data()
        AAP_dm.setup(stage='test')
        trainer = pl.Trainer.from_argparse_args(loaded_args)
        test_result = trainer.test(model=model, datamodule=AAP_dm)

        best_ACC = test_result[0]['test_ACC']
        best_SE = test_result[0]['test_SE']
        best_SP = test_result[0]['test_SP']
        best_AUC = test_result[0]['test_AUC']
        best_MCC = test_result[0]['test_MCC']

        log_text = '\n' + '=' * 20 + ' Independent Test Performance' + '=' * 20 \
                   + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            best_ACC, best_SE, best_SP, best_AUC, best_MCC) + '\n' + '=' * 60
        print('=' * 50)
        print(log_text)
