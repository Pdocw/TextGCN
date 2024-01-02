import gc
import warnings
from time import time
import sys
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import argparse
from layer import GCN
from utils import accuracy
from utils import macro_f1
from utils import CudaUse
from utils import EarlyStopping
from utils import LogResult
from utils import preprocess_adj
from utils import print_graph_detail
from utils import read_file
from utils import return_seed

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")


def get_train_test(target_fn):
    train_lst = list()
    test_lst = list()
    with read_file(target_fn, mode="r") as fin:
        for indx, item in enumerate(fin):
            if item.split("\t")[1] in {"train", "training", "20news-bydate-train"}:
                train_lst.append(indx)
            else:
                test_lst.append(indx)
    return train_lst, test_lst


class PrepareData:
    def __init__(self, args):
        print("prepare data")
        self.graph_path = "data/graph"
        self.args = args

        # graph
        graph = nx.read_weighted_edgelist(f"{self.graph_path}/{args.dataset}.txt"
                                          , nodetype=int)
        print_graph_detail(graph)
        adj = nx.to_scipy_sparse_matrix(graph,
                                        nodelist=list(range(graph.number_of_nodes())),
                                        weight='weight',
                                        dtype=np.float)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        self.adj = preprocess_adj(adj, is_sparse=True)

        # features
        self.nfeat_dim = graph.number_of_nodes()
        row = list(range(self.nfeat_dim))
        col = list(range(self.nfeat_dim))
        value = [1.] * self.nfeat_dim
        shape = (self.nfeat_dim, self.nfeat_dim)
        indices = torch.from_numpy(
                np.vstack((row, col)).astype(np.int64))
        values = torch.FloatTensor(value)
        shape = torch.Size(shape)

        self.features = torch.sparse.FloatTensor(indices, values, shape)

        # target
        target_fn = f"data/text_dataset/{self.args.dataset}.txt"
        target = np.array(pd.read_csv(target_fn,
                                      sep="\t",
                                      header=None)[2])
        target2id = {label: indx for indx, label in enumerate(set(target))}
        self.target = [target2id[label] for label in target]
        self.nclass = len(target2id)

        # train val test split
        self.train_lst, self.test_lst = get_train_test(target_fn)


class TextGCNTrainer:
    def __init__(self, args, model, pre_data):
        self.args = args
        self.model = model
        self.device = args.device

        self.max_epoch = self.args.max_epoch
        self.set_seed()

        self.dataset = args.dataset
        self.predata = pre_data
        self.earlystopping = EarlyStopping(args.early_stopping)

    def set_seed(self):
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

    def fit(self):
        self.prepare_data()
        self.model = self.model(nfeat=self.nfeat_dim,
                                nhid=self.args.nhid,
                                nclass=self.nclass,
                                dropout=self.args.dropout)
        print(self.model.parameters)
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.model_param = sum(param.numel() for param in self.model.parameters())
        print('# model parameters:', self.model_param)
        self.convert_tensor()

        start = time()
        self.train()
        self.train_time = time() - start

    @classmethod
    def set_description(cls, desc):
        string = ""
        for key, value in desc.items():
            if isinstance(value, int):
                string += f"{key}:{value} "
            else:
                string += f"{key}:{value:.4f} "
        print(string)

    def prepare_data(self):
        self.adj = self.predata.adj
        self.nfeat_dim = self.predata.nfeat_dim
        self.features = self.predata.features
        self.target = self.predata.target
        self.nclass = self.predata.nclass

        self.train_lst, self.val_lst = train_test_split(self.predata.train_lst,
                                                        test_size=self.args.val_ratio,
                                                        shuffle=True,
                                                        random_state=self.args.seed)
        self.test_lst = self.predata.test_lst

    def convert_tensor(self):
        self.model = self.model.to(self.device)
        self.adj = self.adj.to(self.device)
        self.features = self.features.to(self.device)
        self.target = torch.tensor(self.target).long().to(self.device)
        self.train_lst = torch.tensor(self.train_lst).long().to(self.device)
        self.val_lst = torch.tensor(self.val_lst).long().to(self.device)

    def train(self):
        for epoch in range(self.max_epoch):
            self.model.train()
            self.optimizer.zero_grad()
            """
            features 是一个对角矩阵 
            """
            logits = self.model.forward(self.features, self.adj)
            loss = self.criterion(logits[self.train_lst],
                                  self.target[self.train_lst])

            loss.backward()
            self.optimizer.step()

            val_desc = self.val(self.val_lst)

            desc = dict(**{"epoch"     : epoch,
                           "train_loss": loss.item(),
                           }, **val_desc)

            self.set_description(desc)

            if self.earlystopping(val_desc["val_loss"]):
                break

    @torch.no_grad()
    def val(self, x, prefix="val"):
        self.model.eval()
        with torch.no_grad():
            logits = self.model.forward(self.features, self.adj)
            loss = self.criterion(logits[x],
                                  self.target[x])
            acc = accuracy(logits[x],
                           self.target[x])
            f1, precision, recall = macro_f1(logits[x],
                                             self.target[x],
                                             num_classes=self.nclass)

            desc = {
                f"{prefix}_loss": loss.item(),
                "acc"           : acc,
                "macro_f1"      : f1,
                "precision"     : precision,
                "recall"        : recall,
            }
        return desc

    @torch.no_grad()
    def test(self):
        self.test_lst = torch.tensor(self.test_lst).long().to(self.device)
        test_desc = self.val(self.test_lst, prefix="test")
        test_desc["train_time"] = self.train_time
        test_desc["model_param"] = self.model_param
        return test_desc


def main():
    parser = argparse.ArgumentParser(description="Run .")
    parser.add_argument('--dataset', default='20ng', type=str, help='Dataset')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Device to use')
    parser.add_argument('--nhid', default=200, type=int, help='Hidden layer dimension')
    parser.add_argument('--max_epoch', default=200, type=int, help='Maximum number of epochs')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout rate')
    parser.add_argument('--val_ratio', default=0.1, type=float, help='Validation set ratio')
    parser.add_argument('--early_stopping', default=10, type=int, help='Early stopping criteria')
    parser.add_argument('--lr', default=0.02, type=float, help='Learning rate')

    args = parser.parse_args()
    model = GCN

    print(args)

    predata = PrepareData(args)
    cudause = CudaUse()

    record = LogResult()
    seed_lst = list()
    times = 1
    for ind, seed in enumerate(return_seed(times)):
        print(f"\n\n==> {ind}, seed:{seed}")
        args.seed = seed
        seed_lst.append(seed)

        framework = TextGCNTrainer(model=model, args=args, pre_data=predata)
        framework.fit()

        if torch.cuda.is_available():
            gpu_mem = cudause.gpu_mem_get(_id=0)
            record.log_single(key="gpu_mem", value=gpu_mem)

        record.log(framework.test())

        del framework
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("==> seed set:")
    print(seed_lst)
    record.show_str()


if __name__ == '__main__':
    main()
