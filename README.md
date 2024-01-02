# Graph Convolutional Networks for Text Classification in PyTorch [PDF](https://arxiv.org/abs/1809.05679)

## Dataset
| 数据集 | 描述 | 下载 |
| --- | --- | --- |
| **20NG** | 这是一个用于文本分类、文本挖据和信息检索研究的国际标准数据集之一。数据集收集了大约20,000左右的新闻组文档，均匀分为20个不同主题的新闻组集合 | [20NG](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz) |
| **R8** | R8是路透社21578数据集的两个子集之一。R8有8个类别，分为5,485个训练文档和2,189个测试文档 | [R8](https://www.cs.umb.edu/˜smimarog/textmining/datasets/) |
| **R52** | R52也是路透社21578数据集的子集之一。R52有52个类别，分为6,532个训练样本和2,568个测试样本 | [R52](https://www.cs.umb.edu/˜smimarog/textmining/datasets/) |
| **Ohsumed** | Ohsumed数据集是MEDLINE医药信息数据库的一部分，包含了从1987年到1991年五年间270个医药类杂志的标题和/或摘要，总共包含了348,566个文档 | [Ohsumed](http://disi.unitn.it/moschitti/corpora.htm) |
| **MR** | MR是电影评论数据集，其中每个样本对应一个句子。语料库有5,331个积极样本和5,331个消极样本 | [MR](https://github.com/mnqu/PTE/tree/master/data/mr) |

## Benchmark

| dataset       | 20NG | R8 | R52 | Ohsumed | MR  |
|---------------|----------|------|--------|--------|--------|
| Text GCN | 0.8634±0.0009    | 0.9707±0.0010 | 0.9356±00018   | 0.6836±0.0056   | 0.7674±0.0020   |
| Results  | 0.8617±0.0014    | 0.9710±0.0014 | 0.9357±0.0016  | 0.6807±0.0040   | 0.7579±0.0047   |

NOTE: The result of the experiment is to repeat the run 10 times, and then take the average of accuracy.

## Requirements
* PyTorch==1.8.0

## Usage
1. Run `python data_processor.py 20ng`
2. Run `python build_graph.py 20ng`
3. Run `python train.py --dataset 20ng`
4. Change `20ng` in above 3 command lines to `R8`, `R52`, `ohsumed` and `mr` when producing results for other datasets.
## References
[1] [Yao, L. , Mao, C. , & Luo, Y. . (2018). Graph convolutional networks for text classification.](https://arxiv.org/abs/1809.05679)
