import argparse

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()

    def set_arguments(self):
        # GPU ID
        self.parser.add_argument('--gpu',           type=str, help='gpu ids to use e.g. 0,1,2,...')
        # 分布式半监督学习算法
        self.parser.add_argument('--algorithm',     type=str, help='distributed semi-supervised learning algorithm i.e. my_dssl, gossip_fixmatch, gossip, gossip_uy_known_fixmatch')
        # 连接拓扑
        self.parser.add_argument('--topology',      type=str, help='devices connectting topology i.e. random, fc(fully connected), star, round, rt(random tree)')
        # 任务类型
        self.parser.add_argument('running_mode',    type=str, help='running_mode i.e. generate or train')
        # 数据集选择
        self.parser.add_argument('dataset',         type=str, help='dataset i.e. mnist, fmnist (fashion mnist) or cifar10')
        # 是否IID
        self.parser.add_argument('dist',          type=str, help='distribution i.e. iid, non_iid')
        # 每轮客户端数量
        # self.parser.add_argument('--frac-clients', type=float, help='fraction of clients per round')

    def parse(self):
        args, unparsed = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args
