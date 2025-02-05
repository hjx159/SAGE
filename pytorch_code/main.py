#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import pickle
import time
from utils import Data, split_validation
from model import *


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="diginetica",
    help="dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample",
)
parser.add_argument("--batchSize", type=int, default=100, help="input batch size")
parser.add_argument("--hiddenSize", type=int, default=100, help="hidden state size")
parser.add_argument(
    "--epoch", type=int, default=30, help="the number of epochs to train for"
)
parser.add_argument(
    "--lr", type=float, default=0.001, help="learning rate"
)  # [0.001, 0.0005, 0.0001]
parser.add_argument("--lr_dc", type=float, default=0.1, help="learning rate decay rate")
parser.add_argument(
    "--lr_dc_step",
    type=int,
    default=3,
    help="the number of steps after which the learning rate decay",
)
parser.add_argument(
    "--l2", type=float, default=1e-5, help="l2 penalty"
)  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument("--step", type=int, default=1, help="gnn propogation steps")
parser.add_argument(
    "--patience",
    type=int,
    default=3,  # 原本是10
    help="the number of epoch to wait before early stop ",
)
parser.add_argument(
    "--nonhybrid", action="store_true", help="only use the global preference to predict"
)
parser.add_argument("--validation", action="store_true", help="validation")
parser.add_argument(
    "--valid_portion",
    type=float,
    default=0.1,
    help="split the portion of training set as validation set",
)
# 新增参数
parser.add_argument("--seed", type=int, default=3407, help="random seed")
parser.add_argument(
    "--n_layers", type=int, default=2, help="num of layers of lintransformer"
)
parser.add_argument(
    "--contrast_loss_weight", type=float, default=0.2, help="Weight of contrastive loss"
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.1,
    help="Initial temperature for InfoNCE loss",
)
parser.add_argument(
    "--temperature_decay", type=float, default=0.95, help="Decay rate for temperature"
)
parser.add_argument(
    "--min_temperature", type=float, default=0.01, help="Minimum temperature value"
)
parser.add_argument(
    "--num_neg", type=int, default=8, help="Number of negative samples per session"
)
parser.add_argument(
    "--similarity_threshold", type=float, default=0.1, help="Similarity threshold"
)
parser.add_argument("--fusion_factor", type=float, default=0.8, help="fusion factor")
opt = parser.parse_args()

# 配置日志
from logger_config import setup_logger

# 获取当前时间戳（整数）
current_time = str(int(time.time()))
log_filename = (
    "log/"
    + opt.dataset
    + "_current_time="
    + current_time
    + "_batchSize="
    + str(opt.batchSize)
    + "_hiddenSize="
    + str(opt.hiddenSize)
    + "_lr_dc_step="
    + str(opt.lr_dc_step)
    + "_seed="
    + str(opt.seed)
    + "_n_layers="
    + str(opt.n_layers)
    + "_contrast_loss_weight="
    + str(opt.contrast_loss_weight)
    + "_temperature="
    + str(opt.temperature)
    + "_temperature_decay="
    + str(opt.temperature_decay)
    + "_min_temperature="
    + str(opt.min_temperature)
    + "_num_neg="
    + str(opt.num_neg)
    + "_similarity_threshold="
    + str(opt.similarity_threshold)
    + ".log"
)

logger = setup_logger(log_filename)

print(opt)
logger.info(opt)


# 设置随机种子
import random

seed = opt.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    train_data = pickle.load(open("../datasets/" + opt.dataset + "/train.txt", "rb"))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open("../datasets/" + opt.dataset + "/test.txt", "rb"))
    # 加载数据时，新增了两个参数
    train_data = Data(
        train_data,
        shuffle=True,
        num_neg=opt.num_neg,
        similarity_threshold=opt.similarity_threshold,
    )
    test_data = Data(
        test_data,
        shuffle=False,
        num_neg=opt.num_neg,
        similarity_threshold=opt.similarity_threshold,
    )
    # 计算最大长度
    opt.len_max = max(train_data.len_max, test_data.len_max)
    if opt.dataset == "diginetica":
        n_node = 43098
    elif opt.dataset == "yoochoose1_64" or opt.dataset == "yoochoose1_4":
        n_node = 37484
    else:
        n_node = 310

    model = trans_to_cuda(SessionGraph(opt, n_node))
    # 初始化memory_bank
    model.memory_bank = np.random.random([train_data.length, 2, opt.hiddenSize])

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        # f = open('output_abla_weakneg_yc.txt', 'a')
        print("-------------------------------------------------------")
        print("epoch: ", epoch)
        logger.info("-------------------------------------------------------")
        logger.info("epoch: %d" % epoch)
        # 传入epoch参数
        hit, mrr = train_test(model, train_data, test_data, epoch)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print("Best Result:")
        print(
            "\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d"
            % (best_result[0], best_result[1], best_epoch[0], best_epoch[1])
        )
        # print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]), file=f)
        logger.info("Best Result:")
        logger.info(
            (
                "\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d"
                % (best_result[0], best_result[1], best_epoch[0], best_epoch[1])
            )
        )

        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
        # 在每个epoch结束时，填充memory_bank
        print("---filling memory bank---")
        logger.info("---filling memory bank---")
        fill_memory_bank(model, train_data)
        # 保存模型
        model.save(epoch)
        # 更新温度系数
        model.update_temperature()
    print("-------------------------------------------------------")
    logger.info("-------------------------------------------------------")
    end = time.time()
    print("Run time: %f s" % (end - start))
    logger.info("Run time: %f s" % (end - start))


if __name__ == "__main__":
    main()
