from model import *
from utils import Data, split_validation
import torch
import random
import pickle
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="diginetica",
    help="dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample",
)
parser.add_argument("--batchSize", type=int, default=512, help="input batch size")
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
    default=10,
    help="the number of steps after which the learning rate decay",
)  # 原本是3、10
parser.add_argument(
    "--l2", type=float, default=1e-5, help="l2 penalty"
)  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument("--step", type=int, default=1, help="gnn propogation steps")
parser.add_argument(
    "--patience",
    type=int,
    default=3,
    help="the number of epoch to wait before early stop ",
)
parser.add_argument(
    "--nonhybrid", action="store_true", help="only use the global preference to predict"
)
parser.add_argument("--validation", action="store_true", help="validation")
parser.add_argument("--seed", type=int, default=3407, help="random seed")
parser.add_argument(
    "--n_layers", type=int, default=2, help="num of layers of lintransformer"
)
parser.add_argument(
    "--valid_portion",
    type=float,
    default=0.1,
    help="split the portion of training set as validation set",
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
    "--temperature_decay", type=float, default=0.9, help="Decay rate for temperature"
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
parser.add_argument(
    "--item_id", type=int, default=33889, help="Item ID for comparison"
)
opt = parser.parse_args()
print(opt)

train_data = pickle.load(open("../datasets/" + opt.dataset + "/train.txt", "rb"))
# train_data = Data(train_data)
train_data = Data(
    train_data,
    shuffle=True,
    num_neg=opt.num_neg,
    similarity_threshold=opt.similarity_threshold,
)
test_data = pickle.load(open("../datasets/" + opt.dataset + "/test.txt", "rb"))
# test_data = Data(test_data)
test_data = Data(
    test_data,
    shuffle=False,
    num_neg=opt.num_neg,
    similarity_threshold=opt.similarity_threshold,
)
# opt.len_max = 145
# n_node = 37484
opt.len_max = max(train_data.len_max, test_data.len_max)
n_node = 43098


model = trans_to_cuda(SessionGraph(opt, n_node))
# model.load_state_dict(torch.load('./output/diginetica_batchsize_512/epoch_12.pth'))
model.load_state_dict(torch.load("./output/diginetica/epoch_21.pth"))
print("---model loaded---")

model.eval()
slices = test_data.generate_batch(model.batch_size)
res = []
# last_items = np.array(test_data.last_items)
# for i in tqdm.tqdm(slices):
#     targets, scores, global_session = forward(model, i, test_data)
#     sub_scores = scores.topk(20)[1]

#     targets = (targets - 1).tolist()
#     l_i = last_items[i].tolist()
#     sub_scores = trans_to_cpu(sub_scores).detach().numpy().reshape(-1, 20).tolist()

#     res += list(zip(l_i, targets, sub_scores))
#     print("res:",res)

# 修改：对于每个session，选择前20个预测项，并记录其出现次数与排名
top_k = 20
item_id = opt.item_id  # Get the specified item ID
last_items = np.array(test_data.last_items)
for i in tqdm.tqdm(slices):
    targets, scores, global_session = forward(model, i, test_data)
    sub_scores = scores.topk(top_k)[1]  # 选择前20个预测项

    targets = (targets - 1).tolist()
    l_i = last_items[i].tolist()
    sub_scores = trans_to_cpu(sub_scores).detach().numpy().reshape(-1, top_k).tolist()

    # 只考虑与指定Item ID相关的会话
    if item_id in l_i:
        for rank, item in enumerate(sub_scores[0]):
            res.append((item, rank))  # 存储预测项和其排名
            print("res:",res)

np.save("res." + str(opt.dataset) + ".npy", res)