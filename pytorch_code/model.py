#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import tqdm
import time
import collections

from main import logger


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = (
            torch.matmul(A[:, :, : A.shape[1]], self.linear_edge_in(hidden))
            + self.b_iah
        )
        input_out = (
            torch.matmul(
                A[:, :, A.shape[1] : 2 * A.shape[1]], self.linear_edge_out(hidden)
            )
            + self.b_oah
        )
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class LinearSelfAttention(Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(LinearSelfAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)  # row-wise
        self.softmax_col = nn.Softmax(dim=-2)  # column-wise
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.scale = np.sqrt(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Our Elu Norm Attention
        elu = nn.ELU()
        # relu = nn.ReLU()
        elu_query = elu(query_layer)
        elu_key = elu(key_layer)
        query_norm_inverse = 1 / torch.norm(elu_query, dim=3, p=2)  # (L2 norm)
        key_norm_inverse = 1 / torch.norm(elu_key, dim=2, p=2)
        normalized_query_layer = torch.einsum(
            "mnij,mni->mnij", elu_query, query_norm_inverse
        )
        normalized_key_layer = torch.einsum("mnij,mnj->mnij", elu_key, key_norm_inverse)
        context_layer = (
            torch.matmul(
                normalized_query_layer, torch.matmul(normalized_key_layer, value_layer)
            )
            / self.sqrt_attention_head_size
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(
        self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
    ):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": torch.nn.functional.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = LinearSelfAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states):
        attention_output = self.multi_head_attention(hidden_states, None)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.dataset = opt.dataset
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.len_max = opt.len_max
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.pos_emb = PositionalEncoding(self.hidden_size, 0, self.len_max+1)
        self.pos_emb = nn.Embedding(self.len_max + 1, self.hidden_size)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(
            self.hidden_size * 2, self.hidden_size, bias=True
        )
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=opt.lr, weight_decay=opt.l2
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc
        )
        self.dropout = nn.Dropout(0.1)
        self.memory_bank = None
        self.fusion_factor = opt.fusion_factor

        transformer_encoder = []
        for i in range(opt.n_layers):
            transformer_encoder.append(
                (
                    f"trans{i+1}",
                    TransformerLayer(
                        n_heads=4,
                        hidden_size=self.hidden_size,
                        intermediate_size=self.hidden_size,
                        hidden_dropout_prob=0.2,
                        attn_dropout_prob=0.2,
                        hidden_act="gelu",
                        layer_norm_eps=1e-12,
                    ),
                )
            )
        self.transformer_encoder = nn.Sequential(
            collections.OrderedDict(transformer_encoder)
        )

        # 对比学习相关超参数
        self.contrast_loss_weight = opt.contrast_loss_weight  # 对比损失权重
        self.temperature = opt.temperature  # 温度参数
        self.temperature_decay = opt.temperature_decay  # 温度衰减率
        self.min_temperature = opt.min_temperature  # 最小温度值
        self.num_neg = opt.num_neg  # 负样本数量

        self.reset_parameters()

    def update_temperature(self):
        """
        根据衰减率更新温度参数，但不低于最小温度。
        """
        new_temperature = self.temperature * self.temperature_decay
        if new_temperature < self.min_temperature:
            new_temperature = self.min_temperature
        self.temperature = new_temperature
        print(f"Updated temperature: {self.temperature:.4f}")  # 可选：打印更新后的温度
        # 如果使用日志记录，可以取消注释以下行
        logger.info(f"Updated temperature: {self.temperature:.4f}")

    def save(self, epoch):
        torch.save(
            self.state_dict(),
            "./output/" + str(self.dataset) + "/epoch_" + str(epoch) + ".pth",
        )

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        """
        计算推荐系统的分数。

        参数:
        - hidden: 隐藏状态矩阵，表示用户对物品的潜在特征。
        - mask: 掩码矩阵，用于区分有效项和填充项。

        返回:
        - scores: 物品的推荐分数。
        - a: 用户的潜在特征向量。
        """
        # 创建一个与mask形状相同的张量，表示每个用户的交互序列的逆序索引
        lens = trans_to_cuda(
            torch.LongTensor(
                [
                    list(reversed(range(1, le + 1))) + [0] * (hidden.shape[1] - le)
                    for le in mask.sum(-1)
                ]
            )
        )
        # 将位置嵌入添加到隐藏状态中，以考虑序列中每个位置的重要性
        hidden = hidden + self.pos_emb(lens)
        # 提取每个用户的最后一个交互的隐藏状态
        ht = hidden[
            torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1
        ]  # batch_size x latent_size
        # 使用Transformer编码器对隐藏状态进行编码
        Ek = self.transformer_encoder(hidden)
        # 提取每个用户的最后一个交互的编码状态
        En = Ek[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]

        # 根据数据集名称来确定fusion_factor的值
        if self.dataset == "diginetica":
            fusion_factor_En = self.fusion_factor
            fusion_factor_ht = 1 - self.fusion_factor
        elif self.dataset == "yoochoose1_64" or self.dataset == "yoochoose1_4":
            fusion_factor_En = self.fusion_factor
            fusion_factor_ht = self.fusion_factor

        # 根据是否使用非混合模型来决定如何计算用户表示
        if not self.nonhybrid:
            a = F.normalize(fusion_factor_En * En + fusion_factor_ht * ht)
        else:
            a = F.normalize(En)

        # 对物品嵌入进行标准化
        b = F.normalize(self.embedding.weight[1:], dim=-1)  # n_nodes x latent_size
        # 计算用户表示和物品嵌入之间的点积，以得到推荐分数
        scores = torch.matmul(a, b.transpose(1, 0)) * 16
        return scores, a

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.dropout(F.normalize(hidden, dim=-1))
        hidden = self.dropout(self.gnn(A, hidden))
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.to(torch.device("cuda:1"))
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    """
    定义模型的前向传播过程。

    参数:
    - model: 使用的模型实例。
    - i: 当前切片的索引。
    - data: 包含训练数据的数据集。

    返回:
    - targets: 目标值。
    - scores: 模型的得分。
    - global_session: 全局会话信息。
    """
    # 从数据集中获取指定索引i的数据切片，包括输入别名、邻接矩阵、项目列表、掩码、位置和目标值。
    alias_inputs, A, items, mask, pos, targets = data.get_slice(i)

    # 将输入别名转换为CUDA张量，下同。
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(np.array(A)).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    pos = trans_to_cuda(torch.Tensor(pos).long())

    # 将项目和邻接矩阵传递给模型，获取隐藏层状态。
    hidden = model(items, A)

    # 定义一个lambda函数来获取隐藏层状态中与输入别名对应的序列隐藏状态。
    get = lambda i: hidden[i][alias_inputs[i]]

    # 使用上述lambda函数，迭代所有输入别名，并将结果堆叠起来形成序列隐藏状态。
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

    # 模型根据序列隐藏状态和掩码计算得分和全局会话信息。
    scores, global_session = model.compute_scores(seq_hidden, mask)

    # 返回目标值、得分和全局会话信息。
    return targets, scores, global_session


def fill_memory_bank(model, train_data):
    """
    为模型填充记忆库。

    该函数通过使用训练数据生成的批次进行前向传播，来填充模型的记忆库。
    它首先将模型设置为训练模式，尽管这里没有更新模型参数的操作。
    使用tqdm包装生成的批次以显示进度条，提高用户体验。

    参数:
    - model: 模型对象，拥有记忆库和批处理大小属性。
    - train_data: 训练数据对象，包含生成批次的方法。

    返回:
    无返回值，但模型的记忆库会被填充。
    """
    # 将模型设置为训练模式
    model.train()

    # 生成训练数据的批次索引
    slices = train_data.generate_batch(model.batch_size)

    # 禁止自动求导，因为不需要更新模型参数
    with torch.no_grad():
        # 遍历每个批次索引
        for i in tqdm.tqdm(slices):
            # 前向传播两次以获取两个不同的目标和分数，以及全局会话表示
            targets_1, scores_1, global_session_1 = forward(model, i, train_data)
            targets_2, scores_2, global_session_2 = forward(model, i, train_data)

            # 将全局会话表示从GPU转移到CPU，并转换为numpy数组，然后存储到记忆库中
            model.memory_bank[i, 0, :] = global_session_1.detach().cpu().numpy()
            model.memory_bank[i, 1, :] = global_session_2.detach().cpu().numpy()


def train_test(model, train_data, test_data, epoch):
    """
    对模型进行训练和测试。

    参数:
    - model: 训练和测试的模型。
    - train_data: 训练数据。
    - test_data: 测试数据。
    - epoch: 当前训练的轮数。

    返回:
    - hit: 命中率。
    - mrr: 排名倒数的平均值。
    """
    # 学习率调度
    model.scheduler.step()
    # 打印开始训练的时间
    print("start training: ", datetime.datetime.now())
    logger.info("start training: %s" % datetime.datetime.now())
    # 开始训练
    model.train()
    total_loss = 0.0
    total_contrast_loss = 0.0
    # 生成训练数据的批次
    slices = train_data.generate_batch(model.batch_size)
    # 调整对比损失的比例，第一轮训练时，对比损失为0，而后均为0.2。这里可以试试调参。
    # contrast_loss_ratio = model.contrast_loss_weight
    contrast_loss_ratio = min(epoch, model.contrast_loss_weight)
    # 对于每个批次的数据进行训练
    for i, j in zip(slices, range(len(slices))):
        model.optimizer.zero_grad()
        # 第一次前向传播，生成一个会话表示
        targets_1, scores_1, global_session_1 = forward(model, i, train_data)
        targets_1 = trans_to_cuda(torch.Tensor(targets_1).long())
        # 计算分类损失
        loss = model.loss_function(scores_1, targets_1 - 1)
        # 第二次前向传播，生成另一个会话表示，与第一次前向传播的会话表示结成对比学习的正样本对
        targets_2, scores_2, global_session_2 = forward(model, i, train_data)
        # 更新记忆库中的会话表示
        model.memory_bank[i, 0, :] = global_session_1.detach().cpu().numpy()
        model.memory_bank[i, 1, :] = global_session_2.detach().cpu().numpy()
        # 初始化对比损失
        loss_contrast = torch.tensor(0.0).to(scores_1.device)
        if contrast_loss_ratio >= 0:
            # 获取负样本
            neg_indices = train_data.neg_inputs[i]  # [batch_size, num_neg]
            neg_global_vectors = model.memory_bank[
                neg_indices
            ]  # [batch_size, num_neg, 2, hidden_size]
            neg_global_vectors = neg_global_vectors.reshape(
                neg_global_vectors.shape[0], -1, neg_global_vectors.shape[-1]
            )  # [batch_size, num_neg*2, hidden_size]
            # neg_global_vectors = torch.FloatTensor(neg_global_vectors).to(
            #     scores_1.device
            # )  # 转换为Tensor
            # 获取当前批次的全局会话表示，扩展维度以匹配负样本
            query = global_session_1.unsqueeze(1)  # [batch_size, 1, hidden_size]
            key = global_session_2.unsqueeze(1)  # [batch_size, 1, hidden_size]
            # 拼接正样本和负样本
            # samples = torch.cat(
            #     [key, neg_global_vectors], dim=1
            # )  # [batch_size, 1 + num_neg*2, hidden_size]
            samples = torch.hstack(
                [key, trans_to_cuda(torch.FloatTensor(neg_global_vectors))]
            )
            # ? 计算余弦相似度。这里的dim究竟是1还是2？
            logits = F.cosine_similarity(
                query, samples, dim=1
            )  # [batch_size, 1 + num_neg*2]
            # 应用温度
            logits = logits / model.temperature  # [batch_size, 1 + num_neg*2]
            # ? 标签：第一个样本为正样本，其余为负样本。如何理解这里，为什么是全零向量？
            labels = torch.zeros(logits.size(0), dtype=torch.long).to(
                logits.device
            )  # [batch_size]
            # 计算对比损失
            loss_contrast = F.cross_entropy(logits, labels)
            # 将对比损失加权并加入总损失
            loss += contrast_loss_ratio * loss_contrast
        # 反向传播和优化
        loss.backward()
        model.optimizer.step()
        # 累加损失
        total_loss += loss.item()
        if contrast_loss_ratio > 0:
            total_contrast_loss += loss_contrast.item()
        # 打印训练信息
        if j % max(1, int(len(slices) / 5)) == 0:
            print(
                f"[{j}/{len(slices)}] Classification Loss: {loss.item() - (contrast_loss_ratio * loss_contrast.item()):.4f}, Contrast Loss: {loss_contrast.item():.4f}"
            )
            logger.info(
                f"[{j}/{len(slices)}] Classification Loss: {loss.item() - (contrast_loss_ratio * loss_contrast.item()):.4f}, Contrast Loss: {loss_contrast.item():.4f}"
            )
    # 打印总损失
    print(
        f"\tTotal Classification Loss:\t{total_loss:.3f}, Total Contrast Loss:\t{total_contrast_loss:.3f}, Contrast Loss Weight:\t{contrast_loss_ratio:.3f}"
    )
    logger.info(
        f"\tTotal Classification Loss:\t{total_loss:.3f}, Total Contrast Loss:\t{total_contrast_loss:.3f}, Contrast Loss Weight:\t{contrast_loss_ratio:.3f}"
    )
    # 开始测试
    print("start predicting: ", datetime.datetime.now())
    logger.info(f"start predicting: {datetime.datetime.now()}")
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores, _ = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
