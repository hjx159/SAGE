#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import numpy as np

import tqdm
from collections import defaultdict
import random


def jaccard_similarity(session_a, session_b):
    set_a = set(session_a) - {0}
    set_b = set(session_b) - {0}
    if not set_a and not set_b:
        return 0.0
    return float(len(set_a.intersection(set_b))) / float(len(set_a.union(set_b)))


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [
        upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)
    ]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    us_pos = [list(reversed(range(1, le + 1))) + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, us_pos, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype="int32")
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1.0 - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data:
    def __init__(self, data, shuffle=False, num_neg=2, similarity_threshold=0.2):
        inputs = data[0]
        # print("inputs type: ", type(inputs))
        # print("inputs[:5]",inputs[:5])
        inputs, mask, pos, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.pos = np.asarray(pos)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.batch_size = 100
        self.num_neg = num_neg
        self.similarity_threshold = similarity_threshold
        self.item_session_dict = defaultdict(list)
        self.last_items = []
        self._initialize_item_session_dict()
        self._generate_neg_x()

    def _initialize_item_session_dict(self):
        """
        记录每个会话的最后一个项目，并建立项目到会话索引的映射
        """
        for i in tqdm.trange(
            len(self.inputs), desc="Initializing item-session dictionary"
        ):
            last_item_index = np.sum(self.mask[i]) - 1
            last_item = self.inputs[i][last_item_index]
            self.last_items.append(last_item)
            self.item_session_dict[last_item].append(i)

    def _generate_neg_x(self):
        """
        基于会话间Jaccard相似度生成负样本。选择与当前会话相似度低且最后项目相同的会话作为负样本。
        TODO: 另外，在计算相似度时，是不是应该排除最后一个项目？
        其实还可以考虑直接选取除了最后一个项目相同外，其余项目都不同的其他会话作为强负样本。
        """
        self.neg_inputs = []

        for i in tqdm.trange(len(self.inputs), desc="Generating negative samples"):
            last_item = self.last_items[i]
            # 获取具有相同最后项目的会话候选（排除自身）
            neg_candidates = list(set(self.item_session_dict[last_item]) - {i})

            # 获取当前会话的实际项目，排除最后一个项目
            current_session_full = self.inputs[i][self.mask[i] == 1].tolist()
            if len(current_session_full) > 1:
                current_session = current_session_full[:-1]
            else:
                current_session = []

            selected_neg = []

            # Step 1: 优先采样相似度为0的会话作为强负样本
            for cand in neg_candidates:
                if len(selected_neg) >= self.num_neg:
                    break  # 达到负样本数量，提前停止
                candidate_session_full = self.inputs[cand][
                    self.mask[cand] == 1
                ].tolist()
                if len(candidate_session_full) > 1:
                    candidate_session = candidate_session_full[:-1]
                else:
                    candidate_session = []
                similarity = jaccard_similarity(current_session, candidate_session)
                if similarity == 0.0:
                    selected_neg.append(cand)

            # Step 2: 如果强负样本不足，采样相似度小于阈值的会话
            if len(selected_neg) < self.num_neg:
                remaining = self.num_neg - len(selected_neg)
                for cand in neg_candidates:
                    if len(selected_neg) >= self.num_neg:
                        break  # 达到负样本数量，提前停止
                    if cand in selected_neg:
                        continue  # 已选过，跳过
                    candidate_session_full = self.inputs[cand][
                        self.mask[cand] == 1
                    ].tolist()
                    if len(candidate_session_full) > 1:
                        candidate_session = candidate_session_full[:-1]
                    else:
                        candidate_session = []
                    similarity = jaccard_similarity(current_session, candidate_session)
                    if similarity < self.similarity_threshold:
                        selected_neg.append(cand)
                        remaining -= 1
                        if remaining == 0:
                            break  # 达到负样本数量，提前停止

            # Step 3: 如果仍不足，优先采样相似度小于阈值的其他会话（耗时很长）
            if len(selected_neg) < self.num_neg:
                remaining = self.num_neg - len(selected_neg)
                low_sim_other = []
                for cand in range(self.length):
                    if cand == i or cand in neg_candidates or cand in selected_neg:
                        continue  # 排除自身、已有的负样本和同last_item会话
                    candidate_session_full = self.inputs[cand][
                        self.mask[cand] == 1
                    ].tolist()
                    if len(candidate_session_full) > 1:
                        candidate_session = candidate_session_full[:-1]
                    else:
                        candidate_session = []
                    similarity = jaccard_similarity(current_session, candidate_session)
                    if similarity < self.similarity_threshold:
                        low_sim_other.append(cand)
                        if len(low_sim_other) >= remaining:
                            break  # 达到负样本数量，提前停止

                if low_sim_other:
                    sampled_low_sim_other = random.sample(
                        low_sim_other, min(len(low_sim_other), remaining)
                    )
                    selected_neg += sampled_low_sim_other
                    remaining -= len(sampled_low_sim_other)

            # Step 4: 如果仍不足，随机选择其他会话作为补充
            if len(selected_neg) < self.num_neg:
                remaining = self.num_neg - len(selected_neg)
                other_candidates = list(
                    set(range(self.length))
                    - set(self.item_session_dict[last_item])
                    - {i}
                )
                if len(other_candidates) >= remaining:
                    selected_neg += random.sample(other_candidates, remaining)
                else:
                    selected_neg += list(
                        np.random.choice(other_candidates, remaining, replace=True)
                    )

            self.neg_inputs.append(selected_neg)

        self.neg_inputs = np.asarray(self.neg_inputs)

    def _generate_neg_x_random(self):
        self.last_items = []
        self.item_session_dict = defaultdict(list)
        for i in tqdm.trange(len(self.inputs)):
            self.last_items.append(self.inputs[i][np.sum(self.mask[i]) - 1])
            self.item_session_dict[self.inputs[i][np.sum(self.mask[i]) - 1]].append(i)
        self.neg_inputs = []
        for i in tqdm.trange(len(self.inputs)):
            last_item = self.last_items[i]
            strong_negs = list(set(self.item_session_dict[last_item]) - set([i]))[
                : self.num_neg
            ]
            neg_candidates = list(set(self.item_session_dict[last_item]) - set([i]))
            strong_negs = []
            for cand in neg_candidates:
                if self.last_items[cand] != last_item:
                    strong_negs.append(cand)
            strong_negs = strong_negs[: self.num_neg]
            for _ in range(self.num_neg - len(strong_negs)):
                strong_negs.append(random.randint(0, self.length - 1))
            self.neg_inputs.append(strong_negs)
        self.neg_inputs = np.asarray(self.neg_inputs)

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][: (self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, pos, targets = (
            self.inputs[i],
            self.mask[i],
            self.pos[i],
            self.targets[i],
        )
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items, mask, pos, targets
