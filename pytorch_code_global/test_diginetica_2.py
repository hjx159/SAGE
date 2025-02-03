import pandas as pd
import numpy as np
import pickle


test_data = pickle.load(open('../datasets/diginetica/test.txt', 'rb'))

res = []
for i in zip(*test_data):
    sess, gt = i
    if sess[-1] == 33889:
        res.append(gt)

from collections import Counter
res_cnt = Counter(res)
print("res_cnt:\n",res_cnt)
# 计算res_cnt的合计频次总数
total_cnt = sum(res_cnt.values())
print("total_cnt:",total_cnt)
np.save('gt_cnt', res_cnt)

output_list = np.load('res.diginetica.npy', allow_pickle=True)
print('len(output_list):',len(output_list))

res_sage = []
for i in output_list:
    if i[0] == 33889:
        res_sage+=i[-1][:1]
res_sage = list(map(lambda x: x+1, res_sage))

res_sage_cnt = Counter(res_sage)
print("res_sage_cnt:\n", res_sage_cnt)
total_sage_cnt = sum(res_sage_cnt.values())
print("total_sage_cnt:", total_sage_cnt)
np.save('sage_cnt', res_sage_cnt)