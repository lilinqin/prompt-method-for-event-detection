import torch
import os
import numpy as np
import argparse
import pandas as pd
from torch.utils.data import DataLoader


# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 改变label，从字符序列变成01序列，1表示标签有该类别
def change_labels(label, num_labels):
    new_label = [0 for _ in range(num_labels)]
    label = eval(label)
    # print(label)
    for i in label:
        new_label[i] = 1
    return new_label

# 加载标签，获取标签到索引、索引到标签和标签到符号的字典
def load_data(path, num_label=18, manual_template=None):
    data_df = pd.read_csv(path, sep='\t')
    for i in range(len(data_df)):
        data_df.loc[i,'label'] = change_labels(data_df.loc[i,'label'], num_label)
    if manual_template:
        for i in range(len(data_df)):
            data_df.loc[i,'text'] = manual_template + data_df.loc[i,'text']
    # print(data_df['text'])
    return data_df


# 设置参数
def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0) # 随机数种子
    parser.add_argument("--n_tokens", type=int, default=0) # 连续prompt向量个数
    parser.add_argument("--mask_id", type=int, default=0) # mask位置
    parser.add_argument("--split1", type=int, default=0) # 连续prompt分割位置
    parser.add_argument("--manual_template", type=str, default='') # 是否使用手工模板
    parser.add_argument("--model", type=str, default='') # 模型类型
    parser.add_argument('--fine_tuning', action='store_true', default=False) # 微调
    parser.add_argument('--post', action='store_true', default=False) # 模板放在输入后面
    parser.add_argument('--initialize_from_vocab', action='store_true', default=False) # 从词典初始化
    parser.add_argument('--lr', type=float, default=2e-5) # 学习率
    parser.add_argument('--epochs', type=int, default=10) # 迭代次数

    args = parser.parse_args()
    return args

# 获取dataloader
def get_data(mydataset, args, train_df, dev_df, test_df):
    train_dataset = mydataset(train_df, args)
    dev_dataset = mydataset(dev_df, args)
    test_dataset = mydataset(test_df, args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader