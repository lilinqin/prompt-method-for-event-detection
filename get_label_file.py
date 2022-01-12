import csv

label_path = 'data/ACE_event_types.txt'
train_path = 'data/train.txt'
dev_path = 'data/dev.txt'
test_path = 'data/test.txt'

# 加载标签
def load_label(path):
    label2id = {} # 标签到下标  000 --> 0
    label2token = {} # 标签到token  000 --> declare-bankruptcy
    with open(path) as file:
        for line in file.readlines():
            line = line.strip().split()
            label2id[line[0]] = len(label2id)
            label2token[line[0]] = line[2]
    id2label = {v:k for k,v in label2id.items()} # 下标到标签  0 --> 000

    return label2id, id2label, label2token

# 加载数据
def load_data(path):
    sents_list = [] # 句子列表 [s1,...,sn]
    labels_list = [] # 标签下标列表 [[1,2],[],[1],...,[0]] []表示无类别 一个句子可能对应多个标签
    with open(path) as file:
        for line in file.readlines():
            line = line.strip().split('\t')
            tokens = line[0].split()
            sent = ''.join(tokens)
            labels = [i for i in line[1].split() if i!='O']
            if len(labels)==0:
                labels=[]
            sents_list.append(sent)
            labels_list.append(labels)

    return sents_list, labels_list

label2id, id2label, label2token = load_label(label_path) # 加载标签
# print(label2id, id2label, label2token, end='\n')

# 加载数据
train_data, train_label = load_data(train_path)
dev_data, dev_label = load_data(dev_path)
test_data, test_label = load_data(test_path)


labels = [[k, v] for k, v in label2token.items()] # (000, declare-bankruptcy),...
labels.sort()
labels1 = [[i, item[0], item[1]] for i, item in enumerate(labels)]  # (0, 000, declare-bankruptcy)

# 写入标签，格式是 index, label, token
label_file = 'data/labels.csv'
with open(label_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['index', 'label', 'token'])
    writer.writerows(labels1)

# 分别写入训练集、验证集、测试集文件，格式是 text, label
train_csv = 'data/train.csv'
with open(train_csv, 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['text', 'label'])
    for i in range(len(train_data)):
        text = train_data[i]
        label = [label2id[l] for l in train_label[i]]
        writer.writerow([text, label])

dev_csv = 'data/dev.csv'
with open(dev_csv, 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['text', 'label'])
    for i in range(len(dev_data)):
        text = dev_data[i]
        label = [label2id[l] for l in dev_label[i]]
        writer.writerow([text, label])

test_csv = 'data/test.csv'
with open(test_csv, 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['text', 'label'])
    for i in range(len(test_data)):
        text = test_data[i]
        label = [label2id[l] for l in test_label[i]]
        writer.writerow([text, label])
