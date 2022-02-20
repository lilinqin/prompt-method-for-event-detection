import torch
from torch import nn
from transformers import BertModel,BertTokenizer, AdamW,get_linear_schedule_with_warmup
import torch.functional as F
import os
from torch.utils.data import DataLoader
# from torchcrf import CRF
# set GPU memory
import json
import numpy as np
from tqdm import tqdm
import sklearn.metrics as metrics
# from bert4keras.tokenizers import Tokenizer
from argparse import ArgumentParser
# torch.manual_seed(2)
import logging
from time import strftime,localtime


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


def set_seed():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

class Example:
    def __init__(self,sent,ent,label):
        self.sent = sent
        self.ent = ent
        self.label = label

def load_data_example(path):

    examples = []
    with open(path) as file:
        for line in file.readlines():
            # print(line)
            line = line.strip().split('\t')
            tokens = line[0].split()
            labels = [i for i in line[1].split() if i!='O']
            ents = [['negative'] * len(word) for word in tokens]
            idx = 0
            # all_len = len(line)
            if len(line) > 2 :
                ent_ids = line[2].split()
                # print(ent_ids)
                ids = [int(x.split('#')[-1]) for x in ent_ids]
                for id in ids:
                    ents[id] = ['ent'] * len(ents[id])
            # while idx + 2 <= all_len:

            # tokens = [w.split()[0] for w in instance[:-1]]
            # ents = [w.split()[1] for w in instance[:-1]]
            # labels = instance[-1].split()
            tokens = ''.join(tokens)
            tokens = [x for x in tokens]
            ents_temp = []
            for ent in ents:
                ents_temp.extend(ent)
            # print(len(line))
            # print(tokens)
            # print(ents_temp)
            # print(labels)
            # exit()



            examples.append(Example(tokens,ents_temp,labels))

    return examples



def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logfile = os.path.join(res_path,'res.log')

    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    return logger

class EAE_MODEL(nn.Module):

    def __init__(self, word_dim, ent_size, ent_dim,event_size, alpha):
        super(EAE_MODEL, self).__init__()
        self.alpha = alpha
        self.bert_model = BertModel.from_pretrained(args.bert_path)
        self.ent_emb = nn.Embedding(ent_size,ent_dim)
        self.event_emb = nn.Embedding(event_size,word_dim+ent_dim)
        self.event_emb_last = nn.Embedding(event_size,word_dim+ent_dim)
        self.drop = nn.Dropout(0.1)
        self.dense = nn.Linear(word_dim+ent_dim,word_dim+ent_dim)
        self.norm = nn.LayerNorm(word_dim+ent_dim)


    def forward_raw(self, x1, x2, ent_ids, event_type, att_mask):
        # print(x1, x2, ent_ids, event_type)
        word_hidden = self.bert_model(x1,token_type_ids=x2,attention_mask=att_mask).last_hidden_state
        ent_hidden = self.ent_emb(ent_ids)
        event_hidden = self.event_emb(event_type)
        event_hidden_last = self.event_emb_last(event_type)
        word_hidden = torch.cat([word_hidden,ent_hidden],2)


        #attention
        attention1_logits = torch.matmul(word_hidden,event_hidden.permute(0,2,1))
        attention1_logits = attention1_logits.view(-1,att_mask.shape[1])*att_mask
        attention1 = torch.exp(attention1_logits)*att_mask

        #attention_score
        _score1 = attention1_logits*attention1/attention1.sum(1,keepdim=True)
        score1 = torch.sum(_score1,1,keepdim=True)

        #global score
        event_hidden_last = event_hidden_last.view(-1,word_hidden.shape[2])
        score2 = torch.sum(word_hidden[:,0]*event_hidden_last,axis=1,keepdim=True)

        score = score1*self.alpha+score2*(1-self.alpha)

        return score

    def forward_first(self, x1, x2, ent_ids, event_type, att_mask):
        # print(x1, x2, ent_ids, event_type)
        word_hidden = self.bert_model(x1,token_type_ids=x2,attention_mask=att_mask).last_hidden_state

        ent_hidden = self.ent_emb(ent_ids)
        event_hidden = self.event_emb(event_type)
        event_hidden_last = self.event_emb_last(event_type)
        # print(word_hidden,ent_hidden)
        word_hidden = torch.cat([word_hidden,ent_hidden],2)


        #attention
        attention1_logits = torch.matmul(word_hidden,event_hidden.permute(0,2,1))
        attention1_logits = attention1_logits.view(-1,att_mask.shape[1])*att_mask
        attention1 = torch.exp(attention1_logits)*att_mask

        #attention_score
        _score1 = attention1_logits*attention1/attention1.sum(1,keepdim=True)
        score1 = torch.sum(_score1,1,keepdim=True)

        return score1




    def forward_last(self, x1, x2, ent_ids, event_type, att_mask):

        word_hidden = self.bert_model(x1,token_type_ids=x2,attention_mask=att_mask).last_hidden_state

        ent_hidden = self.ent_emb(ent_ids)
        event_hidden = self.event_emb(event_type)
        event_hidden_last = self.event_emb_last(event_type)
        word_hidden = torch.cat([word_hidden,ent_hidden],2)

        #global score
        event_hidden_last = event_hidden_last.view(-1,word_hidden.shape[2])
        score2 = torch.sum(word_hidden[:,0]*event_hidden_last,axis=1,keepdim=True)

        return score2




def sequence_padding(inputs, length=None, padding=0):
    if length is None:
        length = max([len(x) for x in inputs])

    outputs = np.array([
        np.concatenate([x, [padding] *
                        (length - len(x))]) if len(x) < length else x[:length]
        for x in inputs
    ])
    return outputs


class Dataset:
    def __init__(self, data, mode=None):
        self.mode = mode
        self.process(data)
    def __len__(self):
        return len(self.x1)

    def __getitem__(self, item):
        return self.x1[item], self.x2[item], self.ent_ids[item], self.event_type[item], self.att_mask[item], self.label[item]

    def process(self, data):
        self.x1 = []
        self.x2 = []
        self.ent_ids = []
        self.event_type = []
        self.att_mask = []
        self.label = []

        for i in range(len(data)):
            tokens = data[i].sent[:args.maxlen]
            ent = data[i].ent[:args.maxlen]
            label_ = data[i].label

            x1 = tokenizer.convert_tokens_to_ids(tokens)

            x1 = [101] + x1 + [102]
            x2 = [0]*len(x1)
            ent_ids = [0]+[ent2id[e] if e in ent2id else 0 for e in ent]+[0]

            assert len(x1)==len(x2) and len(x1)==len(ent_ids)

            for ly in label2id.keys():
                lb = 1 if ly in label_ else 0
                if self.mode!='test':
                    if lb == 0 and np.random.random() > args.neg_prob: continue
                self.x1.append(x1)
                self.x2.append(x2)
                self.ent_ids.append(ent_ids)
                self.event_type.append(label2id[ly])
                self.att_mask.append([1]*len(x1))
                self.label.append(lb)



def my_collate(batch):
    transposed = list(zip(*batch))
    x1 = sequence_padding(transposed[0])
    x2 = sequence_padding(transposed[1])
    ent_ids = sequence_padding(transposed[2])
    event_type = np.array(transposed[3])[:,np.newaxis]
    att_mask = sequence_padding(transposed[4])
    label = np.array(transposed[5])[:,np.newaxis]

    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
    ent_ids = torch.from_numpy(ent_ids)
    event_type = torch.from_numpy(event_type)
    att_mask = torch.from_numpy(att_mask)
    label = torch.FloatTensor(label)

    return [x1, x2, ent_ids, event_type, att_mask, label]


def evaluate1(model,data_loader):

    model.eval()
    pred = []
    gold = []
    for batch in tqdm(data_loader):
        batch = [i.to(args.device) for i in batch]
        out = model(batch[0], batch[1],batch[2],batch[3])
        out = torch.sigmoid(out)
        target = batch[4]
        out = np.where(out.cpu().detach().numpy()>0.5,1,0)
        pred.extend(out.tolist())
        gold.extend(target.cpu().detach().numpy().tolist())
    pred = np.array(pred)
    gold = np.array(gold)
    logger.info(metrics.classification_report(gold,pred,target_names=list(['negative','event']),digits=4))
    res = metrics.precision_recall_fscore_support(gold,pred,average='micro',labels=list(range(1,2)))


    return res[0],res[1],res[2]


def evaluate(model,data):

    model.eval()
    pred = []
    gold = []
    for i in range(len(data)):
        x1 = []
        x2 = []
        entities = []
        event_type = []
        att_mask = []
        label = []


        tokens = data[i].sent
        ent = data[i].ent
        label_ = data[i].label

        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        token_ids = [101] + token_ids + [102]
        segment_ids = [0] * len(token_ids)
        ent_ids = [0] + [ent2id[e] if e in ent2id else 0 for e in ent] + [0]


        for ly in label2id.keys():
            lb = 1 if ly in label_ else 0
            x1.append(token_ids)
            x2.append(segment_ids)
            entities.append(ent_ids)
            event_type.append(label2id[ly])
            att_mask.append([1]*len(token_ids))
            label.append(lb)

        x1 = sequence_padding(x1)
        x2 = sequence_padding(x2)
        ent_ids = sequence_padding(entities)
        event_type = np.array(event_type)[:, np.newaxis]
        att_mask = sequence_padding(att_mask)
        # label = label

        x1 = torch.from_numpy(x1).cuda()
        x2 = torch.from_numpy(x2).cuda()
        ent_ids = torch.from_numpy(ent_ids).cuda()
        event_type = torch.from_numpy(event_type).cuda()
        att_mask = torch.from_numpy(att_mask).cuda()


        if args.forward == 'forward_raw':
            out = model.forward_raw(x1,x2,ent_ids,event_type,att_mask)

        elif args.forward == 'forward_first':
            out = model.forward_first(x1,x2,ent_ids,event_type,att_mask)

        elif args.forward == 'forward_last':
            out = model.forward_last(x1,x2,ent_ids,event_type,att_mask)

        # out = model(x1,x2,ent_ids,event_type,att_mask)

        out = torch.sigmoid(out)
        out = np.where(out.cpu().detach().numpy()>0.5,1,0)

        out = out.squeeze()

        pred.append(out.tolist())
        gold.append(label)
    pred = np.array(pred)
    gold = np.array(gold)
    logger.info(metrics.classification_report(gold,pred,target_names=list(label2id.keys()),digits=4))
    res = metrics.precision_recall_fscore_support(gold,pred,average='micro',labels=list(range(1,34)))


    return res[0],res[1],res[2]

def get_res_path():

    log_dir_path = 'log/'+'_'.join([
                    args.config,
                    str(args.forward),
                    'maxlen'+str(args.maxlen),
                    'batch'+str(args.batch_size),
                    'lr'+str(args.learning_rate),
                    'neg_prob'+str(args.neg_prob),
                    'epochs'+str(args.epochs),
                    'seed'+str(args.seed)])
    if not os.path.exists(log_dir_path):
        os.mkdir(log_dir_path)

    save_name = os.path.join(log_dir_path,'model.weights')
    return log_dir_path,save_name

def load_embedding(path):
    '''
    load word embedding or
    position embedsing or
    label embedding
    '''

    with open(path) as file:
        data = []
        for line in file.readlines():
            if not line.strip(): continue
            data.append(list(map(float, line.strip().split(' '))))
    return np.array(data, dtype='float32')

def train():
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.02},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    if args.sched:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * args.epochs
        )
        logger.info('use scheduler')
    else:
        logger.info('not use scheduler')


    best_dev_f1 = 0
    best_test_f1 = 0
    best_test_p = 0
    best_test_r = 0
    count = 0
    for epoch in range(args.epochs):
        logger.info('epoch:{}'.format(epoch+1))
        model.train()
        trian_loss = 0
        train_step = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            batch = [i.to(args.device) for i in batch]

            if args.forward == 'forward_raw':
                out = model.forward_raw(batch[0],batch[1],batch[2],batch[3],batch[4]).view(-1)


            elif args.forward == 'forward_first':
                out = model.forward_first(batch[0],batch[1],batch[2],batch[3],batch[4]).view(-1)


            elif args.forward == 'forward_last':
                out = model.forward_last(batch[0],batch[1],batch[2],batch[3],batch[4]).view(-1)


            target = batch[5].view(-1)

            loss = criterion(out,target)
            loss.backward()
            optimizer.step()

            if args.sched:
                scheduler.step()

            trian_loss += loss.item()
            train_step += 1
        train_loss = trian_loss / train_step
        logger.info('loss:{}'.format(train_loss))

        p_dev,r_dev,f_dev = evaluate(model, dev_data)
        p_test,r_test,f_test = evaluate(model, test_data)
        # p_dev,r_dev,f_dev = evaluate(model, dev_loader)
        # p_test,r_test,f_test = evaluate(model, test_loader)

        logger.info('dev: p:{}, r:{}, f:{}'.format(p_dev, r_dev, f_dev))
        logger.info('test: p:{}, r:{}, f:{}'.format(p_test, r_test, f_test))
        if f_dev>best_dev_f1:
            best_dev_f1 = f_dev
            best_test_f1 = f_test
            best_test_p = p_test
            best_test_r = r_test
            logger.info('save best')
            logger.info('test {}\t{}\t{}'.format(p_test, r_test, f_test))
            # torch.save(model.state_dict(),save_name)
        logger.info('best dev f1:{}, test f1:{}'.format(best_dev_f1, best_test_f1))
        if f_dev==0:
            count+=1
            if count==2:
                break
    with open('log/summary.log','a') as file:
        file.write(res_path+'\n')
        file.write('P:{} R:{} F:{}\n'.format(best_test_p,best_test_r,best_test_f1))

def init():
    parser = ArgumentParser()

    model_name = 'hfl/chinese-bert-wwm'
    parser.add_argument('--config', default='tbnnam_bert_230',help='用来记录每次实验的参数')
    parser.add_argument('--bert',default=model_name,help='bert的名字，需要自己手动下载bert')

    parser.add_argument('--mode', default='train', help='train or test')

    parser.add_argument('--train_path', default='data/train.txt')
    parser.add_argument('--dev_path', default='data/dev.txt')
    parser.add_argument('--test_path', default='data/test.txt')

    parser.add_argument("--tqdm", default=False)
    parser.add_argument('--maxlen', default=230,type=int)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--device',default='cuda',help='cuda or cpu')
    parser.add_argument('--epochs', default=20,type=int)
    parser.add_argument('--learning_rate', default=2e-5,type=float)
    parser.add_argument('--batch_size', default=16,type=int)
    parser.add_argument('--hidden_dim', default=768,type=int)
    parser.add_argument('--neg_prob', default=0.4,type=float)

    parser.add_argument('--bert_path', default='../{}')
    parser.add_argument('--dict_path', default='../{}/vocab.txt')
    parser.add_argument('--use',default='2')
    parser.add_argument('--seed', default=0,type=int)
    parser.add_argument('--sched',  action="store_true")

    parser.add_argument('--forward', default='foward')

    args = parser.parse_args()


    args.bert_path = args.bert_path.format(args.bert)
    args.dict_path = args.dict_path.format(args.bert)


    return args




if __name__ == '__main__':
    args = init()
    set_seed()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    res_path,save_name = get_res_path()
    logger = get_logger()

    logger.info('========================begin:{}========================='.format(strftime('%Y-%m-%d %H:%M:%S',localtime())))
    logger.info(args)

    # label2id = {'negative': 0, 'declare-bankruptcy': 1, 'end-org': 2, 'merge-org': 3, 'start-org': 4, 'attack': 5,
    #             'demonstrate': 6, 'meet': 7, 'phone-write': 8, 'acquit': 9, 'appeal': 10, 'arrest-jail': 11, 'charge-indict': 12,
    #             'convict': 13, 'execute': 14, 'extradite': 15, 'fine': 16, 'pardon': 17, 'release-parole': 18, 'sentence': 19, 'sue': 20,
    #             'trial-hearing': 21, 'be-born': 22, 'die': 23, 'divorce': 24, 'injure': 25, 'marry': 26, 'transport': 27, 'elect': 28,
    #             'end-position': 29, 'nominate': 30, 'start-position': 31, 'transfer-money': 32, 'transfer-ownership': 33}

    # id2label = {v: k for k, v in label2id.items()}

    label_path = 'data/ACE_event_types.txt'
    label2id, id2label, label2token = load_label(label_path) # 加载标签
    ent2id = {"negative": 0, "ent": 1}
    tokenizer = BertTokenizer.from_pretrained(args.bert_path, do_lower_case=True)


    num_labels = len(label2id)

    # word2id = json.load(open('word_dict_hm.txt'))
    # word_emb = load_embedding('word_embedding.txt')

    train_data = load_data_example(args.train_path)
    dev_data = load_data_example(args.dev_path)
    test_data = load_data_example(args.test_path)


    train_dataset = Dataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate)

    dev_dataset = Dataset(dev_data,'test')
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate)

    test_dataset = Dataset(test_data,'test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate)




    model = EAE_MODEL(768,2,50,len(label2id),0.25).to(args.device)

    train()
    logger.info('========================end:{}========================='.format(strftime('%Y-%m-%d %H:%M:%S',localtime())))



