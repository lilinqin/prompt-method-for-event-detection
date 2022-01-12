import torch
import torch.nn as nn
import numpy as np
from transformers import  AdamW, BertTokenizer ,get_linear_schedule_with_warmup
from tqdm import tqdm
import sklearn.metrics as metrics
from util import set_seed, load_data, get_parse, get_data
from dataset import PromptDataset, BERTDataset, BERTDatasetPost
from models import EAE_MODEL_Prompt, EAE_MODEL, EAE_MODEL_MLM, EAE_MODEL_Prompt_MLM

args = get_parse()

args.max_len = 128  
args.batch_size = 16
args.num_labels = 33 # 标签个数
args.hidden_size = 1024 # 隐藏层大小

# args.model_name = 'hfl/chinese-bert-wwm'
# args.model_name = 'hfl/chinese-bert-wwm-ext'
# args.model_name = 'hfl/chinese-roberta-wwm-ext'
args.model_name = 'hfl/chinese-roberta-wwm-ext-large'
args.tokenizer = BertTokenizer.from_pretrained(args.model_name, do_lower_case=True)
args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

args.train_path = 'data/train.csv'
args.dev_path = 'data/dev.csv'
args.test_path = 'data/test.csv'
args.label_path = 'data/labels.csv'
args.save_name = 'model_save'


# 测试
def evaluate(model, dataloader, end=False):
    model.eval()
    pred = [] # 预测
    gold = [] # 标签
    # for batch in tqdm(dataloader):
    for batch in dataloader:
        input_ids = batch['ids'].to(args.device)
        input_mask = batch['mask'].to(args.device) # mask位置
        input_type_ids = batch['type_ids'].to(args.device)
        labels = batch['label'].to(args.device)
        if 'mask_pos' in batch: # 每个句子有单独的mask位置
            mask_pos = batch['mask_pos'].to(args.device)
            out = model(input_ids, input_type_ids,input_mask,mask_pos)
        else:
            out = model(input_ids, input_type_ids,input_mask)
        out = torch.sigmoid(out)
        out = np.where(out.cpu().detach().numpy()>0.5,1,0)
        target = labels

        pred.extend(out.tolist())
        gold.extend(target.cpu().detach().numpy().tolist())
        # print(target)
        # print(out)
        
    pred = np.array(pred, dtype='int')
    gold = np.array(gold, dtype='int')
    res = metrics.precision_recall_fscore_support(gold,pred,average='micro',labels=list(range(args.num_labels))) # precision_recall_fscore
    res_all = metrics.classification_report(gold,pred,digits=4,labels=list(range(args.num_labels)),zero_division=0)
    # print(res_all)
    # if end:
    #     with open('r.txt','w') as f:
    #         for i in range(len(pred)):
    #             if not all(pred[i] == gold[i]):
    #                 f.write(str(gold[i]) + '\n')
    #                 f.write(str(pred[i]) + '\n\n')
            

    return res[0],res[1],res[2],res_all

# 随机数
set_seed(args.seed) 

# 加载数据
if args.post: # 模板放在输入后面
    train_df = load_data(args.train_path, args.num_labels, None)
    dev_df = load_data(args.dev_path, args.num_labels, None)
    test_df = load_data(args.test_path, args.num_labels, None)
else:
    train_df = load_data(args.train_path, args.num_labels, args.manual_template)
    dev_df = load_data(args.dev_path, args.num_labels, args.manual_template)
    test_df = load_data(args.test_path, args.num_labels, args.manual_template)

# print(args.model, args.mask_id)

if args.post: # 模板在输入后面
    if args.model == 'manual_prompt': # 人工提示 + MLMhead
        train_dataloader, dev_dataloader, test_dataloader = get_data(BERTDatasetPost, args, train_df, dev_df, test_df)
        model = EAE_MODEL_MLM(args).to(args.device)
    elif args.model == 'bert_dense': # bert_dense
        train_dataloader, dev_dataloader, test_dataloader = get_data(BERTDatasetPost, args, train_df, dev_df, test_df)
        model = EAE_MODEL(args).to(args.device)
elif args.model == 'soft_prompt_mlm': # 可学习提示 + MLMhead
    train_dataloader, dev_dataloader, test_dataloader = get_data(PromptDataset, args, train_df, dev_df, test_df)
    model = EAE_MODEL_Prompt_MLM(args).to(args.device)
elif args.model == 'soft_prompt': # 可学习提示 + CLS
    train_dataloader, dev_dataloader, test_dataloader = get_data(PromptDataset, args, train_df, dev_df, test_df)
    model = EAE_MODEL_Prompt(args).to(args.device)
elif args.model == 'manual_prompt': # 人工提示 + MLMhead
    train_dataloader, dev_dataloader, test_dataloader = get_data(BERTDataset, args, train_df, dev_df, test_df)
    model = EAE_MODEL_MLM(args).to(args.device)
elif args.model == 'bert_dense': # bert_dense
    train_dataloader, dev_dataloader, test_dataloader = get_data(BERTDataset, args, train_df, dev_df, test_df)
    model = EAE_MODEL(args).to(args.device)


no_decay = ['bias', "LayerNorm.bias", 'LayerNorm.weight']

if args.fine_tuning: # 微调
    parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(parameters, lr=args.lr)
else: # frozen
    for name, param in model.bert_model.named_parameters():
        if "embeddings.word_embeddings.learned_embedding" in name or "cls.predictions.decoder" in name:
            # print(name)
            continue
        param.requires_grad = False
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)


# 学习率衰减策略
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(0.1*len(train_dataloader)*args.epochs), 
    num_training_steps=len(train_dataloader)*args.epochs
)

criterion = nn.BCEWithLogitsLoss()

best_dev_f1 = 0
best_test_f1 = 0
best_epoch = 0
best_p, best_r = 0, 0
best_res = ''

# print(args.lr)
# 训练
flag = False # 判断是否是最后一次迭代
for epoch in range(args.epochs):
    print('epoch:{}'.format(epoch+1))
    model.train()

    if epoch == args.epochs - 1:
        flag = True
    
    pred = []
    gold = []
    # for batch in tqdm(train_dataloader):
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['ids'].to(args.device)
        input_mask = batch['mask'].to(args.device)
        input_type_ids = batch['type_ids'].to(args.device)
        labels = batch['label'].to(args.device)
        if 'mask_pos' in batch:
            mask_pos = batch['mask_pos'].to(args.device)
            out = model(input_ids, input_type_ids,input_mask,mask_pos)
        else:
            out = model(input_ids, input_type_ids,input_mask)
        #.view(-1)
        target = labels
        
        loss = criterion(out,target)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        out = torch.sigmoid(out)
        out = np.where(out.cpu().detach().numpy()>0.5,1,0)
        pred.extend(out.tolist())
        gold.extend(target.cpu().detach().numpy().tolist())
    
    pred = np.array(pred, dtype='int')
    gold = np.array(gold, dtype='int')
    res = metrics.precision_recall_fscore_support(gold,pred,average='micro',labels=list(range(args.num_labels)))
    p_dev,r_dev,f_dev,_ = evaluate(model, dev_dataloader)
    
    p_test,r_test,f_test,res_all = evaluate(model, test_dataloader,flag)
    print('train: p:{}, r:{}, f:{}'.format(res[0], res[1], res[2]))
    print('dev: p:{}, r:{}, f:{}'.format(p_dev, r_dev, f_dev))
    print('test: p:{}, r:{}, f:{}'.format(p_test, r_test, f_test))

    # 取第六次迭代之后效果最好的模型
    if epoch >= 5:
        if f_dev>best_dev_f1:
            best_dev_f1 = f_dev
            best_test_f1 = f_test
            best_epoch = epoch + 1
            best_p = p_test
            best_r = r_test
            best_res = res_all
            print('save best')
            # torch.save(model.state_dict(), args.save_name)

print('best epoch: {}, dev f1:{}, test p:{}, r:{}, f1:{}, '.format(best_epoch, best_dev_f1, best_p, best_r, best_test_f1))
with open('result/result_1220.txt','a+') as f:
    s = ''
    s += args.model
    if 'soft_prompt' in args.model:
        if args.initialize_from_vocab:
            s += ', ' + 'initialize from vocab'
        else:
            s += ', ' + 'initialize randomly'
    if args.fine_tuning:
        s += ', ' + 'fine-tuning'
    else:
        s += ', ' + 'not fine-tuning'
    if args.n_tokens:
        s += ', n_tokens:' + str(args.n_tokens)
    
    
    s1 = f'{s}, seed:{args.seed}, lr:{args.lr}, mask_id:{args.mask_id}, post:{args.post}, split:{args.split1}\n'
    s2 = 'best epoch: {}, dev f1:{:.4f}, test p:{:.4f}, r:{:.4f}, f1:{:.4f} \n'.format(best_epoch, best_dev_f1, best_p, best_r, best_test_f1)
    f.write(s1)
    f.write(s2)
    f.write(best_res)
    f.write('\n')

with open('result/prf1_1220.csv', 'a+') as f:
    if args.seed == 13:
        f.write(s1)
    f.write('{:.4f},{:.4f},{:.4f} \n'.format(best_p, best_r, best_test_f1)) 
    # f.write(best_res)
    # f.write('\n')

