import torch.nn as nn
from transformers import BertModel, BertForMaskedLM
import torch

# 普通BERT模型，未使用MLM_head，直接用BERT输出的hidden_state进行预测
class EAE_MODEL(nn.Module):
    def __init__(self, args):
        super(EAE_MODEL, self).__init__()
        self.mask_id = args.mask_id # 被mask的token位置
        self.bert_model = BertModel.from_pretrained(args.model_name)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(args.hidden_size, args.num_labels)

    def forward(self, input_ids, token_type_ids, input_mask, mask_pos=None):
        last_hidden_state = self.bert_model(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask).last_hidden_state
        if mask_pos != None: # 若每个句子有不同的mask位置，使用每个句子对应的位置
            pooled = self.drop(last_hidden_state[list(range(input_ids.shape[0])),mask_pos])
        else: # 否则实验预先定义的mask位置
            pooled = self.drop(last_hidden_state[:,self.mask_id])
        out = self.fc(pooled)

        return out

# 连续提示嵌入
class PROMPTEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 20, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True,
                split1 = 0):
        super(PROMPTEmbedding, self).__init__()
        self.wte = wte # 词嵌入矩阵
        self.n_tokens = n_tokens # 连续prompt向量的个数
        self.split1 = split1 # 连续向量的分割处，一个放在[MASK]前面，一个放在[MASK]后面
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                               n_tokens, 
                                                                               random_range, 
                                                                               initialize_from_vocab))

    # 初始化prompt向量    
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 20, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True,):
        if initialize_from_vocab: # 从词典更新
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range) # 随机更新（默认）
            
    def forward(self, tokens):
        input_embedding_1 = self.wte(tokens[:, 0:1]) # [CLS]
        input_embedding_2 = self.wte(tokens[:, self.n_tokens+1:self.n_tokens+2]) # [MASK]
        
        input_embedding_3 = self.wte(tokens[:, self.n_tokens+2:]) # x
        
        # 分割prompt向量
        learned_embedding = self.learned_embedding.repeat(input_embedding_1.size(0), 1, 1)
        learned_embedding1 = learned_embedding[:,0:self.split1]
        learned_embedding2 = learned_embedding[:,self.split1:]

        # 拼接向量，[CLS] p1 [MASK] p2 x
        return torch.cat([input_embedding_1, learned_embedding1, input_embedding_2, learned_embedding2, input_embedding_3], 1) 

# 连续prompt向量模型，未使用MLM_head
class EAE_MODEL_Prompt(nn.Module):
    def __init__(self, args):
        super(EAE_MODEL_Prompt, self).__init__()
        self.mask_id = args.mask_id # mask位置
        self.bert_model = BertModel.from_pretrained(args.model_name)
        # 可学习连续prompt向量
        self.prompt_emb = PROMPTEmbedding(self.bert_model.get_input_embeddings(), 
                      n_tokens=args.n_tokens, 
                      initialize_from_vocab=args.initialize_from_vocab, split1=args.split1)
        
        self.bert_model.set_input_embeddings(self.prompt_emb) # 设置BERT的词嵌入矩阵
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(args.hidden_size, args.num_labels)
    
    def forward(self, input_ids, token_type_ids, input_mask):
        last_hidden_state = self.bert_model(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask).last_hidden_state
        pooled = self.drop(last_hidden_state[:,self.mask_id])
        out = self.fc(pooled)

        return out

# 添加了MLM_head的BERT模型
class EAE_MODEL_MLM(nn.Module):
    def __init__(self, args):
        super(EAE_MODEL_MLM, self).__init__()
        self.mask_id = args.mask_id # mask位置
        self.bert_model = BertForMaskedLM.from_pretrained(args.model_name)
        self.bert_model.cls.predictions.decoder = nn.Linear(args.hidden_size, args.num_labels) # 修改mlm的decoder

    def forward(self, input_ids, token_type_ids, input_mask, mask_pos=None):
        if mask_pos != None: # 若每个句子有不同的mask位置，使用每个句子对应的位置
            out = self.bert_model(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask).logits[list(range(input_ids.shape[0])),mask_pos,:].contiguous()
        else: # 否则使用固定的mask位置
            out = self.bert_model(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask).logits[:,self.mask_id,:].contiguous()

        return out

# 添加了MLM_head的连续prompt方法
class EAE_MODEL_Prompt_MLM(nn.Module):
    def __init__(self, args):
        super(EAE_MODEL_Prompt_MLM, self).__init__()
        self.mask_id = args.mask_id # mask位置
        self.bert_model = BertForMaskedLM.from_pretrained(args.model_name)
        # 连续prompt向量
        self.prompt_emb = PROMPTEmbedding(self.bert_model.get_input_embeddings(), 
                      n_tokens=args.n_tokens, 
                      initialize_from_vocab=args.initialize_from_vocab, split1=args.split1)
        
        self.bert_model.set_input_embeddings(self.prompt_emb) # 设置BERT的嵌入层
        self.bert_model.cls.predictions.decoder = nn.Linear(args.hidden_size, args.num_labels) # 修改mlm的decoder
        
    
    def forward(self, input_ids, token_type_ids, input_mask):
        out = self.bert_model(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask).logits[:,self.mask_id,:].contiguous()

        return out