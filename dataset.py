from torch.utils.data import Dataset
import torch
import numpy as np

# 普通BERT的dataset
class BERTDataset(Dataset):
    def __init__(self, df, args):
        self.df = df
        self.max_len = args.max_len
        self.tokenizer = args.tokenizer
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.df.loc[index, 'text']
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )

        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'] ,dtype=torch.long),
            'type_ids': torch.tensor(inputs['token_type_ids'] ,dtype=torch.long),
            'label': torch.tensor(self.df.loc[index, 'label'], dtype=torch.float)
        }

# 模板在后面的dataset
class BERTDatasetPost(Dataset):
    def __init__(self, df, args):
        self.df = df
        self.max_len = args.max_len # 句子最大长度
        self.tokenizer = args.tokenizer
        self.template = args.manual_template # 模板
        self.template_enc = self.tokenizer.encode(self.template, add_special_tokens=False)
        self.attention_mask = [1 for _ in range(len(self.template_enc))]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.df.loc[index, 'text']
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )

        # 获取模板的填充起止位置
        if 0 in inputs['attention_mask']:
            start = inputs['attention_mask'].index(0)
            end = start + len(self.template_enc)
            # 若长度超出最大长度，只取输入x的前部分
            if end > self.max_len:
                start = self.max_len - len(self.template_enc)
                end = self.max_len
        else:
            start = self.max_len - len(self.template_enc)
            end = self.max_len
        # 将模板放到输入最后
        inputs['input_ids'] = inputs['input_ids'][:start] + self.template_enc + inputs['input_ids'][end:]
        inputs['attention_mask'] = inputs['attention_mask'][:start] + self.attention_mask + inputs['attention_mask'][end:]
        mask_pos = start + 2
 
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'] ,dtype=torch.long),
            'type_ids': torch.tensor(inputs['token_type_ids'] ,dtype=torch.long),
            'label': torch.tensor(self.df.loc[index, 'label'], dtype=torch.float),
            'mask_pos': mask_pos
        }

# 连续prompt的dataset
class PromptDataset(Dataset):
    def __init__(self, df, args):
        self.df = df
        self.max_len = args.max_len
        self.tokenizer = args.tokenizer
        self.n_tokens=args.n_tokens
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.df.loc[index, 'text']
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )

        # [CLS] p_1 P_2 ... P_n x
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        inputs['input_ids'] = torch.cat((input_ids[0:1] ,torch.full((1,self.n_tokens), 500).reshape(self.n_tokens), input_ids[1:]))
        inputs['attention_mask'] = torch.cat((torch.full((1,self.n_tokens), 1).reshape(self.n_tokens), torch.tensor(inputs['attention_mask'], dtype=torch.long)))
        inputs['token_type_ids'] = torch.cat((torch.full((1,self.n_tokens), 0).reshape(self.n_tokens), torch.tensor(inputs['token_type_ids'], dtype=torch.long)))


        return {
            'ids': inputs['input_ids'].long(),
            'mask': inputs['attention_mask'].long(),
            'type_ids': inputs['token_type_ids'].long(),
            'label': torch.tensor(self.df.loc[index, 'label'], dtype=torch.float)
        }