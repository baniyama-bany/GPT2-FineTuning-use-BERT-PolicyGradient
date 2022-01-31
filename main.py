import copy
import random
import numpy as np

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.nn.functional as F

SEQ_LEN = 30
BATCH_SIZE = 2

from transformers import T5Tokenizer, AutoModelForCausalLM
gpt_tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
gpt = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium").to(device)
gpt_optimizer = torch.optim.Adam(gpt.parameters(),lr=1e-4)
BOS_IDX = gpt_tokenizer.bos_token_id
EOS_IDX = gpt_tokenizer.eos_token_id
PAD_IDX = gpt_tokenizer.pad_token_id
UNK_IDX = gpt_tokenizer.unk_token_id
VOCAB_SIZE = gpt_tokenizer.vocab_size

from transformers import AutoTokenizer, AutoModelForSequenceClassification
senti_tokenizer = AutoTokenizer.from_pretrained("daigo/bert-base-japanese-sentiment")
senti = AutoModelForSequenceClassification.from_pretrained("daigo/bert-base-japanese-sentiment").to(device)

class Rollout():
    def __init__(self,gpt):
        self.old_gpt = copy.deepcopy(gpt)
        self.pre_gpt = copy.deepcopy(gpt)
        self.now_gpt = gpt
        self.update_rate = 0.01

    #GPT-2の単語IDsから評価用BERTのtokenizer出力に変換 [B,L] -> tokenized output
    def index_translate_gpt2bert(self,input_ids):
        input_ids = self.sample_pad_process(input_ids)
        input_ids_decoded = gpt_tokenizer.batch_decode(input_ids,skip_special_token=True)
        roll_samples = senti_tokenizer(input_ids_decoded,return_tensors='pt',max_length=512, padding='max_length',truncation=True).to(device)
        return roll_samples

    #GPT-2の単語IDsからGPT-2のtokenizer出力に変換 [B,L] -> tokenized output
    def index_translate_gpt2gpt(self,input_ids):
        input_ids = self.sample_pad_process(input_ids)
        input_ids_decoded = gpt_tokenizer.batch_decode(input_ids,skip_special_token=True)
        roll_samples = gpt_tokenizer(input_ids_decoded,return_tensors='pt',max_length=512, padding='max_length',truncation=True).to(device)
        return roll_samples

    def mle_old_gpt_reward(self,samples):  #Input ids of GPT2 [B,L] -> [B]
        inputs = self.index_translate_gpt2gpt(samples)
        with torch.no_grad():
            logits = self.old_gpt(input_ids=inputs["input_ids"][:,:-1].to(device),attention_mask=inputs["attention_mask"][:,:-1].to(device)).logits

        mle_rewards = []
        for i in range(logits.size(0)):
            loss = F.cross_entropy(logits[i,:,:],inputs["input_ids"][i,1:].reshape(-1).to(device))
            mle_rewards.append(-loss.cpu())

        return torch.tensor(mle_rewards)

    def disc_model_reward(self,roll_samples,disc_model):  #Input ids of GPT2 [B,L] -> [B]
        roll_samples = self.index_translate_gpt2bert(roll_samples)
        with torch.no_grad():
            logits = disc_model(input_ids=roll_samples["input_ids"],token_type_ids=roll_samples["token_type_ids"],attention_mask=roll_samples["attention_mask"]).logits
        negative_preds = F.softmax(logits,dim=-1)[:,1]
        # negative_preds = (negative_preds>0.5).float()  #報酬の離散化
        return negative_preds.cpu()

    def update_params(self):
        dic = {}
        for name, param in self.now_gpt.named_parameters():
            dic[name] = param.data
        for name, param in self.pre_gpt.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
    
    #EOSトークンの後は滅茶苦茶なのでPADに置き換える
    #[B,L]->[B,L]
    def sample_pad_process(self,samples):
        batch_size = samples.size(0)
        eos_index = [100000000]*batch_size
        for i in range(samples.size(0)):
            for j in range(samples.size(1)):
                if samples[i,j].item()==EOS_IDX and j<eos_index[i]: eos_index[i]=j
        
        for i in range(samples.size(0)):
            for j in range(samples.size(1)):
                if j > eos_index[i]:samples[i,j]=PAD_IDX

        return samples
    
    def get_reward(self, x, num, discriminator):
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)

        for i in range(num):
            for l in range(2, seq_len+1):#bos token is ignore
                data = x[:, 0:l]

                if data.size(-1)<seq_len:
                    roll_samples = self.pre_gpt.generate(input_ids=data,max_length=seq_len,pad_token_id=PAD_IDX,eos_token_id=EOS_IDX)
                else:
                    roll_samples = data

                disc_model_preds = self.disc_model_reward(roll_samples,discriminator)  #判別モデルの出力
                mle_model_preds = self.mle_old_gpt_reward(roll_samples)  #従来のGPT2の単語確率に沿わす
                model_preds = disc_model_preds + mle_model_preds*0.01

                if i==0:rewards.append(model_preds.cpu().numpy())
                else:rewards[l-2] += model_preds.cpu().numpy()

        rewards = np.transpose(np.array(rewards)) / (1.0 * num)
        rewards = torch.tensor(rewards).to(device)
        return rewards

class PGLoss(nn.Module):
    def __init__(self):
        super(PGLoss, self).__init__()

    def forward(self, preds, targets, rewards):
        targets_onehot = F.one_hot(targets,num_classes=VOCAB_SIZE)
        loss = -(torch.log(preds)*targets_onehot).sum(-1)*rewards
        return loss.mean()

rollout = Rollout(gpt)
pgloss = PGLoss()
nllloss = nn.NLLLoss(ignore_index=PAD_IDX)

for epoch in range(10):
    step=0
    for iter in range(100):
        step+=1

        if random.random()<1.0:  #確率でサンプリング時にtop_kを行わない
            samples = gpt.generate(do_sample=True, max_length=SEQ_LEN, num_return_sequences=BATCH_SIZE,top_k=100,bad_words_ids=[[UNK_IDX]])
        else:
            samples = gpt.generate(do_sample=True, max_length=SEQ_LEN, num_return_sequences=BATCH_SIZE,bad_words_ids=[[UNK_IDX]])

        targets = samples[:,1:]

        rewards = rollout.get_reward(samples,1,senti)

        samples_pad = rollout.sample_pad_process(samples)  #修了トークン以降のトークンをPADに置き換える
        attention_mask = torch.ones(samples_pad.shape).to(device)

        gpt_optimizer.zero_grad()
        logits = gpt(input_ids=samples_pad[:,:-1],attention_mask=attention_mask[:,:-1]).logits
        preds = F.softmax(logits,dim=-1)
        loss = pgloss(preds,targets,rewards)
        loss.backward()
        gpt_optimizer.step()

        if step%1==0:
            print(gpt_tokenizer.batch_decode(samples_pad))

        if step%50==0:
            rollout.update_params()

    torch.save(gpt.state_dict(),"./gpt_reanforce.bin")
    print("save!")