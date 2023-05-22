from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5ForTextToSpeech
from datasets import load_dataset
import torch
import librosa
import pandas as pd
import copy, tqdm
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5ForTextToSpeech
from datasets import load_dataset



text_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
speech_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
sampling_rate = 16000




labels_csv = pd.read_csv('./datas/labels.csv')
label_set = {cat:idx for idx,cat in enumerate(['surprise', 'angry', 'fear', 'neutral','sad','disgust','happy'])}
label_set['disqust'] = label_set['disgust']



train_dataset = load_dataset("json", data_files='./datas/en_train.json', split="train")
test_dataset = load_dataset("json", data_files='./datas/en_test.json', split="train")




def train_load_audio(datas):
    arrs = []
    srs = []
    dlabels = []
    clabels = []
    for wav_path in datas['wav']:
        dirc = wav_path.split('_')[0]
        if 'User' in wav_path:
            y, sr= librosa.load(f'./datas/20_wavs/{dirc}/'+wav_path,sr=16000)
        else:
            y, sr= librosa.load(f'./datas/19_wavs/{dirc}/'+wav_path,sr=16000)
        arrs.append(y)
        srs.append(sr)
        
        row = labels_csv[labels_csv['filename'] == wav_path[:-4]]
        categories = row['dlabel'].tolist()[0].split(';')
        label = [0 for _ in range(7)]
        for cat in categories:
            label[label_set[cat]] = 1
        dlabels.append(label)
        clabels.append([row['valence'].tolist()[0],row['Arousal'].tolist()[0]])
        
    datas['arrs'] = arrs
    datas['srs'] = srs        
    datas['dlabels'] = torch.Tensor(dlabels)
    datas['clabels'] = torch.Tensor(clabels)
    return datas


def test_load_audio(datas):
    arrs = []
    srs = []    
    dlabels = []
    clabels = []
    for wav_path in datas['wav']:
        y, sr= librosa.load(f'./datas/wavs_test/'+wav_path,sr=16000)
        arrs.append(y)
        srs.append(sr)
        row = labels_csv[labels_csv['filename'] == wav_path[:-4]]        
        categories = row['dlabel'].tolist()[0].split(';')
        label = [0 for _ in range(7)]
        for cat in categories:
            label[label_set[cat]] = 1
        dlabels.append(label)
        clabels.append([row['valence'].tolist()[0],row['Arousal'].tolist()[0]])

    datas['arrs'] = arrs
    datas['srs'] = srs        
    datas['dlabels'] = torch.Tensor(dlabels)
    datas['clabels'] = torch.Tensor(clabels)
    return datas


def collate_fn(batch):
    arrs = [b['arrs'] for b in batch]
    text = [b['en_text'] for b in batch]
    arrs = speech_processor(audio=arrs, sampling_rate=sampling_rate, return_tensors="pt",padding=True,max_length=None)
    text = text_processor(text=text, return_tensors="pt",padding=True,truncation=True)
    
    
    res = dict()
    res['arrs'] = arrs
    res['text'] = text
    res['dlabels'] = torch.Tensor([b['dlabels'] for b in batch])
    res['clabels'] = torch.Tensor([ b['clabels'] for b in batch])
    return res





processed_tr_dataset = train_dataset.map(
            train_load_audio,
            batched=True,
            num_proc=8,
            remove_columns=['wav'],
        )
cnt_label = [0 for _ in range(7)]
for label in processed_tr_dataset['dlabels']:
    for idx,value in enumerate(label):
        if value == 1:
            cnt_label[idx] += 1

cnt_weight = [1 / i for i in cnt_label]
class_each_weight = []
for label in processed_tr_dataset['dlabels']:
    temp = min(cnt_weight)
    for idx,value in enumerate(label):
        if value == 1:
            curr = cnt_weight[idx]
        if curr >= temp:
            res = curr
    class_each_weight.append(res)
    
processed_tt_dataset = test_dataset.map(
            test_load_audio,
            batched=True,
            num_proc=8,
            remove_columns=['wav'],
        )


text_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
speech_model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
text_encoder = text_model.get_encoder()
speech_encoder = speech_model.get_encoder()
speech_encoder.prenet.freeze_feature_encoder()


class TextT5(torch.nn.Module):
    def __init__(self,text_encoder) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.textdlinear = torch.nn.Linear(768,7)
        self.textclinear = torch.nn.Linear(768,2)
    
    @staticmethod
    def mean_pool(emb, attention_mask):
        """Mean pooling while ignoring padded parts
        emb: [B, T, D]
        attention_mask: binary tensor of [B, T] where 0 if padded else 1
        """     
        emb_sum = (emb * attention_mask.unsqueeze(-1)).sum(dim=1)
        count =  torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1e-9)
        return emb_sum / count
        
    
    def forward(self,text_batch=None):
        outputs = self.text_encoder(
            input_values=text_batch['input_ids'],
            attention_mask=text_batch['attention_mask'],
            output_hidden_states=True,
        )            
        
        return outputs['last_hidden_state'],text_batch['attention_mask']
    
    def calcul_cls_loss(self,logits,labels):
        logits = self.textdlinear(logits)
        loss = torch.nn.BCEWithLogitsLoss()(logits.view(-1,7),labels)
        return loss
    def calcul_reg_loss(self,logits,labels):
        logits = self.textclinear(logits)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(logits.view(-1,2),labels)
        return loss
    
    def freeze_encoders(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder._requires_grad = False
        
    def unfreeze_encoders(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = True
        self.text_encoder._requires_grad = True        
        



class SpeechT5(torch.nn.Module):
    def __init__(self,speech_encoder) -> None:
        super().__init__()
        self.speech_encoder = speech_encoder
        self.speechdlinear = torch.nn.Linear(768,7)
        self.speechclinear = torch.nn.Linear(768,2)

    
    @staticmethod
    def mean_pool(emb, attention_mask):
        """Mean pooling while ignoring padded parts
        emb: [B, T, D]
        attention_mask: binary tensor of [B, T] where 0 if padded else 1
        """     
        emb_sum = (emb * attention_mask.unsqueeze(-1)).sum(dim=1)
        count =  torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1e-9)
        return emb_sum / count
        
    
    def forward(self,speech_batch=None):
        outputs = self.speech_encoder(
            input_values=speech_batch['input_values'],
            attention_mask=speech_batch['attention_mask'],
            output_hidden_states=True,
        )
        encoder_attention_mask = self.speech_encoder.prenet._get_feature_vector_attention_mask(
            outputs[0].shape[1], speech_batch['attention_mask']
        )
        return outputs['last_hidden_state'],encoder_attention_mask

    
    def calcul_cls_loss(self,logits,labels):
        logits = self.speechdlinear(logits)
        loss = torch.nn.BCEWithLogitsLoss()(logits.view(-1,7),labels)
        return loss
    def calcul_reg_loss(self,logits,labels):
        logits = self.speechclinear(logits)
        loss = torch.nn.MSELoss()(logits.view(-1,2),labels)
        return loss
    
    def freeze_encoders(self):
        for param in self.speech_encoder.parameters():
            param.requires_grad = False
        self.speech_encoder._requires_grad = False
        
    def unfreeze_encoders(self):
        for param in self.speech_encoder.parameters():
            param.requires_grad = True
        self.speech_encoder._requires_grad = True
        self.speech_encoder.prenet.freeze_feature_encoder()
    

class Multi_Model(torch.nn.Module):
    def __init__(self,speecht5, textt5) -> None:
        super().__init__()
        self.speecht5 = speecht5
        self.textt5 = textt5
        self.final_attn = torch.nn.MultiheadAttention(768, 8,dropout=0.1,batch_first=True)
        self.dlinear = torch.nn.Linear(768,7)
        self.clinear = torch.nn.Linear(768,2)        
    
    @staticmethod
    def mean_pool(emb, attention_mask):
        """Mean pooling while ignoring padded parts
        emb: [B, T, D]
        attention_mask: binary tensor of [B, T] where 0 if padded else 1
        """     
        emb_sum = (emb * attention_mask.unsqueeze(-1)).sum(dim=1)
        count =  torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1e-9)
        return emb_sum / count
        
    
    def forward(self,sbatch,tbatch):
        speech_repr, speech_mask = self.speecht5.forward(sbatch)
        
        text_repr, text_mask = self.textt5.forward(tbatch)
        if len(text_repr.size()) == 2:
            speech_repr = speech_repr.unsqueeze(0)
            text_repr = text_repr.unsqueeze(0)

        concat_repr = torch.cat([speech_repr,text_repr],dim=1)
        concat_attn = torch.cat([speech_mask,text_mask],dim=-1)
        dim_size = concat_repr.size()

        attn_output, _ = self.final_attn(query=concat_repr,key=concat_repr,value=concat_repr,key_padding_mask=concat_attn.float())
        logits = Multi_Model.mean_pool(attn_output,concat_attn)
        return logits,concat_attn
    
    def calcul_cls_loss(self,logits,labels):
        logits = self.dlinear(logits)
        loss = torch.nn.BCEWithLogitsLoss()(logits.view(-1,7),labels)
        return loss
    
    def calcul_reg_loss(self,logits,labels):
        logits = self.clinear(logits)
        loss = torch.nn.MSELoss()(logits.view(-1,2),labels)
        return loss
    
    def freeze_encoders(self):
        self.speecht5.freeze_encoders()
        self.textt5.freeze_encoders()
        
    def unfreeze_encoders(self):
        self.speecht5.unfreeze_encoders()
        self.textt5.unfreeze_encoders()




# tmodel = TextT5(text_encoder).load_state_dict(torch.load('./out/text_best_check.pt'))
# smodel = SpeechT5(speech_encoder).load_state_dict(torch.load('./out/speech_best_check.pt'))
tmodel = TextT5(text_encoder)
smodel = SpeechT5(speech_encoder)
model = Multi_Model(smodel,tmodel)
model.unfreeze_encoders()
tr_bs = 1
sampler = torch.utils.data.WeightedRandomSampler(class_each_weight,len(class_each_weight))
tr_loader = torch.utils.data.DataLoader(processed_tr_dataset,batch_size=tr_bs,collate_fn=collate_fn,sampler=sampler)
test_bs = 32
tt_loader = torch.utils.data.DataLoader(processed_tt_dataset,batch_size=test_bs,collate_fn=collate_fn)


optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

best_acc = 0
cnt = 0
accumulation_steps = 16
for epoch in range(30):
    # with accelerator.accumulate(model):
    for idx,batch in tqdm.tqdm(enumerate(tr_loader)):
        speech_batch = batch['arrs'].to(device)
        text_batch = batch['text'].to(device)
        dlabels = batch['dlabels'].to(device)
        clabels = batch['clabels'].to(device)
        logits,_ = model.forward(speech_batch,text_batch)
        dloss=model.calcul_cls_loss(logits,dlabels)
        closs=model.calcul_reg_loss(logits,clabels)
        loss = (dloss+closs) / accumulation_steps
        loss.backward()
        # accelerator.backward(log_loss)
        if (idx +1) & accumulation_steps == 0:        
            log_loss = {"train_loss": loss.detach().float()}
            optimizer.step()
            optimizer.zero_grad()
        
    with torch.no_grad():
        acc = 0
        for batch in tqdm.tqdm(tt_loader):
            text_batch = batch['text'].to(device)
            speech_batch = batch['arrs'].to(device)
            dlabels = batch['dlabels'].to(device)
            clabels = batch['clabels'].to(device)
            logits,_ = model.forward(speech_batch,text_batch)
            
            
            output = model.dlinear(logits)
            output = torch.where(torch.sigmoid(output)>0.5,1,0)
            distance = model.calcul_reg_loss(logits,clabels)
            
            acc += torch.sum(torch.all(output==dlabels,dim=-1))
    print(distance)
    acc = acc / ((len(tt_loader)-1) * test_bs)
    print(acc)
    if acc > best_acc:
        cnt = 0
        best_distance = distance
        best_acc = acc
        best_epoch = epoch
        best_states = copy.deepcopy(model.state_dict())
        
    else:
        cnt += 1

        
        
        

torch.save(best_states,'./out/multi_modal_best_check.pt')
print(best_acc)
print(best_epoch)
print(best_distance)
with open('./out/multi_modal_log_info.txt','w',encoding='utf-8') as f:
    f.write(str(best_acc.detach().cpu().numpy()))
    f.write('\n')
    f.write(str(best_epoch))
    f.write('\n')
    f.write(str(best_distance))
    
        


