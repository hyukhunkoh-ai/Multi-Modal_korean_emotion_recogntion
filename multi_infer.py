from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5ForTextToSpeech
from datasets import load_dataset
import torch
import librosa
import pandas as pd
import copy
speech_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
text_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
sampling_rate = 16000

labels_csv = pd.read_csv('./datas/labels.csv')
label_set = {cat:idx for idx,cat in enumerate(['surprise', 'angry', 'fear', 'neutral','sad','disgust','happy'])}
label_set['disqust'] = label_set['disgust']



train_dataset = load_dataset("json", data_files='./datas/train.json', split="train")
test_dataset = load_dataset("json", data_files='./datas/test.json', split="train")



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
    arrs = speech_processor(audio=arrs, sampling_rate=sampling_rate, return_tensors="pt",padding=True,max_length= None)
    # text = [b['en_text'] for b in batch]
    # text = text_processor(text=text, return_tensors="pt",padding=True,max_length= None)
    res = dict()
    res['arrs'] = arrs
    # res['text'] = text
    res['dlabels'] = torch.Tensor([b['dlabels'] for b in batch])
    res['clabels'] = torch.Tensor([ b['clabels'] for b in batch])
    return res


processed_tr_dataset = train_dataset.map(
            train_load_audio,
            batched=True,
            num_proc=8,
            remove_columns=['wav'],
        )


processed_tt_dataset = test_dataset.map(
            test_load_audio,
            batched=True,
            num_proc=8,
            remove_columns=['wav'],
        )


class Multi_Model(torch.nn.Module):
    def __init__(self,speech_encoder,text_encoder) -> None:
        super().__init__()
        self.speech_encoder = speech_encoder
        self.text_encoder = text_encoder
        self.clslinear = torch.nn.Linear(768,7)
        self.reglinear = torch.nn.Linear(768,2)
    
    @staticmethod
    def mean_pool(emb, attention_mask):
        """Mean pooling while ignoring padded parts
        emb: [B, T, D]
        attention_mask: binary tensor of [B, T] where 0 if padded else 1
        """     
        emb_sum = (emb * attention_mask.unsqueeze(-1)).sum(dim=1)
        count =  torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1e-9)
        return emb_sum / count
        
    
    def forward(self,speech_batch=None,text_batch=None):

        if speech_batch is not None:
            outputs = self.speech_encoder(
                input_values=speech_batch['input_values'],
                attention_mask=speech_batch['attention_mask'],
                output_hidden_states=True,
            )
            encoder_attention_mask = self.speech_encoder.prenet._get_feature_vector_attention_mask(
                outputs[0].shape[1], speech_batch['attention_mask']
            )
            logits = self.mean_pool(outputs['last_hidden_state'],encoder_attention_mask)
            return logits,encoder_attention_mask
        if text_batch is not None:
            outputs = self.text_encoder(
                input_values=text_batch['input_ids'],
                attention_mask=text_batch['attention_mask'],
                output_hidden_states=True,
            )            
            logits = Multi_Model.mean_pool(outputs['last_hidden_state'],text_batch['attention_mask'])
            return logits,text_batch['attention_mask']
    
    def calcul_cls_loss(self,logits,labels):
        logits = self.clslinear(logits)
        loss = torch.nn.BCEWithLogitsLoss()(logits.view(-1,7),labels)
        return loss
    def calcul_reg_loss(self,logits,labels):
        # logits = self.reglinear(logits)
        # loss_fn = torch.nn.MSELoss()
        logits = self.clslinear(logits)
        loss = torch.nn.BCEWithLogitsLoss()(logits.view(-1,7),labels)
        # return loss_fn(logits,labels)
        return loss
    
    def freeze_encoders(self):
        for param in self.speech_encoder.parameters():
            param.requires_grad = False
        self.speech_encoder._requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder._requires_grad = False
        
    def unfreeze_encoders(self):
        for param in self.speech_encoder.parameters():
            param.requires_grad = True
        self.speech_encoder._requires_grad = True
        for param in self.text_encoder.parameters():
            param.requires_grad = True
        self.text_encoder._requires_grad = True
        self.speech_encoder.prenet.freeze_feature_encoder()
        
        
# from accelerate import Accelerator
import tqdm
# accelerator = Accelerator(gradient_accumulation_steps=8)


speech_model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
speech_encoder = speech_model.get_encoder()
speech_encoder.prenet.freeze_feature_encoder()
text_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
text_encoder = text_model.get_encoder()

model = Multi_Model(speech_encoder,text_encoder)
tr_bs = 2
tr_loader = torch.utils.data.DataLoader(processed_tr_dataset,batch_size=tr_bs,collate_fn=collate_fn)
test_bs = 32
tt_loader = torch.utils.data.DataLoader(processed_tt_dataset,batch_size=test_bs,collate_fn=collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# model, optimizer, training_dataloader = accelerator.prepare(
#      model, optimizer, tr_loader
# )
# model.freeze_encoders()
best_acc = 0
cnt = 0
accumulation_steps = 8
# for epoch in range(10):
#     # with accelerator.accumulate(model):
for idx,batch in tqdm.tqdm(enumerate(tr_loader)):
    speech_batch = batch['arrs'].to(device)
    dlabels = batch['dlabels'].to(device)
    logits,_ = model.forward(speech_batch=speech_batch)
    # sloss=model.calcul_reg_loss(logits,clabels)
    sloss=model.calcul_reg_loss(logits,dlabels)
    loss = sloss / accumulation_steps
    
    break
    
model.load_state_dict(torch.load('./out/test.pt',map_location=device))
with torch.no_grad():
    acc = 0
    for batch in tqdm.tqdm(tt_loader):
        speech_batch = batch['arrs'].to(device)
        # text_batch = batch['text'].to(device)
        dlabels = batch['dlabels'].to(device)
        # clabels = batch['clabels'].to(device)
        # tlogits,_ = model.forward(text_batch=text_batch)
        # tloss=model.calcul_cls_loss(tlogits,dlabels)
        logits,_ = model.forward(speech_batch=speech_batch)
        output = model.clslinear(logits)
        output = torch.where(torch.sigmoid(output)>0.5,1,0)
        
        acc += torch.sum(torch.sum(output==dlabels,dim=-1))
            
print(acc)
# torch.save(best_states,'./out/best_ckpg.pt')
# with open('./out/log_info.txt','w',encoding='utf-8') as f:
#     f.write(best_acc)
#     f.write('\n')
#     f.write(best_epoch)
    
        
        
            
            
            
    
    
    
    
    









'''
encoder_outputs = self.encoder(
    input_values=input_values,
    attention_mask=attention_mask,
    head_mask=head_mask,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=return_dict,
)

encoder_outputs = self.encoder(
    input_values=input_values,
    attention_mask=attention_mask,
    head_mask=head_mask,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=return_dict,
)

'''




