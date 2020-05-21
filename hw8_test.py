
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.data.sampler as sampler
import torchvision
from torchvision import datasets, transforms

import numpy as np
import sys
import os
import random
import re
import json

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 判斷是用 CPU 還是 GPU 執行運算

data_dir = sys.argv[1]
output_path = sys.argv[2]

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)  # if you are using multi-GPU.
np.random.seed(0)  # Numpy module.
random.seed(0)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class LabelTransform(object):
  def __init__(self, size, pad):
    self.size = size
    self.pad = pad

  def __call__(self, label):
    label = np.pad(label, (0, (self.size - label.shape[0])), mode='constant', constant_values=self.pad)
    return label



class EN2CNDataset(data.Dataset):
  def __init__(self, root, max_output_len, set_name):
    self.root = root

    self.word2int_cn, self.int2word_cn = self.get_dictionary('cn')
    self.word2int_en, self.int2word_en = self.get_dictionary('en')

    # 載入資料
    self.data = []
    with open(os.path.join(self.root, f'{set_name}.txt'), "r") as f:
      for line in f:
        self.data.append(line)
    print (f'{set_name} dataset size: {len(self.data)}')

    self.cn_vocab_size = len(self.word2int_cn)
    self.en_vocab_size = len(self.word2int_en)
    self.transform = LabelTransform(max_output_len, self.word2int_en['<PAD>'])

  def get_dictionary(self, language):
    # 載入字典
    with open(os.path.join(self.root, f'word2int_{language}.json'), "r") as f:
      word2int = json.load(f)
    with open(os.path.join(self.root, f'int2word_{language}.json'), "r") as f:
      int2word = json.load(f)
    return word2int, int2word

  def __len__(self):
    return len(self.data)

  def __getitem__(self, Index):
    # 先將中英文分開
    sentences = self.data[Index]
    sentences = re.split('[\t\n]', sentences)
    sentences = list(filter(None, sentences))
    #print (sentences)
    assert len(sentences) == 2

    # 預備特殊字元
    BOS = self.word2int_en['<BOS>']
    EOS = self.word2int_en['<EOS>']
    UNK = self.word2int_en['<UNK>']

    # 在開頭添加 <BOS>，在結尾添加 <EOS> ，不在字典的 subword (詞) 用 <UNK> 取代
    en, cn = [BOS], [BOS]
    # 將句子拆解為 subword 並轉為整數
    sentence = re.split(' ', sentences[0])
    sentence = list(filter(None, sentence))
    #print (f'en: {sentence}')
    for word in sentence:
      en.append(self.word2int_en.get(word, UNK))
    en.append(EOS)

    # 將句子拆解為單詞並轉為整數
    # e.g. < BOS >, we, are, friends, < EOS > --> 1, 28, 29, 205, 2
    sentence = re.split(' ', sentences[1])
    sentence = list(filter(None, sentence))
    #print (f'cn: {sentence}')
    for word in sentence:
      cn.append(self.word2int_cn.get(word, UNK))
    cn.append(EOS)

    en, cn = np.asarray(en), np.asarray(cn)

    # 用 <PAD> 將句子補到相同長度
    en, cn = self.transform(en), self.transform(cn)
    en, cn = torch.LongTensor(en), torch.LongTensor(cn)

    return en, cn

class Encoder(nn.Module):
  def __init__(self, en_vocab_size, emb_dim, hid_dim, n_layers, dropout):
    super().__init__()
    self.embedding = nn.Embedding(en_vocab_size, emb_dim)
    self.hid_dim = hid_dim
    self.n_layers = n_layers
    self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
    self.dropout = nn.Dropout(dropout)

  def forward(self, input):
    # input = [batch size, sequence len, vocab size]
    embedding = self.embedding(input)
    outputs, hidden = self.rnn(self.dropout(embedding))
    # outputs = [batch size, sequence len, hid dim * directions]
    # hidden =  [num_layers * directions, batch size  , hid dim]
    # outputs 是最上層RNN的輸出
        
    return outputs, hidden
class Decoder(nn.Module):
  def __init__(self, cn_vocab_size, emb_dim, hid_dim, n_layers, dropout, isatt):
    super().__init__()
    self.cn_vocab_size = cn_vocab_size
    self.hid_dim = hid_dim * 2                                                 #bidirectional??
    self.n_layers = n_layers
    self.embedding = nn.Embedding(cn_vocab_size, config.emb_dim)
    self.isatt = isatt
    self.attention = Attention(hid_dim, n_layers)                   #增加參數      
    # 如果使用 Attention Mechanism 會使得輸入維度變化，請在這裡修改
    # e.g. Attention 接在輸入後面會使得維度變化，所以輸入維度改為
    self.input_dim = emb_dim + hid_dim * 2 if isatt else emb_dim                             #切換
    self.input_dim_ori = emb_dim
    self.rnn = nn.GRU(self.input_dim_ori, self.hid_dim, self.n_layers, dropout = dropout, batch_first=True)
    if self.isatt == True:
      self.embedding2vocab1 = nn.Linear(self.hid_dim * 2, self.hid_dim * 4)                              #全體乘2
      self.embedding2vocab2 = nn.Linear(self.hid_dim * 4, self.hid_dim * 8)
      self.embedding2vocab3 = nn.Linear(self.hid_dim * 8, self.cn_vocab_size)
    elif self.isatt == False:
      self.embedding2vocab1 = nn.Linear(self.hid_dim, self.hid_dim * 2)                           
      self.embedding2vocab2 = nn.Linear(self.hid_dim * 2, self.hid_dim * 4)
      self.embedding2vocab3 = nn.Linear(self.hid_dim * 4, self.cn_vocab_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, input, hidden, encoder_outputs):
    # input = [batch size, vocab size]
    # hidden = [batch size, n layers * directions, hid dim]
    # Decoder 只會是單向，所以 directions=1
    input = input.unsqueeze(1)
    embedded = self.dropout(self.embedding(input))                              #把吃進來的東西做embedding
    # embedded = [batch size, 1, emb dim]
    #print('shape of encoder_outputs:',encoder_outputs.shape)
    if self.isatt:
      attn = self.attention(encoder_outputs, hidden)
      attn = attn.unsqueeze(1)
      #print('attn.shape:',attn.shape)
      # TODO: 在這裡決定如何使用 Attention，e.g. 相加 或是 接在後面， 請注意維度變化                      #todo:用attn，可以接在output後面
    hidden = hidden.contiguous()
    output, hidden = self.rnn(embedded, hidden)
    # output = [batch size, 1, hid dim]
    # hidden = [num_layers, batch size, hid dim]              #應該是hid dim*2
    
    #自家：進行output和attn對接
    #print('output.shape:',output.shape)
    #print('hidden:',hidden.shape)
    if self.isatt:
      output = torch.cat((output,attn),dim=2)

    # 將 RNN 的輸出轉為每個詞出現的機率
    #print('output.shape:',output.shape)
    output = self.embedding2vocab1(output.squeeze(1))
    output = self.embedding2vocab2(output)
    prediction = self.embedding2vocab3(output)
    # prediction = [batch size, vocab size]
    return prediction, hidden

class Attention(nn.Module):
  def __init__(self, hid_dim, num_layers, sequence_len=50):                              #增加參數
    super(Attention, self).__init__()
    self.hid_dim = hid_dim
    
    self.dnn = nn.Sequential(
            nn.Linear(hid_dim*2*(sequence_len+num_layers), 1024),                 
            nn.Dropout(p=0.3),                          
            nn.ReLU(),
            nn.Linear(1024, 512),                
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(512, hid_dim*2),  
            nn.Softmax()                                                         #step2:pass softmax
        ).to(device)                                                             #to device?
    
  
  def forward(self, encoder_outputs, decoder_hidden):
    # encoder_outputs = [batch size, sequence len, hid dim * directions]
    # decoder_hidden = [num_layers, batch size, hid dim]                #應該是hid dim*2
    # 一般來說是取 Encoder 最後一層的 hidden state 來做 attention
    ########
    # TODO #
    ########
    #stap1:get original weight
    #print('encoder_outputs.shape:',encoder_outputs.shape)
    #print('decoder_hidden.shape:',decoder_hidden.shape)
    decoder_hidden = decoder_hidden.permute(1, 0, 2)           #把batch_size維度移到最上面
    encoder_outputs = encoder_outputs.reshape(encoder_outputs.shape[0],-1)       #攤平
    decoder_hidden = decoder_hidden.reshape(decoder_hidden.shape[0],-1)
    nn_input = torch.tensor(torch.cat((encoder_outputs,decoder_hidden), dim=1))
    
    #print('nn_input.shape:',nn_input.shape)                
    attention=self.dnn(nn_input)
    #attention = None
    return attention
class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, device, topk):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.device = device
    self.topk = topk
    assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
            
  def forward(self, input, target, teacher_forcing_ratio):
    # input  = [batch size, input len, vocab size]
    # target = [batch size, target len, vocab size]
    # teacher_forcing_ratio 是有多少機率使用正確答案來訓練
    batch_size = target.shape[0]
    target_len = target.shape[1]
    vocab_size = self.decoder.cn_vocab_size

    # 準備一個儲存空間來儲存輸出
    outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
    # 將輸入放入 Encoder
    encoder_outputs, hidden = self.encoder(input)
    # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
    # encoder_outputs 主要是使用在 Attention
    # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
    # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
    hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
    hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)           #在hid dim對接
    #print('the shape of hidden',hidden.shape)                              
    # 取的 <BOS> token
    input = target[:, 0]
    preds = []
    for t in range(1, target_len):
      output, hidden = self.decoder(input, hidden, encoder_outputs)
      #print('output.shape:',output.shape)
      #print('outputs.shape:',outputs.shape)
      outputs[:, t] = output
      # 決定是否用正確答案來做訓練 
      teacher_force = random.random() <= teacher_forcing_ratio                            
      # 取出機率最大的單詞
      top1 = output.argmax(1)
      # 如果是 teacher force 則用正解訓練，反之用自己預測的單詞做預測
      input = target[:, t] if teacher_force and t < target_len else top1
      preds.append(top1.unsqueeze(1))
    preds = torch.cat(preds, 1)
    return outputs, preds

  def inference(self, input, target):
    ########
    # TODO #
    ########
    # 在這裡實施 Beam Search
    # 此函式的 batch size = 1  
    # input  = [batch size, input len, vocab size]
    # target = [batch size, target len, vocab size]
    batch_size = input.shape[0]
    input_len = input.shape[1]        # 取得最大字數
    vocab_size = self.decoder.cn_vocab_size

    # 準備一個儲存空間來儲存輸出
    record = torch.zeros(batch_size,self.topk,input_len).to(self.device)
    #record首項先填上<BOS>的序號
    for j in range(self.topk):
        record[:,j,0] = int(target[:, 0])
    outputs = torch.zeros(batch_size, self.topk, input_len, vocab_size).to(self.device)                
    acprob = torch.ones(batch_size, self.topk).to(self.device)
    problist = torch.zeros(batch_size,self.topk**2).to(self.device)
    print(problist)
    daodao = torch.zeros(batch_size,self.topk,self.topk).to(self.device)
    print(daodao)
    hita = torch.zeros(self.encoder.n_layers, batch_size, self.topk, self.encoder.hid_dim*2).to(self.device).contiguous()
    hibf =  torch.zeros(self.encoder.n_layers, batch_size, self.topk, self.encoder.hid_dim*2).to(self.device).contiguous()

    encoder_outputs, hidden = self.encoder(input)

    hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
    hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2).contiguous() 

    for i in range(self.topk):
        hita[:,:,i] = hidden 


    input = torch.zeros([batch_size,self.topk],dtype = torch.long).to(self.device)
    for j in range(self.topk):
        input[:,j] = target[:, 0]                                #becareful
    preds = []
    for t in range(1, input_len):
      for j in range(self.topk):
          input = input.type(torch.long)
          output, hibf[:,:,j] = self.decoder(input[:,j], hita[:,:,j], encoder_outputs) 
          outputs[:,j,t] = output
          pros ,candi = torch.topk(output, self.topk, dim=1)  
          
          for i in range(self.topk):
              pros[:,i] = pros[:,i]*acprob[:,j]                       
          problist[:,j*self.topk:j*self.topk+self.topk] = pros[:]
          print(candi.size())
          daodao[:,j] = candi[:] 


      values, indices = torch.topk(problist, self.topk,dim=1)

      hita[:,:,:] = hibf[:,:,(indices[:,:]//self.topk).squeeze(0)]

      acprob = values[:]

      outputs_buffer  = outputs[:]
      for j in range(self.topk):
          outputs[:,j] = outputs_buffer[:,int(indices[:,j]/self.topk)]
      

      record_buffer = record[:]
      record[:,:] = record_buffer[:,int(indices[:,:]/self.topk)]
      record[:,:,t] = daodao[:, int(indices[:,:]/self.topk), indices[:,:]%self.topk]    
      
      input = record[:,:,t]

    for t in range(input_len):
        preds.append(record[:,0,t].unsqueeze(1))
    preds = torch.cat(preds, 1)
    return outputs[:,0], preds
    '''
    batch_size = input.shape[0]
    input_len = input.shape[1]        # 取得最大字數
    vocab_size = self.decoder.cn_vocab_size

    # 準備一個儲存空間來儲存輸出
    outputs = torch.zeros(batch_size, input_len, vocab_size).to(self.device)
    # 將輸入放入 Encoder
    encoder_outputs, hidden = self.encoder(input)
    # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
    # encoder_outputs 主要是使用在 Attention
    # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
    # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
    hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
    hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
    # 取的 <BOS> token
    input = target[:, 0]
    preds = []
    for t in range(1, input_len):
      output, hidden = self.decoder(input, hidden, encoder_outputs)
      # 將預測結果存起來
      outputs[:, t] = output
      # 取出機率最大的單詞
      top1 = output.argmax(1)
      input = top1
      preds.append(top1.unsqueeze(1))
    preds = torch.cat(preds, 1)
    return outputs, preds'''

def load_model(model, load_model_path):
  print(f'Load model from {load_model_path}')
  model.load_state_dict(torch.load(f'{load_model_path}.ckpt'))
  return model

def build_model(config, en_vocab_size, cn_vocab_size, topk):
  # 建構模型
  encoder = Encoder(en_vocab_size, config.emb_dim, config.hid_dim, config.n_layers, config.dropout)
  decoder = Decoder(cn_vocab_size, config.emb_dim, config.hid_dim, config.n_layers, config.dropout, config.attention)
  model = Seq2Seq(encoder, decoder, device, topk)
  print(model)
  # 建構 optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
  print(optimizer)
  if config.load_model:
    model = load_model(model, config.load_model_path)
  model = model.to(device)

  return model, optimizer

def tokens2sentence(outputs, int2word):
  sentences = []
  for tokens in outputs:
    sentence = []
    for token in tokens:
      word = int2word[str(int(token))]
      if word == '<EOS>':
        break
      sentence.append(word)
    sentences.append(sentence)
  
  return sentences




def computebleu(sentences, targets):
  score = 0 
  assert (len(sentences) == len(targets))

  def cut_token(sentence):
    tmp = []
    for token in sentence:
      if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
        tmp.append(token)
      else:
        tmp += [word for word in token]
    return tmp 
  
  smoothie = SmoothingFunction().method4
  for sentence, target in zip(sentences, targets):
    sentence = cut_token(sentence)
    target = cut_token(target)
    score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0), smoothing_function=smoothie)                                                                                          
  
  return score

def infinite_iter(data_loader):
  it = iter(data_loader)
  while True:
    try:
      ret = next(it)
      yield ret
    except StopIteration:
      it = iter(data_loader)

def test(model, dataloader, loss_function, beam_search):
  model.eval()
  loss_sum, bleu_score= 0.0, 0.0
  n = 0
  result = []
  for sources, targets in dataloader:
    sources, targets = sources.to(device), targets.to(device)
    batch_size = sources.size(0)
    outputs, preds = model.inference(sources, targets, beam_search)
    # targets 的第一個 token 是 <BOS> 所以忽略
    outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
    targets = targets[:, 1:].reshape(-1)

    loss = loss_function(outputs, targets)
    loss_sum += loss.item()

    # 將預測結果轉為文字
    targets = targets.view(sources.size(0), -1)
    preds = tokens2sentence(preds, dataloader.dataset.int2word_cn)
    sources = tokens2sentence(sources, dataloader.dataset.int2word_en)
    targets = tokens2sentence(targets, dataloader.dataset.int2word_cn)
    for source, pred, target in zip(sources, preds, targets):
      result.append((source, pred, target))
    # 計算 Bleu Score
    bleu_score += computebleu(preds, targets)

    n += batch_size

  return loss_sum / len(dataloader), bleu_score / n, result

def test_process(config):
  # 準備測試資料
  test_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'testing')
  test_loader = data.DataLoader(test_dataset, batch_size=1)
  # 建構模型
  model, optimizer = build_model(config, test_dataset.en_vocab_size, test_dataset.cn_vocab_size, config.topk)
  print ("Finish build model")
  loss_function = nn.CrossEntropyLoss(ignore_index=0)
  model.eval()
  # 測試模型
  test_loss, bleu_score, result = test(model, test_loader, loss_function, config.beam_search)
  # 儲存結果
  with open(output_path, 'w') as f:
    for line in result:
      print (line, file=f)

  return test_loss, bleu_score

class configurations(object):
  def __init__(self):
    self.batch_size = 60
    self.emb_dim = 256
    self.hid_dim = 512
    self.n_layers = 3
    self.dropout = 0.5
    self.learning_rate = 0.00005
    self.max_output_len = 50              # 最後輸出句子的最大長度
    self.num_steps = 12000                # 總訓練次數                                
    self.store_steps = 800                # 訓練多少次後須儲存模型                            
    self.summary_steps = 800              # 訓練多少次後須檢驗是否有overfitting            
    self.load_model = True               # 是否需載入模型
    self.store_model_path = "."      # 儲存模型的位置
    self.load_model_path = './model_12000'                           #更改為客製化
    self.data_path = data_dir          # 資料存放的位置                         #客製化 
    self.attention = True                # 是否使用 Attention Mechanism      
    self.beam_search = True                     
    self.topk = 2                         #自家:beam search 的k值
    

# 在執行 Test 之前，請先行至 config 設定所要載入的模型位置
if __name__ == '__main__':
  config = configurations()
  print ('config:\n', vars(config))
  test_loss, bleu_score = test_process(config)
  print (f'test loss: {test_loss}, bleu_score: {bleu_score}')

