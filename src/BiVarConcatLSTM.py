#!/usr/bin/env python
# -*- coding: utf-8 -*-
import chainer.functions as F
import numpy as np
xp = np
from util.NNCommon import *
import util.generators as gens
from util.vocabulary import Vocabulary
from util.NNCommon import transferWordVector,LSTM
import os,random

class VaeLSTM(Chain):

    def __init__(self,args,cudnn_flag=False):
        self.n_vocab = args.n_vocab
        self.n_embed = args.embed
        self.n_layers = args.layer
        self.n_latent = args.n_latent
        self.out_size = args.hidden
        self.sample_size= args.sample_size
        self.kl_zero_epoch = args.kl_zero_epoch
        self.drop_ratio = args.dropout

        super(VaeLSTM, self).__init__(
                embed = L.EmbedID(self.n_vocab,self.n_embed),
                #VAEenc
                enc_f = LSTM(self.n_layers,self.n_embed, self.out_size, dropout=self.drop_ratio, use_cudnn=cudnn_flag),
                enc_b = LSTM(self.n_layers,self.n_embed, self.out_size, dropout=self.drop_ratio, use_cudnn=cudnn_flag),

                le2_mu=L.Linear(4*self.out_size, self.n_latent),
                le2_ln_var=L.Linear(4*self.out_size, self.n_latent),
                #VAEdec
                ld_h = L.Linear(self.n_latent,2*self.out_size),
                ld_c = L.Linear(self.n_latent,2*self.out_size),

                dec = LSTM(self.n_layers,self.n_embed, 2*self.out_size, dropout=self.drop_ratio, use_cudnn=cudnn_flag),
                h2w = L.Linear(2*self.out_size,self.n_vocab),
        )


    def setEpochNow(self,epoch_now):
        self.epoch_now = epoch_now

    def setMaxEpoch(self,epoch):
        self.epoch = epoch

    def setBatchSize(self,batch_size):
        self.batch_size = batch_size

    def setVocab(self,vocab):
        self.vocab = vocab

    def makeEmbedBatch(self,xs,reverse=False):
        if reverse:
            xs = [xp.asarray(x[::-1],dtype=xp.int32) for x in xs]
        elif not reverse:
            xs = [xp.asarray(x,dtype=xp.int32) for x in xs]
        section_pre = np.array([len(x) for x in xs[:-1]], dtype=np.int32)
        sections = np.cumsum(section_pre)
        xs = F.split_axis(self.embed(F.concat(xs, axis=0)), sections, axis=0)
        return xs

    def encode(self,xs):
        xs = [x+[2] for x in xs]#2は</s>を指す。decには<s>から入れる。
        xs_f = self.makeEmbedBatch(xs)
        xs_b = self.makeEmbedBatch(xs,True)

        self.enc_f.reset_state()
        self.enc_b.reset_state()
        ys_f = self.enc_f(xs_f)
        ys_b = self.enc_b(xs_b)
        
        mu_arr = [self.le2_mu(F.concat((hx_f,cx_f,hx_b,cx_b))) for hx_f,cx_f,hx_b,cx_b in zip(self.enc_f.hx,self.enc_f.cx,self.enc_b.hx,self.enc_b.cx)]
        var_arr= [self.le2_ln_var(F.concat((hx_f,cx_f,hx_b,cx_b))) for hx_f,cx_f,hx_b,cx_b in zip(self.enc_f.hx,self.enc_f.cx,self.enc_b.hx,self.enc_b.cx)]
        return mu_arr,var_arr

    def __call__(self,xs):
        mu_arr,var_arr = self.encode(xs)
        t = [[1]+x for x in xs]#1は<s>を指す。decには<s>から入れる。</s>まで予測する。
        loss = None
        for mu,var in zip(mu_arr,var_arr):
            if loss is None:
                loss = self.calcLoss(t,mu,var)
            else:
                loss+= self.calcLoss(t,mu,var)
        return loss

    def calcLoss(self,t,mu,ln_var):
        k = self.sample_size;kl_zero_epoch = self.kl_zero_epoch
        loss = None
        t_pred = [t_e[1:]+[2] for t_e in t]
        t_pred = [xp.asarray(tp_e,dtype=xp.int32) for tp_e in t_pred]
        t = self.denoiseInput(t)
        print("t:{}".format([self.vocab.itos(t_e) for t_e in t[0]]))        
        t_vec = self.makeEmbedBatch(t)
        for l in range(k):
            z = F.gaussian(mu, ln_var)
            if loss is None:loss = self.decode(z,t_vec,t_pred) / (k * self.batch_size)
            elif loss is not None:loss += self.decode(z,t_vec,t_pred) / (k * self.batch_size)
        C = 0.06 *(self.epoch_now-kl_zero_epoch)/self.epoch
        if self.epoch_now>kl_zero_epoch:loss += C * F.gaussian_kl_divergence(mu, ln_var) / self.batch_size
        return loss
    
    def denoiseInput(self,t,noise_rate=0.3):###WordDropOut
        if noise_rate>0.0:
            for t_i,t_e in enumerate(t):
                ind_arr = [t_i for t_i in range(1,min(len(t_e),10))]
                random.shuffle(ind_arr)
                unk_ind_arr = ind_arr[:int(len(ind_arr)*noise_rate)]
                for unk_ind in unk_ind_arr:t[t_i][unk_ind]=self.vocab.stoi("<unk>")
        return t
        
    def decode(self,z,t_vec,t_pred):
        self.dec.hx = F.reshape(self.ld_h(z),(1,self.batch_size,2*self.out_size))#1,30,100
        self.dec.cx = F.reshape(self.ld_c(z),(1,self.batch_size,2*self.out_size))

        ys_d = self.dec(t_vec)
        ys_w = [self.h2w(y) for y in ys_d]
        loss = F.softmax_cross_entropy(ys_w[0],t_pred[0])#/len(t[0])

        print("t:{}".format([self.vocab.itos(tp_e) for tp_e in t_pred[0].tolist()]))
        print("y:{}\n".format([self.vocab.itos(vec.argmax()) for vec in np.array(ys_w[0].data.tolist())]))
        for ti in range(1,len(t_pred)):
            loss+=F.softmax_cross_entropy(ys_w[ti],t_pred[ti])
        print("before_loss:{}".format(loss.data))
        
        return loss

    def loadW(self,premodel_name):
        src_vocab = self.vocab
        src_w2ind = {}; trg_w2ind = {}
        src_ind2w = {}; trg_ind2w = {}
        src_size = self.n_vocab
        print(src_size)
        for vi in range(src_size):
            src_ind2w[vi] = src_vocab.itos(vi)
            src_w2ind[src_ind2w[vi]] = vi
        print("pre:{}".format(self.embed.W.data[0][:5]))
        self.embed.W = Variable(xp.array(transferWordVector(src_w2ind,src_ind2w,premodel_name),dtype=xp.float32))
        print("pos:{}".format(self.embed.W.data[0][:5]))
        

    def predict(self,batch,vocab,z=None,randFlag=True):
        print(xp)
        if z is None:
            z = Variable(xp.random.normal(0,1,(batch,self.n_latent)).astype(xp.float32))
        else:
            pass
        self.dec.hx = F.reshape(self.ld_h(z),(1,batch,2*self.out_size))
        self.dec.cx = F.reshape(self.ld_c(z),(1,batch,2*self.out_size))
        t = [[bi] for bi in [1]*batch]
        t = self.makeEmbedBatch(t)
        ys_d = self.dec(t,train=False)
        ys_w = [self.h2w(y) for y in ys_d]
        name_arr_arr = []
        if randFlag:
            t = [predictRandom(F.softmax(y_each)) for y_each in ys_w]
        else:
            t = [y_each.data[-1].argmax(0) for y_each in ys_w]
            if t[0].__class__.__name__!=int:
                t = [int(t_each) for t_each in t]
        name_arr_arr.append(t)
        t = [self.embed(xp.array([t_each],dtype=xp.int32)) for t_each in t]
        count_len=0
        while count_len<100:
            ys_d = self.dec(t,train=False)
            ys_w = [self.h2w(y) for y in ys_d]
            if randFlag:
                t = [predictRandom(F.softmax(y_each)) for y_each in ys_w]
            else:
                t = [y_each.data[-1].argmax(0) for y_each in ys_w]
                if t[0].__class__.__name__!=int:
                    t = [int(t_each) for t_each in t]
            name_arr_arr.append(t)
            t = [self.embed(xp.array([t_each],dtype=xp.int32)) for t_each in t]
            count_len+=1
        tenti = xp.array(name_arr_arr).T
        z_sum = [sum([z_e2**2 for z_e2 in z_e]) for z_e in z.data]
        for ni,name in enumerate(tenti):
            name = [vocab.itos(nint) for nint in name.tolist()]
            if "</s>" in name:
                print("name:{}".format(" ".join(name[:name.index("</s>")])))


def train(args,dataname="sent",wordvec_model=""):
    if args.gpu>=0:
        import cupy as cp
        global xp;xp=cp

    src_vocab = Vocabulary.new(gens.word_list(args.source), args.n_vocab)
    src_vocab.save('./{}/vocab_{}_l{}.bin'.format(dataname,dataname,args.n_latent))
    encdec = VaeLSTM(args)

    encdec.setBatchSize(args.batchsize)
    encdec.setVocab(src_vocab)
    encdec.setMaxEpoch(args.epoch)
    
    first_e = 0
    model_name=""
    for e in range(args.epoch):
        model_name_tmp = "./{}/model/biconcatlstm_vae_kl_{}_{}_l{}.npz".format(dataname,dataname,e,args.n_latent)
        if os.path.exists(model_name_tmp):
            model_name = model_name_tmp
            encdec.setEpochNow(e+1)
            
    if os.path.exists(model_name):
        serializers.load_npz(model_name,encdec)
        print("loaded_{}".format(model_name))           
        first_e = encdec.epoch_now
    else:
        print("loadW2V")
        if os.path.exists(wordvec_model):
            encdec.loadW()
        else:
            print("wordvec model doesnt exists.")
      
    if args.gpu>=0:encdec.to_gpu() 
    
    optimizer = optimizers.Adam()
    optimizer.setup(encdec)
    for e_i in range(first_e,args.epoch):
        encdec.setEpochNow(e_i)
        loss_sum = 0
        tt_now = ([src_vocab.stoi(char) for char in char_arr] for char_arr in gens.word_list_rand(args.source))
        tt_gen = gens.batch(tt_now,args.batchsize)
        for tt in tt_gen:
            if len(tt)!=args.batchsize:
                print("len_tt:{}".format(len(tt)))
                continue
            loss = encdec(tt)
            loss_sum+=loss.data

            optimizer.target.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
        print("e{}:loss_sum:{}".format(e_i,loss_sum))
        model_name_save = "./{}/model/biconcatlstm_vae_kl_{}_{}_l{}.npz".format(dataname,dataname,e_i,args.n_latent)
        serializers.save_npz(model_name_save, encdec)
        
def test(args,dataname,epoch):
    if args.gpu>=0:
        import cupy as cp
        global xp;xp=cp
    model_name="./{}/model/biconcatlstm_vae_kl_{}_{}_l{}.npz".format(dataname,dataname,epoch,args.n_latent)
    print("model:{}".format(model_name))
    encdec = VaeLSTM(args)
    serializers.load_npz(model_name,encdec)
    if args.gpu>=0:encdec.to_gpu()
    else:encdec.to_cpu()
    src_vocab = Vocabulary.load('./{}/vocab_{}_l{}.bin'.format(dataname,dataname,args.n_latent))
    encdec.setVocab(src_vocab)
    print("xp:{}".format(xp))
    encdec.predict(args.batchsize,src_vocab,randFlag=False)
    return encdec

if __name__=="__main__":
    args = Args()
    #train(args)
    test(args ,"sent",32)

 

