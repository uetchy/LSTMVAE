import chainer.functions as F
import numpy as np
xp = np
import util.generators as gens
import random
from chainer import Variable
import chainer.links as L

from model_common import VAECommon,LSTM

class VAE(VAECommon):

    def __init__(self,args):
        self.setArgs(args)
        super(VAE, self).__init__(
                embed = L.EmbedID(self.n_vocab,self.n_embed),
                #VAEenc
                enc_f = LSTM(self.n_layers,self.n_embed, self.out_size, dropout=self.drop_ratio),
                enc_b = LSTM(self.n_layers,self.n_embed, self.out_size, dropout=self.drop_ratio),

                le2_mu=L.Linear(4*self.out_size, self.n_latent),
                le2_ln_var=L.Linear(4*self.out_size, self.n_latent),
                #VAEdec
                ld_h = L.Linear(self.n_latent,2*self.out_size),
                ld_c = L.Linear(self.n_latent,2*self.out_size),

                dec = LSTM(self.n_layers,self.n_embed, 2*self.out_size, dropout=self.drop_ratio),
                h2w = L.Linear(2*self.out_size,self.n_vocab),
        )

    def setArgs(self,args):
        self.n_vocab = args.n_vocab
        self.n_embed = args.embed
        self.n_layers = args.layer
        self.n_latent = args.n_latent
        self.out_size = args.hidden
        self.sample_size= args.sample_size
        self.kl_zero_epoch = args.kl_zero_epoch
        self.drop_ratio = args.dropout

        self.setBatchSize(args.batchsize)
        self.setVocab(args)
        self.setMaxEpoch(args.epoch)
        self.setEpochNow(0)

    def getBatchGen(self,args):
        tt_now_list = [[self.vocab.stoi(char) for char in char_arr] for char_arr in gens.word_list(args.source)]
        ind_arr = list(range(len(tt_now_list)))
        random.shuffle(ind_arr)
        tt_now = (tt_now_list[ind] for ind in ind_arr)
        tt_gen = gens.batch(tt_now, args.batchsize)
        for tt in tt_gen:
            yield tt

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

    def decode(self,z,t_vec,t_pred):
        self.dec.hx = F.reshape(self.ld_h(z),(1,self.batch_size,2*self.out_size))#1,30,100
        self.dec.cx = F.reshape(self.ld_c(z),(1,self.batch_size,2*self.out_size))
        loss = super().decode(t_vec,t_pred)
        return loss

    def predict(self,batch,z=None,randFlag=True):
        if z is None:
            z = Variable(xp.random.normal(0,1,(batch,self.n_latent)).astype(xp.float32))
        self.dec.hx = F.reshape(self.ld_h(z),(1,batch,2*self.out_size))
        self.dec.cx = F.reshape(self.ld_c(z),(1,batch,2*self.out_size))
        super().predict(batch,randFlag)




