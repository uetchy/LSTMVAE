import chainer
from chainer import Chain,Variable
import chainer.links as L
import chainer.functions as F
import numpy as np
import numpy as xp
import random,os
from chainer import serializers,optimizers
from util.vocabulary import Vocabulary
import util.generators as gens
from model_common import VAECommon,LSTM
from calc_vector import cosSim

class CVAEHidden(VAECommon):
    # args.n_vocab,args.layer,args.embed,args.hidden,args.n_latent
    def __init__(self,args):
        self.setArgs(args)
        super(CVAEHidden, self).__init__(
                categ_enc_b_h = L.EmbedID(self.categ_size,self.out_size),
                categ_enc_b_c = L.EmbedID(self.categ_size,self.out_size),
                categ_enc_f_h = L.EmbedID(self.categ_size,self.out_size),
                categ_enc_f_c = L.EmbedID(self.categ_size,self.out_size),
                categ_dec_h   = L.EmbedID(self.categ_size,2*self.out_size),
                categ_dec_c   = L.EmbedID(self.categ_size,2*self.out_size),
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
        self.categ_size= args.categ_size
        self.setCateg(args)
        if args.gpu>=0:
            global xp
            import cupy as xp
        super().setArgs(args)


    def getBatchGen(self,args):
        tt_now_list = [[self.vocab.stoi(char) for char in char_arr] for char_arr in gens.word_list(args.source)]
        cat_now_list = [[self.categ_vocab.stoi(cat[0])] for cat in gens.word_list(args.category)]
        ind_arr = list(range(len(tt_now_list)))
        random.shuffle(ind_arr)
        tt_now = (tt_now_list[ind] for ind in ind_arr)
        cat_now = (cat_now_list[ind] for ind in ind_arr)
        tt_gen = gens.batch(tt_now, args.batchsize)
        cat_gen = gens.batch(cat_now, args.batchsize)
        for tt,cat in zip(tt_gen,cat_gen):
            yield (tt,cat)

    # def encode(self, xs, categ_vec_f_h, categ_vec_f_c, categ_vec_b_h, categ_vec_b_c):
    def encode(self, xs, cat):
        self.enc_f.hx = self.categ_enc_f_h(xp.array(cat, dtype=xp.int32))
        self.enc_f.cx = self.categ_enc_f_c(xp.array(cat, dtype=xp.int32))
        self.enc_b.hx = self.categ_enc_b_h(xp.array(cat, dtype=xp.int32))
        self.enc_b.cx = self.categ_enc_b_c(xp.array(cat, dtype=xp.int32))
        mu_arr,var_arr = super().encode(xs)
        return mu_arr, var_arr

    def __call__(self,tupl):
        xs = tupl[0];cat = tupl[1]
        print(self.categ_vocab.itos(cat[0][0]))
        mu_arr, var_arr = self.encode(xs, cat)

        categ_vec_dec_h = self.categ_dec_h(xp.array(cat, dtype=xp.int32))
        categ_vec_dec_c = self.categ_dec_c(xp.array(cat, dtype=xp.int32))

        t = [[1] + x for x in xs]  # 1は<s>を指す。decには<s>から入れる。</s>まで予測する。
        loss = None
        for mu, var in zip(mu_arr, var_arr):
            if loss is None:
                loss = self.calcLoss(t, categ_vec_dec_h, categ_vec_dec_c, mu, var)
            else:
                loss += self.calcLoss(t, categ_vec_dec_h, categ_vec_dec_c, mu, var)
        return loss

    def calcLoss(self, t, categ_vec_h, categ_vec_c, mu, ln_var):
        k = self.sample_size;
        kl_zero_epoch = self.kl_zero_epoch
        loss = None
        t_pred = [t_e[1:] + [2] for t_e in t]
        t_pred = [xp.asarray(tp_e, dtype=xp.int32) for tp_e in t_pred]
        t = self.denoiseInput(t)
        t_vec = self.makeEmbedBatch(t)

        for l in range(k):
            z = F.gaussian(mu, ln_var)
            if loss is None:
                loss = self.decode(z, categ_vec_h, categ_vec_c, t_vec, t_pred) / (k * self.batch_size)
            elif loss is not None:
                loss += self.decode(z, categ_vec_h, categ_vec_c, t_vec, t_pred) / (k * self.batch_size)
        C = 0.005 * (self.epoch_now - kl_zero_epoch) / self.epoch  # 0.02
        if self.epoch_now > kl_zero_epoch: loss += C * F.gaussian_kl_divergence(mu, ln_var) / self.batch_size
        return loss

    def decode(self, z, categ_vec_h, categ_vec_c, t_vec, t_pred):
        categ_vec_h = F.reshape(categ_vec_h, (1, self.batch_size, 2 * self.out_size))
        categ_vec_c = F.reshape(categ_vec_c, (1, self.batch_size, 2 * self.out_size))

        self.dec.hx = categ_vec_h + F.reshape(self.ld_h(z), (1, self.batch_size, 2 * self.out_size))  # 1,30,100
        self.dec.cx = categ_vec_c + F.reshape(self.ld_c(z), (1, self.batch_size, 2 * self.out_size))
        loss = super().decode(t_vec,t_pred)
        return loss

    def predict(self, batch, tag=1, randFlag=True, z=None):
        categ_vec_h = self.categ_dec_h(xp.array([[tag for i in range(batch)]], dtype=xp.int32))
        categ_vec_c = self.categ_dec_c(xp.array([[tag for i in range(batch)]], dtype=xp.int32))
        if z is None:
            z = Variable(xp.random.normal(0, 1, (batch, self.n_latent)).astype(xp.float32))
        self.dec.hx = categ_vec_h + F.reshape(self.ld_h(z), (1, self.batch_size, 2 * self.out_size))  # 1,30,100
        self.dec.cx = categ_vec_c + F.reshape(self.ld_c(z), (1, self.batch_size, 2 * self.out_size))
        super().predict(batch,randFlag)

    def shiftStyle(self,sent_arr,enc_tag,dec_tag,randFlag=False):
        xs = [[self.vocab.stoi(char) for char in word_arr.split(" ")] for word_arr in sent_arr]
        mu_arr,var_arr = self.encode(xs,enc_tag)
        categ_vec_h = F.reshape(self.categ_dec_h(xp.array([dec_tag for i in range(len(xs))], dtype=xp.int32)),(1, len(xs), 2 * self.out_size))
        categ_vec_c = F.reshape(self.categ_dec_c(xp.array([dec_tag for i in range(len(xs))], dtype=xp.int32)),(1, len(xs), 2 * self.out_size))
        self.dec.hx = categ_vec_h + F.reshape(self.ld_h(mu_arr[0]), (1, len(xs), 2 * self.out_size))  # 1,30,100
        self.dec.cx = categ_vec_c + F.reshape(self.ld_c(mu_arr[0]), (1, len(xs), 2 * self.out_size))
        super().predict(len(xs),randFlag)

