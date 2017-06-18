from chainer import Chain,Variable
import chainer.links as L
import chainer.functions as F
from chainer import serializers,optimizers
import os,random
import numpy as np
import numpy as xp
#xp 問題をどうするか。
from util.vocabulary import Vocabulary
import util.generators as gens
from util.NNCommon import transferWordVector,predictRandom

class VAECommon(Chain):
    # args.n_vocab,args.layer,args.embed,args.hidden,args.n_latent
    def __init__(self,**args):
        super(VAECommon, self).__init__(**args)

    def setEpochNow(self, epoch_now):
        self.epoch_now = epoch_now

    def setMaxEpoch(self, epoch):
        self.epoch = epoch

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size

    def denoiseInput(self, t, noise_rate=0.4):  ###WordDropOut
        if noise_rate > 0.0:
            for t_i, t_e in enumerate(t):
                ind_arr = [t_i for t_i in range(1, min(len(t_e), 12))]
                random.shuffle(ind_arr)
                unk_ind_arr = ind_arr[:int(len(ind_arr) * noise_rate)]
                for unk_ind in unk_ind_arr: t[t_i][unk_ind] = self.vocab.stoi("<unk>")
        return t

    def setVocab(self, args):
        vocab_name = "./{}/vocab_{}.bin".format(args.dataname, args.dataname)
        if os.path.exists(vocab_name):
            src_vocab = Vocabulary.load(vocab_name)
        else:
            set_vocab = set()
            [[set_vocab.add(word) for word in word_arr] for word_arr in gens.word_list(args.source)]
            n_vocab = len(set_vocab) + 3
            print("n_vocab:{}".format(n_vocab))
            print("arg_vocab:{}".format(args.n_vocab))
            src_vocab = Vocabulary.new(gens.word_list(args.source), args.n_vocab)
            src_vocab.save(vocab_name)
        self.vocab = src_vocab
        return src_vocab

    def setCateg(self, args):
        categ_name = "./{}/categ_{}.bin".format(args.dataname, args.dataname)
        if os.path.exists(categ_name):
            categ_vocab = Vocabulary.load(categ_name)
        else:
            set_cat = set()
            [[set_cat.add(word) for word in word_arr] for word_arr in gens.word_list(args.category)]
            n_categ = len(set_cat) + 3
            print("n_categ:{}".format(n_categ))
            categ_vocab = Vocabulary.new(gens.word_list(args.category), n_categ)
            categ_vocab.save(categ_name)
        self.categ_vocab = categ_vocab
        return categ_vocab

    def loadW(self, premodel_name):
        src_vocab = self.vocab
        src_w2ind = {}
        src_ind2w = {}
        src_size = self.n_vocab
        print(src_size)
        for vi in range(src_size):
            src_ind2w[vi] = src_vocab.itos(vi)
            src_w2ind[src_ind2w[vi]] = vi
        print("pre:{}".format(self.embed.W.data[0][:5]))
        self.embed.W = Variable(xp.array(transferWordVector(src_w2ind, src_ind2w, premodel_name), dtype=xp.float32))
        print("pos:{}".format(self.embed.W.data[0][:5]))

    def makeEmbedBatch(self, xs, reverse=False):
        if reverse:
            xs = [xp.asarray(x[::-1], dtype=xp.int32) for x in xs]
        elif not reverse:
            # xs = xp.asarray(xs,dtype=xp.int32)
            xs = [xp.asarray(x, dtype=xp.int32) for x in xs]
        section_pre = np.array([len(x) for x in xs[:-1]], dtype=np.int32)
        sections = np.cumsum(section_pre)  # CuPy does not have cumsum()
        xs = F.split_axis(self.embed(F.concat(xs, axis=0)), sections, axis=0)
        return xs

    def loadModel(self,model_name_base,args):
        first_e = 0
        for e in range(args.epoch):
            model_name_tmp = model_name_base.format(args.dataname, args.dataname, e,args.n_latent)
            if os.path.exists(model_name_tmp):
                model_name = model_name_tmp
                self.setEpochNow(e + 1)

        if os.path.exists(model_name):
            print(model_name)
            # serializers.load_npz(model_name, encdec)
            serializers.load_npz(model_name, self)
            print("loaded_{}".format(model_name))
            first_e = self.epoch_now
        else:
            print("loadW2V")
            if os.path.exists(args.premodel):
                self.loadW(args.premodel)
            else:
                print("wordvec model doesnt exists.")
        return first_e

    def predict(self,batch,randFlag):
        t = [[bi] for bi in [1] * batch]
        t = self.makeEmbedBatch(t)

        ys_d = self.dec(t, train=False)
        ys_w = [self.h2w(y) for y in ys_d]
        name_arr_arr = []
        if randFlag:
            t = [predictRandom(F.softmax(y_each)) for y_each in ys_w]
        else:
            t = [y_each.data[-1].argmax(0) for y_each in ys_w]
        name_arr_arr.append(t)
        t = [self.embed(xp.array([t_each], dtype=xp.int32)) for t_each in t]
        count_len = 0
        while count_len < 50:
            ys_d = self.dec(t, train=False)
            ys_w = [self.h2w(y) for y in ys_d]
            if randFlag:
                t = [predictRandom(F.softmax(y_each)) for y_each in ys_w]
            else:
                t = [y_each.data[-1].argmax(0) for y_each in ys_w]
            name_arr = [self.vocab.itos(t_each) for t_each in t]
            name_arr_arr.append(t)
            # print("t:{}".format(name_arr))
            t = [self.embed(xp.array([t_each], dtype=xp.int32)) for t_each in t]
            count_len += 1
        tenti = xp.array(name_arr_arr).T
        for name in tenti:
            name = [self.vocab.itos(nint) for nint in name]
            if "</s>" in name:
                print("     Gen:{}".format("".join(name[:name.index("</s>")])))

    def encode(self,xs):
        xs = [x + [2] for x in xs]  # 1は<s>を指す。decには<s>から入れる。
        xs_f = self.makeEmbedBatch(xs)
        xs_b = self.makeEmbedBatch(xs, True)

        self.enc_f.reset_state()
        self.enc_b.reset_state()
        ys_f = self.enc_f(xs_f)
        ys_b = self.enc_b(xs_b)

        # VAE
        mu_arr = [self.le2_mu(F.concat((hx_f, cx_f, hx_b, cx_b))) for hx_f, cx_f, hx_b, cx_b in
                  zip(self.enc_f.hx, self.enc_f.cx, self.enc_b.hx, self.enc_b.cx)]
        var_arr = [self.le2_ln_var(F.concat((hx_f, cx_f, hx_b, cx_b))) for hx_f, cx_f, hx_b, cx_b in
                   zip(self.enc_f.hx, self.enc_f.cx, self.enc_b.hx, self.enc_b.cx)]
        return mu_arr,var_arr

    def decode(self,t_vec,t_pred):
        ys_d = self.dec(t_vec)
        ys_w = self.h2w(F.concat(ys_d, axis=0))
        t_all = []
        for t_each in t_pred: t_all += t_each.tolist()
        t_all = xp.array(t_all, dtype=xp.int32)
        loss = F.softmax_cross_entropy(ys_w, t_all)  # /len(t_all)
        print("t:{}".format([self.vocab.itos(tp_e) for tp_e in t_pred[0].tolist()]))
        print("y:{}\n".format([self.vocab.itos(int(ys_w.data[ri].argmax())) for ri in range(len(t_pred[0]))]))
        return loss

class LSTM(L.NStepLSTM):

    def __init__(self,n_layer, in_size, out_size, dropout=0.5):
        n_layers = 1
        super(LSTM, self).__init__(n_layers, in_size, out_size, dropout)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(LSTM, self).to_cpu()
        if self.cx is not None:
            self.cx.to_cpu()
        if self.hx is not None:
            self.hx.to_cpu()

    def to_gpu(self, device=None):
        super(LSTM, self).to_gpu(device)
        if self.cx is not None:
            self.cx.to_gpu(device)
        if self.hx is not None:
            self.hx.to_gpu(device)

    def set_state(self, cx, hx):
        assert isinstance(cx, Variable)
        assert isinstance(hx, Variable)
        cx_ = cx
        hx_ = hx
        if self.xp == np:
            cx_.to_cpu()
            hx_.to_cpu()
        else:
            cx_.to_gpu()
            hx_.to_gpu()
        self.cx = cx_
        self.hx = hx_

    def reset_state(self):
        self.cx = self.hx = None

    def __call__(self, xs, train=True):
        batch = len(xs)
        if self.hx is None:
            xp = self.xp
            self.hx = Variable(
                    xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype))
        if self.cx is None:
            xp = self.xp
            self.cx = Variable(
                    xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype))

        # hy, cy, ys = super(LSTM, self).__call__(self.hx, self.cx, xs, train)
        hy, cy, ys = super(LSTM, self).__call__(self.hx, self.cx, xs)
        self.hx, self.cx = hy, cy
        return ys



