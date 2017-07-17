from model_cvae import CVAEHidden
from chainer import links as L
import random
import util.generators as gens
from chainer import functions as F

#M1+M2
##損失関数の作り方
# ラベル有り
#   普通のCVAEの損失関数L(x,y)＋xからクラスを推定する損失関数q(y|x)
# ラベル無し
#   普通のCVAEの損失関数L(x,y)にq(y|x)かけたものをyで周辺化。+q(y|x)自体の条件付きエントロピー。

##バッチはどうするのか？ バッチサイズはそれぞれ異なりそう。

class CVAESemiSup(CVAEHidden):
    def __init__(self,args):
        self.setArgs(args)
        super().__init__(args)
        with self.init_scope():
            self.x2y = L.Linear(2*self.out_size,self.categ_size)

    def setArgs(self,args):
        super().setArgs(args)

    def __call__(self,tupl):
        #calcLabelLoss
        #L(x,y)
        xs_label = tupl[0];cat_label = tupl[1];
        print("cat_label:{}".format(cat_label))
        loss =self.calcLabelLoss(xs_label,cat_label)

        #calcUnlabelLoss
        xs_unlabel= tupl[2]
        loss+=self.calcUnlabelLoss(xs_unlabel)

        return loss


    def calcLoss(self,xs,cat,wei_arr=None):
        t = [[1] + x for x in xs]  # 1は<s>を指す。decには<s>から入れる。</s>まで予測する。
        mu_arr, var_arr = self.encode(xs, cat)

        categ_vec_dec_h = self.categ_dec_h(self.xp.array(cat, dtype=self.xp.int32))
        categ_vec_dec_c = self.categ_dec_c(self.xp.array(cat, dtype=self.xp.int32))
        loss = None
        for mu, var in zip(mu_arr, var_arr):
            if loss is None:
                loss = super().calcLoss(t, categ_vec_dec_h, categ_vec_dec_c, mu, var,wei_arr)
            else:
                loss += super().calcLoss(t, categ_vec_dec_h, categ_vec_dec_c, mu, var,wei_arr)
        return loss


    def calcUnlabelLoss(self,xs_unlabel):
        def calcEntropy(p):
            # p = F.softmax(self.xp.array(p,dtype=self.float32))
            log_p = F.log(p)
            # entropy = F.matmul(p,log_p,transb=True)
            entropy = F.sum(p*log_p,axis=1)
            print("entropy:{}".format(entropy.data))
            entropy = F.sum(entropy,axis=0)/entropy.data.shape[0]
            print("entropy:{}".format(entropy.data))
            return entropy

        no_categ2 = [self.categ_vocab.stoi("<unk>")]*len(xs_unlabel)
        self.calcLoss(xs_unlabel,no_categ2)
        h_concat = F.concat((self.enc_f.hx,self.enc_b.hx),axis=2)
        h_concat = F.reshape(h_concat,(h_concat.data.shape[1],h_concat.data.shape[2]))
        y_prob = F.softmax(self.x2y(h_concat))
        #H(y|x)
        loss=calcEntropy(y_prob)
        print("loss_shape:{}".format(loss.data.shape))
        # #q(y|x)*L(x,y)
        print("yprob:{}".format(y_prob.data.shape))
        y_prob_spl = F.split_axis(y_prob,y_prob.data.shape[1],axis=1)
        print("yprob_spl:{}".format(y_prob_spl[0].data.shape))

        for ci in range(self.categ_size):
            cat_unlabel = [ci]*len(xs_unlabel)
            loss_Lxy = self.calcLoss(xs_unlabel,cat_unlabel,wei_arr=y_prob_spl[ci])
            print("loss_Lxy:{}".format(loss_Lxy.data))#data.shape))
            # y_prob_ci = F.transpose(F.concat([y_prob_spl[ci] for ri in range(2*self.out_size)],axis=0))
            # loss+=y_prob_ci*loss_Lxy
            # print(y_prob_spl[ci].data.shape)
            # loss+=y_prob_spl[ci]*loss_Lxy
            loss+=loss_Lxy
        return loss


    def calcLabelLoss(self,xs_label,cat_label):
        print(self.categ_vocab.itos(cat_label[0][0]))
        loss = self.calcLoss(xs_label,cat_label)
        self.reset_state()
        #q(y|x)
        no_categ = [self.categ_vocab.stoi("<unk>")]*len(xs_label)
        self.calcLoss(xs_label,no_categ)
        # print("h_concat:{}".format(self.enc_f.hx.data.shape))
        h_concat = F.concat((self.enc_f.hx,self.enc_b.hx),axis=2)
        h_concat = F.reshape(h_concat,(h_concat.data.shape[1],h_concat.data.shape[2]))
        print("h_concat:{}".format(h_concat.data.shape))
        cat_label = [cat[0] for cat in cat_label]
        loss += F.softmax_cross_entropy(self.x2y(h_concat),self.xp.array(cat_label,dtype=self.xp.int32))
        self.reset_state()
        return loss

    def getBatchGen(self,args):
        def getBatchLabeled(args):
            return super().getBatchGen(args)

        def getBatchUnlabeled(args):
            tt_now_list = [[self.vocab.stoi(char) for char in char_arr] for char_arr in gens.word_list(args.unlabeled)]
            ind_arr = list(range(len(tt_now_list)))
            random.shuffle(ind_arr)
            tt_now = (tt_now_list[ind] for ind in ind_arr)
            tt_gen = gens.batch(tt_now, args.batchsize)
            for tt in tt_gen:
                yield tt

        # for lbl,unlbl in zip(getBatchLabeled(args),getBatchUnlabeled(args)):
        for lbl,unlbl in zip(super().getBatchGen(args),getBatchUnlabeled(args)):
            print("lbel:{}".format(lbl))
            yield (lbl[0],lbl[1],unlbl)

    def reset_state(self):
        self.enc_f.reset_state()
        self.enc_b.reset_state()
        self.dec.reset_state()

if __name__=="__main__":
    def fortest(r_len):
        for ri in range(r_len):
            yield ri

    def fortestSup(r_len):
        return fortest(r_len)

    def cross_entropy(t,p,weight_arr,weigh_flag=True):
        b = np.zeros(p.shape,dtype=np.float32)
        b[np.arange(p.shape[0]), t] = 1
        soft_arr = F.softmax(p)
        log_arr = -F.log(soft_arr)
        xent = b*log_arr
        xent = F.max(xent,axis=1)/p.shape[0]
        print("xent:{}".format(xent.data))
        if not weigh_flag:
            return F.sum(xent)
        wxent= F.matmul(weight_arr,xent,transa=True)
        print("wxent:{}".format(wxent.data))
        return wxent

    import numpy as np
    arr_ = np.array([[2,3,4,5,56],[2,3,4,5,56],[2,3,4,5,56]],dtype=np.float32)
    t = np.array([1,2,4],dtype=np.int32)
    loss = F.softmax_cross_entropy(arr_,t)
    print(loss.data)
    wei_arr = np.array([1,2,0],dtype=np.float32)
    ent = cross_entropy(t,arr_,wei_arr,False)
    xent = cross_entropy(t,arr_,wei_arr,True)




