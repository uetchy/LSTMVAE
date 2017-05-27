import chainer.links as L
import numpy
from chainer import cuda
from chainer import link
import numpy as np
from gensim.models import word2vec
from chainer import Variable,optimizers,serializers,Chain

class LSTM(L.NStepLSTM):

    def __init__(self,n_layers, in_size, out_size, dropout=0.5, use_cudnn=True):
        # n_layers = 1
        super(LSTM, self).__init__(n_layers, in_size, out_size, dropout, use_cudnn)
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
                xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype),
                volatile='auto')
        if self.cx is None:
            xp = self.xp
            self.cx = Variable(
                xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype),
                volatile='auto')
        hy, cy, ys = super(LSTM, self).__call__(self.hx, self.cx, xs, train)
        self.hx, self.cx = hy, cy
        return ys

def copy_model(src, dst):
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    for child in src.children():
        if child.name not in dst.__dict__: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        if isinstance(child, link.Chain):
            copy_model(child, dst_child)
        if isinstance(child, link.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print('Ignore %s because of parameter mismatch' % child.name)
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print('Copy %s' % child.name)

#gensimのwordvectorの重みをLSTMのembedに転移する
def transferWordVector(w2ind_post,ind2w_post,premodel_name):
    premodel = word2vec.Word2Vec.load(premodel_name).wv
    vocab = premodel.vocab
    sims = premodel.most_similar("医者",topn=5)
    error_count=0
    print("ind2len:"+str(len(ind2w_post)))
    for ind in range(len(ind2w_post)):
        try:
            vocab[ind2w_post[ind]]
        except:
            error_count+=1
    unk_ind = vocab["<unk>"]
    print("unk_ind:"+str(unk_ind))
    print("errcount:"+str(error_count))
    W = [premodel.syn0norm[vocab.get(ind2w_post[ind],unk_ind).index].tolist() for ind in range(len(ind2w_post))]
    return W            
            
def predictRandom(prob):
    probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
    probability /= np.sum(probability)
    index = np.random.choice(range(len(probability)), p=probability)
    return index