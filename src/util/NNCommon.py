from chainer import cuda
from chainer import link
import numpy as np
# from gensim.models import word2vec

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