#coding:utf-8
from chainer import serializers
from chainer import functions as F

from util.NNCommon import *
import util.generators as gens
from util.vocabulary import Vocabulary

def loadModel(args,model_name="./model/biconcatlstm_vae_kl_sent_38_l1200.npz"):
    encdec = VaeLSTM(args.n_vocab,args.layer,args.embed,args.hidden,args.n_latent,drop_ratio=0.0)
    serializers.load_npz(model_name,encdec)
    vocab = Vocabulary.load("vocab_sent_l1200.bin")
    args.batchsize=2
    
    encdec.setBatchSize(args.batchsize)
    encdec.setVocab(vocab)
    return encdec

def testAdd(args,encdec,sent_arr,times=10):#"lstm_vae_model_14.npz"
    mu_arr,var_arr = vectorize(args,encdec,sent_arr)
    #mu_arr = [F.gaussian(mu,xp.zeros(mu.data.shape,dtype=xp.float32)) for mu,var in zip(mu_arr,var_arr)]
    #mu_arr = [Variable(xp.random.normal(mu.data,0,(1,1200)).astype(xp.float32)) for mu,var in zip(mu_arr,var_arr)]
    mu_arr = [mu for mu,var in zip(mu_arr,var_arr)]
    ratio = 1.0/times
    vec_arr = [mu_arr[1]]+[ri*ratio*mu_arr[0]+(1.0-ri*ratio)*mu_arr[1] for ri in range(times+1)]+[mu_arr[0]]
    #print("vec_len:{}".format(len(vec_arr)))
    if len(mu_arr)>1:
        encdec.predict(len(vec_arr),encdec.vocab,randFlag=False,z=F.reshape(F.concat(vec_arr),(len(vec_arr),args.n_latent)))
    encdec.dec.reset_state()

def vectorize(args,encdec,sent_arr):
    tt_batch = [[encdec.vocab.stoi(char) for char in word_arr.split(" ")] for word_arr in sent_arr]
    mu_arr,var_arr = encdec.encode(tt_batch)

    mu_arr = mu_arr[0]
    mu_arr = F.split_axis(mu_arr, len(sent_arr), axis=0)
    var_arr = var_arr[0]
    var_arr = F.split_axis(var_arr, len(sent_arr), axis=0)
    print("cossim:{}".format(cosSim(mu_arr[0].data[0],mu_arr[1].data[0])))
    return mu_arr,var_arr
    
def cosSim(v1,v2):
    v1 = np.array(v1.tolist()); v2 = np.array(v2.tolist());
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    