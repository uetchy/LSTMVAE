from chainer import serializers, optimizers
from calc_vector import cosSim
import numpy as xp
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def train(args,
          encdec,
          model_name_base="./{}/model/cvaehidden_kl_{}_{}_l{}.npz"):
    encdec.loadModel(model_name_base, args)
    if args.gpu >= 0:
        import cupy as cp
        global xp
        xp = cp
        encdec.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(encdec)
    for e_i in range(encdec.epoch_now, args.epoch):
        encdec.setEpochNow(e_i)
        loss_sum = 0
        for tupl in encdec.getBatchGen(args):
            loss = encdec(tupl)
            loss_sum += loss.data

            encdec.cleargrads()
            loss.backward()
            optimizer.update()
        print("epoch{}:loss_sum:{}".format(e_i, loss_sum))
        model_name = model_name_base.format(args.dataname, args.dataname, e_i,
                                            args.n_latent)
        serializers.save_npz(model_name, encdec)


def test(args, encdec, model_name, categ_arr=[], predictFlag=False):
    serializers.load_npz(model_name, encdec)
    if args.gpu >= 0:
        import cupy as cp
        global xp
        xp = cp
        encdec.to_gpu()
    encdec.setBatchSize(args.batchsize)

    if "cvae" in model_name:
        for categ in categ_arr:
            print("categ:{}".format(encdec.categ_vocab.itos(categ)))
            if predictFlag:
                encdec.predict(args.batchsize, tag=categ, randFlag=False)
    elif predictFlag:
        encdec.predict(args.batchsize, randFlag=False)
    return encdec


def cosSimCategVec(model, categ_arr=range(3, 101)):
    categ_hash = {
        ri: model.categ_dec_h(xp.array([ri], dtype=xp.int32))
        for ri in categ_arr
    }
    cossim_hash = {}
    for ri in categ_arr:
        for rj in range(ri + 1, categ_arr[-1]):
            name1 = model.categ_vocab.itos(ri)
            name2 = model.categ_vocab.itos(rj)
            cossim = cosSim(categ_hash[ri][0].data, categ_hash[rj][0].data)
            #print("{}:{}:{}".format(name1,name2,cossim))
            cossim_hash[name1 + ":" + name2] = cossim
    cossim_tupl = sorted(cossim_hash.items(), key=lambda x: x[1])
    for tupl in cossim_tupl[::-1][:200]:
        print(tupl)
