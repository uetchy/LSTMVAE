#coding:utf-8
from chainer import functions as F
from util.NNCommon import *

def testAdd(args,encdec,sent_arr,times=10):
    mu_arr,var_arr = vectorize(args,encdec,sent_arr)
    mu_arr = [mu for mu,var in zip(mu_arr,var_arr)]
    ratio = 1.0/times
    vec_arr = [mu_arr[1]]+[ri*ratio*mu_arr[0]+(1.0-ri*ratio)*mu_arr[1] for ri in range(times+1)]+[mu_arr[0]]
    if len(mu_arr)>1:
        encdec.predict(len(vec_arr),randFlag=False,z=F.reshape(F.concat(vec_arr),(len(vec_arr),args.n_latent)))
    encdec.dec.reset_state()

def vectorize(args,encdec,sent_arr):
    tt_batch = [[encdec.vocab.stoi(char) for char in word_arr.split(" ")] for word_arr in sent_arr]
    mu_arr,var_arr = encdec.encode(tt_batch)

    mu_arr = mu_arr[0]
    mu_arr = F.split_axis(mu_arr, len(sent_arr), axis=0)
    var_arr = var_arr[0]
    var_arr = F.split_axis(var_arr, len(sent_arr), axis=0)
    # print("cossim:{}".format(cosSim(mu_arr[0].data[0],mu_arr[1].data[0])))
    return mu_arr,var_arr


def cosSim(v1,v2):
    v1 = np.array(v1.tolist()); v2 = np.array(v2.tolist());
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def weighted_cross_entropy(p,t,weight_arr,sec_arr,weigh_flag=True):
    print("p:{}".format(p.data.shape))
    b = np.zeros(p.shape,dtype=np.float32)
    b[np.arange(p.shape[0]), t] = 1
    soft_arr = F.softmax(p)
    log_arr = -F.log(soft_arr)
    xent = b*log_arr

    #
    # print("sec_arr:{}".format(sec_arr))
    # print("xent_shape:{}".format(xent.data.shape))
    xent = F.split_axis(xent,sec_arr,axis=0)
    print([xent_e.data.shape[0] for xent_e in xent])
    x_sum = [F.reshape(F.sum(xent_e)/xent_e.data.shape[0],(1,1)) for xent_e in xent]
    # print("x_sum:{}".format([x_e.data for x_e in x_sum]))
    xent = F.concat(x_sum,axis=0)
    #
    # print("xent1:{}".format(xent.data))
    xent = F.max(xent,axis=1)/p.shape[0]
    # print("xent2:{}".format(xent.data))
    if not weigh_flag:
        return F.sum(xent)
    # print("wei_arr:{}".format(weight_arr))
    # print("wei_arr:{}".format(weight_arr.data.shape))

    print("xent3:{}".format(xent.data.shape))
    wxent= F.matmul(weight_arr,xent,transa=True)
    wxent = F.sum(F.sum(wxent,axis=0),axis=0)
    print("wxent:{}".format(wxent.data))
    return wxent