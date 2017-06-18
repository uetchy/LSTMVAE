import sys
sys.path.append("../")
from model_cvae import CVAEHidden
from traintest import train,test
import argparse

class Args():
    def __init__(self,train=True):
        dataname = "serif"
        self.dataname =dataname
        self.source ="./{}/all_{}16000_fixed.txt".format(dataname,dataname)
        self.category="./{}/all_chara.txt".format(dataname)

        self.epoch = 30
        self.n_vocab =  15628#15593
        self.embed = 300
        self.categ_size=221#101
        self.premodel="/home/fx30045/git/LibraryCommon/model/gensim3/w2v_raw_iki{}_e10_200000.w2v".format(self.embed)

        self.hidden= 300
        self.n_latent = 600#1200
        self.layer = 1
        self.batchsize=60
        self.sample_size = 10
        self.kl_zero_epoch = 15
        self.dropout = 0.5
        if train:
            self.gpu = 0
        else:
            self.gpu = -1
        self.gradclip = 5

def sampleTrain():
    args = Args(False)
    train(args)

def sampleTest():
    args = Args(False)
    categ_arr = [ri for ri in range(3,217)]
    # categ_arr = [ri for ri in range(3,17)]
    for e_i in [22]:#[18,19,20]:
        model_name="./{}/model/cvaehidden_kl_{}_{}_l{}.npz".format(args.dataname,args.dataname,e_i,args.n_latent)
        encdec = CVAEHidden(args)
        encdec = test(args,encdec,model_name,categ_arr)
        # cosSimCategVec(encdec)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--train",
                        help="train mode",
                        action="store_true")
    args = parser.parse_args()

    if args.train:
        sampleTrain()
    else:
        sampleTest()
