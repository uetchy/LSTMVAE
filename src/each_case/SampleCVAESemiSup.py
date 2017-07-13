import sys
sys.path.append("../")
from model_semisup import CVAESemiSup
from traintest import train,test,cosSimCategVec
import argparse

class Args():
    def __init__(self,train=True):
        dataname = "serif_dummy"
        self.dataname =dataname
        self.source ="./{}/all_serif16000_fixed.txt".format(dataname)
        self.category="./{}/all_chara.txt".format(dataname)
        self.unlabeled="./{}/all_serif16000_fixed.txt".format(dataname)

        self.epoch = 30
        self.n_vocab =  156#28
        self.embed = 300
        self.categ_size=22#1
        #points directory to transfer gensim w2v model
        self.premodel=""

        self.hidden= 300
        self.n_latent = 600
        self.layer = 1
        self.batchsize=3
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
    model_name_base="./{}/model/cvae_semisup_{}_{}_l{}.npz"
    encdec = CVAESemiSup(args)
    train(args,encdec,model_name_base)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--train",
                        help="train mode",
                        action="store_true")
    args = parser.parse_args()

    sampleTrain()
    # if args.train:
    #     pass
    # else:
    #     sampleTestGen()
    #     sampleTestTransfer()
