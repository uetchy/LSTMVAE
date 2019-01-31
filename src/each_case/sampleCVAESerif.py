import sys
sys.path.append("../")
from model_cvae import CVAEHidden
from traintest import train, test, cosSimCategVec
import argparse


class Args():

    def __init__(self, train=True):
        dataname = "serif"
        self.dataname = dataname
        self.source = "./{}/all_{}16000_fixed.txt".format(dataname, dataname)
        self.category = "./{}/all_chara.txt".format(dataname)

        self.epoch = 30
        self.n_vocab = 15628
        self.embed = 300
        self.categ_size = 221
        #points directory to transfer gensim w2v model
        self.premodel = ""

        self.hidden = 300
        self.n_latent = 600
        self.layer = 1
        self.batchsize = 60
        self.sample_size = 10
        self.kl_zero_epoch = 15
        self.dropout = 0.5
        if train:
            self.gpu = 0
        else:
            self.gpu = -1
        self.gradclip = 5


def sampleTrain():
    args = Args(True)
    model_name_base = "./{}/model/cvaehidden_kl_{}_{}_l{}.npz"
    encdec = CVAEHidden(args)
    train(args, encdec, model_name_base)


def sampleTestGen():
    args = Args(False)
    categ_arr = [16, 24, 25, 30, 31, 37, 40, 43, 46, 53, 55, 57]
    for e_i in [28]:
        model_name = "./{}/model/cvaehidden_kl_{}_{}_l{}.npz".format(
            args.dataname, args.dataname, e_i, args.n_latent)
        encdec = CVAEHidden(args)
        encdec = test(args, encdec, model_name, categ_arr, predictFlag=True)


def sampleTestTransfer():
    args = Args(False)
    categ_arr = [ri for ri in range(91, 92)]
    sent_arr_arr = [["皆さん も 見て 下さい"]]
    sent_arr_arr.append(["ハロ 〜 ォ 、 ミサト ! 元気 してた ?"])
    sent_arr_arr.append(["ギュネイ が 敵の 核ミサイル 群 を 阻止 してくれた 。 あれが 強化人間 の仕事 だ"])
    enc_dec_tupls = [("友利奈緒", "イカ娘"), ("アスカ", "シンジ"), ("シャア", "クェス")]
    for e_i in [28]:
        model_name = "./{}/model/cvaehidden_kl_{}_{}_l{}.npz".format(
            args.dataname, args.dataname, e_i, args.n_latent)
        encdec = CVAEHidden(args)
        encdec = test(
            args, encdec, model_name, categ_arr[:1], predictFlag=False)
        for (enc_chara, dec_chara), sent_arr in zip(enc_dec_tupls,
                                                    sent_arr_arr):
            print("{}→{}".format(enc_chara, dec_chara))
            enc_cat = [encdec.categ_vocab.stoi(enc_chara)]
            dec_cat = [encdec.categ_vocab.stoi(dec_chara)]
            encdec.shiftStyle(sent_arr, enc_cat, dec_cat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--train", help="train mode", action="store_true")
    args = parser.parse_args()

    if args.train:
        sampleTrain()
    else:
        sampleTestGen()
        sampleTestTransfer()
