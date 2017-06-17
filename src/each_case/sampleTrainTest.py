import sys
sys.path.append("../")
from BiVarConcatLSTM import train,test
from AddVaeConcat import testAdd

import argparse

class Args():
    def __init__(self,dataname,train=True):
        self.source ="./{}/prof16000fixed_longcut.txt".format(dataname)
        self.epoch = 40
        self.n_vocab = 16940#16640
        self.embed = 300
        self.hidden= 300#600
        self.n_latent = 200#1200
        self.layer = 1
        self.batchsize=10
        self.sample_size = 10
        self.kl_zero_epoch = 15
        self.dropout = 0.5
        if train:
            self.gpu = 0
        else:
            self.gpu = -1
        self.gradclip = 5

def testSentAdd(args,encdec):
    sent_arr = []
    sent_arr.append("イ リ ヤ の クラスの 先生 。 破天荒 な性格 は 原作 同様 。 士 郎 とも面識があ るよう で 、 好意を寄せ ている 節 も 見られる 。")
    sent_arr.append("イ リ ヤ のクラスメート 。 眼 が 細 く 、 能天気 そうな 外見 をした 少女 。 頭の回転 は 速い らしく 、 あまり 動じない 性格 。 龍 子 に対する 突っ込み 役 で 、 過剰な ツッコミ も多い 。")
    testAdd(args,encdec,sent_arr)
    
    sent_arr = []
    sent_arr.append("プラズマ ジ カ で ヴォ ー カル 、 ギター を担当 。 伝説の 名 楽器 と噂され る 「 スト ロ ベリー ハート 」 と言う ハート 型の ギター を愛用している 。")
    sent_arr.append("プラズマ ジ カ で ヴォ ー カル 、 ベース を担当 。 普段は クール に 装 っているが 実は 恥ずかしがり屋 な性格")
    testAdd(args,encdec,sent_arr)
    
    sent_arr = []                 
    sent_arr.append("のび太 と 静 香 の間に できた 一人 息子 。 知能 や 容姿は のび太 に似て おり 、 髪の色 、 体力 や 運動神経 については 静 香 に似ている 。 のび太 と 比べ 活発 な ためか のび太 や そ の子孫 である セ ワシ に 比べ 肌 の 色 が 少し 黒い 。")
    sent_arr.append("のび太 の ママ 。 旧姓 は 片 岡 。 やや 厳しい 専業主婦 。 極度の 動物 嫌い で のび太 が 犬 や 猫 を拾って き ても すぐに 発見 して あ の セリフ である 。 幼い頃 は非常に 勝ち 気 な 女の子 で 0 点 も 取 っていた ようである 。")
    testAdd(args,encdec,sent_arr)
    

def sampleTrain():
    dataname= "prof"
    args = Args(dataname,False)
    train(args,dataname)

def sampleTest():
    dataname= "prof"
    args = Args(dataname,False)
    for ei in [29]:
        encdec = test(args,dataname,ei)
        testSentAdd(args,encdec)


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
