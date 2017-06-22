import sys
sys.path.append("../")
from model_cvae import CVAEHidden
from traintest import train,test,cosSimCategVec
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
    # categ_arr = [ri for ri in range(3,217)]
    categ_arr = [ri for ri in range(91,92)]
    sent_arr = ["皆さん も 見て 下さい"]
    # sent_arr = ["ミサトさん から は 何も 聞いてない ..."]
    sent_arr = ["ハロ 〜 ォ 、 ミサト ! 元気 してた ?","...... 知ってる んでしょ 、 私のこと も 、 みんな"]
    sent_arr = ["私","なのに あなたは 彼が 破 いた と言う","おかしく ありません か ?","他の も 見せて あげます よ","あなたは 他人に 乗り 移 る 特殊 能力 がある","それ を使って 毎回 カンニングし ている","そうですよね ?","あぁぁ ...","面倒 な ヤツだ な ...","今度は こっち","はい","あなたの 健康 診 断 書 の コピー","これ を見る 限り 、 そんな 病気 なー し !","さっ 、 テスト の問題 を解いて 下さい","我々の 学校に 転 入 して もらい ます"]
    sent_arr = ["ねぇ ? お姉ちゃん","ずっと いっしょに 遊んで たのに ―― ?","今 ... 何 時 ... ?","おはよう ...","途中で 寝 ちゃったの","そうだよね 自分のため にならない もんね ...","えっと ...","数学 が 23 ページ","世界 史 が 20 ページ","英 語 は 30 ページ で","お休み が 5 日 間","一日 5 時間 いや 、 4 時間 やる として","数学 1 時間 1 . 15 ページ","いけない 今日は 全然 してない や","えっと 残り 三日 だから","一日 6 時間は するか ...","あと 二日"]
    sent_arr = ["諸君 、 み ず から の道を 拓 く 為 、 難 民 の 為 の 政治 を手に入れる 為に 、 あと 一 息 、 諸君ら の力を 私に 貸して いただきたい 。 そして 私は 、 父 ジオンの も と に 召 される であろう","海 軍の 連中 は 、 船 の 数 が 合ってい れば 安心 するもの さ","ああ 、 アクシズ を 加速 するの にも 、 地球を汚染 させる にも な","α ・ アジール 、 牽 引","ようし 、 モビルスーツ 部隊 、 アクシズ を 偵察","そうか 、 来たか","了解 だ","意外 に 遅 かったな","第二波 が来た 。 モビルスーツ 部隊 は 機雷 源 の上に","これ にも 核ミサイル が 一発 だけ 。 やる な 、 ブライト","ギュネイ が 敵の 核ミサイル 群 を 阻止 してくれた 。 あれが 強化人間 の仕事 だ","暴力 は いけない な 。 ナナイ には 言っておく","クェス 、 パイロットスーツ も なし で","ああ 、 本当だ","調子に乗るな","実戦 の 恐 さ は 体験 しなかった ようだな","それで 、 私の 所 に来たの か","その 感じ 方 、 本物の ニュータイプ かもしれん 。 いい子 だ","第一 波 は 引き上げ た ようだ"]
    enc_chara = "アスカ";dec_chara="レイ"
    enc_chara = "友利奈緒";dec_chara="カミーユ"
    enc_chara = "つかさ";dec_chara="ルルーシュ"
    enc_chara = "シャア";dec_chara="クェス"
    for e_i in [28]:#[18,19,20]:
        model_name="./{}/model/cvaehidden_kl_{}_{}_l{}.npz".format(args.dataname,args.dataname,e_i,args.n_latent)
        encdec = CVAEHidden(args)
        encdec = test(args,encdec,model_name,categ_arr[:1])
        print("{}→{}".format(enc_chara,dec_chara))

        enc_cat=[encdec.categ_vocab.stoi(enc_chara)]
        dec_cat=[encdec.categ_vocab.stoi(dec_chara)]

        encdec.shiftStyle(sent_arr,enc_cat,dec_cat)
        cosSimCategVec(encdec)

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
