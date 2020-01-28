B4M1勉強会 - High Performance Programming
====

概要
----
MICやGPUを最大限利用した行列演算を作成し、競ってください。

内容
----
このテストプログラムをリンクし、測定を行います。
その際に、MICとGPU向けの設定項目が存在するので注意してください。(コンパイラが異なるため)

準備・はじめてのビルド
----
まず[オリジナルのリポジトリ](https://gitlab.eidos.ic.i.u-tokyo.ac.jp/benkyokai-maintainers/mm)からforkします。
その後以下のコマンドを入力してください。
```
git clone --recursive git@gitlab.eidos.ic.i.u-tokyo.ac.jp:(Your Name)/mm.git
make
./main.bin --version
./main.bin --help
```

最も簡単な使い方
----
* 適当なx86やamd64のマシン(自分のラップトップなど)でg++によりビルドする場合
  1. srcの中にある、mymulmat.cppを変更する
  2. このREADME.mdがあるディレクトリに移動して、`make`コマンドを実行
  3. 生成されるmain.binを実行する
  4. 問題セットを変更したい場合は、data/list-intel.txtを変更する。

* MIC向けにビルドする場合
  1. include.mkを開き、PLATFORMの項目をXEONPHIに変更する
  2. srcの中にある、mymulmat.cppを変更する
  3. このREADME.mdがあるディレクトリに移動して、`make`コマンドを実行
  4. 出来上がったmain.bin、及びdataディレクトリをmicに転送し、実行する。
  5. 問題セットを変更したい場合は、data/list-intel.txtを変更する。

* GPU向けにビルドする場合
  1. include.mkを開き、PLATFORMの項目をCUDAに変更する
  2. srcの中にある、mymulmat.cppとcuda.cuを変更する
  3. このREADME.mdがあるディレクトリに移動して、`make`コマンドを実行
  4. 問題セットを変更したい場合は、data/list-intel.txtを変更する。

また、main.binを始めとする副生成物を削除したい場合は、
```
make clean
```
を実行してください。

include.mkを変更したり、ライブラリが変更されたと先輩から言われたら
----
このReadme.mdがあるディレクトリで以下のコマンドを入力し、ライブラリもまとめてアップデートしてください。
```
cd mm-core
git pull
cd ..
make full_clean
```

MPIを利用する際の注意
----
MPIを利用する際には、include.mkに以下の変更を加える必要があります。
1. USEMPIオプションを1にする
2. 利用するプラットフォームのCXXを変更する
   * MYLOCALであれば、g++からmpic++に変更する
   * XEONPHIであれば、iccからmpiiccに変更する

プログラムの全体の実行の流れとしては以下のようになるので注意してください。
1. [ALL]   MPI::Init()
2. [ALL]   mymulmatのコンストラクタ
3. [ALL]   mymulmat.init n,m,kは全てのプロセスに同じものが渡されます。
4. [RANK0] A, Bに値がセットされる
5. [ALL]   mymulmat.multiply A, Bのブロードキャストは各自で実行してください。
6. [RANK0] Cをテスト
7. [ALL]   mymulmatのデストラクタ
8. [ALL]   MPI::Finalize()

オプション
----
* `./main.bin` : リスト中の<free>の中からランダムにファイルを取ってきて実行
* `./main.bin -h` : ヘルプ
* `./main.bin -v` : バージョン
* `./main.bin -t type` : タイプを指定して実行。typeとしてはfree, mv(mat-vec), trmm(Aが上三角), symm(Aが対称), square(正方)がある。

リポジトリ管理方法
----
Forkを行い、自分のユーザーでリポジトリをかんりするといい感じです。

テスト
----
適当なデータセットでプログラムを走らせ、WRONGが0であればそのデータセットにおいてテストが通っていると言えます。

競い方
----
以下の2点を総合的に評価します。
1. 最大効率を発揮（1ノードあたり）
2. 最大FLOPSを発揮（複数ノードで）

最大FLOPSの場合ノードの無駄遣いにご注意ください。
スケールしなければ、多くとも100ノード程度に抑えましょう。

データサイズ・データの型はfloat/double/だいたい正方行列系であれば、メモリに乗る限り自由です。
データの作り方は、`mm-answer-generator`を参照。

過去のスライドもあるので、見てもいいし、見なくてもいいです。その辺は競技者同士で話し合って決めてください。
頑張ってください！

結果の参照
----
mm-result-senderをつかって結果ファイル(result.dat)を送信できます。
実装は少しお待ちください。

作者
----
Makoto Shimazu <shimazu@eidos.ic.i.u-tokyo.ac.jp>
