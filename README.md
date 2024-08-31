# atmaCup17
## コンペティションURL
https://www.guruguru.science/competitions/24/

## 問題概要
洋服のレビューや衣服情報から、ユーザーが洋服をおすすめするかどうかを判定する。  
数値、カテゴリ、テキストが混ざった特徴量となっている。

## 取り組みの概要
レビューのテキストが最も重要と判断したため、Bertをファインチューニングして文書分類を行った。  
分類結果を数値として表に載せて、lightgbmで予測を行った。  
本githubのコードはDistilBertを使っているが、コンペで提出したファイルでは、Debertaを使っている。(Kaggle Notebook上で計算)

## 結果
Debertaで予測した結果のROC AUCが0.95程度であり、その特徴をlightgbmに入れて学習させたが、スコアは上昇しなかった。

## 環境構築
Docker V2で動作します。
### イメージのビルド
```
docker compose up --build -d
```
### アクセス
以下にアクセスすることで、JupyterLabが起動します。  
http://127.0.0.1:8888

### コンテナのシャットダウン
```
docker compose down
```

## 動かし方
1. fine tuningを行います。モデルは`/models`に保存されます。
```
cd src
python fine_tune.py
```
2. 上記で得られたモデルを使って、テキストの埋め込みを行います。結果は`/data/processed`に格納されます。
```
python embedding.py
```
3. lightgbmで学習して予測します。最終結果は`/data/processed/submission.csv`です。
```
python light_gbm.py --dir processed
```
ちなみに、`/data/deberta`は、モデルを`microsoft/deberta-v3-large`に変更して埋め込みを行った結果です。  
また、`/notebooks`はアドボックな分析を行ったものです。
