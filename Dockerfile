FROM python:latest

WORKDIR /app

COPY requirements.txt /app/
RUN pip3 install -r requirements.txt

# 作業ディレクトリの設定
WORKDIR /src/notebooks

# ポートの公開
EXPOSE 8888
