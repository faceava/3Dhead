FROM python:3.7-slim
RUN ln -snf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo 'Asia/Shanghai' > /etc/timezone
# 已通过前一条命令设置时区，移除重复配置

WORKDIR /app
COPY . /app

# 暴露端口
EXPOSE 5000

# 设置环境变量
ENV FLASK_APP=3Dapp.py
ENV FLASK_ENV=production

RUN pip install -r requirements.txt -i http://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com

CMD ["python", "3Dapp.py"]