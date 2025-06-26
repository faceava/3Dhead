FROM python:3.7

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone

WORKDIR /app
ADD . /app

# 安装 OpenCV 运行依赖
RUN apt-get update && apt-get install -y libgl1

# 先安装除 mediapipe 以外的依赖
RUN pip install -r requirements.txt --no-deps --ignore-installed mediapipe -i http://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com

# 单独安装 mediapipe（用官方源）
RUN pip install mediapipe==0.8.9

EXPOSE 5000

CMD ["python", "3Dapp.py"]