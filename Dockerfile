FROM centos
COPY --from=centos  /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN echo "Asia/Shanghai" > /etc/timezone

FROM python:3.7
# 使用官方提供的 Python 开发镜像作为基础镜像


WORKDIR /app
ADD . /app

RUN pip install -r requirements.txt -i http://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com

# 暴露端口
EXPOSE 5000

CMD ["python", "3Dapp.py"]