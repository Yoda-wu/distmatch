FROM tensorflow/tensorflow:2.9.1
RUN apt update && apt install apt iproute2 net-tools iperf3 iputils-ping nano -y
COPY dml_req.txt /home
WORKDIR /home
RUN pip3 install --upgrade pip
RUN pip3 install -r dml_req.txt
CMD ["/bin/bash"]
