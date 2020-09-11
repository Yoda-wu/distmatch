#!/bin/bash
docker build -t etree-node:v1.0 .

sudo mkdir -p /etc/cni/net.d
sudo cp 10-flannel.conflist /etc/cni/net.d/10-flannel.conflist

k=v1.19.1
g=k8s.grc.io
a=registry.aliyuncs.com/google_containers
images=(kube-proxy:${k}
pause:3.2)
for i in ${images[@]} ; do
	docker pull ${a}/${i}
	docker tag ${a}/${i} ${g}/${i}
	docker rmi ${a}/${i}
done
docker pull quay.io/coreos/flannel:v0.12.0-amd64
docker images