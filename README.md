### Overview

EdgeTB is a hybrid testbed for distributed machine learning at the edge. It allows using Docker containers and physical
nodes to create hybrid test environments. On the one hand, the existence of physical nodes improves the computing
fidelity and network fidelity of EdgeTB, making it close to the physical testbed. On the other hand, compared with the
physical testbed, the adoption of emulators makes it easier for EdgeTB to generate large-scale and network-flexible test
environments.

### Installation

1. At least 2 computing devices, one acts as Controller, and others act as Workers (as physical nodes, class:
   PhysicalNode or as emulator, class:Emulator).
2. Software requirement for computing devices.

| Computing devices     | Requirement                                                  |
|-----------------------|--------------------------------------------------------------|
| Controller            | python3, python3-pip, NFS-Server                             |
| Worker (PhysicalNode) | python3, python3-pip, NFS-Client, iproute (iproute2)         |
| Worker (Emulator)     | python3, python3-pip, NFS-Client, iproute (iproute2), Docker |

3. Copy ```controller``` into Controller and install the python packages defined in ```controller/ctl_req.txt```.
4. Copy ```worker``` into Worker and install the python packages defined in ```worker/agent_req.txt```.

### File structure

```
controller
   ├─ dml_app  >>  Where we prepare roles, static files, shared by NFS
      ├─ dml_req.txt  >>  Role's execution environment
      ├─ Dockerfile  >>  Role's execution environment
      ├─ gl_peer.py  >>  Role's functions, an example
      ├─ nns  >>  Neural networks
      ├─ dml_utils.py
      └─ worker_utils.py
   ├─ dml_file  >>  Dynamically generated files for each node, transmitted over the network
      ├─ conf  >>  Generated by dml_tool/*_conf.py before running test, send to each node
      └─ log  >>  Received from each node
   ├─ dataset  >>  Splitted dataset, static files, shared by NFS
   ├─ dml_tool
      ├─ gl_dataset.json  >>  Dataset definition of all Gossip peer nodes, an example
      ├─ gl_structure.json  >>  Structure definition of all Gossip peer nodes, an example
      ├─ dataset_conf.py  >>  Used to generate dataset conf file for each node
      ├─ gl_structure_conf.py  >>  Used to generate structure conf file for each Gossip peer  node
      ├─ conf_utils.py
      ├─ splitter_utils.py
      └─ splitter_fashion_mnist.py  >>  Used to download and/or split dataset, an example
   ├─ gl_manager.py  >>  Runtime manager, an example
   ├─ gl_run.py  >>  Test environment definition, an example
   └─ links.json  >>  Network links definition
   
worker
   ├─ agent.py  >>  Used to communicate with controller/*_run.py
   ├─ dml_app  >>  mount point of controller/dml_app, over NFS
   ├─ dml_file
      ├─ conf  >>  Received from controller
      └─ log  >>  Generated by each node while running test, send to controller
   └─ dataset  >>  mount point of controller/dataset, over NFS
```

### Usage

#### Workflow overview

Prepare roles, neural networks, dataset >> Define test environment >> Run it >>  Collect result.

#### Workflow detail

1. The only things you need to do in Worker is to run ```worker/agent.py``` with python3 with root privileges. We need
   to mount NFS and install python packages via ```python3-pip```, which require root privileges.
2. All the following operations should be completed in the Controller.
3. Prepare roles, just like what ```controller/dml_app/gl_peer.py``` does.
4. Prepare neural network model, just like what ```controller/dml_app/nns/nn_fashion_mnist.py``` does.
5. Prepare datasets and split it, just like what ```controller/dml_tool/nn_fashion_mnist.py``` does.
6. Update ```controller/dml_app/Dockerfile``` and ```controller/dml_app/dml_req.txt``` to meet your DML.
7. Prepare test environment ```controller/run.py```, just like what ```controller/gl_run.py``` does.
8. Prepare network ```controller/links.json```.
9. Prepare Runtime Manager, just like what ```controller/gl_manager.py``` does.
10. Run ```controller/run.py``` with python3 with root privileges and keep it running on a terminal (called Term).
11. It takes a while to deploy the tc settings, so please set your DML to start running after receiving a certain
    message, such as receiving a ```GET``` request for ```/start```.
12. Wait until Term displays ```tc finish```, and then start your DML.
13. Clear the test environment.

### Examples

#### Gossip Learning

1. Same with above 1-4.
2. Just use the ```controller/dml_app/gl_peer.py```, ```controller/dml_app/Dockerfile```,
   and ```controller/dml_app/dml_req.txt```
3. Modify ```controller/gl_run.py```  to define the test environment.
4. Modify ```controller/linlks.json```  to define the network.
5. Modify ```controller/dml_tool/gl_dataset.json``` to define the data used by each node and
   modify ```controller/dml_tool/gl_structure.json``` to define the DML structure of each node,
   see ```controller/dml_tool/README.md``` for more.
6. Run ```controller/gl_run.py``` with python3 with root privileges and keep it running on a terminal (called Term).
7. In path ```controller/dml_tool```, type ```python3 dataset_conf.py -d gl_dataset.json``` in terminal to generate
   dataset conf files and type ```python3 gl_structure_conf.py -s gl_structure.json```  generate DML structure conf
   files.
8. Type ```curl localhost:3333/conf/dataset``` in a terminal to send those dataset conf files to each node. Wait until
   all nodes have received the dataset conf file. This function is defined in ```controller/base/manager.py```.
9. Type ```curl localhost:3333/conf/structure``` to send those DML structure conf files to each node. Wait until all
   nodes have received the structure conf file. This function is defined in ```controller/base/manager.py```.
10. Wait until Term displays ```tc finish```.
11. Type ```curl localhost:3333/start``` in a terminal to start all nodes. This function is defined
    in ```controller/base/manager.py``` and ```controller/gl_manager.py```.
12. When there is no node _Gossip_, type ```curl localhost:3333/finish``` in a terminal to stop all nodes and collect
    result files. This function is defined in ```controller/base/manager.py``` and ```controller/gl_manager.py```.
13. Commands such as ```curl localhost:3333/emulated/reset``` and ```curl localhost:3333/physical/reset```are used to
    remove all the emulated nodes and physical nodes. These functions are defined in ```controller/base/manager```,
    and ```worker/agent.py```.

#### Federated Learning

1. Same with above 1-4.
2. Just use the ```controller/dml_app/fl_trainer.py```, ```controller/dml_app/fl_aggregator.py```,
   ```controller/dml_app/Dockerfile``` and ```controller/dml_app/dml_req.txt```
3. Modify ```controller/fl_run.py```  to define the test environment.
4. Modify ```controller/linlks.json```  to define the network.
5. Modify ```controller/dml_tool/fl_dataset.json``` to define the data used by each node and
   modify ```controller/dml_tool/fl_structure.json``` to define the DML structure of each node,
   see ```controller/dml_tool/README.md``` for more.
6. Run ```controller/fl_run.py``` with python3 with root privileges and keep it running on a terminal (called Term).
7. In path ```controller/dml_tool```, type ```python3 dataset_conf.py -d fl_dataset.json``` in terminal to generate
   dataset conf files and type ```python3 fl_structure_conf.py -s fl_structure.json```  generate DML structure conf
   files.
8. Type ```curl localhost:3333/conf/dataset``` in a terminal to send those dataset conf files to each node. Wait until
   all nodes have received the dataset conf file. This function is defined in ```controller/base/manager.py```.
9. Type ```curl localhost:3333/conf/structure``` to send those DML structure conf files to each node. Wait until all
   nodes have received the structure conf file. This function is defined in ```controller/base/manager.py```.
10. Wait until Term displays ```tc finish```.
11. Type ```curl localhost:3333/start?root=n1``` in a terminal to start all nodes. This function is defined
    in ```controller/base/manager.py``` and ```controller/fl_manager.py```. The root should be the first node defined
    in ```controller/dml_tool/fl_structure.json```.
12. When the pre-set training round is met, it will automatically stop all nodes and collect result files. This function
    is defined in ```controller/base/manager.py``` and ```controller/fl_manager.py```.
13. Commands such as ```curl localhost:3333/emulated/reset``` and ```curl localhost:3333/physical/reset```are used to
    remove all the emulated nodes and physical nodes. These functions are defined in ```controller/base/manager```,
    and ```worker/agent.py```.

#### E-Tree Learning

1. Same with above 1-4.
2. Just use the ```controller/dml_app/el_peer.py```, ```controller/dml_app/Dockerfile```,
   and ```controller/dml_app/dml_req.txt```
3. Modify ```controller/el_run.py```  to define the test environment.
4. Modify ```controller/linlks.json```  to define the network.
5. Modify ```controller/dml_tool/el_dataset.json``` to define the data used by each node and
   modify ```controller/dml_tool/el_structure.json``` to define the DML structure of each node,
   see ```controller/dml_tool/README.md``` for more.
6. Run ```controller/el_run.py``` with python3 with root privileges and keep it running on a terminal (called Term).
7. In path ```controller/dml_tool```, type ```python3 dataset_conf.py -d el_dataset.json``` in terminal to generate
   dataset conf files and type ```python3 el_structure_conf.py -s el_structure.json```  generate DML structure conf
   files.
8. Type ```curl localhost:3333/conf/dataset``` in a terminal to send those dataset conf files to each node. Wait until
   all nodes have received the dataset conf file. This function is defined in ```controller/base/manager.py```.
9. Type ```curl localhost:3333/conf/structure``` to send those DML structure conf files to each node. Wait until all
   nodes have received the structure conf file. This function is defined in ```controller/base/manager.py```.
10. Wait until Term displays ```tc finish```.
11. Type ```curl localhost:3333/start?root=n1``` in a terminal to start all nodes. This function is defined
    in ```controller/base/manager.py``` and ```controller/el_manager.py```. The root should be the first node defined
    in ```controller/dml_tool/el_structure.json```.
12. When the pre-set training round is met, it will automatically stop all nodes and collect result files. This function
    is defined in ```controller/base/manager.py``` and ```controller/el_manager.py```.
13. Commands such as ```curl localhost:3333/emulated/reset``` and ```curl localhost:3333/physical/reset```are used to
    remove all the emulated nodes and physical nodes. These functions are defined in ```controller/base/manager```,
    and ```worker/agent.py```.

#### Ring All-Reduce

1. Same with above 1-4.
2. Just use the ```controller/dml_app/ra_peer.py```, ```controller/dml_app/Dockerfile```,
   and ```controller/dml_app/dml_req.txt```
3. Modify ```controller/ra_run.py```  to define the test environment.
4. Modify ```controller/linlks.json```  to define the network.
5. Modify ```controller/dml_tool/ra_dataset.json``` to define the data used by each node and
   modify ```controller/dml_tool/ra_structure.json``` to define the DML structure of each node,
   see ```controller/dml_tool/README.md``` for more.
6. Run ```controller/ra_run.py``` with python3 with root privileges and keep it running on a terminal (called Term).
7. In path ```controller/dml_tool```, type ```python3 dataset_conf.py -d ra_dataset.json``` in terminal to generate
   dataset conf files and type ```python3 ra_structure_conf.py -s ra_structure.json``` to generate DML structure conf
   files.
8. Type ```curl localhost:3333/conf/dataset``` in a terminal to send those dataset conf files to each node. Wait until
   all nodes have received the dataset conf file. This function is defined in ```controller/base/manager.py```.
9. Type ```curl localhost:3333/conf/structure``` to send those DML structure conf files to each node. Wait until all
   nodes have received the structure conf file. This function is defined in ```controller/base/manager.py```.
10. Wait until Term displays ```tc finish```.
11. Type ```curl localhost:3333/start``` in a terminal to start all nodes. This function is defined
    in ```controller/base/manager.py``` and ```controller/ra_manager.py```.
12. When the pre-set training round is met, it will automatically stop all nodes and collect result files. This function
    is defined in ```controller/base/manager.py``` and ```controller/ra_manager.py```.
13. Commands such as ```curl localhost:3333/emulated/reset``` and ```curl localhost:3333/physical/reset```are used to
    remove all the emulated nodes and physical nodes. These functions are defined in ```controller/base/manager```,
    and ```worker/agent.py```.

### Citing

Please cite our paper if you find *EdgeTB* is useful in your research.  
Lei Yang, Fulin Wen, Jiannong Cao, Zhenyu Wang. "EdgeTB: a Hybrid Testbed for Distributed Machine Learning at the Edge
with High Fidelity." IEEE Transactions on Parallel and Distributed Systems. DOI: 10.1109/TPDS.2022.3144994.

### Contact

EdgeTB is designed and developed by the joint research team at School of Software Engineering, South China University of
Technology, and the Department of Computing, The Hong Kong Polytechnic University. If you have any question, please
contact with us: Fulin Wen <201921043987@mail.scut.edu.cn> and Lei Yang <sely@scut.edu.cn>.
