
### Send data to the Cluster

Open 3 terminals:

1) **Local**
Send data to DICE:
rsync -ua --progress /Users/zhouyi/DatasetCondensation/result/vis_DM_TinyVIRAT_test_10_1_ConvNet_1ipc_exp0_iter900.png s1908422@student.ssh.inf.ed.ac.uk:/home/s1908422/<>

2) **Log into DICE**
ssh s1908422@student.ssh.inf.ed.ac.uk
Send data to Cluster:
rsync -ua --progress /home/s1908422/vis_DM_TinyVIRAT_test_10_1_ConvNet_1ipc_exp0_iter900.png s1908422@mlp.inf.ed.ac.uk:/home/s1908422

3) **Log into Cluster**
ssh s1908422@student.ssh.inf.ed.ac.uk
ssh mlp1
Check data and run code



### Goals

