# ====== Cora ====== #
# MaskedEdge (same as the pre-training strategy of GPPT)
python main.py --dataset Cora --subgraph_file Cora_4hop_svd100_10shots_nnl5-150_seed0_split.dat --gnn_file GPPT_Cora.pth\
               --train_loader_size 8 --batch_size 8 --minibatch_size 64\
               --total_epochs 100 --head_layers 2 --tasknet_epochs 1 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay static --max_z 0.1 --penalty_alpha_d 1e5
# ContraEdge (same as the pre-training strategy of Gprompt)
python main.py --dataset Cora --subgraph_file Cora_4hop_svd100_10shots_nnl5-150_seed0_split.dat --gnn_file Gprompt_Cora.pth\
               --train_loader_size 8 --batch_size 8 --minibatch_size 64\
               --total_epochs 100 --head_layers 2 --tasknet_epochs 1 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay static --max_z 0.05 --penalty_alpha_d 1e5


# ====== CiteSeer ====== #
# MaskedEdge (same as the pre-training strategy of GPPT)
python main.py --dataset CiteSeer --subgraph_file CiteSeer_3hop_svd100_10shots_nnl20-150_seed0_split.dat --gnn_file GPPT_CiteSeer.pth\
               --train_loader_size 7 --batch_size 7 --minibatch_size 256\
               --total_epochs 100 --head_layers 1 --tasknet_epochs 1 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay static --max_z 0.05 --penalty_alpha_d 1e5
# ContraEdge (same as the pre-training strategy of Gprompt)
python main.py --dataset CiteSeer --subgraph_file CiteSeer_3hop_svd100_10shots_nnl20-150_seed0_split.dat --gnn_file Gprompt_CiteSeer.pth\
               --train_loader_size 7 --batch_size 7 --minibatch_size 256\
               --total_epochs 50 --head_layers 2 --tasknet_epochs 3 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay static --max_z 0.05 --penalty_alpha_d 1e1


# ====== PubMed ====== #
# MaskedEdge (same as the pre-training strategy of GPPT)
python main.py --dataset PubMed --subgraph_file PubMed_2hop_svd100_10shots_nnl50-300_seed0_split.dat --gnn_file GPPT_PubMed.pth\
               --train_loader_size 7 --batch_size 7 --minibatch_size 256\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 1 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay static --max_z 0.05 --penalty_alpha_d 1e5
# ContraEdge (same as the pre-training strategy of Gprompt)
python main.py --dataset PubMed --subgraph_file PubMed_2hop_svd100_10shots_nnl50-300_seed0_split.dat --gnn_file Gprompt_PubMed.pth\
               --train_loader_size 7 --batch_size 7 --minibatch_size 256\
               --total_epochs 100 --head_layers 2 --tasknet_epochs 1 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay static --max_z 0.05 --penalty_alpha_d 1e5


# ====== Photo ====== #
# MaskedEdge (same as the pre-training strategy of GPPT)
python main.py --dataset Photo --subgraph_file Photo_2hop_svd100_10shots_nnl200-2000(20_10)_seed0_split.dat --gnn_file GPPT_Photo.pth\
               --train_loader_size 10 --batch_size 10 --minibatch_size 256\
               --total_epochs 100 --head_layers 3 --tasknet_epochs 5 --tasknet_lr 1e-3 --warmup_epochs 100\
               --tasknet_decay down --policy_decay static --max_z 0.05 --penalty_alpha_d 1e5
# ContraEdge (same as the pre-training strategy of Gprompt)
python main.py --dataset Photo --subgraph_file Photo_2hop_svd100_10shots_nnl200-2000(20_10)_seed0_split.dat --gnn_file Gprompt_Photo.pth\
               --train_loader_size 10 --batch_size 10 --minibatch_size 256\
               --total_epochs 100 --head_layers 1 --tasknet_epochs 5 --tasknet_lr 1e-3 --warmup_epochs 50\
               --tasknet_decay down --policy_decay static --max_z 0.1 --penalty_alpha_d 1e5


# ====== Computers ====== #
# MaskedEdge (same as the pre-training strategy of GPPT)
python main.py --dataset Computers --subgraph_file Computers_2hop_svd100_10shots_nnl100-500(10_10)_seed0_split.dat --gnn_file GPPT_Computers.pth\
               --train_loader_size 12 --batch_size 12 --minibatch_size 512\
               --total_epochs 300 --head_layers 2 --tasknet_epochs 5 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay static --max_z 0.1 --penalty_alpha_d 1e5
# ContraEdge (same as the pre-training strategy of Gprompt)
python main.py --dataset Computers --subgraph_file Computers_2hop_svd100_10shots_nnl100-500(10_10)_seed0_split.dat --gnn_file Gprompt_Computers.pth\
               --train_loader_size 12 --batch_size 12 --minibatch_size 512\
               --total_epochs 100 --head_layers 1 --tasknet_epochs 3 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay static --max_z 0.1 --penalty_alpha_d 1e5
