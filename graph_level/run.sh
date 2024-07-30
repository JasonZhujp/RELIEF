# ====== BBBP ====== #
# Infomax
python main.py --dataset bbbp --seed 0 --minibatch_size 64 --gnn_file infomax.pth\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 1 --tasknet_lr 5e-4\
               --tasknet_decay static --policy_decay down --max_z 0.1 --penalty_alpha_d 1e0
# AttrMasking
python main.py --dataset bbbp --seed 0 --minibatch_size 64 --gnn_file masking.pth\
               --total_epochs 100 --head_layers 3 --tasknet_epochs 1 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay down --max_z 0.1 --penalty_alpha_d 1e0
# ContextPred
python main.py --dataset bbbp --seed 0 --minibatch_size 64 --gnn_file contextpred.pth\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 3 --tasknet_lr 1.5e-3\
               --tasknet_decay static --policy_decay static --max_z 0.5 --penalty_alpha_d 1e0
# GCL
python main.py --dataset bbbp --seed 0 --minibatch_size 64 --gnn_file gcl.pth\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 3 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay static --max_z 0.1 --penalty_alpha_d 1e5


# ====== Tox21 ====== #
# Infomax
python main.py --dataset tox21 --seed 2 --eval_loader_size 128 --minibatch_size 64 --gnn_file infomax.pth\
               --total_epochs 50 --head_layers 2 --tasknet_epochs 2 --tasknet_lr 1.5e-3\
               --tasknet_decay down --policy_decay down --max_z 1.0 --penalty_alpha_d 1e5
# AttrMasking
python main.py --dataset tox21 --seed 2 --eval_loader_size 128 --minibatch_size 64 --gnn_file masking.pth\
               --total_epochs 100 --head_layers 2 --tasknet_epochs 3 --tasknet_lr 1e-3\
               --tasknet_decay down --policy_decay static --max_z 0.1 --penalty_alpha_d 1e0
# ContextPred
python main.py --dataset tox21 --seed 2 --eval_loader_size 128 --minibatch_size 64 --gnn_file contextpred.pth --tasknet_train_mode 0\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 1 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay down --max_z 0.1 --penalty_alpha_d 1e0
# GCL
python main.py --dataset tox21 --seed 2 --eval_loader_size 128 --minibatch_size 64 --gnn_file gcl.pth\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 2 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay down --max_z 1.0 --penalty_alpha_d 1e0


# ====== ToxCast ====== #
# Infomax
python main.py --dataset toxcast --seed 5 --eval_loader_size 256 --minibatch_size 64 --gnn_file infomax.pth\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 1 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay down --max_z 0.1 --penalty_alpha_d 1e5
# AttrMasking
python main.py --dataset toxcast --seed 5 --eval_loader_size 256 --minibatch_size 64 --gnn_file masking.pth\
               --total_epochs 100 --head_layers 3 --tasknet_epochs 1 --tasknet_lr 5e-4\
               --tasknet_decay static --policy_decay static --max_z 1.0 --penalty_alpha_d 1e0
# ContextPred
python main.py --dataset toxcast --seed 5 --eval_loader_size 256 --minibatch_size 64 --gnn_file contextpred.pth\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 1 --tasknet_lr 5e-4\
               --tasknet_decay static --policy_decay down --max_z 0.5 --penalty_alpha_d 1e5
# GCL
python main.py --dataset toxcast --seed 5 --eval_loader_size 256 --minibatch_size 64 --gnn_file gcl.pth\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 1 --tasknet_lr 5e-4\
               --tasknet_decay static --policy_decay down --max_z 0.1 --penalty_alpha_d 1e5


# ====== SIDER ====== #
# Infomax
python main.py --dataset sider --seed 2 --minibatch_size 128 --gnn_file infomax.pth --tasknet_train_mode 0\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 2 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay static --max_z 0.1 --penalty_alpha_d 1e0
# AttrMasking
python main.py --dataset sider --seed 2 --minibatch_size 128 --gnn_file masking.pth\
               --total_epochs 100 --head_layers 3 --tasknet_epochs 2 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay static --max_z 1.0 --penalty_alpha_d 1e0
# ContextPred
python main.py --dataset sider --seed 2 --minibatch_size 128 --gnn_file contextpred.pth\
               --total_epochs 100 --head_layers 2 --tasknet_epochs 3 --tasknet_lr 5e-4\
               --tasknet_decay static --policy_decay static --max_z 1.0 --penalty_alpha_d 1e5
# GCL
python main.py --dataset sider --seed 2 --minibatch_size 128 --gnn_file gcl.pth\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 1 --tasknet_lr 1e-3 --tasknet_train_mode 0\
               --tasknet_decay static --policy_decay static --max_z 1.0 --penalty_alpha_d 1e5


# ====== ClinTox ====== #
# Infomax
python main.py --dataset clintox --seed 3 --minibatch_size 64 --gnn_file infomax.pth\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 2 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay static --max_z 0.1 --penalty_alpha_d 1e0
# AttrMasking
python main.py --dataset clintox --seed 3 --minibatch_size 64 --gnn_file masking.pth\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 3 --tasknet_lr 1.5e-3\
               --tasknet_decay static --policy_decay static --max_z 0.1 --penalty_alpha_d 1e5
# ContextPred
python main.py --dataset clintox --seed 3 --minibatch_size 64 --gnn_file contextpred.pth\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 3 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay static --max_z 1.0 --penalty_alpha_d 1e0
# GCL
python main.py --dataset clintox --seed 3 --minibatch_size 64 --gnn_file gcl.pth\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 2 --tasknet_lr 1.5e-3\
               --tasknet_decay down --policy_decay down --max_z 1.0 --penalty_alpha_d 1e5


# ====== MUV ====== #
# Infomax
python main.py --dataset muv --seed 3 --eval_loader_size 512 --minibatch_size 64 --gnn_file infomax.pth\
               --total_epochs 100 --head_layers 2 --tasknet_epochs 2 --tasknet_lr 5e-4\
               --tasknet_decay static --policy_decay down --max_z 0.1 --penalty_alpha_d 1e0
# AttrMasking
python main.py --dataset muv --seed 3 --eval_loader_size 512 --minibatch_size 64 --gnn_file masking.pth\
               --total_epochs 100 --head_layers 2 --tasknet_epochs 1 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay down --max_z 1.0 --penalty_alpha_d 1e0
# ContextPred
python main.py --dataset muv --seed 3 --eval_loader_size 512 --minibatch_size 64 --gnn_file contextpred.pth\
               --total_epochs 100 --head_layers 2 --tasknet_epochs 2 --tasknet_lr 1.5e-3\
               --tasknet_decay static --policy_decay static --max_z 0.1 --penalty_alpha_d 1e5
# GCL
python main.py --dataset muv --seed 3 --eval_loader_size 512 --minibatch_size 64 --gnn_file gcl.pth\
               --total_epochs 100 --head_layers 2 --tasknet_epochs 3 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay static --max_z 1.0 --penalty_alpha_d 1e5


# ====== HIV ====== #
# Infomax
python main.py --dataset hiv --seed 0 --eval_loader_size 512 --minibatch_size 128 --gnn_file infomax.pth\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 2 --tasknet_lr 1e-3\
               --tasknet_decay down --policy_decay down --max_z 0.1 --penalty_alpha_d 1e5
# AttrMasking
python main.py --dataset hiv --seed 0 --eval_loader_size 512 --minibatch_size 128 --gnn_file masking.pth\
               --total_epochs 50 --head_layers 2 --tasknet_epochs 1 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay down --max_z 1.0 --penalty_alpha_d 1e5
# ContextPred
python main.py --dataset hiv --seed 0 --eval_loader_size 512 --minibatch_size 128 --gnn_file contextpred.pth\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 2 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay down --max_z 0.1 --penalty_alpha_d 1e5
# GCL
python main.py --dataset hiv --seed 0 --eval_loader_size 512 --minibatch_size 128 --gnn_file gcl.pth\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 1 --tasknet_lr 5e-4\
               --tasknet_decay down --policy_decay down --max_z 1.0 --penalty_alpha_d 1e0


# ====== BACE ====== #
# Infomax
python main.py --dataset bace --seed 1 --minibatch_size 64 --gnn_file infomax.pth\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 1 --tasknet_lr 5e-4\
               --tasknet_decay static --policy_decay static --max_z 0.1 --penalty_alpha_d 1e5
# AttrMasking
python main.py --dataset bace --seed 1 --minibatch_size 64 --gnn_file masking.pth\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 1 --tasknet_lr 1e-3\
               --tasknet_decay static --policy_decay static --max_z 1.0 --penalty_alpha_d 1e5
# ContextPred
python main.py --dataset bace --seed 1 --minibatch_size 64 --gnn_file contextpred.pth\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 1 --tasknet_lr 5e-4\
               --tasknet_decay down --policy_decay static --max_z 0.1 --penalty_alpha_d 1e0
# GCL
python main.py --dataset bace --seed 1 --minibatch_size 64 --gnn_file gcl.pth\
               --total_epochs 50 --head_layers 3 --tasknet_epochs 1 --tasknet_lr 5e-4\
               --tasknet_decay down --policy_decay static --max_z 0.1 --penalty_alpha_d 1e0