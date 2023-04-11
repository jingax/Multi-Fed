N=200
S=0
noise=0.3
data=COVID
r=100
fea_dim=200
mdl=LeNet5
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 4 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset IL --data_size $N --noise $noise
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 4 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FL --data_size $N --noise $noise
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 1 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset CL --data_size $N --noise $noise
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 4 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FD --data_size $N --noise $noise --kd_type feature
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 4 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FD --data_size $N --noise $noise --kd_type output


echo "----------------------------------------------------"
noise=0.2
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 4 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset IL --data_size $N --noise $noise
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 4 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FL --data_size $N --noise $noise
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 1 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset CL --data_size $N --noise $noise
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 4 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FD --data_size $N --noise $noise --kd_type feature
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 4 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FD --data_size $N --noise $noise --kd_type output

echo "----------------------------------------------------"
noise=0.1
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 4 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset IL --data_size $N --noise $noise
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 4 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FL --data_size $N --noise $noise
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 1 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset CL --data_size $N --noise $noise
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 4 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FD --data_size $N --noise $noise --kd_type feature
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 4 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FD --data_size $N --noise $noise --kd_type output

echo "----------------------------------------------------"
noise=0.0
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 4 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset IL --data_size $N --noise $noise
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 4 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FL --data_size $N --noise $noise
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 1 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset CL --data_size $N --noise $noise
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 4 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FD --data_size $N --noise $noise --kd_type feature
CUDA_VISIBLE_DEVICES=3 python runC.py --n_clients 4 --dataset $data --model $mdl --feature_dim $fea_dim --rounds $r --seed $S --preset FD --data_size $N --noise $noise --kd_type output
