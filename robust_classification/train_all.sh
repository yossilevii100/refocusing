#python main.py --model rpc
#python main.py --model rpc --use_wolfmix 1
#python main.py --model dgcnn
#python main.py --model dgcnn --use_wolfmix 1 --epochs 600
#python main.py --model gdanet --batch_size 32
#python main.py --model pct
python main.py --model gdanet --is_pointguard 1 --batch_size 32
python main.py --model gdanet --use_wolfmix 1 --epochs 600 --batch_size 32
python main.py --model dgcnn --use_wolfmix 1 --epochs 600 --is_pointguard 1
python main.py --model rpc --use_wolfmix 1 --epochs 600 --is_pointguard 1
python main.py --model gdanet --use_wolfmix 1 --epochs 600 --is_pointguard 1 --batch_size 32
#python main.py --model pct --use_wolfmix 1
#python main.py --model pointnet
#python main.py --model pointnet --use_wolfmix 1