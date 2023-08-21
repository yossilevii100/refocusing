cp checkpoints/leanrd2/models/*.t7 pretrained/
#python main.py --model rpc --eval 1
#python main.py --model rpc --eval 1 --use_ensemble 1
python main.py --model gdanet --is_pointguard 1 --eval 1 --test_batch_size 1024
python main.py --model dgcnn --is_pointguard 1 --eval 1 --test_batch_size 1024 --use_wolfmix 1
python main.py --model rpc --is_pointguard 1 --eval 1 --test_batch_size 1024 --use_wolfmix 1
python main.py --model gdanet --is_pointguard 1 --eval 1 --test_batch_size 1024 --use_wolfmix 1
#python main.py --model rpc --is_pointguard 1 --epochs 1000 --batch_size 256 --eval 1
#python main.py --model dgcnn --eval 1\
#python main.py --model dgcnn --eval 1 --use_ensemble 1
#python main.py --model dgcnn --use_wolfmix 1 --eval 1
#python main.py --model dgcnn --is_pointguard 1 --epochs 1000 --batch_size 256 --eval 1
#python main.py --model gdanet --eval 1
#python main.py --model gdanet --eval 1 --use_ensemble 1
#python main.py --model gdanet --use_wolfmix 1 --eval 1
#python main.py --model gdanet --is_pointguard 1 --epochs 1000 --batch_size 256 --eval 1
#python main.py --model pct --eval 1 
#python main.py --model pct --use_wolfmix 1 --eval 1
#python main.py --model pct --eval 1 --use_ensemble 1
#python main.py --model pct --is_pointguard 1 --epochs 1000 --batch_size 256 --eval 1
#python main.py --model pointnet --eval 1
#python main.py --model pointnet --use_wolfmix 1 --eval 1
#python main.py --model pointnet --is_pointguard 1 --epochs 1000 --batch_size 256 --eval 1