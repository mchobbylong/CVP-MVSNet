@rem Dataset
@set DATASET_ROOT="./dataset/dtu-train-128/"

@rem Logging
@set CKPT_DIR="./checkpoints/"
@set LOG_DIR="./logs/"

python train.py ^
--info="train_dtu_128" ^
--mode="train" ^
--dataset_root=%DATASET_ROOT% ^
--imgsize=128 ^
--nsrc=2 ^
--nscale=2 ^
--epochs=40 ^
--lr=0.001 ^
--lrepochs="10,12,14,20:2" ^
--batch_size=16 ^
--loadckpt="" ^
--logckptdir=%CKPT_DIR% ^
--loggingdir=%LOG_DIR% ^
--resume=0
