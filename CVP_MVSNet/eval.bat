@rem Dataset
@set dataset_name=dtu-test-1200
@set dataset_root=./dataset
@set dataset_path=%dataset_root%/%dataset_name%/
@set output_path=outputs/%dataset_name%/

@rem Checkpoint
@set CKPT_PATH="./checkpoints/pretrained/model_000027.ckpt"

@rem Logging
@set LOG_DIR=./logs/

@set CUDA_LAUNCH_BLOCKING=1

python eval.py ^
--info="eval_pretrained_e27" ^
--mode="test" ^
--dataset_root=%dataset_path% ^
--imgsize=1200 ^
--nsrc=4 ^
--nscale=5 ^
--batch_size=1 ^
--loadckpt=%CKPT_PATH% ^
--loggingdir=%LOG_DIR% ^
--outdir=%output_path%
