@rem Dataset
@set DATASET_ROOT="./dataset/dtu-test-1200/"

@rem Checkpoint
@set LOAD_CKPT_DIR="./checkpoints/pretrained/model_000027.ckpt"

@rem Logging
@set LOG_DIR="./logs/"

@rem Output dir
@set OUT_DIR="./outputs_pretrained/"

python eval.py ^
--info="eval_pretrained_e27" ^
--mode="test" ^
--dataset_root=%DATASET_ROOT% ^
--imgsize=1200 ^
--nsrc=4 ^
--nscale=5 ^
--batch_size=1 ^
--loadckpt=%LOAD_CKPT_DIR% ^
--loggingdir=%LOG_DIR% ^
--outdir=%OUT_DIR%
