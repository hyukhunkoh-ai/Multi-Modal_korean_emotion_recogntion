#!/bin/bash
#SBATCH --time      96:00:00
#SBATCH --constraint a6000
#SBATCH --gpus      1
#SBATCH --mail-type END
#SBATCH --mail-user hyukhunkoh-ai@snu.ac.kr
conda activate etri

python /home/hyukhunkoh-ai/workspace/spt5/multi_modal_train.py