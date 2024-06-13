#!/bin/bash
#SBATCH --job-name=whisper-waikhana
#SBATCH --mail-user=liu.ying@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output whisper-waikhana-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1

module load conda
conda activate whisper
module load pytorch/2.0.1



cd /blue/liu.ying/word_making

python scripts/eval_whisper_heavy.py rand

date

