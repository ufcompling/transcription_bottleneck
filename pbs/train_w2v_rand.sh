#!/bin/bash
#SBATCH --job-name=hupa-30-5-0-al
#SBATCH --mail-user=liu.ying@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output hupa-30-5-0-al-%j.out
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


cd /blue/liu.ying/transcription_bottleneck

python scripts/train_w2v.py --lang hupa --size 30 --interval 5 --select 0 --method al

python scripts/eval_w2v_confidence.py --lang hupa --size 30 --interval 5 --select 0 --method al

date
