#!/usr/bin/bash
#
# Project/Account (use your own)
#SBATCH -A scw1701
#SBATCH --job-name=26-register     # Job name
#SBATCH --output register-26-%j.out     # Job name
#SBATCH --error register-26-%j.err     # Job name
#
# We ask for 1 tasks with 1 core only.
# We ask for a GPU
#SBATCH -p gpu_v100
#SBATCH --gres=gpu:2
#
# Number of tasks per node
#SBATCH --ntasks-per-node=1
#
# Number of cores per task
#SBATCH --cpus-per-task=1
#
# Use one node
#SBATCH --nodes=1
#
# Runtime of this jobs is less than 5 hours.
#SBATCH --time=01:00:00
module purge > /dev/null 2>&1
module load cmake compiler/gnu/8/1.0 CUDA python/3.7.0

export EGL_PLATFORM=
export PYTHONPATH=/home/b.eese10/gvirtualxray-install/gvxrWrapper-1.0.4/python3:/home/b.eese10/gvirtualxray-install/gvxrWrapper-1.0.4/python3:/apps/languages/python/3.7.0/el7/AVX512/intel-2018/lib/python3.7/site-packages
date > registration-matrix-then-fibres2-SCW/run_SCW_26/runtime_26
python3 registration-matrix-then-fibres.py --input Tutorial2/sino.raw --output registration-matrix-then-fibres2-SCW/run_SCW_26/
date >> registration-matrix-then-fibres2-SCW/run_SCW_26/runtime_26
