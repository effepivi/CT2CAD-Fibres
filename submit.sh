#!/bin/bash

for i in {1..25}
do
	echo "#!/usr/bin/bash" > job_$i.sh
	echo "#" >> job_$i.sh
	echo "# Project/Account (use your own)" >> job_$i.sh
	echo "#SBATCH --job-name=register-$i     # Job name" >> job_$i.sh
	echo "#SBATCH --output register-$i-%j.out     # Job name" >> job_$i.sh
	echo "#SBATCH --error register-$i-%j.err     # Job name" >> job_$i.sh
	echo "#" >> job_$i.sh
	echo "# We ask for 1 tasks with 1 core only." >> job_$i.sh
	echo "# We ask for a GPU" >> job_$i.sh
	echo "#SBATCH -p gpu_v100" >> job_$i.sh
	echo "#SBATCH --gres=gpu:2" >> job_$i.sh
	echo "#" >> job_$i.sh
	echo "# Number of tasks per node" >> job_$i.sh
	echo "#SBATCH --ntasks-per-node=1" >> job_$i.sh
	echo "#" >> job_$i.sh
	echo "# Number of cores per task" >> job_$i.sh
	echo "#SBATCH --cpus-per-task=1" >> job_$i.sh
	echo "#" >> job_$i.sh
	echo "# Use one node" >> job_$i.sh
	echo "#SBATCH --nodes=1" >> job_$i.sh
	echo "#" >> job_$i.sh
	echo "# Runtime of this jobs is less than 5 hours." >> job_$i.sh
	echo "#SBATCH --time=00:40:00" >> job_$i.sh
	echo "module purge > /dev/null 2>&1" >> job_$i.sh
	echo "module load cmake compiler/gnu/8/1.0 CUDA python/3.7.0" >> job_$i.sh
	echo "" >> job_$i.sh
	echo "export EGL_PLATFORM=" >> job_$i.sh
	echo "export PYTHONPATH=$HOME/gvirtualxray-install/gvxrWrapper-1.0.4/python3:$PYTHONPATH" >> job_$i.sh
	echo "date > validation/runtime_$i" >> job_$i.sh
	echo "python3 validation.py --input Tutorial2/sino.raw --output validation/run_SCW_$i/" >> job_$i.sh
	echo "date >> validation/runtime_$i" >> job_$i.sh

	chmod +x 	 job_$i.sh
	sbatch job_$i.sh
done










