#!/bin/bash

OUTPUT_DIR=registration-matrix-then-fibres2-SCW

if [ ! -d $OUTPUT_DIR ]
then
    mkdir $OUTPUT_DIR
fi

for i in {1..10}
do
    if [ ! -d $OUTPUT_DIR/run_SCW_$i ]
    then
        mkdir $OUTPUT_DIR/run_SCW_$i

	    echo "#!/usr/bin/bash" > $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "#" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "# Project/Account (use your own)" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "#SBATCH -A scw1701" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "#SBATCH --job-name=$i-register     # Job name" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "#SBATCH --output register-$i-%j.out     # Job name" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "#SBATCH --error register-$i-%j.err     # Job name" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "#" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "# We ask for 1 tasks with 1 core only." >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "# We ask for a GPU" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "#SBATCH -p gpu_v100" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "#SBATCH --gres=gpu:2" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "#" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "# Number of tasks per node" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "#SBATCH --ntasks-per-node=1" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "#" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "# Number of cores per task" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "#SBATCH --cpus-per-task=1" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "#" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "# Use one node" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "#SBATCH --nodes=1" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "#" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "# Runtime of this jobs is less than 5 hours." >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "#SBATCH --time=01:00:00" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "module purge > /dev/null 2>&1" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "module load cmake compiler/gnu/8/1.0 CUDA python/3.7.0" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "export EGL_PLATFORM=" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "export PYTHONPATH=$HOME/gvirtualxray-install/gvxrWrapper-1.0.4/python3:$PYTHONPATH" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "date > $OUTPUT_DIR/run_SCW_$i/runtime_$i" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "python3 registration-matrix-then-fibres.py --input Tutorial2/sino.raw --output $OUTPUT_DIR/run_SCW_$i/" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    echo "date >> $OUTPUT_DIR/run_SCW_$i/runtime_$i" >> $OUTPUT_DIR/run_SCW_$i/job_$i.sh

	    chmod +x 	 $OUTPUT_DIR/run_SCW_$i/job_$i.sh
	    sbatch $OUTPUT_DIR/run_SCW_$i/job_$i.sh
    fi
done

