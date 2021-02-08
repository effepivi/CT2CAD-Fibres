#!/bin/bash

echo "i,RECONSTRUCTION_RUNTIME (in min),CUBE_FIBRES_RUNTIME (in min),CUBE_FIBRES_RUNTIME (in MM:SS),CUBE_FIBRES_ZNCC,X (in um),Y (in um),ROT (in degree),W (in um),H (in um),RADIUS_CORE (in um),RADIUS_FIBRE (in um),RADIUS_CORE (in pixels),RADIUS_FIBRE (in pixels),FIBRE_ITER,FIBRE_FEVAL,FIBRE_IND,AVG,STDDEV,MIN,MAX"

for i in {1..29}
do

    CT_slice_file=run_SCW_$i/reconstruction_cube_CT_fibres.mha

    if [ -f $CT_slice_file ]
    then
        # Move the job script
        if [ -f ../job_$i.sh ]
        then
            mv ../job_$i.sh run_SCW_$i/
        fi

        # Move SLURM output
        FILES=`ls ../register-$i-*.out ../register-$i-*.err 2>/dev/null`
        if [ $? -eq 0 ]; then
            for file in $FILES
            do
                mv $file run_SCW_$i/
            done
        fi

        # Get the reconstruction time in second
        RECONSTRUCTION_RUNTIME=`grep "RECONSTRUCTION:" run_SCW_$i/register-$i-*.out | cut -d " " -f 2  -`
        CUBE_FIBRES_RUNTIME=`grep "FIBRES " run_SCW_$i/register-$i-*.out | cut -d " " -f 2  -`

        RECONSTRUCTION_RUNTIME=`echo "$RECONSTRUCTION_RUNTIME/60.0" | bc -l`
        CUBE_FIBRES_RUNTIME=`echo "$CUBE_FIBRES_RUNTIME/60.0" | bc -l`
        CUBE_FIBRES_RUNTIME_DAYS=`echo "$CUBE_FIBRES_RUNTIME/60.0/24.0" | bc -l`

        MATRIX_PARAMS=`grep "Matrix params: " run_SCW_$i/register-$i-*.out`
        MATRIX_PARAMS=${MATRIX_PARAMS#"Matrix params: ["}
        MATRIX_PARAMS=${MATRIX_PARAMS::-1}
        X=`echo $MATRIX_PARAMS | cut -d "," -f 1`
        Y=`echo $MATRIX_PARAMS | cut -d "," -f 2`
        ROT=`echo $MATRIX_PARAMS | cut -d "," -f 3`
        W=`echo $MATRIX_PARAMS | cut -d "," -f 4`
        H=`echo $MATRIX_PARAMS | cut -d "," -f 5`

        X=`echo "$X * 1024 * 1.9" | bc -l`
        Y=`echo " $Y * 1024 * 1.9" | bc -l`
        ROT=`echo " $ROT + 0.5 * 360.0" | bc -l`
        W=`echo " ($W + 0.5) * 1024 * 1.9" | bc -l`
        H=`echo " ($H + 0.5) * $W" | bc -l`

        RADIUS_CORE=`grep "Radii: " run_SCW_$i/register-$i-*.out | cut -d " " -f 2`
        RADIUS_FIBRE=`grep "Radii: " run_SCW_$i/register-$i-*.out | cut -d " " -f 3`


        RADIUS_CORE_PX=`echo "$RADIUS_CORE/1.9" | bc -l`
        RADIUS_FIBRE_PX=`echo "$RADIUS_FIBRE/1.9" | bc -l`

        TEMP=`grep -n "FIBRES " run_SCW_$i/register-$i-*.out | cut -d ":" -f 1`
        TEMP=`echo $TEMP - 1 | bc`
        TEMP=`tail -n+$TEMP run_SCW_$i/register-$i-*.out | head -n1 | tr -s ' '`
        FIBRE_ITER=`echo $TEMP | cut -d " " -f 1`
        FIBRE_FEVAL=`echo $TEMP | cut -d " " -f 2`
        FIBRE_IND=`echo $FIBRE_FEVAL / $FIBRE_ITER | bc`



        FIBRE_ZNCC=`grep "Fibres CT ZNCC: " run_SCW_$i/register-$i-*.out | cut -d " " -f 4`

        STATS=`python3 ../imageStats.py $CT_slice_file`

        echo $i,$RECONSTRUCTION_RUNTIME,$CUBE_FIBRES_RUNTIME,$CUBE_FIBRES_RUNTIME_DAYS,$FIBRE_ZNCC,$X,$Y,$ROT,$W,$H,$RADIUS_CORE,$RADIUS_FIBRE,$RADIUS_CORE_PX,$RADIUS_FIBRE_PX,$FIBRE_ITER,$FIBRE_FEVAL,$FIBRE_IND,$STATS
    fi
done
