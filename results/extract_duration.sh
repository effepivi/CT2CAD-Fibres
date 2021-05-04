#!/bin/bash

echo "objective,i,MATRIX_ZNCC,FIBRE1_ZNCC,FIBRE2_ZNCC,FIBRE3_ZNCC,HARMONICS_ZNCC,NOISE_ZNCC,LAPLACIAN1_ZNCC,LAPLACIAN_LSF_ZNCC,CUBE1_RUNTIME (in min),FIBRES1_RUNTIME (in min),FIBRES3_RUNTIME (in min),HARMONICS_RUNTIME (in min),NOISE_RUNTIME (in min),LAPLACIAN1_RUNTIME (in min),LAPLACIAN_LSF_RUNTIME (in min),OVERALL_RUNTIME (in min),CUBE1_RUNTIME (in hh:mm:ss),FIBRES1_RUNTIME (in hh:mm:ss),FIBRES3_RUNTIME (in hh:mm:ss),HARMONICS_RUNTIME (in hh:mm:ss),NOISE_RUNTIME (in hh:mm:ss),LAPLACIAN1_RUNTIME (in hh:mm:ss),LAPLACIAN_LSF_RUNTIME (in hh:mm:ss),OVERALL_RUNTIME (in hh:mm:ss),X1 (in um),Y1 (in um),ROT1 (in degree),W1 (in um),H1 (in um),RADIUS1_CORE (in um),RADIUS1_FIBRE (in um),RADIUS1_CORE_PX (in pixels),RADIUS1_FIBRE_PX (in pixels),RADIUS3_CORE (in um),RADIUS3_FIBRE (in um),RADIUS3_CORE_PX (in pixels),RADIUS3_FIBRE_PX (in pixels),33keV_WEIGHT,66keV_WEIGHT,99keV_WEIGHT,NOISE_BIAS,NOISE_GIAN,NOISE_SCALE,LAPLACIAN1_SIGMA_CORE,LAPLACIAN1_K_CORE,LAPLACIAN1_SIGMA_FIBRE,LAPLACIAN1_K_FIBRE,LAPLACIAN1_SIGMA_MATRIX,LAPLACIAN1_K_MATRIX,LAPLACIAN1_RADIUS_CORE (in um),LAPLACIAN1_RADIUS_FIBRE (in um),LAPLACIAN1_RADIUS_CORE_PX (in pixels),LAPLACIAN1_RADIUS_FIBRE_PX (in pixels),LAPLACIAN_LSF_K_CORE,LAPLACIAN_LSF_K_FIBRE,LSF_a,LSF_b,LSF_c,LSF_d,LSF_e,LSF_f,LAPLACIAN_LSF_K_MATRIX,MEAN_CORE_REF,MEAN_CORE_SIM,STDDEV_CORE_REF,STDDEV_CORE_SIM,MEAN_FIBRE_REF,MEAN_FIBRE_SIM,STDDEV_FIBRE_REF,STDDEV_FIBRE_SIM,MEAN_MATRIX_REF,MEAN_MATRIX_SIM,STDDEV_MATRIX_REF,STDDEV_MATRIX_SIM,AVG,STDDEV,MIN,MAX"

for i in {1..25}
do
    CT_slice_file=run_SCW_$i/simulated_CT_after_noise.mha

    if [ -f $CT_slice_file ]
    then

        # Get the ZNCC
        MATRIX_ZNCC=`grep "ZNCC matrix registration: " run_SCW_$i/optimisation-$i.out | cut -d ":" -f 2`
        FIBRE1_ZNCC=`grep "ZNCC matrix registration with fibres: " run_SCW_$i/optimisation-$i.out | cut -d ":" -f 2`
        FIBRE2_ZNCC=`grep "ZNCC radii registration 1: " run_SCW_$i/optimisation-$i.out | cut -d ":" -f 2`
        FIBRE3_ZNCC=`grep "ZNCC radii registration 2: " run_SCW_$i/optimisation-$i.out | cut -d ":" -f 2`
        HARMONICS_ZNCC=`grep "ZNCC spectrum registration 1: " run_SCW_$i/optimisation-$i.out | cut -d ":" -f 2`
        LAPLACIAN1_ZNCC=`grep "ZNCC phase contrast registration 1: " run_SCW_$i/optimisation-$i.out | cut -d ":" -f 2`
        LAPLACIAN_LSF_ZNCC=`grep "ZNCC phase contrast and LSF registration: " run_SCW_$i/optimisation-$i.out | cut -d ":" -f 2`
        NOISE_ZNCC=`grep "ZNCC noise registration: " run_SCW_$i/optimisation-$i.out | cut -d ":" -f 2`


        # Get the runtime in seconds
        CUBE1_RUNTIME=`grep "Matrix execution time" run_SCW_$i/optimisation-$i.out | cut -d ":" -f 2  -`
        FIBRES1_RUNTIME=`grep "Fibre1 execution time" run_SCW_$i/optimisation-$i.out | cut -d ":" -f 2  -`
        FIBRES3_RUNTIME=`grep "Fibre3 execution time" run_SCW_$i/optimisation-$i.out | cut -d ":" -f 2  -`
        HARMONICS_RUNTIME=`grep "Spectrum1 execution time" run_SCW_$i/optimisation-$i.out | cut -d ":" -f 2  -`
        LAPLACIAN1_RUNTIME=`grep "Laplacian1 execution time" run_SCW_$i/optimisation-$i.out | cut -d ":" -f 2  -`
        LAPLACIAN_LSF_RUNTIME=`grep "Laplacian2 execution time" run_SCW_$i/optimisation-$i.out | cut -d ":" -f 2  -`
        NOISE_RUNTIME=`grep "Noise execution time" run_SCW_$i/optimisation-$i.out | cut -d ":" -f 2  -`

        OVERALL_RUNTIME=`echo $CUBE1_RUNTIME + $FIBRES1_RUNTIME + $FIBRES3_RUNTIME + $HARMONICS_RUNTIME + $LAPLACIAN1_RUNTIME + $LAPLACIAN_LSF_RUNTIME + $NOISE_RUNTIME |bc -l`

        CUBE1_RUNTIME_DAYS=`echo "$CUBE1_RUNTIME/60.0/24.0" | bc -l`
        FIBRES1_RUNTIME_DAYS=`echo "$FIBRES1_RUNTIME/60.0/24.0" | bc -l`
        FIBRES3_RUNTIME_DAYS=`echo "$FIBRES3_RUNTIME/60.0/24.0" | bc -l`
        HARMONICS_RUNTIME_DAYS=`echo "$HARMONICS_RUNTIME/60.0/24.0" | bc -l`
        LAPLACIAN1_RUNTIME_DAYS=`echo "$LAPLACIAN1_RUNTIME/60.0/24.0" | bc -l`
        LAPLACIAN_LSF_RUNTIME_DAYS=`echo "$LAPLACIAN_LSF_RUNTIME/60.0/24.0" | bc -l`
        NOISE_RUNTIME_DAYS=`echo "$NOISE_RUNTIME/60.0/24.0" | bc -l`

        OVERALL_RUNTIME_DAYS=`echo "$OVERALL_RUNTIME/60.0/24.0" | bc -l`

        MATRIX1_PARAMS=`cat run_SCW_$i/cube.dat`
        X1=`echo $MATRIX1_PARAMS | cut -d " " -f 3  -`
        Y1=`echo $MATRIX1_PARAMS | cut -d " " -f 4  -`
        ROT1=`echo $MATRIX1_PARAMS | cut -d " " -f 5  -`
        W1=`echo $MATRIX1_PARAMS | cut -d " " -f 6  -`
        H1=`echo $MATRIX1_PARAMS | cut -d " " -f 7  -`

        X1=`printf '%.4f' $X1`
        Y1=`printf '%.4f' $Y1`
        ROT1=`printf '%.4f' $ROT1`
        W1=`printf '%.4f' $W1`
        H1=`printf '%.4f' $H1`

        X1=`echo "$X1 * 1024 * 1.9" | bc -l`
        Y1=`echo " $Y1 * 1024 * 1.9" | bc -l`
        ROT1=`echo "($ROT1 + 0.5) * 180.0" | bc -l`
        W1=`echo " ($W1 + 0.5) * 1024 * 1.9" | bc -l`
        H1=`echo " ($H1 + 0.5) * $W1" | bc -l`

        FIBRE1_PARAMS=`cat run_SCW_$i/fibre1_radii.dat`
        RADIUS1_CORE=`echo $FIBRE1_PARAMS | cut -d " " -f 3  -`
        RADIUS1_FIBRE=`echo $FIBRE1_PARAMS | cut -d " " -f 4  -`

        RADIUS1_CORE=`printf '%.4f' $RADIUS1_CORE`
        RADIUS1_FIBRE=`printf '%.4f' $RADIUS1_FIBRE`

        RADIUS1_CORE_PX=`echo "$RADIUS1_CORE/1.9" | bc -l`
        RADIUS1_FIBRE_PX=`echo "$RADIUS1_FIBRE/1.9" | bc -l`

        FIBRE3_PARAMS=`cat run_SCW_$i/fibre3_radii.dat`
        RADIUS3_CORE=`echo $FIBRE3_PARAMS | cut -d " " -f 3  -`
        RADIUS3_FIBRE=`echo $FIBRE3_PARAMS | cut -d " " -f 4  -`

        RADIUS3_CORE=`printf '%.4f' $RADIUS3_CORE`
        RADIUS3_FIBRE=`printf '%.4f' $RADIUS3_FIBRE`

        RADIUS3_CORE_PX=`echo "$RADIUS3_CORE/1.9" | bc -l`
        RADIUS3_FIBRE_PX=`echo "$RADIUS3_FIBRE/1.9" | bc -l`

        # TEMP=`grep -n "CUBE1 " run_SCW_$i/optimisation-$i.out | cut -d ":" -f 1`
        # TEMP=`echo $TEMP - 1 | bc`
        # TEMP=`tail -n+$TEMP run_SCW_$i/optimisation-$i.out | head -n1 | tr -s ' '`
        # MATRIX1_ITER=`echo $TEMP | cut -d " " -f 1`
        # MATRIX1_FEVAL=`echo $TEMP | cut -d " " -f 2`
        # MATRIX1_IND=`echo $MATRIX1_FEVAL / $MATRIX1_ITER | bc`
        #

        LAPLACIAN1_PARAMS=`cat run_SCW_$i/laplacian1.dat`
        LAPLACIAN1_SIGMA_CORE=`echo $LAPLACIAN1_PARAMS | cut -d " " -f 10  -`
        LAPLACIAN1_K_CORE=`echo $LAPLACIAN1_PARAMS | cut -d " " -f 11  -`
        LAPLACIAN1_SIGMA_FIBRE=`echo $LAPLACIAN1_PARAMS | cut -d " " -f 13  -`
        LAPLACIAN1_K_FIBRE=`echo $LAPLACIAN1_PARAMS | cut -d " " -f 14  -`
        LAPLACIAN1_SIGMA_MATRIX=`echo $LAPLACIAN1_PARAMS | cut -d " " -f 14  -`
        LAPLACIAN1_K_MATRIX=`echo $LAPLACIAN1_PARAMS | cut -d " " -f 15  -`

        LAPLACIAN1_RADIUS_CORE=`echo $LAPLACIAN1_PARAMS | cut -d " " -f 16  -`
        LAPLACIAN1_RADIUS_CORE=`printf '%.4f' $LAPLACIAN1_RADIUS_CORE`
        LAPLACIAN1_RADIUS_CORE_PX=`echo "$LAPLACIAN1_RADIUS_CORE/1.9" | bc -l`

        LAPLACIAN1_RADIUS_FIBRE=`echo $LAPLACIAN1_PARAMS | cut -d " " -f 17  -`
        LAPLACIAN1_RADIUS_FIBRE=`printf '%.4f' $LAPLACIAN1_RADIUS_FIBRE`
        LAPLACIAN1_RADIUS_FIBRE_PX=`echo "$LAPLACIAN1_RADIUS_FIBRE/1.9" | bc -l`


        LAPLACIAN_LSF_PARAMS=`cat run_SCW_$i/laplacian2.dat`
        LAPLACIAN_LSF_K_CORE=`echo $LAPLACIAN_LSF_PARAMS | cut -d " " -f 5  -`
        LAPLACIAN_LSF_K_FIBRE=`echo $LAPLACIAN_LSF_PARAMS | cut -d " " -f 6  -`
        LAPLACIAN_LSF_K_MATRIX=`echo $LAPLACIAN_LSF_PARAMS | cut -d " " -f 7  -`

        LAPLACIAN_LSF_PARAMS=`cat run_SCW_$i/lsf2.dat`
        LSF_a=`echo $LAPLACIAN_LSF_PARAMS | cut -d " " -f 8  -`
        LSF_b=`echo $LAPLACIAN_LSF_PARAMS | cut -d " " -f 9  -`
        LSF_c=`echo $LAPLACIAN_LSF_PARAMS | cut -d " " -f 10  -`
        LSF_d=`echo $LAPLACIAN_LSF_PARAMS | cut -d " " -f 11  -`
        LSF_e=`echo $LAPLACIAN_LSF_PARAMS | cut -d " " -f 13  -`
        LSF_f=`echo $LAPLACIAN_LSF_PARAMS | cut -d " " -f 14  -`

        NOISE_PARAMS=`cat run_SCW_$i/poisson-noise.dat`
        NOISE_BIAS=`echo $NOISE_PARAMS | cut -d " " -f 5  -`
        NOISE_GIAN=`echo $NOISE_PARAMS | cut -d " " -f 6  -`
        NOISE_SCALE=`echo $NOISE_PARAMS | cut -d " " -f 7  -`


        # TEMP=`grep -n "FIBRES1 " run_SCW_$i/optimisation-$i.out | cut -d ":" -f 1`
        # TEMP=`echo $TEMP - 1 | bc`
        # TEMP=`tail -n+$TEMP run_SCW_$i/optimisation-$i.out | head -n1 | tr -s ' '`
        # FIBRE1_ITER=`echo $TEMP | cut -d " " -f 1`
        # FIBRE1_FEVAL=`echo $TEMP | cut -d " " -f 2`
        # FIBRE1_IND=`echo $FIBRE1_FEVAL / $FIBRE1_ITER | bc`
        #
        # TEMP=`grep -n "FIBRES2 " run_SCW_$i/optimisation-$i.out | cut -d ":" -f 1`
        # TEMP=`echo $TEMP - 1 | bc`
        # TEMP=`tail -n+$TEMP run_SCW_$i/optimisation-$i.out | head -n1 | tr -s ' '`
        # FIBRE3_ITER=`echo $TEMP | cut -d " " -f 1`
        # FIBRE3_FEVAL=`echo $TEMP | cut -d " " -f 2`
        # FIBRE3_IND=`echo $FIBRE3_FEVAL / $FIBRE3_ITER | bc`
        #
        # TEMP=`grep -n "LAPLACIAN1 " run_SCW_$i/optimisation-$i.out | cut -d ":" -f 1`
        # TEMP=`echo $TEMP - 1 | bc`
        # TEMP=`tail -n+$TEMP run_SCW_$i/optimisation-$i.out | head -n1 | tr -s ' '`
        # LAPLACIAN1_ITER=`echo $TEMP | cut -d " " -f 1`
        # LAPLACIAN1_FEVAL=`echo $TEMP | cut -d " " -f 2`
        # LAPLACIAN1_IND=`echo $LAPLACIAN1_FEVAL / $LAPLACIAN1_ITER | bc`
        #
        # TEMP=`grep -n "LAPLACIAN-LSF " run_SCW_$i/optimisation-$i.out | cut -d ":" -f 1`
        # TEMP=`echo $TEMP - 1 | bc`
        # TEMP=`tail -n+$TEMP run_SCW_$i/optimisation-$i.out | head -n1 | tr -s ' '`
        # LAPLACIAN_LSF_ITER=`echo $TEMP | cut -d " " -f 1`
        # LAPLACIAN_LSF_FEVAL=`echo $TEMP | cut -d " " -f 2`
        # LAPLACIAN_LSF_IND=`echo $LAPLACIAN_LSF_FEVAL / $LAPLACIAN_LSF_ITER | bc`




        MEAN_CORE_REF=`grep "After noise CORE REF (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 13`
        STDDEV_CORE_REF=`grep "After noise CORE REF (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 14`

        MEAN_CORE_SIM=`grep "After noise CORE SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 13`
        STDDEV_CORE_SIM=`grep "After noise CORE SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 14`

        MEAN_FIBRE_REF=`grep "After noise FIBRE REF (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 13`
        STDDEV_FIBRE_REF=`grep "After noise FIBRE REF (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 14`

        MEAN_FIBRE_SIM=`grep "After noise FIBRE SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 13`
        STDDEV_FIBRE_SIM=`grep "After noise FIBRE SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 14`

        MEAN_MATRIX_REF=`grep "After noise MATRIX REF (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 13`
        STDDEV_MATRIX_REF=`grep "After noise MATRIX REF (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 14`

        MEAN_MATRIX_SIM=`grep "After noise MATRIX SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 13`
        STDDEV_MATRIX_SIM=`grep "After noise MATRIX SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 14`


        STATS=`python3 ../../imageStats.py $CT_slice_file`

        cols=`pwd | awk -F '/' '{print NF-1}'`
        cols=`echo $cols+1|bc`

        echo `pwd |cut -d "/" -f $cols`,$i,$MATRIX_ZNCC,$FIBRE1_ZNCC,$FIBRE2_ZNCC,$FIBRE3_ZNCC,$HARMONICS_ZNCC,$NOISE_ZNCC,$LAPLACIAN1_ZNCC,$LAPLACIAN_LSF_ZNCC,$CUBE1_RUNTIME,$FIBRES1_RUNTIME,$FIBRES3_RUNTIME,$HARMONICS_RUNTIME,$NOISE_RUNTIME,$LAPLACIAN1_RUNTIME,$LAPLACIAN_LSF_RUNTIME,$OVERALL_RUNTIME,$CUBE1_RUNTIME_DAYS,$FIBRES1_RUNTIME_DAYS,$FIBRES3_RUNTIME_DAYS,$HARMONICS_RUNTIME_DAYS,$NOISE_RUNTIME_DAYS,$LAPLACIAN1_RUNTIME_DAYS,$LAPLACIAN_LSF_RUNTIME_DAYS,$OVERALL_RUNTIME_DAYS,$X1,$Y1,$ROT1,$W1,$H1,$RADIUS1_CORE,$RADIUS1_FIBRE,$RADIUS1_CORE_PX,$RADIUS1_FIBRE_PX,$RADIUS3_CORE,$RADIUS3_FIBRE,$RADIUS3_CORE_PX,$RADIUS3_FIBRE_PX,`sed -n '2p' < run_SCW_$i/spectrum1.dat`,`sed -n '3p' < run_SCW_$i/spectrum1.dat`,`sed -n '4p' < run_SCW_$i/spectrum1.dat`,$NOISE_BIAS,$NOISE_GIAN,$NOISE_SCALE,$LAPLACIAN1_SIGMA_CORE,$LAPLACIAN1_K_CORE,$LAPLACIAN1_SIGMA_FIBRE,$LAPLACIAN1_K_FIBRE,$LAPLACIAN1_SIGMA_MATRIX,$LAPLACIAN1_K_MATRIX,$LAPLACIAN1_RADIUS_CORE,$LAPLACIAN1_RADIUS_FIBRE,$LAPLACIAN1_RADIUS_CORE_PX,$LAPLACIAN1_RADIUS_FIBRE_PX,$LAPLACIAN_LSF_K_CORE,$LAPLACIAN_LSF_K_FIBRE,$LAPLACIAN_LSF_K_MATRIX,$LSF_a,$LSF_b,$LSF_c,$LSF_d,$LSF_e,$LSF_f,$MEAN_CORE_REF,$MEAN_CORE_SIM,$STDDEV_CORE_REF,$STDDEV_CORE_SIM,$MEAN_FIBRE_REF,$MEAN_FIBRE_SIM,$STDDEV_FIBRE_REF,$STDDEV_FIBRE_SIM,$MEAN_MATRIX_REF,$MEAN_MATRIX_SIM,$STDDEV_MATRIX_REF,$STDDEV_MATRIX_SIM,$STATS
    else
        echo `pwd |cut -d "/" -f $cols`,$i,MISSING
    fi
done
