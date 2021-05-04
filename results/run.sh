#!/bin/bash

DIRS=`ls`

echo "objective,i,MATRIX_ZNCC,FIBRE1_ZNCC,FIBRE2_ZNCC,FIBRE3_ZNCC,HARMONICS_ZNCC,NOISE_ZNCC,LAPLACIAN1_ZNCC,LAPLACIAN_LSF_ZNCC,CUBE1_RUNTIME (in min),FIBRES1_RUNTIME (in min),FIBRES3_RUNTIME (in min),HARMONICS_RUNTIME (in min),NOISE_RUNTIME (in min),LAPLACIAN1_RUNTIME (in min),LAPLACIAN_LSF_RUNTIME (in min),OVERALL_RUNTIME (in min),CUBE1_RUNTIME (in hh:mm:ss),FIBRES1_RUNTIME (in hh:mm:ss),FIBRES3_RUNTIME (in hh:mm:ss),HARMONICS_RUNTIME (in hh:mm:ss),NOISE_RUNTIME (in hh:mm:ss),LAPLACIAN1_RUNTIME (in hh:mm:ss),LAPLACIAN_LSF_RUNTIME (in hh:mm:ss),OVERALL_RUNTIME (in hh:mm:ss),X1 (in um),Y1 (in um),ROT1 (in degree),W1 (in um),H1 (in um),RADIUS1_CORE (in um),RADIUS1_FIBRE (in um),RADIUS1_CORE_PX (in pixels),RADIUS1_FIBRE_PX (in pixels),RADIUS3_CORE (in um),RADIUS3_FIBRE (in um),RADIUS3_CORE_PX (in pixels),RADIUS3_FIBRE_PX (in pixels),33keV_WEIGHT,66keV_WEIGHT,99keV_WEIGHT,NOISE_BIAS,NOISE_GIAN,NOISE_SCALE,LAPLACIAN1_SIGMA_CORE,LAPLACIAN1_K_CORE,LAPLACIAN1_SIGMA_FIBRE,LAPLACIAN1_K_FIBRE,LAPLACIAN1_SIGMA_MATRIX,LAPLACIAN1_K_MATRIX,LAPLACIAN1_RADIUS_CORE (in um),LAPLACIAN1_RADIUS_FIBRE (in um),LAPLACIAN1_RADIUS_CORE_PX (in pixels),LAPLACIAN1_RADIUS_FIBRE_PX (in pixels),LAPLACIAN_LSF_K_CORE,LAPLACIAN_LSF_K_FIBRE,LSF_a,LSF_b,LSF_c,LSF_d,LSF_e,LSF_f,LAPLACIAN_LSF_K_MATRIX,MEAN_CORE_REF,MEAN_CORE_SIM,STDDEV_CORE_REF,STDDEV_CORE_SIM,MEAN_FIBRE_REF,MEAN_FIBRE_SIM,STDDEV_FIBRE_REF,STDDEV_FIBRE_SIM,MEAN_MATRIX_REF,MEAN_MATRIX_SIM,STDDEV_MATRIX_REF,STDDEV_MATRIX_SIM,AVG,STDDEV,MIN,MAX" \
  >summary.csv


for dir in $DIRS
do
    if [ -d $dir ]
    then
        #if [ ! -f $dir/summary.csv ]
        #then
            echo $dir

            #echo "cp /home/fpvidal/PROGRAMMING/GitHub/Fly4Cyl/DONE/FIBRES1/$dir/summary.ods $dir/summary.ods"
            #cp /home/fpvidal/PROGRAMMING/GitHub/Fly4Cyl/DONE/LAPLACIAN/REGISTRATION_SINOGRAM_NORMALISED_RMSE/summary.ods $dir/summary.ods

            cd $dir;sh ../extract_duration.sh > summary.csv 2>/dev/null;cd ..
            tail -n +2 $dir/summary.csv >> summary.csv
       #fi
    fi
done
