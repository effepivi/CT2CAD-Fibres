#!/bin/bash

export ITK_GLOBAL_NUMBER_OF_THREADS=16
#export PATH=/data/id19/bones3/simon/src/rtk/lin64/bin:$PATH
#export PATH=/data/id19/bones3/simon/src/superbuild/Build/VV/bin:$PATH
#MAIN_DIR=/data/id19/bones3/simon/phantom_0.7um_20.5_1_
#OUTPUT_DIR=/tmp

SPACING=0.5 #0019 # in mm
SID=0. #`echo 140*1000 |bc` # in mm
SDD=0. #`echo $SID+80 |bc`  # in mm
DETECTOR_WIDTH=1024 # in px
DETECTOR_HEIGHT=10 # in px
ORIGIN=`echo $DETECTOR_WIDTH*$SPACING/2.-$SPACING/2. | bc -l` 


rtksimulatedgeometry \
       -o $PWD/geometry \
       -a 180 \
       -f 0 \
       -n 900 \
       --sdd $SDD \
       --sid $SID \
       --proj_iso_x=0. \
       --proj_iso_y=0.


#--proj_iso_x=-1033.417253
#(-1033.417253-0.5)*0.7


rtkfdk -p . \
       -r projections_3d.mha \
       -g $PWD/geometry \
       -o $PWD/fdk.mha \
       --lowmem \
       --dimension $DETECTOR_WIDTH,$DETECTOR_WIDTH,$DETECTOR_HEIGHT \
       --spacing $SPACING,$SPACING,$SPACING \
       --origin -$ORIGIN,-$ORIGIN,0. \
       --verbose \
       --direction 1,0,0,0,0,1,0,1,0 \
       --hardware cuda

exit
clitkImageArithm \
      --input1 $OUTPUT_DIR/fdk.mha \
      --output $OUTPUT_DIR/fdk.mha \
      --scalar 10000 \
      -t 1

rtkfieldofview -g $MAIN_DIR/geometry \
               -p $MAIN_DIR \
               -r "phantom_0.7um_20.5_1_[0-9]*edf" \
               --reconstruction $OUTPUT_DIR/fdk.mha \
               -o $OUTPUT_DIR/fdk.mha

sed -i "s/TransformMatrix = 1 0 0 0 0 1 0 1 0/TransformMatrix = 1 0 0 0 1 0 0 0 1/1" $OUTPUT_DIR/fdk.mha
sed -i "s/Offset = .*/Offset = 0 0 0/1" $OUTPUT_DIR/fdk.mha 

