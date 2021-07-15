#/usr/bin/env python3

python3 \
    tutorial.py \
    --input data/sino.raw \
    --output user_study/results-median \
    --metrics RMSE \
    --normalisation \
    --sinogram  > user_study/results-median/output.txt
