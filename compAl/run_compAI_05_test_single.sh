
clear

Policy='individual'

if [ -d ./submit ]; then
    rm -fr ./submit
fi

mkdir -p ./submit/merge

echo 'ImageId,EncodedPixels'  > ./submit/merge/stage2_submit.csv
#declare -a arr=("Brightfield" "Fluorescence" "HE" "IHC" "GEN")
declare -a arr=("GEN")
## now loop through the above array
for i in "${arr[@]}"
do
   #echo "$i"
   # or do whatever with individual element of the array
    echo "............$Policy policy for $i............"
    sleep 1

    if [ ! -d ./submit/$i ]; then
        mkdir ./submit/$i
    fi

    cd config
    result=`python config_setting.py $Policy $i`

    if [ "$result" == "UNET" ]; then
        echo "use UNET model for $i"
        cd ../DSB2018-cam-ex5
        python valid_category.py
        cp ./submit/test_result.csv ../submit/$i/
        sed '1d' ./submit/test_result.csv >> ../submit/merge/stage2_submit.csv
        cd ..
    fi
    if [ "$result" == "MASK_RCNN" ]; then
        echo "use MASK RCNN model for $i"
        cd ../mask_rcnn_v01
        python submit_category.py
        cp ./results/submission_test_category_test.csv ../submit/$i/
        sed '1d' ./results/submission_test_category_test.csv >> ../submit/merge/stage2_submit.csv        
        cd ..
    fi
    
done
