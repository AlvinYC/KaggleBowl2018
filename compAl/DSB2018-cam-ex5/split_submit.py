import os
import argparse
import numpy as np
import csv
from pathlib import Path
import PIL
from skimage.morphology import label

def main(csv_src, category_ids_file):
    cate_ids_file = Path(category_ids_file).expanduser()
    with cate_ids_file.open('r') as f:
        reader = csv.reader(f)
        cate_ids = []
        for row in reader:
            cate_ids.extend(row)
    # print(cate_ids)

    csv_src_file = Path(csv_src).expanduser()
    csv_out_file = csv_src_file.parent / ('submit_' + cate_ids_file.stem + '.csv')
    print('Saving %s...  Done!'%csv_out_file)
    with csv_src_file.open('r') as srcf:
        with csv_out_file.open('w') as outf:
            reader = csv.reader(srcf, delimiter=',')
            writer = csv.writer(outf)

            header = next(reader) # header = 'ImageId,EncodedPixels'
            writer.writerow(header)
            _ids = [] # to store unconcern image id
            for row in reader:
                if len(row) > 2:
                    print('[ERROR] ', row)
                    raise ValueError
                img_id = row[0]
                if img_id in cate_ids:
                    writer.writerow(row)
                else:
                    if img_id not in _ids:
                        _ids.append(img_id)
                        writer.writerow([img_id, '0 0'])

if __name__ == '__main__':
    # > python split_submit.py -src data/unet_0.498.csv -ca data/split/stage1_test_HE[4]
    #

    parser = argparse.ArgumentParser()
    parser.add_argument('-src', type=str, help='the mask files path')
    parser.add_argument('-ca', type=str, help='category image id file')
#    parser.set_defaults(src='test_result.csv', ca='data/split/stage1_test_IHC[8]')
    parser.set_defaults(src='test_result.csv', ca='data/split/stage1_test_HE[4]')
    args = parser.parse_args()

    print("Categorize different submission files")
    if 0:
        main(args.src, args.ca)
    else:
        main('submit/test_result.csv', 'data/split/stage1_test_HE[4]')
        main('submit/test_result.csv', 'data/split/stage1_test_IHC[8]')
        main('submit/test_result.csv', 'data/split/stage1_test_Cloud[4]')
        main('submit/test_result.csv', 'data/split/stage1_test_Drosophilidae[1]')
        main('submit/test_result.csv', 'data/split/stage1_test_easyFluorescence[48]')
        
