from common import *
from utility.file import *
from utility.draw import *
from dataset.reader import *
import glob, xlwt, os

def run_make_test_annotation():
    #image_files = glob.glob(DATA_DIR + '/__download__/stage1_test/*')
    image_files = glob.glob(config['param'].get('img_folder')+'/*')
    ids = [x.split('/')[-1] for x in image_files]
    data_dir = DATA_DIR + '/image/stage1_test'
    os.makedirs(data_dir + '/multi_masks', exist_ok=True)
    os.makedirs(data_dir + '/overlays', exist_ok=True)
    os.makedirs(data_dir + '/images', exist_ok=True)
    for i in range(len(ids)):
        name   = ids[i]
        image_file = DATA_DIR + '/__download__/stage1_test/%s/images/%s.png'%(name,name)
        #image
        image = cv2.imread(image_file,cv2.IMREAD_COLOR)
        ## save and show -------------------------------------------
        #image_show('image',image)
        cv2.imwrite(DATA_DIR + '/image/stage1_test/images/%s.png'% name,image)
        #cv2.waitKey(1)

def run_make_train_annotation():

    # split = 'train1_ids_external'
    # ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')
    image_files = glob.glob(DATA_DIR + '/__download__/stage1_train/*')
    ids = [x.split('/')[-1] for x in image_files]
    data_dir = DATA_DIR + '/image/stage1_train'
    os.makedirs(data_dir + '/multi_masks', exist_ok=True)
    os.makedirs(data_dir + '/overlays', exist_ok=True)
    os.makedirs(data_dir + '/images', exist_ok=True)
    for i in range(len(ids)):
        name = ids[i]
        image_files = glob.glob(DATA_DIR + '/__download__/stage1_train/%s/images/*.png'%(name))
        assert(len(image_files)==1)
        image_file=image_files[0]
        #image
        image = cv2.imread(image_file,cv2.IMREAD_COLOR)
        H,W,C      = image.shape
        multi_mask = np.zeros((H,W), np.int32)
        mask_files = glob.glob(DATA_DIR + '/__download__/stage1_train/%s/masks/*.png'%(name))
        mask_files.sort()
        num_masks = len(mask_files)
        for i in range(num_masks):
            mask_file = mask_files[i]
            mask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
            multi_mask[np.where(mask>128)] = i+1
        #check
        color_overlay   = multi_mask_to_color_overlay  (multi_mask,color='summer')
        color1_overlay  = multi_mask_to_contour_overlay_no_detection(multi_mask,color_overlay,[255,255,255])
        contour_overlay = multi_mask_to_contour_overlay_no_detection(multi_mask,image,[0,255,0])
        all = np.hstack((image, contour_overlay,color1_overlay,)).astype(np.uint8)
        # cv2.imwrite(data_dir +'/images/%s.png'%(name),image)
        np.save(data_dir + '/multi_masks/%s.npy' % name, multi_mask)
        cv2.imwrite(data_dir + '/multi_masks/%s.png' % name, color_overlay)
        cv2.imwrite(data_dir + '/overlays/%s.png' % name, all)
        cv2.imwrite(data_dir + '/images/%s.png' % name, image)

        #image_show('all', all)
        #cv2.waitKey(1)

#
def get_excel_id():
    data = '/home/sychen/code/kaggle/data/__download__/stage1_train/'
    files = glob.glob(os.path.join(data + 'TCGA'+'*'))
    for i in range(len(files)):
        files[i] = files[i].split('/')[-1]
    data=xlwt.Workbook()
    table=data.add_sheet('name')
    excel = '/home/sychen/code/kaggle/data/split/TCGA_ID.xlsx'
    for i in range(len(files)):
        table.write(i,0, files[i])
    data.save(excel)

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_make_train_annotation()
    run_make_test_annotation()
    # get_excel_id()
    print( 'sucess!')
