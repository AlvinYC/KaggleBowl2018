from common import *
from net.lib.box.process import *
import matplotlib.cm


# draw -----------------------------------
def image_show(name, image, resize=1):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))


def draw_shadow_text(img, text, pt,  fontScale, color, thickness, color1=None, thickness1=None):
    if color1 is None: color1=(0,0,0)
    if thickness1 is None: thickness1 = thickness+2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color1, thickness1, cv2.LINE_AA)
    cv2.putText(img, text, pt, font, fontScale, color,  thickness,  cv2.LINE_AA)



##http://stackoverflow.com/questions/26690932/opencv-rectangle-with-dotted-or-dashed-lines
def draw_dotted_line(image, pt1, pt2, color, thickness=1, gap=20):

    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if gap==1:
        for p in pts:
            cv2.circle(image,p,thickness,color,-1,cv2.LINE_AA)
    else:
        def pairwise(iterable):
            "s -> (s0, s1), (s2, s3), (s4, s5), ..."
            a = iter(iterable)
            return zip(a, a)

        for p, q in pairwise(pts):
            cv2.line(image,p, q, color,thickness,cv2.LINE_AA)


def draw_dotted_poly(image, pts, color, thickness=1, gap=20):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        draw_dotted_line(image,s,e,color,thickness,gap)


def draw_dotted_rect(image, pt1, pt2, color, thickness=1, gap=3):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])]
    draw_dotted_poly(image, pts, color, thickness, gap)

def draw_screen_rect(image, pt1, pt2, color, alpha=0.5):
    x1, y1 = pt1
    x2, y2 = pt2
    image[y1:y2,x1:x2,:] = (1-alpha)*image[y1:y2,x1:x2,:] + (alpha)*np.array(color, np.uint8)



# def draw_mask(image, mask, color=(255,255,255), α=1,  β=0.25, λ=0., threshold=32 ):
#     # image * α + mask * β + λ
#
#     if threshold is None:
#         mask = mask/255
#     else:
#         mask = clean_mask(mask,threshold,1)
#
#     mask  = np.dstack((color[0]*mask,color[1]*mask,color[2]*mask)).astype(np.uint8)
#     image[...] = cv2.addWeighted(image, α, mask, β, λ)
#


# def draw_contour(image, mask, color=(0,255,0), thickness=1, threshold=127):
#     ret, thresh = cv2.threshold(mask,threshold,255,0)
#     ret = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     hierarchy = ret[0]
#     contours  = ret[1]
#     #image[...]=image
#     cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)
#     ## drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None): # real signature unknown; restored from __doc__
#
#
def draw_rcnn_detection_nms(image, detections, threshold=0.8):
    
    image = image.copy()
    for det in detections:
        s = det[5].data.cpu().numpy()
        # print ('s', s)
        if s<threshold: continue

        b = det[1:5]
        color = to_color(s, [255,0,255])
        cv2.rectangle(image,(b[0], b[1]), (b[2], b[3]), color, 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # prob_round = round(det[5], 3)
        # Label_round = round(det[6], 1)
        prob_round = np.around(det[5].data.cpu().numpy(), 3)
        # print ('prob', prob_round)
        Label_round = np.around(det[6].data.cpu().numpy(), 0)
        cv2.putText(image, str(prob_round), (b[2],b[3]), font, .3, (0, 255, 0), 1, 2)
        cv2.putText(image,str(Label_round), (b[2],b[1]), font, .3, (0, 255, 0), 1, 2)

            # name  = dataset.annotation.NAMES[j]
            # text  = '%02d %s : %0.3f'%(label,name,s)
            # fontFace  = cv2.FONT_HERSHEY_SIMPLEX
            # fontScale = 0.5
            # textSize = cv2.getTextSize(text, fontFace, fontScale, 2)
            # cv2.putText(img, text,(b[0], (int)((b[1] + 2*textSize[1]))), fontFace, fontScale, (0,0,0), 2, cv2.LINE_AA)
            # cv2.putText(img, text,(b[0], (int)((b[1] + 2*textSize[1]))), fontFace, fontScale, (255,255,255), 1, cv2.LINE_AA)
    return image

def draw_rpn_proposal_before_nms(image, prob_flat, delta_flat, windows, threshold=0.95):
    #prob  = prob_flat.cpu().data.numpy()
    #delta = delta_flat.cpu().data.numpy()

    image = image.copy()
    height,width = image.shape[0:2]
    prob  = prob_flat
    delta = delta_flat
    prob_max = np.max(prob, axis = 1)
    index = np.argsort(prob_max)  #sort descend #[::-1]
    if threshold<0:
        threshold = np.percentile(prob,99.8)

    num_windows = len(windows)
    for i in range(3):
        #if insides[i]==0: continue #ignore bounday
        s = prob[i]
        if s.all()<threshold:  continue

        w = windows[i]
        max_index = np.argmax(s)
        d = delta[i , max_index]

        b = box_transform_inv(w.reshape(1,4), d.reshape(1,4))
        b = clip_boxes(b, width, height)
        b = b.reshape(-1)


        color_w = to_color(s[max_index], [255,255,255])
        color_b = to_color(s[max_index], [0,0,255])
        #draw_dotted_rect(image,(w[0], w[1]), (w[2], w[3]),color_w , 1)
        cv2.rectangle(image,(b[0], b[1]), (b[2], b[3]), color_b, 1)
    print((prob>threshold).sum())
    return image
    
def to_color(s, color=None):

    if type(color) in [str] or color is None:
        #https://matplotlib.org/xkcd/examples/color/colormaps_reference.html

        if color is None: color='cool'
        color = matplotlib.get_cmap(color)(s)
        b = int(255*color[2])
        g = int(255*color[1])
        r = int(255*color[0])

    elif type(color) in [list,tuple]:
        b = int(s*color[0])
        g = int(s*color[1])
        r = int(s*color[2])

    return b,g,r



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    image = np.zeros((50,50,3), np.uint8)
    cv2.rectangle(image, (0,0),(49,49), (0,0,255),1) #inclusive

    image[8,8]=[255,255,255]

    image_show('image',image,10)
    cv2.waitKey(0)


    print('\nsucess!')