import configparser
import logging
from pyzbar import pyzbar
from predict import *
import PIL.Image
import PIL.ImageOps
from pystrich.ean13 import EAN13Encoder

def generate_barcode(assignment_id=""):
    id = "0" * (12 - len(assignment_id)) + assignment_id
    encoder = EAN13Encoder(id)
    encoder.save("barcode/barcode" + str(assignment_id) + ".png")

logger = logging.getLogger(__name__)

# 读入图片，cv读入时会自动旋转，说是和拍照方向有关，没有搞懂
def load_image_file(file, mode='RGB'):
    # Load the image with PIL
    img = PIL.Image.open(file)

    img = img.convert(mode)

    return np.array(img)

# 拍摄照片时图片容易有噪点，扫描好很多，但是也可以在效果不理想时使用
def noise_reduction(input=ORIGINAL_FILE):
    # 读取图片，为了不导致旋转直接读取
    image = load_image_file(input)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    # 通过腐蚀再膨胀去除噪点
    erosion = cv2.erode(gray, kernel, 1)
    gray = cv2.dilate(erosion, kernel, 1)
    # cv2.namedWindow("edges", 0)
    # cv2.resizeWindow("edges", 2000, 2000)
    # cv2.imshow("edges", gray)
    cv2.imwrite(DENOISE_FILE, gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# 删除不符合规则的框, axis指代方向，0为横向排列的数据，1为纵向，fix_gap指代几个框是等距排列的，expect_num为预期数量,
# 其中expect_num仅在fix_gap下产生待检测修复项，返回正确性，修复项，去除项
'''
主要目标: 过滤不对的位置，修复等距排列的上下不连贯导致的分割成两块的问题
           ++++   ++++          ++++               ++++   ++++   ++++   ++++     
           +  +   +  +          +  +               +  +   +  +   +  +   +  +     
           +  +   +  +          +  +   --->        +  +   +  +   +  +   +  +     
           +  +   +  +   ++++   +  +               +  +   +  +   +  +   +  +     
     +++   ++++   ++++   ++++   ++++               ++++   ++++   ++++   ++++     
     +++   
             1      2      3      4                  1      2      3      4
'''
def fix_linear_data(poss, axis=0, fix_gap=True, expect_num=None):
    from collections import defaultdict
    width_dic, length_dic, width_sum = defaultdict(int), defaultdict(int), defaultdict(int)
    width_most_common, width_most_num, length_most_common, length_most_num = 0, 0, 0, 0
    gap_dic = defaultdict(int)
    gap_most_common, gap_most_num = 0, 0
    if axis == 0:
        poss = sorted(poss, key=lambda x:x[0])
        moving_pos = [p[0] for p in poss]
        fix_pos = [p[1] for p in poss]
        last = poss[0][0]
    else:
        poss = sorted(poss, key=lambda x:x[1])
        moving_pos = [p[1] for p in poss]
        fix_pos = [p[0] for p in poss]
        last = poss[0][1]
    newposs, checkposs, removeposs = [], [], []
    print(poss)
    for ind, (x, y, w, h) in enumerate(poss):
        width_dic[w // 10] += 1
        length_dic[h // 10] += 1
        if ind != 0:
            gap_dic[(moving_pos[ind] - last) // 10] += 1
        width_sum[w // 10] += w
        if width_dic[w // 10] > width_most_num:
            width_most_num = width_dic[w // 10]
            width_most_common = w
        if length_dic[h // 10] > length_most_num:
            length_most_num = length_dic[h // 10]
            length_most_common = h
        if gap_dic[(moving_pos[ind] - last) // 10] > gap_most_num:
            gap_most_num = gap_dic[(moving_pos[ind] - last) // 10]
            gap_most_common = (moving_pos[ind] - last)
        last = moving_pos[ind]
    width_most_common = width_sum[width_most_common // 10] // width_dic[width_most_common // 10]
    print(width_most_common, length_most_common, gap_most_common)
    last = moving_pos[0] - gap_most_common
    for ind, (x, y, w, h) in enumerate(poss):
        # 宽度不对
        if relaxed_inrange(width_most_common, w) != 0:
            if expect_num != len(poss) or fix_gap == True:
                removeposs.append((x, y, w, h))
                continue
        # 长宽都对
        if inrange(length_most_common, h) == 0:
            last = moving_pos[ind]
        # 长度不对，进行修复
        elif inrange(length_most_common, h) == -1:
            # 位置重复默认修复过，跳过
            if inrange(last, moving_pos[ind]) == 0:
                removeposs.append((x, y, w, h))
                continue
            h = length_most_common
            # 不固定gap，无法修复进一步修复
            # 固定gap，根据gap修复错漏对象
            if fix_gap and axis == 0:
                while (x - last) / gap_most_common > 1.9:
                    x = last + gap_most_common
                    last = x
                    checkposs.append((x, y, w, h))
                # x = last + gap_most_common
                last = x
            if fix_gap and axis == 1:
                while (y - last) / gap_most_common > 1.9:
                    y = last + gap_most_common
                    last = y
                    checkposs.append((x, y, w, h))
                # y = last + gap_most_common
                last = y
        newposs.append((x, y, w, h))
    if fix_gap and expect_num != None and len(newposs) + len(checkposs) < expect_num:
        i = expect_num - len(newposs) + len(checkposs)
        new_check = []
        while i != 0:
            first = newposs[0]
            if axis == 0:
                new_check.append((first[0] - i * gap_most_common, first[1], first[2], first[3]))
            else:
                new_check.append((first[0], first[1] - i * gap_most_common, first[2], first[3]))
            last = newposs[-1]
            if axis == 0:
                new_check.append((last[0] + i * gap_most_common, last[1], last[2], last[3]))
            else:
                new_check.append((last[0], last[1] + i * gap_most_common, last[2], last[3]))
            i -= 1
        checkposs.extend(new_check)
    return newposs, checkposs, removeposs

# 根据比例判断是否符合大约相等
def inrange(mid, cur):
    if mid == 0:
        return 1
    if cur / mid < 0.9:
        return -1
    elif cur / mid > 1.1:
        return 1
    return 0


# 在比例基础上使用变化量（8 pixel）来放松检测机制
def relaxed_inrange(mid, cur):
    if mid == 0:
        return 1
    if abs(mid - cur) < 8:
        return 0
    if cur / mid < 0.9:
        return -1
    elif cur / mid > 1.1:
        return 1
    return 0


# 提取结果框，left指定答题框是在左侧还是右侧（left指标已废弃），up指定是否要削去首页的记录情况部分，其会影响识别
def extract_check_result_from_page(file, easy, upper_border):
    print(file)
    check_result = []
    image = load_image_file(file)
    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, edges = cv2.threshold(gray, THRESHOLD, 255, 1)
    original_edge = edges.copy()
    # 膨胀边框使其明显
    dilate_edge = cv2.dilate(edges, kernel, 1)
    # if up:
    #     upper_border = image.shape[0] // 3
    # else:
    #     upper_border = 0
    edges = dilate_edge[upper_border:image.shape[0], 0:image.shape[1] // 6]
    # cv2.namedWindow("edges", 0)
    # cv2.resizeWindow("edges", 400, 4000)
    # cv2.imshow("edges", edges)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    boxes = []
    minside, maxwidth = image.shape[1], 0
    oridnary_size = cv2.contourArea(contours[0])
    for i in range(min(len(contours),5)):
        largest_item = contours[i]
        x, y, w, h = cv2.boundingRect(largest_item)
        minside = min(minside, x)
        maxwidth = max(maxwidth, w)
    # print(left, upper_border, edges.shape[0])
    # cv2.namedWindow("edges", 0)
    # cv2.resizeWindow("edges", 400, 4000)
    # cv2.imshow("edges", edges)
    edges = dilate_edge[upper_border:image.shape[0], minside:minside + maxwidth + 15]
    original_edge_slice = original_edge[upper_border:image.shape[0], minside:minside + maxwidth + 15]
    # cv2.imshow("edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > oridnary_size * 0.6:
            boxes.append(contours[i])
    boxes = sorted(boxes, key=lambda x:cv2.boundingRect(x)[1])
    index_list = []
    for i in range(len(boxes)):
        largest_item = boxes[i]
        new_block, x = box_content_remove_margin(original_edge_slice, largest_item, 0)
        if easy:
            check = hasinside(new_block)
            check_result.append(check)
        else:
            check_result.append(new_block)
        x,y,w,h = cv2.boundingRect(largest_item)
        index_image = original_edge[upper_border + y + h - 70:upper_border + y + h - 10, minside + x + 80:minside + x+ w + 60]
        get_index(index_image, index_list)
        # cv2.imshow(str(i), new_block)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(len(check_result))
    return check_result, index_list

def hasinside(image):
    # cv2.imshow("img", image)
    image = image[image.shape[0] // 4 - 5: image.shape[0] // 4 * 3 + 5, image.shape[1] // 4 - 5: image.shape[1] // 4 * 3 + 5]
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    _,image = cv2.threshold(image,100,1, cv2.THRESH_BINARY)
    image_sum = sum(sum(image))
    if image_sum > 10 and image_sum > image.shape[0] * image.shape[1] // 20:
        return 1
    return 0

def inratiorange(item, ratio):
    x,y,w,h = cv2.boundingRect(item)
    if w == 0 or h == 0:
        return False
    if w / h < ratio or h / w < ratio:
        return False
    return True


def get_index(index_image, index_list):
    inner_contours, inner_hierarchy = cv2.findContours(index_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    inner_contours = sorted(inner_contours, key=cv2.contourArea, reverse=True)
    inner_contours = inner_contours[:2]
    # cv2.imshow(str(i + 1), index_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if len(inner_contours) > 1 and inrange(cv2.contourArea(inner_contours[0]), cv2.contourArea(inner_contours[1])) == 0:
        inner_contours = sorted(inner_contours, key=lambda x: cv2.boundingRect(x)[0])
        x, y, w, h = merge_two_contours(inner_contours[0], inner_contours[1])
        print(x, y, w, h)
        index_image = index_image[y: y + h, x:x + w]
    elif len(inner_contours) == 0:
        # cv2.imshow("1", index_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        index_list.append(0)
        return
    else:
        index_image = box_content_remove_margin(index_image, inner_contours[0], 0)[0]
    index = deal([index_image])
    try:
        index_list.append(index[0])
    except IndexError:
        index_list.append(0)



def merge_two_contours(contours1, contours2):
    x1,y1,w1,h1 = cv2.boundingRect(contours1)
    x2,y2,w2,h2 = cv2.boundingRect(contours2)
    return x1, min(y1, y2), x2 - x1 + w2, max(h2, h1)

# 从连续的方框中提取内容序列
def extract_data_sequence_from_rect(image=None, double_line=False, remove_boarder=False):
    if not isinstance(image,np.ndarray):
        image = load_image_file("grade_block.png")
        kernel = np.ones((5, 5), np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, image = cv2.threshold(image, THRESHOLD, 255, 1)
        # 膨胀边框使其明显
        edges = cv2.dilate(image, kernel, 1)
        edges = cv2.Canny(edges, 50, 150, apertureSize=3)
    else:
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges.copy(), 1, np.pi / 360,  edges.shape[1] // 4)
    upper_boarder, lower_boarder = 0, edges.shape[0]
    if lines is None:
        return None
    for line in lines:
        # print(line)
        rho = int(abs(line[0][0]))  # 第一个元素是距离rho
        theta = line[0][1]  # 第二个元素是角度theta
        if double_line and (theta < (6 * np.pi / 10)) and (theta > (4 * np.pi / 10)) and rho > edges.shape[0] / 4 \
                and rho < edges.shape[0] / 4 * 3:  # 水平直线
            upper_boarder = rho
            break
        if remove_boarder and (theta < (6 * np.pi / 10)) and (theta > (4 * np.pi / 10)) and rho < edges.shape[0] / 4 :
            upper_boarder = max(upper_boarder, rho)
        if remove_boarder and (theta < (6 * np.pi / 10)) and (theta > (4 * np.pi / 10)) and rho > edges.shape[0] / 4 * 3:
            lower_boarder = min(lower_boarder, rho)
    edges = edges[upper_boarder:lower_boarder,:]
    image = image[upper_boarder:lower_boarder,:]
    image[0:4,] = 0
    image[-4:,] = 0

    if lower_boarder - upper_boarder > 50:
        return None
    inner_contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = list(filter(lambda x:cv2.boundingRect(x)[2] > 10 and cv2.boundingRect(x)[3] > (lower_boarder - upper_boarder) // 2, inner_contours))
    contours = list(filter(lambda x:cv2.boundingRect(x)[3] > 10 , contours))
    contours = sorted(contours, key=lambda x:cv2.boundingRect(x)[0])
    image_list = []
    last_box = (0,0,0,0)
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        # print(x,y,w,h)
        new_block = image[y:y + h, x:x + w]
        # print(x, last_box[0])
        if x - last_box[0] > 25:
            image_list.append(new_block)
        else:
            image_list = image_list[:-1]
            x, y, w, h = merge_two_contours(contours[i - 1], contours[i])
            new_block = image[y:y + h, x:x + w]
            image_list.append(new_block)
            # try:
            #     cv2.imshow("a", new_block)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            # except:
            #     pass
        last_box = (x, y, w, h)
    return image_list


# 根据线的识别完成方格提取，按最大外边框提取
def box_by_line_detection(image_shape, lines):
    x,y,xr,yb = 0, 0, image_shape[1], image_shape[0]
    # x,y,xr,yb = image_shape[1] / 2, image_shape[0] / 2, image_shape[1] / 2, image_shape[0] / 2
    # print("rho")
    # print(x,y,xr,yb)
    for line in lines:
        rho = abs(line[0][0])  # 第一个元素是距离rho
        theta = line[0][1]  # 第二个元素是角度theta
        if (theta < (1 * np.pi / 10)) or (theta > (9 * np.pi / 10)):  # 垂直直线
            # pt1 = (int(rho / np.cos(theta)), 0)  # 该直线与第一行的交点
            # # 该直线与最后一行的焦点
            # pt2 = (int((rho - image_shape[0] * np.sin(theta)) / np.cos(theta)), image_shape[0])
            # cv2.line(result, pt1, pt2, (0, 150, 0))  # 绘制一条白线
            if rho > image_shape[1] // 2:
                xr = min(xr, rho)
            else:
                x = max(x, rho)
        elif (theta < (6 * np.pi / 10)) and (theta > (4 * np.pi / 10)):  # 水平直线
            # print(rho)
            # pt1 = (0, int(rho / np.sin(theta)))  # 该直线与第一列的交点
            # # 该直线与最后一列的交点
            # pt2 = (image_shape[1], int((rho - image_shape[1] * np.cos(theta)) / np.sin(theta)))
            # cv2.line(result, pt1, pt2, (0, 150, 0), 1)  # 绘制一条直线
            if rho > image_shape[0] // 2:
                yb = min(yb, rho)
            else:
                y = max(y, rho)

            # yb = max(yb, rho)
            # y = min(y, rho)
    return int(x),int(y),int(xr),int(yb)


# 去除图片的特定边框宽度
def box_content_remove_margin(edges, largest_item, margin):
    x, y, w, h = cv2.boundingRect(largest_item)
    new_block = edges[y + margin:y + h - margin, x + margin:x + w - margin]
    return new_block, x


# 根据试卷页眉右上方试卷编号读取试卷编号
def extract_assignment_number(file):
    image = load_image_file(file)
    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, edges = cv2.threshold(gray, THRESHOLD, 255, 1)
    edges = edges[0:image.shape[0] // 8, image.shape[1] // 2:]
    lines = cv2.HoughLines(edges.copy(), 1, np.pi / 360, edges.shape[1] // 4)
    for line in lines:
        rho = line[0][0]  # 第一个元素是距离rho
        theta = line[0][1]  # 第二个元素是角度theta
        if (theta < (6 * np.pi / 10)) and (theta > (4 * np.pi / 10)):
            edges = edges[0:int(rho) - 8, :]
            break
    # cv2.imshow("a", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 膨胀边框使其明显
    # edges = cv2.dilate(edges, kernel, 1)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda x:cv2.boundingRect(x)[0])
    image_sequence = []
    box = []
    for i in contours[-4:]:
        x, y, w, h = cv2.boundingRect(i)
        new_block = edges[y:y + h, x:x + w]
        box.append((x, y, w, h))
        # inner_edge = cv2.Canny(new_block, 50, 150, apertureSize=3)
        image_sequence.append(new_block)
        cv2.imwrite("assign_id.png", new_block)
        # cv2.imshow("a", new_block)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    newposs, checkposs, removeposs = fix_linear_data(box, axis=0, expect_num=4)
    check_expect_in_contours(checkposs, contours, newposs)
    newposs = sorted(newposs, key=lambda z:z[0])
    if newposs != box:
        print(newposs, checkposs, removeposs)

    # for x, y, w, h in newposs:
    #     new_block = edges[y:y + h, x:x + w]
    #     cv2.imshow("a", new_block)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ret = deal(image_sequence)
    ret = "".join(map(str,ret))
    return ret


# 检查一个预期位置是否是正确的对象
def check_expect_in_contours(checkposs, contours, newposs):
    if len(newposs) != 4:
        for i in checkposs:
            for j in contours:
                x, y, w, h = cv2.boundingRect(j)
                if not relaxed_inrange(i[0], x) and not relaxed_inrange(i[1], y) and not relaxed_inrange(i[2], w) \
                        and not relaxed_inrange(i[3], h):
                    # print(x, y, w, h)
                    newposs.append(i)


# 条形码识别
def BarcodeDetector(file):
    img = cv2.imread(file)
    # 条码识别
    barcodes = pyzbar.decode(img[:img.shape[0] // 2, :])
    if len(barcodes) == 0:
        student_id = None
    else:
        student_id = barcodes[0].data.decode("utf-8")

    barcodes = pyzbar.decode(img[img.shape[0] // 2:, :])
    if len(barcodes) == 0:
        print("None")
        assignment_id = None
        assignment_id = "0"
    else:
        assignment_id = barcodes[0].data.decode("utf-8")

    # 获取图片所有的条码
    # barcodes.sort(key=lambda x:x.rect[0])
    # student_id, assignment_id = barcodes[0].data.decode("utf-8"), None

    return assignment_id, student_id



def isfirstpage(file):
    assignment_id, student_id = BarcodeDetector(file)
    if student_id != None:
        return True
    # return False
    # 如果上面那个忘记贴二维码，会漏检测，也可以用这个，但是依赖识别会有漏
    image = load_image_file(file)
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, edges = cv2.threshold(gray, THRESHOLD, 255, 1)
    # 膨胀边框使其明显
    dilate_edge = cv2.dilate(edges, kernel, 1)
    edges = dilate_edge[image.shape[0] // 16 * 15:, image.shape[1] // 5 * 2:image.shape[1] // 5 * 3]
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = list(filter(lambda x:cv2.contourArea(x) > 150, contours))
    contours = sorted(contours, key=lambda x:cv2.boundingRect(x)[0])
    if len(contours) < 2:
        return False
    index_item = contours[1]
    x,y,w,h = cv2.boundingRect(index_item)
    index_image = edges[y:y + h, x:x + w]
    a = deal([index_image])
    if a == [1]:
        return True
    return False


# 两图象逐像素对比的函数
def compare(src, sample):
    same = 0.0
    resized_src = cv2.resize(src, (sample.shape[1], sample.shape[0]))
    resized_src = 255 - resized_src
    ret, resized_src = cv2.threshold(resized_src, THRESHOLD, 255, 1)
    row = resized_src.shape[0]
    col = resized_src.shape[1]
    for i in range(row):
        for j in range(col):
            a = resized_src[i][j]
            b = sample[i][j]
            if(a == b):
                same += 1
    return same / (row * col)


# 选取符合程度最高的数字
def deal(images):
    result = []
    samples = []
    for i in range(20):
        img = cv2.imread("printed_digit/" + str(i) + ".png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        samples.append(img)
    for i in range(len(images)):
        base = []
        for j in range(len(samples)):
            base.append((compare(images[i], samples[j]), j))
        base.sort(reverse=True)
        if base[0][0] > 0.6:
            result.append(base[0][1])
    return result


# 按排序选取变化程度最高的数字
def choose_change(images):
    samples = []
    for i in range(20):
        img = cv2.imread("printed_digit/" + str(i) + ".png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        samples.append(img)
    base = []
    for i in range(min(20, len(images))):
        base.append((compare(images[i], samples[i]), i))
    base.sort()
    if base[0][0] > 0.7:
        logging.warning("Nothing has changed significantly.")
    return base[0][1]


# 检查是否包含上方成绩统计框
def has_grade_block(file):
    image = load_image_file(file)
    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, edges = cv2.threshold(gray, THRESHOLD, 255, 1)
    # 膨胀边框使其明显
    dilate_edge = cv2.dilate(edges, kernel, 1)
    edges = dilate_edge[image.shape[0] // 4:image.shape[0] // 3 * 2, :]
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for i in range(2):
        x, y, w, h = cv2.boundingRect(contours[i])
        print(h)
        if h > image.shape[0] // 20 and x < image.shape[1] // 4 and x + w > image.shape[1] // 4 * 3:
            return image.shape[0] // 4 + y + h
    return 0


# 提取结果框，up指定是否要削去首页的记录情况部分，其会影响识别
def extract_score_result_from_page(file, upper_border):
    image = load_image_file(file)
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, edges = cv2.threshold(gray, THRESHOLD, 255, 1)
    edges = cv2.dilate(edges, kernel, 1)
    # if up:
    #     upper_border = image.shape[0] // 3
    # else:
    #     upper_border = 0
    edges = edges[upper_border:, :]
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = list(filter(lambda x:cv2.boundingRect(x)[2] > edges.shape[1] // 2, contours))
    contours = sorted(contours, key=lambda x:cv2.boundingRect(x)[1])
    result = []
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        item = edges[y:y+h, x :x +w ]
        images = extract_data_sequence_from_rect(item, remove_boarder=True)
        if images != None:
            result.append(choose_change(images))
    print(result)
    return result

# 试卷分析入口（main entry）
def extract_paper(file, require_denoise=True):
    if require_denoise:
        noise_reduction(file)
        file = DENOISE_FILE
    assignment_id, student_id = BarcodeDetector(file)
    up = 0
    if student_id != None:
        up = has_grade_block(file)
    check_results = extract_score_result_from_page(file, up)
    return student_id, assignment_id, check_results


config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")
easy = config.get("Prediction", "type")
# 主入口，作业分析（main entry）
def extract_assignment(file, require_denoise=True):
    # 现在未使用，原为标记区分两个模板
    # 图片中有大量噪声时建议使用，一般不使用
    if require_denoise:
        noise_reduction(file)
        file = DENOISE_FILE
    # 检测试卷号，学号，如无学号认为承接上一张卷子，有学号认为上一个人的卷子结束，进行分析
    assignment_id, student_id = BarcodeDetector(file)
    # assignment_id = extract_assignment_number(file)
    #检测是否包含成绩框
    up = 0
    if student_id != None:
        up = has_grade_block(file)
    a = easy =="easy"
    check_results, index_list = extract_check_result_from_page(file, a, up)
    print(index_list)
    return student_id, assignment_id, check_results, index_list
