from abc import abstractmethod
from extract_result import noise_reduction, BarcodeDetector, load_image_file, inrange, box_content_remove_margin, \
    merge_two_contours, hasinside,inratiorange
from constants import *
import numpy as np
import cv2
from image_comparison import Image_Comparsion

class Model:
    ic = Image_Comparsion()
    def __init__(self):
        self.student_id = None
        self.assignment_id = None
        self.result = []
        self.questions_number = 0
        self.grade_block=False
        self.grade_lower_boarder = 0
        self.pics = []
        self.boxes = []

    def print_all_papers(self):
        for ind, paper in enumerate(self.pics):
            cv2.imshow("paper " + str(ind), paper)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @abstractmethod
    def extract(self, file, require_denoise=True):
        pass

    # 检查是否包含上方成绩统计框
    def has_grade_block(self, file):
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
            if h > image.shape[0] // 20 and x < image.shape[1] // 4 and x + w > image.shape[1] // 4 * 3:
                self.grade_block = True
                self.grade_lower_boarder = image.shape[0] // 4 + y + h
                return image.shape[0] // 4 + y + h
        self.grade_lower_boarder = 0
        return 0

    def remove_side(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,gray = cv2.threshold(gray,100,1, cv2.THRESH_BINARY)
        up = 0
        for i in range(gray.shape[0]// 20):
            if sum(gray[i]) < gray.shape[1] // 4 * 3:
                up += 1
            else:
                break
        gray = np.transpose(gray)
        left = 0
        for i in range(gray.shape[0]// 20):
            if sum(gray[i]) < gray.shape[1] // 4 * 3:
                left += 1
            else:
                break
        return image[up:, left:]


    # 根据试卷页眉右上方试卷编号读取试卷编号
    # def extract_assignment_number(self, file):
    #     image = load_image_file(file)
    #     kernel = np.ones((5, 5), np.uint8)
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     ret, edges = cv2.threshold(gray, THRESHOLD, 255, 1)
    #     edges = edges[0:image.shape[0] // 8, image.shape[1] // 2:]
    #     lines = cv2.HoughLines(edges.copy(), 1, np.pi / 360, edges.shape[1] // 4)
    #     for line in lines:
    #         rho = line[0][0]  # 第一个元素是距离rho
    #         theta = line[0][1]  # 第二个元素是角度theta
    #         if (theta < (6 * np.pi / 10)) and (theta > (4 * np.pi / 10)):
    #             edges = edges[0:int(rho) - 8, :]
    #             break
    #     # cv2.imshow("a", edges)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     # 膨胀边框使其明显
    #     # edges = cv2.dilate(edges, kernel, 1)
    #     contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    #     image_sequence = []
    #     box = []
    #     for i in contours[-4:]:
    #         x, y, w, h = cv2.boundingRect(i)
    #         new_block = edges[y:y + h, x:x + w]
    #         box.append((x, y, w, h))
    #         # inner_edge = cv2.Canny(new_block, 50, 150, apertureSize=3)
    #         image_sequence.append(new_block)
    #         cv2.imwrite("assign_id.png", new_block)
    #         # cv2.imshow("a", new_block)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #     newposs, checkposs, removeposs = fix_linear_data(box, axis=0, expect_num=4)
    #     check_expect_in_contours(checkposs, contours, newposs)
    #     newposs = sorted(newposs, key=lambda z: z[0])
    #     if newposs != box:
    #         print(newposs, checkposs, removeposs)
    #
    #     # for x, y, w, h in newposs:
    #     #     new_block = edges[y:y + h, x:x + w]
    #     #     cv2.imshow("a", new_block)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     ret = deal(image_sequence)
    #     ret = "".join(map(str, ret))
    #     return ret


class Assignment_Model(Model):
    def __init__(self, easy=False):
        Model.__init__(self)
        self.index_list = []
        self.easy = easy

    # 主入口，作业分析（main entry）
    def extract(self, file, require_denoise=True):
        # 图片中有大量噪声时建议使用，一般不使用
        if require_denoise:
            noise_reduction(file)
            file = DENOISE_FILE
        # assignment_id = extract_assignment_number(file)
        # 检测试卷号，学号，如无学号认为承接上一张卷子，有学号认为上一个人的卷子结束，进行分析
        assignment_id, student_id = BarcodeDetector(file)
        if self.assignment_id == None:
            self.assignment_id =  assignment_id if assignment_id != None else "000000000001"
        up = 0
        # 检测是否包含成绩框
        if student_id != None:
            up = self.has_grade_block(file)
            self.student_id = student_id
        self.extract_check_result_from_page(file, self.easy, up)

    def get_index(self, index_image, index_list):
        inner_contours, inner_hierarchy = cv2.findContours(index_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        inner_contours = sorted(inner_contours, key=cv2.contourArea, reverse=True)
        inner_contours = list(filter(lambda x: cv2.contourArea(x) > index_image.size / 40, inner_contours))
        inner_contours = inner_contours[:2]
        # cv2.imshow(str(i + 1), index_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if len(inner_contours) > 1:
            # and inrange(cv2.contourArea(inner_contours[0]), cv2.contourArea(inner_contours[1])) == 0:
            inner_contours = sorted(inner_contours, key=lambda x: cv2.boundingRect(x)[0])
            x, y, w, h = merge_two_contours(inner_contours[0], inner_contours[1])
            # print(x, y, w, h)
            index_image = index_image[y: y + h, x:x + w]
        elif len(inner_contours) == 0:
            index_list.append(0)
            return
        else:
            index_image = box_content_remove_margin(index_image, inner_contours[0], 0)[0]
        index = Model.ic.deal([index_image])
        try:
            self.index_list.append(index[0])
        except IndexError:
            self.index_list.append(0)

    # 提取结果框，left指定答题框是在左侧还是右侧（left指标已废弃），up指定是否要削去首页的记录情况部分，其会影响识别
    def extract_check_result_from_page(self, file, easy, upper_border):
        print(file)
        check_result = []
        image = load_image_file(file)
        # image = image[:, image.shape[1] // 12:]
        # new_image = self.remove_side(image)
        # if image.shape != new_image.shape:
        #     print(image.shape, new_image.shape)
            # cv2.namedWindow("edges", 0)
            # cv2.resizeWindow("edges", 800, 1000)
            # cv2.imshow("edges", image)
            # cv2.namedWindow("edges1", 0)
            # cv2.resizeWindow("edges1", 800, 1000)
        #    cv2.imshow("edges1", new_image)
        # image = new_image
        self.pics.append(image)
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
        # print(upper_border)
        # cv2.namedWindow("edges", 0)
        # cv2.resizeWindow("edges", 200, 1500)
        # cv2.imshow("edges", edges)
        contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = list(filter(lambda x: inratiorange(x, 1 / 3), contours))
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        boxes = []
        minside, maxwidth = image.shape[1], 0
        oridnary_size = cv2.contourArea(contours[0])
        for i in range(min(len(contours), 5)):
            largest_item = contours[i]
            x, y, w, h = cv2.boundingRect(largest_item)
            minside = min(minside, x)
            maxwidth = max(maxwidth, w)
        # print(left, upper_border, edges.shape[0])
        # cv2.namedWindow("edges", 0)
        # cv2.resizeWindow("edges", 200, 1500)
        # cv2.imshow("edges", edges)
        # minside = image.shape[1] // 12 + minside
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
        boxes = sorted(boxes, key=lambda x: cv2.boundingRect(x)[1])
        index_list = []
        poses = []
        for i in range(len(boxes)):
            largest_item = boxes[i]
            new_block, x = box_content_remove_margin(original_edge_slice, largest_item, 0)
            x, y, w, h = cv2.boundingRect(largest_item)
            poses.append((x, y, w, h))
            index_image = original_edge[upper_border + y + h // 8:upper_border + y + h // 8 * 7,
                          minside + x + w // 8 * 10:minside + x + w // 8 * 17]
            self.get_index(index_image, index_list)
            if easy:
                check = hasinside(new_block)
                check_result.append(check)
            else:
                check_result.append(new_block)
            # cv2.imshow(str(i), index_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.boxes.extend(poses)
        self.result.extend(check_result)
        self.index_list.extend(index_list)


class Paper_Model(Model):
    def __init__(self):
        Model.__init__(self)

    # 试卷分析入口（main entry）
    def extract(self, file, require_denoise=True):
        if require_denoise:
            noise_reduction(file)
            file = DENOISE_FILE
        up = 0
        self.assignment_id, student_id = BarcodeDetector(file)
        if student_id != None:
            up = self.has_grade_block(file)
            self.student_id = student_id
        self.extract_score_result_from_page(file, up)

    # 提取结果框，up指定是否要削去首页的记录情况部分，其会影响识别
    def extract_score_result_from_page(self, file, upper_border):
        image = load_image_file(file)
        image = image[:, image.shape[1] // 12:]
        self.pics.append(image)
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
        contours = list(filter(lambda x: cv2.boundingRect(x)[2] > edges.shape[1] // 2, contours))
        contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[1])
        result = []
        poses = []
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            item = edges[y:y + h, x:x + w]
            images = self.extract_data_sequence_from_rect(item, remove_boarder=True)
            poses.append((x, y, w, h))
            if images != None:
                result.append(Model.ic.choose_change(images))
        print(result)
        self.boxes.append(poses)
        self.result.extend(result)

    # 从连续的方框中提取内容序列
    def extract_data_sequence_from_rect(self, image=None, double_line=False, remove_boarder=False):
        if not isinstance(image, np.ndarray):
            image = load_image_file("grade_block.png")
            kernel = np.ones((5, 5), np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, image = cv2.threshold(image, THRESHOLD, 255, 1)
            # 膨胀边框使其明显
            edges = cv2.dilate(image, kernel, 1)
            edges = cv2.Canny(edges, 50, 150, apertureSize=3)
        else:
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges.copy(), 1, np.pi / 360, edges.shape[1] // 4)
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
            if remove_boarder and (theta < (6 * np.pi / 10)) and (theta > (4 * np.pi / 10)) and rho < edges.shape[
                0] / 4:
                upper_boarder = max(upper_boarder, rho)
            if remove_boarder and (theta < (6 * np.pi / 10)) and (theta > (4 * np.pi / 10)) and rho > edges.shape[
                0] / 4 * 3:
                lower_boarder = min(lower_boarder, rho)
        edges = edges[upper_boarder:lower_boarder, :]
        image = image[upper_boarder:lower_boarder, :]
        image[0:4, ] = 0
        image[-4:, ] = 0

        if lower_boarder - upper_boarder > 50:
            return None
        inner_contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = list(filter(
            lambda x: cv2.boundingRect(x)[2] > 10 and cv2.boundingRect(x)[3] > (lower_boarder - upper_boarder) // 2,
            inner_contours))
        contours = list(filter(lambda x: cv2.boundingRect(x)[3] > 10, contours))
        contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
        image_list = []
        last_box = (0, 0, 0, 0)
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
