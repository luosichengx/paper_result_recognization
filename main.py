from predict import *
from extract_result import *
import os
import shutil
import pandas as pd
from collections import Counter
import time
import configparser
import logging

logger = logging.getLogger(__name__)

def add_with_index(indexes, result):
    if indexes[-1] == len(indexes):
        return result
    elif indexes[-2] == len(indexes) - 1:
        if indexes[-1] == len(indexes) + 1:
            result.append(result[-1])
            result[-2] = 0
        else:
            return result
    else:
        assignment_ind = 1
        ret = []
        indexes_ind = 0
        while(indexes_ind != len(indexes)):
            if indexes[indexes_ind] == assignment_ind:
                ret.append(result[indexes_ind])
                indexes_ind += 1
                assignment_ind += 1
                continue
            elif indexes[indexes_ind] < assignment_ind + 2 and indexes[indexes_ind + 1] != assignment_ind + 1:
                ret.extend([0] * (indexes[indexes_ind] - assignment_ind))
                ret.append(result[indexes_ind])
                indexes_ind += 1
                assignment_ind = indexes_ind + 1
            else:
                ret.append(result[indexes_ind])
                indexes_ind += 1
                assignment_ind += 1
        return ret

def check_and_add_assignment_result_for_single_student(student_id, assignment_id, dataset, check_model, assignment_df, indexes):
    if len(dataset) == 0:
        return
    # if len(question_ids) != len(dataset):
    #     logging.error("The questions number {%d} does not match the found question  {%d}, please manually "
    #                   "fix the result for student {%s}.", len(question_ids), len(dataset), student_id)
    if check_model == None:
        pred = [[x] for x in dataset]
    else:
        pred = predict_box_image(dataset, check_model)
    print(pred)
    # if indexes != None:
    #     add_with_index(indexes, pred)
    new_line_index = assignment_df.shape[0] if student_id == None else student_id
    assignment_df.loc[new_line_index] = 0
    assignment_df.loc[new_line_index, "assignment_id"] = assignment_id
    for j in range(len(pred)):
        if pred[j][0] == 0:
            try:
                assignment_df.loc[new_line_index, str(j + 1)] = 1
            except IndexError:
                print("out of index")
        else:
            try:
                assignment_df.loc[new_line_index, str(j + 1)] = -1
            except IndexError:
                print("out of index")


def scanned_assignment_to_result_dataframe():
    try:
        check_model = load_model("./CHECK_with_box_could_demo.pth")
    except FileNotFoundError:
        print("no trained model")
        return
    ids, dataset = [], []
    assignment_df = pd.DataFrame(columns=["assignment_id"] + list(map(str, range(1,50,1))))
    assignment_id_f = -1
    last_student_id, last_assignment_id = "", ""
    assignment_id_list = []
    indexes = []
    for root, dir, files in os.walk(SCAN_ASSIGNMENT_DIRECTORY):
        files = sort_by_ctime(root, files)
        for file in files:
            student_id, assignment_id, new_data, index_list = extract_assignment(root + "/" + file, False)
            assignment_id_list.append(assignment_id)
            if student_id != None:
                check_and_add_assignment_result_for_single_student(last_student_id, last_assignment_id, dataset, check_model, assignment_df, indexes)
                assignment_id_f += 1
                dataset = []
                indexes = []
            dataset.extend(new_data)
            indexes.extend(index_list)
            if student_id != None:
                last_student_id, last_assignment_id = student_id[:-1], assignment_id[:-1]
        check_and_add_assignment_result_for_single_student(last_student_id, last_assignment_id, dataset, check_model, assignment_df, indexes)
    assignment_df.index.name = "student_id"
    ind = 1
    ma = assignment_df.max()
    mi = assignment_df.min()
    for i in range(1, len(ma)):
        ind += 1
        if ma[i] == 0 and mi[i] == 0:
            break
    assignment_df = assignment_df.iloc[:, : ind - 1]
    assignment_df.to_csv("assignment_last.csv")
    t = time.localtime()
    assignment_df.to_csv("assignment_%d_%d_%d_%d.csv"%(t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min))
    print(assignment_df)


def scanned_assignment_to_result_dataframe_no_prediction():
    ids, dataset = [], []
    assignment_df = pd.DataFrame(columns=["assignment_id"] + list(map(str, range(1,50,1))))
    assignment_id_f = -1
    last_student_id, last_assignment_id = "", ""
    assignment_id_list = []
    indexes = []
    for root, dir, files in os.walk(SCAN_EASY_ASSIGNMENT_DIRECTORY):
        files = sort_by_ctime(root, files)
        for file in files:
            student_id, assignment_id, new_data, index_list = extract_assignment(root + "/" + file, False)
            assignment_id_list.append(assignment_id)
            if student_id != None:
                # c = Counter(assignment_id_list)
                # assignment_id = sorted(c.items(), key=lambda d: d[1], reverse=True)[0][0]
                check_and_add_assignment_result_for_single_student(last_student_id, last_assignment_id, dataset, None, assignment_df, indexes)
                assignment_id_f += 1
                dataset = []
                indexes = []
            dataset.extend(new_data)
            indexes.extend(index_list)
            if student_id != None:
                last_student_id, last_assignment_id = student_id[:-1], assignment_id[:-1]
        check_and_add_assignment_result_for_single_student(last_student_id, last_assignment_id, dataset, None, assignment_df, indexes)
    assignment_df.index.name = "student_id"
    ind = 1
    ma = assignment_df.max()
    mi = assignment_df.min()
    for i in range(1, len(ma)):
        ind += 1
        if ma[i] == 0 and mi[i] == 0:
            break
    assignment_df = assignment_df.iloc[:, : ind - 1]
    assignment_df.to_csv("assignment_last.csv")
    t = time.localtime()
    assignment_df.to_csv("assignment_%d_%d_%d_%d.csv"%(t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min))
    print(assignment_df)


def scanned_paper_to_result_dataframe():
    ids, dataset = [], []
    last_student_id, assignment_id = "", ""
    student_id_list = []
    all_data = []
    max_size = 0
    last_assignment_id = 0
    for root, dir, files in os.walk(SCAN_PAPER_DIRECTORY):
        files = sort_by_ctime(root, files)
        for file in files:
            student_id, assignment_id, new_data = extract_paper(root + "/" + file, False)
            if student_id != None and len(dataset) != 0:
                all_data.append(dataset)
                max_size = max(max_size, len(dataset))
                student_id_list.append(last_student_id)
                dataset = []
            dataset.extend(new_data)
            if student_id != None:
                last_student_id = student_id[:-1]
            if assignment_id != None:
                last_assignment_id = assignment_id[:-1]
    if len(dataset) != 0:
        all_data.append(dataset)
        max_size = max(max_size, len(dataset))
        student_id_list.append(last_student_id)
    for i in range(len(all_data)):
        if len(all_data[i]) != max_size:
            all_data[i].extend([0] * (max_size - len(all_data[i])))
    all_data = np.array(all_data)
    all_data = np.insert(all_data, 0, values=0, axis=1)
    assignment_df = pd.DataFrame(data=all_data,index=student_id_list,columns=["paper_id"] + list(map(str, range(1,len(all_data[0] + 1),1))))
    assignment_df.loc[:,"paper_id"] = last_assignment_id
    assignment_df.index.name = "student_id"
    t = time.localtime()
    assignment_df.to_csv("paper_last.csv")
    assignment_df.to_csv("paper_%d_%d_%d_%d.csv"%(t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min))
    print(assignment_df)


# 扫描文件格式未知，需要到时候修改
def sort_by_ctime(root, files):
    files = sorted(files, key=lambda x: os.path.getctime(os.path.join(root, x)))
    return files

def sort_by_name(root, files):
    files = sorted(files)
    return files


# 将分析过的试卷移动走，不确定是我处理还是前端处理
def store_analysed_data_elsewhere(directory):
    for root, dir, files in os.walk(SCAN_ASSIGNMENT_DIRECTORY):
        for file in files:
            shutil.move(root + "/" + file, directory)


def add_student_id():
    count = 0
    for root, dir, files in os.walk(SCAN_ASSIGNMENT_DIRECTORY):
        files = sort_by_name(root, files)
        for file in files:
            print(file)
            image = load_image_file(os.path.join(root, file))
            if count % 2 == 0:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, edges = cv2.threshold(gray, THRESHOLD, 255, 1)
                barcode = load_image_file("barcode/barcode" + str(count // 2 + 1) + ".png")
                edges = edges[image.shape[0]// 8 :image.shape[0]// 4 + 50, image.shape[1]// 2 :image.shape[1]]
                up_boarder, left_boarder = image.shape[0]// 8, image.shape[1]// 2
                contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                largest_item = contours[0]
                x, y, w, h = cv2.boundingRect(largest_item)
                block = edges[y:y + h, x : x+w]
                # cv2.imshow("b",block)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                image[up_boarder + y + 20:up_boarder + y + h - 10, left_boarder+ x + 20:left_boarder+x + w - 20] = cv2.resize(barcode, (w - 40, h - 30))
            count += 1
            cv2.imwrite(os.path.join("new_"+root, file), image)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    try:
        type = config.get("Type", "type")
        pred = config.get("Prediction", "type")
    except configparser.NoOptionError:
        type = "assignment"
        pred = "easy"
    if type == "assignment" or type == "all":
        if pred == "easy":
            scanned_assignment_to_result_dataframe_no_prediction()
        else:
            scanned_assignment_to_result_dataframe()
    if type == "paper" or type == "all":
        scanned_paper_to_result_dataframe()