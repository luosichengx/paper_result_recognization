from scan_model import *
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
    if len(indexes) < 2 or len(indexes) != len(result):
        return result
    if indexes[-1] == len(indexes):
        return result
    elif indexes[-2] == len(indexes) - 1:
        if indexes[-1] == len(indexes) + 1:
            result.append(result[-1])
            result[-2] = 0
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
            elif indexes[indexes_ind] <= assignment_ind + 2 and indexes_ind + 1 != len(indexes) and indexes[indexes_ind + 1] != assignment_ind + 1:
                ret.extend([0] * (indexes[indexes_ind] - assignment_ind))
                ret.append(result[indexes_ind])
                assignment_ind = indexes[indexes_ind] + 1
                indexes_ind += 1
            else:
                ret.append(result[indexes_ind])
                indexes_ind += 1
                assignment_ind += 1
        return ret

def check_and_add_assignment_result_for_single_student(cur_assignment, check_model, assignment_df):
    student_id, assignment_id, dataset, indexes = cur_assignment.student_id[:-1], cur_assignment.assignment_id[:-1],cur_assignment.result, cur_assignment.index_list
    if len(cur_assignment.result) == 0:
        return
    if check_model == None:
        pred = [[x] for x in dataset]
    else:
        pred = predict_box_image(dataset, check_model)
    cur_assignment.result = pred
    print(pred)
    print(indexes)
    # if indexes != None:
    #     add_with_index(indexes, pred)
    if allow_dup_stu:
        use_serial_number_as_index(assignment_df, assignment_id, pred, student_id)
    else:
        use_student_id_as_index(assignment_df, assignment_id, pred, student_id)


def use_serial_number_as_index(assignment_df, assignment_id, pred, student_id):
    new_line_index = assignment_df.shape[0]
    assignment_df.loc[new_line_index] = 0
    assignment_df.iloc[new_line_index, 0] = student_id
    assignment_df.iloc[new_line_index, 1] = assignment_id
    for j in range(len(pred)):
        if pred[j][0] == 0:
            try:
                assignment_df.iloc[new_line_index, j + 2] = 1
            except IndexError:
                print("out of index")
        else:
            try:
                assignment_df.iloc[new_line_index, j + 2] = -1
            except IndexError:
                print("out of index")

def use_student_id_as_index(assignment_df, assignment_id, pred, student_id):
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
    assignment_df = pd.DataFrame(columns=["student_id","assignment_id"] + list(map(str, range(1,50,1))))
    cur_assignment = Assignment_Model()
    for root, dir, files in os.walk(SCAN_ASSIGNMENT_DIRECTORY):
        files = sort_by_name(root, files)
        for file in files:
            path = os.path.join(root, file)
            if isfirstpage(path) and len(cur_assignment.result) != 0:
                check_and_add_assignment_result_for_single_student(cur_assignment, check_model, assignment_df)
                cur_assignment = Assignment_Model()
            cur_assignment.extract(path, False)
        check_and_add_assignment_result_for_single_student(cur_assignment, check_model, assignment_df)
    assignment_df.index.name = "index"
    ind = 1
    ma = assignment_df.max()
    mi = assignment_df.min()
    for i in range(2, len(ma)):
        ind += 1
        if ma[i] == 0 and mi[i] == 0:
            break
    assignment_df = assignment_df.iloc[:, : ind - 1]
    assignment_df.to_csv("assignment_last.csv")
    t = time.localtime()
    assignment_df.to_csv("assignment_%d_%d_%d_%d.csv"%(t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min))
    print(assignment_df)


def scanned_assignment_to_result_dataframe_no_prediction():
    assignment_df = pd.DataFrame(columns=["student_id","assignment_id"] + list(map(str, range(1,50,1))))
    cur_assignment = Assignment_Model(True)
    for root, dir, files in os.walk(SCAN_EASY_ASSIGNMENT_DIRECTORY):
        files = sort_by_name(root, files)
        for file in files:
            path = os.path.join(root, file)
            if isfirstpage(path) and len(cur_assignment.result) != 0:
                check_and_add_assignment_result_for_single_student(cur_assignment, None, assignment_df)
                cur_assignment = Assignment_Model(True)
            cur_assignment.extract(path, False)
        check_and_add_assignment_result_for_single_student(cur_assignment, None, assignment_df)
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
    student_id_list, assignment_id_list = [], []
    all_data = []
    max_size = 0
    cur_paper = Paper_Model()
    for root, dir, files in os.walk(SCAN_PAPER_DIRECTORY):
        files = sort_by_name(root, files)
        for file in files:
            path = os.path.join(root, file)
            if isfirstpage(path) and len(cur_paper.result) != 0:
                max_size = add_data(all_data, cur_paper, max_size, assignment_id_list, student_id_list)
                cur_paper = Paper_Model()
            cur_paper.extract(path, False)
    if len(cur_paper.result) != 0:
        max_size = add_data(all_data, cur_paper, max_size, assignment_id_list, student_id_list)
    for i in range(len(all_data)):
        if len(all_data[i]) != max_size:
            all_data[i].extend([0] * (max_size - len(all_data[i])))
    all_data = np.array(all_data)
    data_len = len(all_data[0])
    all_data = np.insert(all_data, 0, values=0, axis=1)
    if allow_dup_stu:
        all_data = np.insert(all_data, 0, values=0, axis=1)
        assignment_df = pd.DataFrame(data=all_data,index=range(len(all_data)),columns=["student_id","paper_id"] + list(map(str, range(1,data_len + 1,1))))
        assignment_df.loc[:,"student_id"] = student_id_list
        assignment_df.index.name = "index"
    else:
        assignment_df = pd.DataFrame(data=all_data, index=student_id_list,
                                     columns=["paper_id"] + list(map(str, range(1, data_len + 1, 1))))
        assignment_df.index.name = "student_id"
    assignment_df.loc[:,"paper_id"] = assignment_id_list
    t = time.localtime()
    assignment_df.to_csv("paper_last.csv")
    assignment_df.to_csv("paper_%d_%d_%d_%d.csv"%(t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min))
    print(assignment_df)


def add_data(all_data, cur_paper, max_size, assignment_id_list, student_id_list):
    student_id, assignment_id, result = cur_paper.student_id, cur_paper.assignment_id, cur_paper.result
    all_data.append(result)
    max_size = max(max_size, len(result))
    student_id_list.append(student_id[:-1])
    assignment_id_list.append(assignment_id[:-1])
    return max_size


# 扫描文件格式未知，需要到时候修改，现在按创建时间排序
def sort_by_name(root, files):
    files = sorted(files, key=lambda x: os.path.getctime(os.path.join(root, x)))
    return files


# 将分析过的试卷移动走，不确定是我处理还是前端处理
def store_analysed_data_elsewhere(directory):
    for root, dir, files in os.walk(SCAN_ASSIGNMENT_DIRECTORY):
        for file in files:
            shutil.move(root + "/" + file, directory)



if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    try:
        type = config.get("Type", "type")
        pred = config.get("Prediction", "type")
        allow_dup_stu = config.getboolean("Dataframe", "type")
    except configparser.NoOptionError:
        type = "assignment"
        pred = "easy"
        allow_dup_stu=True
    if type == "assignment" or type == "all":
        if pred == "easy":
            scanned_assignment_to_result_dataframe_no_prediction()
        else:
            scanned_assignment_to_result_dataframe()
    if type == "paper" or type == "all":
        scanned_paper_to_result_dataframe()