from predict import *
from extract_result import *
import os
import shutil
import pandas as pd
from collections import Counter
import time
import configparser
import ConvNet

def check_and_add_assignment_result_for_single_student(student_id, assignment_id, dataset, check_model, assignment_df):
    if len(dataset) == 0:
        return
    # question_ids = get_question_id_seq_with_assignment_id(assignment_id)
    # print("assign" + str(assignment_id))
    # if len(question_ids) != len(dataset):
    #     logging.error("The questions number {%d} does not match the found question  {%d}, please manually "
    #                   "fix the result for student {%s}.", len(question_ids), len(dataset), student_id)
    pred = predict_box_image(dataset, check_model)
    print(pred)
    # student_id = assignment_id
    new_line_index = assignment_df.shape[0] if student_id == None else student_id
    assignment_df.loc[new_line_index] = 0
    # for j in range(len(pred)):
    #     if pred[j][0] == 0:
    #         try:
    #             assignment_df.iloc[assignment_df.shape[0] - 1, question_ids[j] - 1] = 1
    #         except:
    #             pass
    #     else:
    #         try:
    #             assignment_df.iloc[assignment_df.shape[0] - 1, question_ids[j] - 1] = -1
    #         except:
    #             pass
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
                pass
    print(assignment_df.shape)

def statistics_for_assignment_result(assignment_df):
    # assignment_df.to_csv("assignment_result.csv")
    # assignment_df = pd.read_csv("assignment_result.csv", index_col=0)
    assignment_df.loc["correct"] = (assignment_df == 1).sum()
    assignment_df.loc["error"] = (assignment_df == -1).sum()
    assignment_df.loc["all"] = assignment_df.loc["error"] + assignment_df.loc["correct"]
    assignment_df.to_csv("assignment_result_with_statistics.csv")
    return assignment_df


def scanned_assignment_to_result_dataframe():
    # mnist_model = load_model("./MNIST.pth")
    check_model = load_model("./CHECK_with_box_could_demo.pth")
    ids, dataset = [], []
    # assignment_id0 = 0
    # assignment_id1 = 1
    # assignment_id2 = 2
    # question_ids0 = get_question_id_seq_with_assignment_id(assignment_id0)
    # question_ids1 = get_question_id_seq_with_assignment_id(assignment_id1)
    # question_ids2 = get_question_id_seq_with_assignment_id(assignment_id2)
    # assignment_df = pd.DataFrame(columns=list(map(str, set(question_ids0 + question_ids1 + question_ids2))))
    assignment_df = pd.DataFrame(columns=["assignment_id"] + list(map(str, range(1,50,1))))
    assignment_id_f = -1
    last_student_id, last_assignment_id = "", ""
    assignment_id_list = []
    for root, dir, files in os.walk("assignment"):
        files = sort_by_name(files)
        for file in files:
            student_id, assignment_id, new_data = extract_assignment(root + "/" + file, False)
            assignment_id_list.append(assignment_id)
            if student_id != None:
                c = Counter(assignment_id_list)
                # assignment_id = sorted(c.items(), key=lambda d: d[1], reverse=True)[0][0]
                check_and_add_assignment_result_for_single_student(last_student_id, last_assignment_id, dataset, check_model, assignment_df)
                assignment_id_f += 1
                dataset = []
            dataset.extend(new_data)
            # id, new_data = extract_paper(file, False)
            # data_len = len(new_data)
            # new_data.extend(0 * [data_len - len(new_data)])
            # ids.extend(id)
            # dataset.extend(new_data)
            if student_id != None:
                last_student_id, last_assignment_id = student_id[:-1], assignment_id[:-1]
        check_and_add_assignment_result_for_single_student(last_student_id, last_assignment_id, dataset, check_model, assignment_df)
    # pred = predict_images(dataset, mnist_model)
    # print(pred)
    # pred = predict_images(ids, mnist_model)
    # print(pred)
    # assignment_df = statistics_for_assignment_result(assignment_df)
    assignment_df.index.name = "student_id"
    t = time.localtime()
    ind = 0
    ma = assignment_df.max()
    mi = assignment_df.min()
    for i in range(len(ma)):
        ind += 1
        if ma[i] == 0 and mi[i] == 0:
            break
    assignment_df = assignment_df.iloc[:, : ind - 1]
    assignment_df.to_csv("assignment_last.csv")
    assignment_df.to_csv("assignment_%d_%d_%d_%d.csv"%(t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min))
    print(assignment_df)


def scanned_paper_to_result_dataframe():
    ids, dataset = [], []
    last_student_id, assignment_id = "", ""
    student_id_list = []
    all_data = []
    for root, dir, files in os.walk(SCAN_PAPER_DIRECTORY):
        files = sort_by_name(files)
        for file in files:
            student_id, assignment_id, new_data = extract_paper(root + "/" + file, False)
            if student_id != None and len(dataset) != 0:
                all_data.append(dataset)
                student_id_list.append(last_student_id)
                dataset = []
            dataset.extend(new_data)
            if student_id != None:
                last_student_id, last_assignment_id = student_id[:-1], assignment_id[:-1]
    if len(dataset) != 0:
        all_data.append(dataset)
        student_id_list.append(last_student_id)
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
def sort_by_name(files):
    files.sort(key=lambda x: int(x[5:-4]))
    return files


# 将分析过的试卷移动走，不确定是我处理还是前段处理
def store_analysed_data_elsewhere(directory):
    for root, dir, files in os.walk(SCAN_ASSIGNMENT_DIRECTORY):
        for file in files:
            shutil.move(root + "/" + file, directory)



if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    type = config.get("Type", "type")
    if type == "assignment" or type == "all":
        scanned_assignment_to_result_dataframe()
    if type == "paper" or type == "all":
        scanned_paper_to_result_dataframe()