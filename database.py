# 模拟数据库访问结果，在数据库实现后请替换成真实情况
def get_question_id_seq_with_assignment_id(assignment_id):
    sql_res = []
    if assignment_id == 0 or assignment_id == 1:
        sql_res = [1,2,3,4,5,6,7,8,9]
    elif assignment_id == 2 or assignment_id == 3:
        sql_res = [1,2,3,4,5,6,10]
    elif assignment_id == 4:
        sql_res = [1,2,3,4,5,6,7,8,10]
    # if assignment_id == 0:
    #     sql_res = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    # elif assignment_id == 1:
    #     sql_res = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23]
    # elif assignment_id == 2:
    #     sql_res = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,24]
    return sql_res

