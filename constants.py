SCAN_ASSIGNMENT_DIRECTORY="assignment"
SCAN_PAPER_DIRECTORY="paper"
ORIGINAL_FILE = 'paper2.png'
THRESHOLD = 150 # very important, should be carefully adjust, affect all image analysis
DENOISE_FILE = 'denoised_paper.jpg'
INFO_BLOCK_FILE = 'info_block.png'
ID_BLOCK_FILE = 'id_block.png'
GRADE_BLOCK_FILE = 'grade_block.png'
TEST_ID_BLOCK_FILE = 'test.png'
STU_ID_BLOCK_FILE = 'stu_id_block.png'
STU_NAME_BLOCK_FILE = 'stu_name_block.png'
TEST_ID_OFFSET = [17,46,220,347]
STU_ID_OFFSET = [17,48,402,532]
STU_NAME_OFFSET = [22,46,376,499]
margin = 20
a = 4
b = 6

# sum = []
# num = [1]
# could = True
# for i in range(2, 100):
#     could = True
#     for s in sum:
#         if s - i in num:
#             could = False
#             break
#     if could == False:
#         continue
#     for j in num:
#         sum.append(i + j)
#     num.append(i)
# sum = []
# b = [1, 2, 3, 5, 8, 13, 21, 30, 39, 53, 74, 95]
# for i in range(len(b)):
#     for j in range(i + 1, len(b)):
#         sum.append(b[i] + b[j])
# sum.sort()
# print(sum)
# a = list(set(sum))
# a.sort()
# print(sum == a)



