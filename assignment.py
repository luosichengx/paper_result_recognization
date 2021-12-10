import cv2

class assignment_papers:
    def __init__(self):
        self.student_id = None
        self.assignment_id = None
        self.result = []
        self.questions_number = 0
        self.assignment_papers = []

    def print_all_papers(self):
        for ind, paper in enumerate(self.assignment_papers):
            cv2.imshow("paper " + str(ind), paper)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
