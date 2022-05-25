# 흡연 확률 클래스


class Smoker:
    def __init__(self):
        self.smoker_dictionary = {}

    def add_dict(self, key, value):
        self.smoker_dictionary[key] = value

    def del_dict(self, key):
        self.smoker_dictionary.pop(key)

    def update_dict(self, key, value):
        self.smoker_dictionary.update({key: value})

    def if_in_dict(self, key):
        if key in self.smoker_dictionary:
            return True
        return False


class Smoking:
    def __init__(self):
        self.ID = 0
        self.ROI = []
        self.bg_sub = any
        self.frame_rate = 0
        self.smoking_point = 0
        self.ROI_message = ""
        self.smoking_count = 0
        self.smoking_flag = False

    def set_data(self, data_list):
        self.ID = data_list[0]
        self.ROI = data_list[1]
        self.bg_sub = data_list[2]
        self.frame_rate = data_list[3]

    def is_smoke(self, count_frame):
        if self.frame_rate*0.5 < count_frame < self.frame_rate*3.5:
            return True
        else:
            return False

    def is_smoking(self):
        if self.smoking_count > 2:
            self.ROI_message = "SMOKING"
        elif 1 < self.smoking_point < 5:
            self.ROI_message = "_WARNING_"
        elif self.smoking_point > 5:
            self.ROI_message = "!!! SMOKING !!!"
        else:
            self.ROI_message = "---"
