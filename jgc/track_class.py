class TDict:
    def __init__(self):
        self.t_dict = {}

    def add_dict(self, key, value):
        self.t_dict[key] = value

    def del_dict(self, key):
        self.t_dict.pop(key)

    def update_dict(self, key, value):
        self.t_dict.update({key: value})

    def if_in_dict(self, key):
        if key in self.t_dict:
            return True
        return False

    def items(self):
        for self.k, self.v in self.t_dict.items():
            print("{} in {}".format(self.k, self.v))


class Tracker:
    def __init__(self):
        self.bbox = []
        self.outer_roi_flag = 0
        self.ROI_R = 0
        self.outer_roi = []
        self.inner_roi = []
        self.bg_sub = any

    def __str__(self):
        return str(self.bbox) + ", " + str(self.outer_roi_flag) + ", " + str(self.ROI_R) + ", " + str(self.outer_roi) + ", " + str(self.inner_roi)

    def set_data(self, t_list):
        self.bbox = t_list[0]
        self.outer_roi_flag = t_list[1]
        self.ROI_R = t_list[2]
        self.outer_roi = t_list[3]
        self.inner_roi = t_list[4]


if __name__ == "__main__":
    t_dict = TDict()
    dic = t_dict.t_dict
    for t_id in range(1, 10):
        t_value = Tracker()
        t_list = [[t_id, 100, 20, 20], 0, 30, [50, 50, 100, 100], [80, 80, 40, 40]]
        t_value.set_data(t_list)
        t_dict.add_dict(t_id, t_value)
    t_dict.items()
    t_dict.del_dict(4)
    t_dict.items()
    t_dict.t_dict[1].bbox = [1, 5, 6, 3]
    print("key is 1, value is {}".format(t_dict.t_dict[1]))
    print("key is 1, value.bbox is {}".format(t_dict.t_dict[1].bbox))
    print("key is 1, value.bbox[0] is {}".format(t_dict.t_dict[1].bbox[0]))