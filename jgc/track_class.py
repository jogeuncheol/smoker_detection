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

    def set_data(self, box, f, r, outer_roi, inner_roi):
        self.bbox = box
        self.outer_roi_flag = f
        self.ROI_R = r
        self.outer_roi = outer_roi
        self.inner_roi = inner_roi


if __name__ == "__main__":
    t_dict = TDict()
    for t_id in range(1, 10):
        t_value = Tracker()
        _box = [100, 100, 20, 20]
        flag = 0
        R = 30
        or_box = [50, 50, 100, 100]
        ir_box = [80, 80, 40, 40]
        t_value.set_data(_box, flag, R, or_box, ir_box)
        t_dict.add_dict(t_id, t_value)
    t_dict.items()
    t_dict.del_dict(4)
    t_dict.items()
