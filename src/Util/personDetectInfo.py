class PersonDetectInfo:


    is_exist_person = False
    detections = None
    class_name = ""

    def get_detections(self):
        return self.detections

    def get_boxes(self):
        return self.boxes

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def get_top(self):
        return self.top

    def get_bottom(self):
        return self.bottom
