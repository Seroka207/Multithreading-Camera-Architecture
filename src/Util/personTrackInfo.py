

class PersonTrackInfo:
    left = 0
    top = 0
    right = 0
    bottom = 0

    cam = 0
    ids = []

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