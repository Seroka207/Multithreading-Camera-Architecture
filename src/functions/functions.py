import time
from threading import Thread, Lock


class Function:
    frame = None
    mymutex = Lock()

    personDetectInfo = None
    personTrackerDeepsortInfo = None

    person_detector = None
    person_tracker_deepsort = None

    def __init__(self, cam_id, frame, cam_name, person_detector, person_tracker_deepsort):

        self.cam_name = cam_name
        self.cam_id = cam_id
        self.frame = frame

        self.person_detector = person_detector
        self.person_tracker_deepsort = person_tracker_deepsort

    def start(self):
        print(f'{self.cam_name} {self.cam_id} Threadi basladi ')
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.cam_id == 0:
                self.camera0_process()
            if self.cam_id == 1:
                self.camera1_process()
            if self.cam_id == 2:
                self.camera2_process()
            time.sleep(0.05)


    def camera0_process(self):
        self.mymutex.acquire()
        tmp_person_info = self.person_detector.personDetection(self.frame)
        tmp_person_tracker_deepsort_info = self.person_tracker_deepsort.personTracker(tmp_person_info.detections,
                                                                                      self.cam_id)

        self.set_personTrackDeepSortInfo(tmp_person_tracker_deepsort_info)
        self.set_personDetectInfo(tmp_person_info)
        self.mymutex.release()

    def camera1_process(self):
        self.mymutex.acquire()
        tmp_person_info = self.person_detector.personDetection(self.frame)
        tmp_person_tracker_deepsort_info = self.person_tracker_deepsort.personTracker(tmp_person_info.detections,
                                                                                      self.cam_id)
        self.set_personTrackDeepSortInfo(tmp_person_tracker_deepsort_info)
        self.set_personDetectInfo(tmp_person_info)
        self.mymutex.release()



    def camera2_process(self):
        self.mymutex.acquire()
        tmp_person_info = self.person_detector.personDetection(self.frame)
        tmp_person_tracker_deepsort_info = self.person_tracker_deepsort.personTracker(tmp_person_info.detections,
                                                                                      self.cam_id)
        self.set_personTrackDeepSortInfo(tmp_person_tracker_deepsort_info)
        self.set_personDetectInfo(tmp_person_info)
        self.mymutex.release()


    def set_frame(self, frame):
        self.frame = frame

    def get_cam_id(self):
        return self.cam_id

    def set_cam_id(self, cam_id):
        self.cam_id = cam_id

    def set_personDetectInfo(self, person_info):
        self.personDetectInfo = person_info

    def get_personDetectInfo(self) -> personDetectInfo:
        return self.personDetectInfo

    def set_personTrackDeepSortInfo(self, personTrackDeepSortInfo):
        self.personTrackerDeepsortInfo = personTrackDeepSortInfo

    def get_personTrackDeepSortInfo(self) -> personTrackerDeepsortInfo:
        return self.personTrackerDeepsortInfo
