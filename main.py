import time

import cv2

from src.streaming.stream import Stream
from threading import Thread, Lock
from src.functions.functions import Function
from src.image_processing.person_tracker_deepsort import PersonTrackerDeepSort
from src.image_processing.person_detector import PersonDetector





class Main(Thread):

    mymutex = Lock()

    def __init__(self):
        super(Main, self).__init__()

        self.stream0 = Stream(0, "rtsp://admin:Istech2021@192.168.1.189", "raf").start()
        self.stream1 = Stream(1, "rtsp://admin:Istech2021@192.168.1.191", "ic_kapi").start()
        self.stream2 = Stream(2, "rtsp://admin:Istech2021@192.168.1.193", "genel_gorunum").start()


        self.person_detector = PersonDetector()
        self.person_tracker_deepsort = PersonTrackerDeepSort()


        self.func0 = Function(0, self.stream0.get_frame(), self.stream0.get_camName(),self.person_detector,self.person_tracker_deepsort).start()
        self.func1 = Function(1, self.stream1.get_frame(), self.stream1.get_camName(),self.person_detector,self.person_tracker_deepsort).start()
        self.func2 = Function(2, self.stream2.get_frame(), self.stream2.get_camName(),self.person_detector,self.person_tracker_deepsort).start()



        self.start()

    def run(self):
        while True:

            self.func0.set_frame(self.stream0.get_frame())
            self.func1.set_frame(self.stream1.get_frame())
            self.func2.set_frame(self.stream2.get_frame())


            if  self.func0.get_personDetectInfo() is not None and self.func0.get_personTrackDeepSortInfo() is not None:
                print(self.func0.get_personTrackDeepSortInfo().ids)

            if  self.func1.get_personDetectInfo() is not None and self.func1.get_personTrackDeepSortInfo() is not None:
                print(self.func1.get_personTrackDeepSortInfo().ids)

            if self.func2.get_personDetectInfo() is not None and self.func2.get_personTrackDeepSortInfo() is not None:
                print(self.func2.get_personTrackDeepSortInfo().ids)






            image1 = cv2.putText(self.stream0.get_frame(),"ids0", (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2, cv2.LINE_AA)
            image2 = cv2.putText(self.stream1.get_frame(),"ids1", (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2, cv2.LINE_AA)
            image3 = cv2.putText(self.stream2.get_frame(),"ids2", (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2, cv2.LINE_AA)



            self.imshow(image1,image2,image3)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.05)
        cv2.destroyAllWindows()





    def imshow(self,*args):
        for i in range(len(args)):
            cv2.imshow("CAM"+str(i), args[i])


if __name__ == '__main__':
    Main()
