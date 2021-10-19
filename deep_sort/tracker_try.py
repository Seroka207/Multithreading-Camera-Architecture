# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter, nn_matching
from . import linear_assignment
from . import iou_matching
from .track import Track

class Tracker:

    def __init__(self, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = {}   #(Ölçümden yola ilişkilendirme için bir mesafe metriği.)
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age #(Bir track silinmeden önce kaçırılan maksimum kayıp sayısı. )
        self.n_init = n_init   #Track onaylanmadan önceki ardışık algılama sayısıdır. lk süre içinde bir ıska meydana gelirse izleme durumu 'Silindi' olarak ayarlnır
        self.kf = kalman_filter.KalmanFilter()    #Görüntü uzayındaki hedef yörüngeleri filtrelemek için bir Kalman filtresi.
        self.tracks = {}   #Geçerli zaman adımındaki aktif Tracklerin listesi
        self._next_id=dict()
        self._next_id[0] = 1
        self._next_id[1] = 1
        self._next_id[2] = 1
        self._next_id[3] = 1

    def get_tracks(self,cam_id):

        return self.tracks[cam_id]   #geçerli trackleri ilgili kameraya ata

    def predict(self,cam_id):
        ### Takip durumu dağılımlarını bir adım ileri yayın. Bu işlev, "güncelleme" işleminden önce her adımda bir kez çağrılmalıdır. ###
        if cam_id in self.tracks:
            pass
        else:
            self.tracks[cam_id] = []
            self.metric[cam_id] = nn_matching.NearestNeighborDistanceMetric("cosine", 0.4, None)


        for track in self.tracks[cam_id]:
            track.predict(self.kf)

    def update(self, detections, cam_id = 0):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
            ### Geçerli zaman adımındaki algılamaların listesi ###

        """

        if cam_id in self.tracks:
            pass
        else:
            self.tracks[cam_id] = []
            """ calculate cosine distance metric """
            ### kosinüs uzaklığı metriğini hesapla
            self.metric[cam_id] = nn_matching.NearestNeighborDistanceMetric("cosine", 0.4, None)

        """ Run matching cascade """
        ### Eşleşen kademeyi çalıştır ###

        matches, unmatched_tracks, unmatched_detections = self._match(detections,cam_id)


        ### Track grubunu güncelle ###

        for track_idx, detection_idx in matches:
            self.tracks[cam_id][track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[cam_id][track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx],cam_id)
        # self.racks = [t for t in self.tracks[cam_id] if not t.is_deleted()]


        """"  Update distance metric """
        ### Mesafe metriğini güncelle ###

        active_targets = [t.track_id for t in self.tracks[cam_id] if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks[cam_id]:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric[cam_id].partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections, cam_id):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric[cam_id].distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        """ Split track set into confirmed and unconfirmed tracks """
        ### onaylanmış ve onaylanmamış track olarak ayır ###
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks[cam_id]) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks[cam_id]) if not t.is_confirmed()]

        """ Associate confirmed tracks using appearance features """
        ### Görünüm özelliklerini kullanarak onaylanmış parçaları ilişkilendirin ###

        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric[cam_id].matching_threshold, self.max_age,
                self.tracks[cam_id], detections, confirmed_tracks)

        """ Associate remaining tracks together with unconfirmed tracks using IOU """
        ### IOU kullanarak kalan izleri doğrulanmamış parçalarla ilişkilendirin ###

        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[cam_id][k].time_since_update == 1]

        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[cam_id][k].time_since_update != 1]


        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks[cam_id],
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, cam_id):
        """ Yeni kişi tespit edildiğinde kalman filter yeni mean ve covarience oluşturur"""
        mean, covariance = self.kf.initiate(detection.to_xyah())
        class_name = detection.get_class()
        self.tracks[cam_id].append(Track(
            mean, covariance, self._next_id[cam_id], self.n_init, self.max_age,
            detection.feature, class_name))
        self._next_id[cam_id] += 1
