import cv2,os
import dlib
import numpy as np

class VideoSplitter(object):

    def __init__(self,raw_video):
        if os._exists(raw_video):
            self.raw_video = cv2.videoCapture(raw_video)
            self.fragment_cnt = 0
            self.face_detector = dlib.get_frontal_face_detector()
            self.shape_predictor = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')
            self.face_encoder = dlib.face_recognition_model_v1('../models/dlib_face_recognition_resnet_model_v1.dat')
            self.hosts = []

            # cv2 default parameters for video process control
            self.CV_CAP_PROP_POS_MSEC = 0
            self.CV_CAP_PROP_POS_FRAMES = 1
            self.CV_CAP_PROP_POS_AVI_RATIO = 2
            self.CV_CAP_PROP_FRAME_WIDTH = 3
            self.CV_CAP_PROP_FRAME_HEIGHT = 4
            self.CV_CAP_PROP_FPS = 5
            self.CV_CAP_PROP_FOURCC = 6
            self.CV_CAP_PROP_FRAME_COUNT = 7
            self.CV_CAP_PROP_FORMAT = 8
            self.CV_CAP_PROP_MODE = 9
            print 'Load video successfully'
        else:
            raise IOError

    def gen_time(fps):
        time = 0
        gap = 1000 / fps  # time gap between frames = 1000ms / fps
        while True:
            yield time
            time += gap

    def frame_capture(self,fps=None):
        '''
        :param raw_video: video object obtained by cv2 from raw video
        :param fps: frames per second in frame capturing
        :return: generator contains frames
        '''
        if fps is not None:
            self.time_sequence = self.gen_time(fps)
            next_time = next(self.time_sequence)

        print("Frame Capturing ...")
        while True:
            ret, frame = self.raw_video.read()

            if ret is False:
                print("Frame Capture Done")
                self.raw_video.release()
                break

            time = self.raw_video.get(self.VCProps.CV_CAP_PROP_POS_MSEC)
            position = self.raw_video.get(self.VCProps.CV_CAP_PROP_POS_FRAMES)
            ratio = self.raw_video.get(self.VCProps.CV_CAP_PROP_POS_AVI_RATIO)

            # capture according to the customized fps
            if fps is not None:
                if time<next_time:
                    continue
                else:
                    next_time = next(self.time_sequence)
                    yield frame, time, position, ratio
            else:
                yield frame,time,position,ratio

    def check_host(self,frame):
        width,height,channel = frame.size()
        faces = self.detector(frame,1)

        if len(faces)==0:
            return False

        elif len(faces)==1:
            face = faces[0]
            x_mid = (face.left()+face.right())/2
            if abs(x_mid-width/2)>width/10:   # the host's face is supposed to be in the horizontal x_mid of the scene
                return False
            else:
                return True
        elif len(faces)==2:
            x_mids = []
            for face in faces:
                x_mid = (face.left()+face.right())/2
                x_mids.append(x_mid)

            if (x_mids[0]+x_mids[1])/2-width/2>width/10:
                return False
            else:
                return True
        else:
            return False

    def check_bgd(self,frame):
        '''
        This function is to check if current scene is inside the broadcasting room
        :param frame:
        :return:
        '''

        return False

    def split_video(self):
        frames = self.frame_capture(1)
        self.split_points = []
        # if_indoor switches to true if current scene is in the broadcasting room, otherwise turns false
        if_indoor = True
        for frame,time,position,ration in frames:
            print 'processing frame at %s ms, %s ration done' %(time,ration)

            # if previous state is indoor, search next outdoor scene
            if if_indoor is True:
                if self.host_check(frame) or self.bgd_check(frame) is False:
                    # record a new if_indoor 'outside the broadcasting room', and the time
                    self.split_points.append((1,time))
                    if_indoor = False
            # vice-versa
            else:
                if self.host_check(frame) and self.bgd_check(frame) is True:
                    # record a new if_indoor 'inside the broadcasting room', and the time
                    self.split_points.append((0,time))  #
                    if_indoor = True
        return self.split_points

    def get_face_code(self,frame,face):
        return np.array(self.face_encoder.compute_face_descriptor(frame,self.shape_predictor(frame,face)))

    def compare_face(self,frame,face):
        face_code = self.get_face_code(frame,face)
        disparity = min(self.hosts,lambda host:np.linalg.norm(face_code-host))
        return True if disparity<0.4 else False

    def add_hosts(self,frame,face):
        face_code = self.get_face_code(frame,face)
        self.hosts.append(face_code)







