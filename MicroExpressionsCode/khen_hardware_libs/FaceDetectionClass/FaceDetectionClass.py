__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '5.10.2021'

# On windows:
# pip install https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl#md5=b7330a5b2d46420343fbed5df69e6a3f
# You can download a trained facial shape predictor from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2


import dlib
import numpy as np
import cv2 as cv

class FaceDetector:
    """
    A class used to represent an Image
    ...

    Attributes
    ----------
    detector : dlib object
        sss

    predictor : dlib object
        sss

    detect_list : list
        list of detected objects, each is list of a rect + 68 marks

    Methods
    -------
    __init__()
        sss
    detect_face_from_img(img, show = 0)
        sss
    show_detection(img)
        sss
    update_detection_by_of_mask(of_mask_x, of_mask_y, show = 0)
        sss

    """

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('FaceDetectionClass/shape_predictor_68_face_landmarks.dat')
        return

    def detect_face_from_img(self, img, show = 0):
        dets = self.detector(img, 1)
        detect_list = []
        for k, d in enumerate(dets):
            shape = self.predictor(img, d)
            detect_obj = [   (d.left(), d.top(), d.right(), d.bottom())   ]
            for part_idx in range(shape.num_parts):
                detect_obj += [  (shape.part(part_idx).x,shape.part(part_idx).y)  ]

            detect_list += [ detect_obj ]

        # return format: [ (x_left, y_top, x_right, y_bottom) , (shape0), (shape1) , ...) ] - obj 1
        #                [ (x_left, y_top, x_right, y_bottom) , (shape0), (shape1) , ...) ] - obj 2 ...
        self.detect_list = detect_list
        if show:
            self.show_detection(img)

        return detect_list


    def show_detection(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        win = dlib.image_window()
        win.clear_overlay()
        win.set_image(img)
        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = self.detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = self.predictor(img, d)
            print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                      shape.part(1)))
            # Draw the face landmarks on the screen.
            win.add_overlay(shape)

        win.add_overlay(dets)
        dlib.hit_enter_to_continue()
        return

    def update_detection_by_of_mask(self, of_mask_x, of_mask_y, show = 0):
        last_detect_list = self.detect_list
        forward_detect_list = []
        for detect_obj in last_detect_list:
            # Generate updated detection location using mask of optical flow:
            new_obj_rect = detect_obj[0] # TODO: update the rect around the object

            new_detect_obj = [ new_obj_rect ]

            for part in last_detect_list[1:]:
                new_part = part.copy()
                new_part[0] += np.multiply( of_mask_x, of_mask_x != 0 )[part].astype(np.int32)
                new_part[1] += np.multiply( of_mask_y, of_mask_y != 0 )[part].astype(np.int32)
                new_detect_obj += [  new_part  ]

            forward_detect_list = [ new_detect_obj ]

            if show:
                for past, next in zip(last_detect_list,forward_detect_list):
                    print( past, ' -> ', next )

        self.detect_list = forward_detect_list
        return forward_detect_list


#
# if __name__ == '__main__':
# #     # from DVSVideoClass.dvsClass2_0 import *
# #     # from VideoClass import *
# #     # from scripts.scripts import *
# #     # import VideoClass
#     from ImageClass.ImageClass import *
#     img = ImageClass('../img2.jpg')
# #     # img.show()
# #     img.img = cv.cvtColor(img.img, cv.COLOR_BGR2RGB)
#     face_detector = FaceDetector()
#     detected_list = face_detector.detect_face_from_img(img.img, show = 1)
#
#     obj_detected_list = detected_list[0]
#     obj_marks = np.array( obj_detected_list[1:] )
#     obj_points = (obj_marks[:, 1], obj_marks[:, 0])
#     print( img.img[ obj_points ] )
#     for rect in detected_list[0]:
#         print(rect)
#     print(len(detected_list[0]))
#
#     # for img in cis_video.frames:
    #     detected_list = face_detector.detect_face_from_img(img)
    #     for detected_obj in detected_list:
    #         rect = detected_obj[0]
    #         print(rect)
    #         for part in detected_obj[1:]:
    #             print( part )
    #     face_detector.show_detection(img)


    # cis_video = VideoClass('C:/Users/khen/PycharmProjects/dvs/object_detection/img2.jpg')
    # face_detector = FaceDetector()
    #
    # img = np.moveaxis(cis_video.frames[0],0,1)
    # img = cv.resize(img, (480, 640), interpolation=cv.INTER_AREA)
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # detected_list = face_detector.detect_face_from_img(img)
    # face_detector.show_detection(img)

