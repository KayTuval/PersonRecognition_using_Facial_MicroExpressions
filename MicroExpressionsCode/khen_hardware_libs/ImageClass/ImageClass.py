__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '5.10.2021'

# from Scripts.Scripts import *
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import imutils
from scipy.interpolate import griddata


class ImageClass():

    """
    A class used to represent an Image
    ...

    Attributes
    ----------
    img : np array
        the image

    Methods
    -------
    load(filename)
        Load an image file.

    show(save_string = '')
        Display the image to the screen, if save_string is different than '', a figure of the image in the name of save_string is saved instead.

    gausian_filter(sigma = 3)
        Apply a gaussian filter with standard deviation of sigma on the image. Returns another object of ImageClass.

    median_filter(ksize = 5)
        Apply a median filter in size (ksize, ksize), ksize must be odd. Returns another object of ImageClass.

    canny_edge_detector(threshold1 = 100, threhold2 = 200)
        Apply a canny edge detector using thesholds threshold1 and threhold2. Returns another object of ImageClass.

    resize_image(width, height)
        Resize the image into a new size of (width, height) with linearly interpulated values. Returns another object of ImageClass.

    resize_by_padding(width, height, padding_value=1.0)
        sss

    crop_image(rect)
        Crop the image in the rect area (x left down, y left down, x right up, y right up). Returns another object of ImageClass.

    conv_with_kernel(kernel, boundary='symm', mode='same')
        Apply 2D image convolution with a kernel, boundary and mode are optional. Returns another object of ImageClass.

    x_diff()
        Derive the image along the X axis. Returns another object of ImageClass.

    y_diff()
        Derive the image along the Y axis. Returns another object of ImageClass.

    img_fft()
        Apply 2D FFT to the image. Returns another object of ImageClass.

    filter_with_spectral_kernel(spectral_kernel)
        Apply 2D image filtering in spectral domain with a spectral_kernel. Returns another object of ImageClass.

    rotate_img(angle)
        Apply a rotation of the image in angle degrees. Returns another object of ImageClass.

    get_good_features(feature_params = None)
        Apply good features to track detector and returns list of cornets in the image. feature_params is a dictionary of attributes.

    get_grid_of_pixels()
        Returns a list of all the pixel points in the image (x,y) couples.

    get_optical_flow_from_me_to_other(other, method = 'LK', p0 = None)
        Calculates the optical flow between the image to the other image using optical flow method and at points p0.
        When p0 == None, get_good_features is applied to detect points. The method can be 'LK' (Lucas-Kanade) and 'Gunnar').
        This methods returns the old points list, their movements, the new point location and an error list for each point.

    interpolate_zeros_in_image(method='linear')
        Fills the image blank points (with exactly zero value) using non-regular interpolation method.
        Might be very useful for sparse images as events-frame.

    show_optical_flow_on_img(p0, delta_p0, magnification=0.0, show=1)
        Getting list of points p0 and their delta_p0, this function displays the optical flow arrows with magnification
        (if magnification is different than 0.0). Use show = 1 to display the image and show = 0 if you want to savefig.

    get_pixel_from_img()
        This function is uesed to detect pixel points by a click

    def __mul__(self, other)
        Uses only for type(other) == int or type(other) == float. Multiply the frame with other value.

    def __add__(self, other)
        Uses only for type(other) == int or type(other) == float. Add to the frame another value.

    def __sub__(self, other)
        Uses only for type(other) == int or type(other) == float. Subtract from the frame another value.

    def __truediv__(self, other)
        Uses only for type(other) == int or type(other) == float. Divide the frame with another value.

    """


    def __init__(self, img = None):
        if type(img) is str:
            self.load(img)
            return
        self.img = img
        return

    def load(self, filename):
        # Load image from a file
        self.img = cv.imread(filename)
        if self.img is None:
            print('Failed to load the image file:',filename)
        return

    def show(self, save_string = ''):
        img2show = self.img
        if img2show.dtype == 'complex':
            img2show = np.abs(img2show).astype(np.float32)
        # RGB vs Grayscale
        if len(self.img.shape) == 3:
            plt.imshow(cv.cvtColor(img2show, cv.COLOR_BGR2RGB))
        else:
            plt.imshow(img2show, cmap = 'gray')

        # Save figure mode
        if save_string != '':
            plt.savefig(save_string+'.png')
        else:
            plt.show()
        plt.close()
        # plt.clf()
        return

    def gausian_filter(self, sigma = 3):
        return ImageClass( cv.GaussianBlur(self.img, ksize=(0, 0), sigmaX=sigma, sigmaY=0) )

    def median_filter(self, ksize = 5):
        return ImageClass( cv.medianBlur(self.img, ksize = ksize) )

    def canny_edge_detector(self, threshold1 = 100, threhold2 = 200):
        return ImageClass( cv.Canny( cv.cvtColor(self.img,cv.COLOR_RGB2GRAY) ,threshold1,threhold2) )

    def resize_image(self, width, height):
        return ImageClass( cv.resize(self.img, (width,height), interpolation=cv.INTER_AREA) )

    def resize_by_padding(self, width, height, padding_value=1.0):
        return ImageClass( np.pad(self.img, [((height - self.img.shape[0]) // 2, (height - self.img.shape[0] + 1) // 2), \
                                             ((width - self.img.shape[1]) // 2, (width - self.img.shape[1] + 1) // 2)], \
                                            constant_values=padding_value) )

    def crop_image(self, rect):
        # rect structure: (x left down, y left down, x right up, y right up)
        if len(self.img.shape) == 2:     # When the Kernel is 2D and img is RGB
            return ImageClass( self.img[ rect[3]:rect[1], rect[0]:rect[2] ] )
        return ImageClass( self.img[ rect[3]:rect[1], rect[0]:rect[2],: ] )

    def conv_with_kernel(self, kernel, boundary='symm', mode='same'):
        img2conv = self.img.astype(np.float32)
        if len(self.img.shape) == 2:     # When the Kernel is 2D and img is RGB
            return ImageClass(signal.convolve2d(img2conv, kernel, boundary=boundary, mode=mode))
        else:   # Normal RGB image
            new_img = np.concatenate( \
                (signal.convolve2d(img2conv[:,:,0], kernel, boundary=boundary, mode=mode)[:,:,np.newaxis],\
                 signal.convolve2d(img2conv[:,:,1], kernel, boundary=boundary, mode=mode)[:,:,np.newaxis],\
                 signal.convolve2d(img2conv[:,:,2], kernel, boundary=boundary, mode=mode)[:,:,np.newaxis]), axis = 2 )
        return ImageClass( new_img.astype( self.img.dtype ) )

    def x_diff(self, boundary='symm', mode='same' ):
        return self.conv_with_kernel( kernel = np.array( [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] ), boundary=boundary, mode=mode)

    def y_diff(self, boundary='symm', mode='same' ):
        return self.conv_with_kernel( kernel = np.array( [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]] ), boundary=boundary, mode=mode)

    def img_fft(self):
        img2transform = self.img
        if np.issubdtype(self.img.dtype, np.uint8):
            img2transform = self.img.astype(np.float32)/255.
        spectral_img = np.fft.fft2(img2transform)
        spectral_img_shift = np.fft.fftshift(spectral_img)
        return ImageClass(spectral_img_shift)

    def filter_with_spectral_kernel(self, spectral_kernel):
        # For now, works for grayscale images only
        # spectral_img = np.fft.fft2(self.img)
        # spectral_img_shift = np.fft.fftshift(spectral_img)
        spectral_img_shift = self.img_fft().img
        crow, ccol = self.img.shape[0] / 2, self.img.shape[1] / 2
        kcrow, kccol = spectral_kernel.shape[0] / 2, spectral_kernel.shape[1] / 2

        img_mask = spectral_img_shift[int(crow - kcrow): int(crow + kcrow), int(ccol - kccol): int(ccol + kccol)]
        spectral_img_shift[int(crow - kcrow): int(crow + kcrow), int(ccol - kccol): int(ccol + kccol)] = np.multiply(
            spectral_kernel, img_mask)

        f_ishift = np.fft.ifftshift(spectral_img_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        if np.issubdtype(self.img.dtype, np.uint8):
            img_back *= 255.
        return ImageClass(img_back.astype(self.img.dtype))

    def grayscale_image(self):
        return ImageClass( cv.cvtColor(self.img, cv.COLOR_RGB2GRAY) )

    def rotate_image(self, angle):
        return ImageClass( imutils.rotate_bound(self.img, angle) )

    def get_good_features(self, feature_params = None):
        if feature_params is None:
            feature_params = dict(maxCorners=1000, qualityLevel=0.3, minDistance=7, blockSize=7)

        if len(self.img.shape) == 3:
            gray = cv.cvtColor(self.img,cv.COLOR_RGB2GRAY)
        else:
            gray = self.img
        p0 = cv.goodFeaturesToTrack( gray , mask=None, **feature_params)
        return p0

    def get_grid_of_pixels(self):
        xv, yv = np.meshgrid(range(self.img.shape[1]), range(self.img.shape[0]))
        xv = xv.reshape(-1, 1)
        yv = yv.reshape(-1, 1)
        total_points = np.int16(np.concatenate((xv, yv), axis=1))
        return total_points

    def get_optical_flow_from_me_to_other(self, other, method = 'LK', p0 = None):
        # Claculates optical flow from one frame to the other with LK or Gunnar method

        img = self.img
        other_img = other.img
        # if len(self.img.shape) == 3:
        #     img = cv.cvtColor(self.img, cv.COLOR_RGB2GRAY)
        # else:
        #     img = self.img
        #
        # if len(self.img.shape) == 3:
        #     other_img = cv.cvtColor(other.img, cv.COLOR_RGB2GRAY)
        # else:
        #     other_img = other.img
        #
        # if img.dtype != 'uint8':
        #     img =

        if p0 is None:
            p0 = self.get_good_features()

        if method == 'LK':
            # Parameters for lucas kanade optical flow
            lk_params = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 100, 0.001))  # 10, 0.03

            # calculate optical flow
            p1, st, err = cv.calcOpticalFlowPyrLK(img , other_img, p0, None, **lk_params)
            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            delta_p0 = good_new - good_old

            p0 = good_old.reshape(-1, 1, 2)
            p1 = good_new.reshape(-1, 1, 2)
            delta_p0 = delta_p0.reshape(-1, 1, 2)
        else: # method == 'Gunnar'

            p0 = np.int16(np.round(p0))
            p0[:, 0, 0] = np.clip(p0[:, 0, 0], 0, other.shape[1] - 1)
            p0[:, 0, 1] = np.clip(p0[:, 0, 1], 0, other.shape[0] - 1)

            # calculate optical flow
            flow = cv.calcOpticalFlowFarneback(img, other_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            delta_p0 = flow[[p0[:, 0, 1], p0[:, 0, 0]]]

            delta_p0 = delta_p0.reshape(-1, 1, 2)
            p1 = None
            st = None

        return p0, delta_p0, p1, st

    def interpolate_zeros_in_image(self, method = 'linear'):
        mat = self.img
        cols_vec, rows_vec = np.meshgrid(np.arange(mat.shape[1]), np.arange(mat.shape[0]))
        points = np.where(mat != 0)
        if len(points[0]) != 0:
            values = mat[points]
            points = np.transpose(points)
            mat = griddata(points, values, (rows_vec, cols_vec), method=method, fill_value = 0)
        return ImageClass( mat )

    def get_pixel_from_img(self):
        def click(event, x, y, flags, param):
            global mouseX, mouseY
            if event == cv.EVENT_LBUTTONDOWN:
                mouseX = x
                mouseY = y
                print('X = ',mouseX,'Y =', mouseY)
                # cv.circle(img, (mouseX, mouseY), 3, (0,0,255))
                # cv.imshow('center', img)
            return

        cv.namedWindow('center', cv.WINDOW_NORMAL)
        cv.setMouseCallback('center', click)
        cv.resizeWindow('center', self.img.shape[1] // 2, self.img.shape[0] // 2)
        cv.imshow('center', self.img)
        q = cv.waitKey(0) & 0xFF
        return


    def show_optical_flow_on_img(self, p0, delta_p0, magnification = 0.0, show = 1):
        if len(self.img.shape) == 2:
            img = cv.cvtColor(self.img, cv.COLOR_GRAY2RGB)
        else:
            img = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)

        for (point, delta) in zip(p0, delta_p0):
            x = int(point[0, 0])
            y = int(point[0, 1])
            delta_x = delta[0, 0]
            delta_y = delta[0, 1]

            x2 = int(x + (1+magnification) * delta_x)
            y2 = int(y + (1+magnification) * delta_y)
            if x >= img.shape[1] or y >= img.shape[0]:
                print('Problem (x, y) =',x, y)
                continue
            img[y, x, 2] = 255
            img = cv.arrowedLine(img, (x, y), (x2, y2), (255, 0, 0), 1)

        # fig = plt.figure(figsize=(15,10))
        plt.figure(figsize=(13, 8))
        print('Points =', len(p0))
        if len(img.shape) == 2:
            plt.imshow(img ,cmap = 'gray')
        else:
            plt.imshow(img)

        if show:
            plt.show()
        # cv.namedWindow('img', cv.WINDOW_NORMAL)
        # cv.resizeWindow('img', img.shape[1] // 2, img.shape[0] // 2)
        # cv.imshow('img', img)
        # cv.waitKey(0)

        return img


    def __mul__(self, other):
        if type(other) != int and type(other) != float:
            return
        return ImageClass( self.img * other )

    def __add__(self, other):
        if type(other) != int and type(other) != float:
            return
        return ImageClass( self.img + other )

    def __sub__(self, other):
        if type(other) != int and type(other) != float:
            return
        return ImageClass( self.img - other )

    def __truediv__(self, other):
        if type(other) != int and type(other) != float:
            return
        return ImageClass( self.img / other )








# if __name__ == '__main__':
#     frame = ImageClass('Correlation_vs_radius_example_class0_5000_1024_CIFAR10.png')
#     rotated = frame.rotate_img(45)
#     rotated = frame.rotate_img(90)
#     rotated.show()
# #     canny_frame = frame.canny_edge_detector()
# #     # print()
# #     # print(frame.__doc__)