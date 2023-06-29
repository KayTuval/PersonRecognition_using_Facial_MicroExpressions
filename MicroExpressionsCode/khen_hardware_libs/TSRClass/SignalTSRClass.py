__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '5.10.2021'

from khen_hardware_libs.VideoClass.VideoClass import *
from khen_hardware_libs.SignalClass.SignalClass import *

class SignalTSRClass:

    '''
    SignalTSRClass performs the up-sampled signal version from a video, using the TSR method
    '''

    def __init__(self, blue_vec, green_vec, red_vec):
        '''
        Initilize the Class
        :param blue_vec: vector of the blue flicker code
        :param green_vec: vector of the green flicker code
        :param red_vec: vector of the red flicker code
        '''

        self.blue_vec = blue_vec
        self.green_vec = green_vec
        self.red_vec = red_vec
        self.N = len(red_vec)

        self.Smat = self._get_Smat()
        self.invMmat = self._get_invMmat()
        self.Itrans = self._get_Itrans()


    def _get_Smat(self):
        '''
        This function creates the S matrix
        '''
        return np.concatenate((self.blue_vec.reshape(-1, 1), \
                               self.green_vec.reshape(-1, 1), \
                               self.red_vec.reshape(-1, 1)), axis=1)

    def _get_invMmat(self):
        '''
        This function creates the M inverse matrix
        '''
        m_mat = np.zeros((self.N, self.N))
        for i in range(self.N):
            m_mat[i, i] = 2
            if i > 0:
                m_mat[i - 1, i] = -1
            if i < self.N - 1:
                m_mat[i + 1, i] = -1

        return np.linalg.inv(2 * m_mat)

    def _get_Itrans(self):
        '''
        This function creates the Transformation matrix
        '''
        return self.invMmat @ self.Smat @ np.linalg.pinv(self.Smat.transpose() @ self.invMmat @ self.Smat)

    def get_upsampled_signal_from_video(self, video_class, pixel, BGR_ratio = [1.0, 1.0, 1.0], \
                                        white_noise_filter = 0.0, neighborhood = 1):
        '''
        This function applies the TSR method on a single pixel of a video to get upsample signal
        :param video_class: VideoClass object
        :param pixel: (x, y) format pixel. e.g. (200, 100)
        :param BGR_ratio: Attenuation / enhancement factors to balance colors
        :param white_noise_filter: Threshold to filter noise (eliminates spectrum amplitude < threshold)
        :param neighborhood: Spatial averaging factor for noise reduction
        :return: Up-sampled signal
        '''
        x, y = pixel
        time_step = 1/video_class.fps
        signal_vec = []
        for frame in video_class.frames:
            BGR = []
            for c in range(3):  # [ R, G, B ]
                BGR += [BGR_ratio[c] * (frame[y - neighborhood:y + neighborhood + 1, \
                                        x - neighborhood:x + neighborhood + 1, c].astype(np.float32)/255.).mean()]
            BGR = np.clip( self.Itrans @ np.array(BGR).reshape(-1, 1), 0.0, 1.0)
            signal_vec += list(BGR.reshape(-1))

        signal_vec = np.array(signal_vec)
        recovered_signal = SignalClass(signal_vec, time_step=time_step/self.N)
        recovered_signal.filter_white_noise(white_noise_filter)  # 0.0175*np.abs(recovered_signal.signal_fft).max()
        return recovered_signal


if __name__ ==  '__main__':

    videoc = VideoClass('../../cis2.avi').resize_video(500, 500)
    TSR = SignalTSRClass(np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))

    # ImageClass(videoc.frames[10]).get_pixel_from_img()
    BGR_ratio = [1.0, 1.0, 1.0]
    pixel = (112, 202)
    white_noise_filter = 0.0
    neighborhood = 1
    lowsig = videoc.get_temporal_profile_for_pixel(pixel)
    highsig = TSR.get_upsampled_signal_from_video(videoc, pixel, BGR_ratio = BGR_ratio, \
                                                  white_noise_filter = white_noise_filter, \
                                                  neighborhood = neighborhood)

    lowsig.show_signal()
    highsig.show_signal()

    lowsig_resampled = lowsig.get_resampled_signal(dense_factor = 5)
    highsig_resampled = highsig.get_resampled_signal(dense_factor = 5)

    lowsig_resampled.show_signal()
    highsig_resampled.show_signal()
