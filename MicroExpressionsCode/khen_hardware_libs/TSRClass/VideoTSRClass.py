__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '5.10.2021'

from khen_hardware_libs.VideoClass.VideoClass import *


class VideoTSRClass:
    '''
    VideoTSRClass performs the up-sampled video version from a video (with spatial correlation), using the TSR method
    '''

    def __init__(self, blue_mat, green_mat, red_mat, temporal_weight = 1.0, spatial_weight = 1.0):
        '''
        Initilize the Class
        :param blue_mat: matrix of the blue flicker code (for all 5 pixels)
        :param green_mat: matrix of the green flicker code (for all 5 pixels)
        :param red_mat: matrix of the red flicker code (for all 5 pixels)
        :param temporal_weight: Temporal correlation weight factor for the optimization term
        :param spatial_weight: Spatial correlation weight factor for the optimization term
        '''
        self.blue_mat = blue_mat
        self.green_mat = green_mat
        self.red_mat = red_mat
        self.temporal_weight = temporal_weight
        self.spatial_weight = spatial_weight
        self.N = blue_mat.shape[0]
        self.number_of_channels = 3

        # Calculate S matrix, M matrix and Minv
        self.Smat = self._get_Smat()
        self.Mmat = self._get_Mmat()
        self.invMmat = self._get_invMmat()

        # Get the total transformation Itrans
        self.Itrans = self._get_Itrans()


    def _get_Mmat(self):
        Mmat = np.zeros((5*self.N, 5*self.N))
        for i in range(5*self.N):
            Mmat[i,i] = 4*(self.spatial_weight+self.temporal_weight)
            if i % self.N > 0:
                Mmat[i - 1, i] = -2*self.temporal_weight
            if i % self.N < self.N - 1:
                Mmat[i + 1, i] = -2*self.temporal_weight
            for j in range(1,5):
                Mmat[i, (i + j*self.N) % (5*self.N)] = -2*self.spatial_weight

        return Mmat

    def _get_Smat(self):
        Smat = np.zeros( (self.N*5, 5*self.number_of_channels) )
        for i in range(5):
            pixel_block = np.concatenate((self.blue_mat[:,i:i+1], \
                                          self.green_mat[:,i:i+1], \
                                          self.red_mat[:,i:i+1] ), axis=1)
            Smat[i*self.N:(i+1)*self.N,i*self.number_of_channels:(i+1)*self.number_of_channels] = pixel_block

        return Smat

    def _get_invMmat(self):
        return np.linalg.inv( self.Mmat )

    def _get_Itrans(self):
        STinvMS = np.matmul( self.Smat.transpose(), np.matmul(self.invMmat, self.Smat) )
        return np.matmul( np.matmul( self.invMmat , self.Smat ), np.linalg.pinv(STinvMS) )


    def apply_tsr_on_illumination_matrix(self, C_color_matrix):
        Cvec = self._from_tensor_to_vector(C_color_matrix)
        return np.matmul( self.Itrans, Cvec)

    def _from_patch_to_tsr(self, patch):
        '''
        :param patch: Patch in size of 3X3 (colored)
        :return: TSR of the middle pixel
        '''
        C_color_matrix = self._from_patch_to_tensor( patch )
        upsampled_intensity = self._from_vector_to_tensor( self.apply_tsr_on_illumination_matrix(C_color_matrix) )
        return self._from_tensor_to_patch( upsampled_intensity )

    def _from_tensor_to_patch(self, tensor):
        ''' Gets tensor of (5, N) and reshape to (3,3,N) '''
        patch = np.zeros( (3,3,self.N) )
        patch[1,1,:] = tensor[0, :]
        patch[0,1,:] = tensor[1, :]
        patch[1,2,:] = tensor[2, :]
        patch[2,1,:] = tensor[3, :]
        patch[1,0,:] = tensor[4, :]
        return patch

    def _from_patch_to_tensor(self, patch):
        ''' Gets patch of (3,3,3) and reshape to (5, 3) '''
        return np.array([
                        [ patch[1,1,2], patch[1,1,1], patch[1,1,0] ],   # Middle pixel - 1
                        [ patch[0,1,2], patch[0,1,1], patch[0,1,0] ],   # Upper pixel - 2
                        [ patch[1,2,2], patch[1,2,1], patch[1,2,0] ],   # Right pixel - 3
                        [ patch[2,1,2], patch[2,1,1], patch[2,1,0] ],   # Bottom pixel - 4
                        [ patch[0,1,2], patch[0,1,1], patch[0,1,0] ]    # Left pixel - 5
                        ])

    def _from_vector_to_tensor(self, vector):
        return np.column_stack( [vector[5*i:5*(i+1)] for i in range(len(vector)//5)] )

    def _from_tensor_to_vector(self, tensor):
        return np.hstack(tensor).reshape(-1,1)


    def from_video_to_tsr(self, video_class, BGR_ratio = [1.0, 1.0, 1.0], frames2check = [0, -1]):
        '''
        from_video_to_tsr creates a video after TSR procedure
        :param video_class: VideoClass object
        :param BGR_ratio: Attenuation / enhancement factors to balance colors
        :param frames2check: Video frames range [start frame, stop frame]
        :return: Up-sampled Video
        '''

        upsampled_video = []
        # Running over all the video frames:
        for frame in video_class.frames[frames2check[0]:frames2check[1]]:
            upsampled_frame = np.zeros((frame.shape[0], frame.shape[1],self.N))
            frame = frame.astype(np.float32)/255.
            frame[:,:,0] = BGR_ratio[0]*frame[:,:,0]
            frame[:,:,1] = BGR_ratio[1]*frame[:,:,1]
            frame[:,:,2] = BGR_ratio[2]*frame[:,:,2]
            # Running over all the patches in the frame:
            for x_coordinate in range(1, frame.shape[1]-1,1):
                for y_coordinate in range(1, frame.shape[0]-1,1):
                    patch = frame[y_coordinate-1:y_coordinate+2,x_coordinate-1:x_coordinate+2]
                    # Upsample the patch using the TSR method:
                    upsampled_patch = self._from_patch_to_tsr(patch)
                    upsampled_frame[y_coordinate,x_coordinate,:] = upsampled_patch[1,1,:]
            upsampled_video += list( ( np.clip(np.moveaxis(upsampled_frame, 2, 0), 0.0, 1.0)*255).astype(np.uint8) )

        return VideoClass(upsampled_video, fps = video_class.fps * self.N)



def get_mat_from_N(N):
    if N == 3:
        blue_mat = np.array([[1, 0, 0]] * 5).T
        green_mat = np.array([[0, 1, 0]] * 5).T
        red_mat = np.array([[0, 0, 1]] * 5).T
    elif N == 4:
        blue_mat = np.array([[1, 0, 0, 1]] * 5).T
        green_mat = np.array([[1, 0, 1, 1]] * 5).T
        red_mat = np.array([[0, 1, 0, 1]] * 5).T
    elif N == 5:
        blue_mat = np.array([[0, 1, 0, 0, 0]] * 5).T
        green_mat = np.array([[1, 0, 1, 0, 1]] * 5).T
        red_mat = np.array([[0, 0, 0, 1, 0]] * 5).T
    elif N == 6:
        blue_mat = np.array([[1, 0, 1, 0, 1, 0]] * 5).T
        green_mat = np.array([[0, 1, 0, 1, 0, 1]] * 5).T
        red_mat = np.array([[1, 1, 1, 1, 1, 1]] * 5).T

    return blue_mat, green_mat, red_mat

def create_videos_for_N(video, N, folder_path = '../../', BGR_ratio = [1.0, 1.0, 1.0], frames2check = [0, -1]):
    '''
    Creates a sweeping results over different weight factors and save for each case
    :param video_class: VideoClass object
    :param N: Level of TSR raising
    :param BGR_ratio: Attenuation / enhancement factors to balance colors
    :param frames2check: Video frames range [start frame, stop frame]
    '''
    blue_mat, green_mat, red_mat = get_mat_from_N(N)

    for SPATIAL_WEIGHT in [0, 1, 2, 3]:
        for TEMPORAL_WEIGHT in [0, 1, 2, 3]:
            if SPATIAL_WEIGHT != 1 and TEMPORAL_WEIGHT == SPATIAL_WEIGHT:
                continue
            tsr_class = VideoTSRClass(blue_mat, green_mat, red_mat, \
                                         temporal_weight = TEMPORAL_WEIGHT, \
                                         spatial_weight = SPATIAL_WEIGHT)

            upsampled_video = tsr_class.from_video_to_tsr(video, BGR_ratio = BGR_ratio, frames2check = frames2check)
            upsampled_video.save_video(folder_path + 'Upsampled_video_N={}_{}_{}'.format(N, SPATIAL_WEIGHT, TEMPORAL_WEIGHT))


if __name__ == '__main__':
    folder_path = '../../'
    video = VideoClass(folder_path + 'cis2.avi').resize_video(width = 128, height = 128)

    frames2check = [5, 10]
    SPATIAL_WEIGHT = 0.0
    TEMPORAL_WEIGHT = 1.0
    blue_mat = np.array([ [1, 0, 0, 1] ]*5).T
    green_mat = np.array([ [0, 1, 0, 0] ]*5).T
    red_mat = np.array([ [0, 0, 1, 0] ]*5).T
    BGR_ratio = [1.0, 1.0, 1.0]
    VideoTSR_class = VideoTSRClass(blue_mat, green_mat, red_mat, \
                              temporal_weight = TEMPORAL_WEIGHT, \
                              spatial_weight = SPATIAL_WEIGHT)

    upsampled_video = VideoTSR_class.from_video_to_tsr(video, BGR_ratio = [1.0, 1.0, 1.0], frames2check = frames2check)

    video.show(display_fps=0)
    upsampled_video.show(display_fps=0)

    upsampled_video.save_video(folder_path + 'upsampled_version')

