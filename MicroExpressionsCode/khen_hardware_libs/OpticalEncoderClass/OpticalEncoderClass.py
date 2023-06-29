__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '9.10.2021'

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.fftpack as sfft
from khen_hardware_libs.ImageClass.ImageClass import *


# Optical Encoder Net
class OpticalEncNet(nn.Module):
    """
    A class used to perform a Optically diffraction-based Imaging convolution
    ...

    Attributes
    ----------
    mask_resolution : int (Odd)
        The mask quantification resolution. Should be Odd number.

    initial_perturbation_rate : float
        The initial random pertubation (standard deviation of the gaussian noise factor) for the learnable PSF function

    wavelengths_list : list of float
        List of different wavelengths in the same order as the image. Grayscale 1 element, color = [R, G, B]

    main_wavelength : float
        Center of mass among the different wavelengths to make the design for.

    h_noise_amp_phase_mean_std : tuple in format ( (*,*) , (*,*) )
        Gaussian noise model for the real and imaginary part of the PSF (h_parameters). Each term is another tupple of
        the expectation value and the standard deviation.

    device : string ('cpu' or 'gpu')
        The device to work on (for optimization process)

    Methods
    -------
    set_mask_resolution()
        From the mask resolution value, calculates the influence of any wavelength

    multiply_mask_size()
        Increase the size of the mask by (roughly) factor of 2 (in each dimension)

    forward(x, add_noise = False)
        Apply the optical transformation on a given image. Set add_noise = True to add noise to h_parameter
        to get more noise-robust model.

    get_filter_score()
        Calculates the score of the optical intensity transmission rate
        (1 - all light is transsmited, 0 - all light is blocked)

    normalize_psf()
        Normalize the PSF function (h_parameters) to get PSF with only-phase values

    normalize_ctf()
        Normalize the PSF function (h_parameters) to get CTF with only-phase values

    get_ctf_score()
        Calculates the score of the optical intensity transmission rate (of the CTF)
        (1 - all light is transsmited, 0 - all light is blocked)

    add_h_noise()
        Add noise for h_parameters according to h_noise_amp_phase_mean_std values

    set_mask(ctf_mask)
        Gets a mask and set it as the CTF, to calculate h_parameters

    forward_no_grad(img_class_input)
        Apply the transformation with no learned process

    from_img_converter(img)
        Transformation from numpy matrix (an image) to the corrected dimension tensor to feed the transformation

    to_ImageClass_converter(img_tensor)
        Transformation from tensor to ImageClass object

    get_coherent_psf(show = 0, save = 0)
        returns the PSF (coherent). If show = 1 displays on screen, if save = 1 saves the figure

    get_incoherent_psf(show = 0, save = 0)
        returns the PSF (incoherent). If show = 1 displays on screen, if save = 1 saves the figure

    get_ctf(show = 0, save = 0)
        returns the CTF (coherent). If show = 1 displays on screen, if save = 1 saves the figure

    get_otf(show=0, save=0)
        returns the OTF (incoherent). If show = 1 displays on screen, if save = 1 saves the figure

    """

    def __init__(self, mask_resolution = 51, initial_perturbation_rate = 0.1, \
                 wavelengths_list = [610.0, 540.0, 450.0], main_wavelength = 540.0, \
                 h_noise_amp_phase_mean_std = ((0.0, 0.0),(0.0, 0.0)), device = 'cpu'):
        '''
        Initialize the Optical Encoder Mask
        :param mask_resolution: Size of the CTF or PSF Grid
        :param initial_perturbation_rate: A deviation from Delta Function (for optimization)
        :param green_wavelength: Wavelength of green's 'center of mass'
        :param red_wavelength: Wavelength of red's 'center of mass'
        :param blue_wavelength: Wavelength of blue's 'center of mass'
        :param h_noise_amp_phase_mean_std: Noise for each forward in format (amplitude - (mean, std), phase - (mean, std))
        :param device: 'cpu' or 'gpu'
        '''
        super(OpticalEncNet, self).__init__()

        # Optical process:
        self.device = device
        self.initial_perturbation_rate = initial_perturbation_rate
        self.h_noise_amp_phase_mean_std = h_noise_amp_phase_mean_std

        self.wavelengths_list = wavelengths_list
        self.main_wavelength = main_wavelength
        self.wavelength_alpha_factor_list = []
        for wavelength in self.wavelengths_list:
            self.wavelength_alpha_factor_list += [ wavelength/self.main_wavelength ]

        if mask_resolution % 2 == 0:
            mask_resolution += 1
            print('Warning: mask_resolution should be Odd! \n\tChanging: {} -> {}'.format(mask_resolution, mask_resolution+1))

        self.mask_resolution = mask_resolution
        self.set_mask_resolution()

        # # Initialization method 1:
        # self.h_parameters = torch.nn.Parameter( \
        #                       torch.randn(1, 1, self.mask_resolution, self.mask_resolution),\
        #                       requires_grad = True )

        # Initialization method 2:
        perturbation_rate = self.initial_perturbation_rate / ( (self.mask_resolution-1)//2 )**2
        init_mask = torch.zeros(1,1,mask_resolution,mask_resolution)
        init_mask[0,0, mask_resolution//2, mask_resolution//2] = 1.0
        # self.h_parameters = torch.nn.Parameter( init_mask + \
        #                       perturbation_rate*torch.randn(1, 1, self.mask_resolution, self.mask_resolution), \
        #                       requires_grad = True )
        self.h_parameters = torch.nn.Parameter( init_mask + \
                              perturbation_rate*( torch.randn(1, 1, self.mask_resolution, self.mask_resolution) + \
                                                  1j * torch.randn(1, 1, self.mask_resolution, self.mask_resolution) ), \
                              requires_grad = True )

        # # TODO: Test - Delta Function Tests:
        # self.h_parameters = torch.nn.Parameter( torch.zeros(1,1,mask_resolution,mask_resolution), requires_grad = False )
        # self.h_parameters[0,0, mask_resolution//2, mask_resolution//2] = 1
        # self.h_parameters[0,0, mask_resolution//2-2:mask_resolution//2+2, mask_resolution//2-2:mask_resolution//2+2] = 1

    def set_mask_resolution(self):
        '''
        Calculate the chromatic response and size for each colors (R, G, B)
        :return:
        '''

        self.pad = int((self.mask_resolution - 1) / 2)
        self.transform_wavelength_size_list = []
        self.transform_wavelength_padding_list = []
        for alpha_factor in self.wavelength_alpha_factor_list:
            new_size = int(self.mask_resolution/alpha_factor)
            r_pad = [ int( np.floor( (self.mask_resolution - new_size)/2 ) ), \
                      int( np.ceil( (self.mask_resolution - new_size)/2 ) )]
            self.transform_wavelength_size_list += [ (new_size, new_size) ]
            # self.transform_red = torchvision.transforms.Resize( (new_size, new_size) )
            self.transform_wavelength_padding_list += [ nn.ZeroPad2d([r_pad[0], \
                                                        r_pad[1], \
                                                        r_pad[0], \
                                                        r_pad[1]]) ]

        return

    def multiply_mask_size(self):
        '''
        Apply mask multiplication to increase the mask size (of the CTF) from multiplication of the
        PSF size (via h_parameters).
        h_parameters size is changed from self.mask_resolution -> 2*self.mask_resolution - 1
        The increasing method is done via zero-padding with perturbation_rate which depands on self.mask_resolution
        and the self.initial_perturbation_rate
        :return:
        '''
        old_mask_res_padding = (self.mask_resolution - 1)//2
        self.mask_resolution = self.mask_resolution*2-1
        self.set_mask_resolution()

        perturbation_rate = self.initial_perturbation_rate / ( (self.mask_resolution-1)//2 )**2   # TODO: Test
        # self.perturbation_rate /= (self.mask_resolution-1)//2   # TODO: Test - changing pertubation
        # Interpolate the mask:
        # self.h_parameters = torch.nn.Parameter( \
        #                       torch.nn.functional.interpolate(self.h_parameters, size=self.mask_resolution, \
        #                                                       mode = 'nearest'), requires_grad = True )\
        # # Interpolate with zero values:
        # self.h_parameters = torch.nn.Parameter( \
        #     torch.nn.ZeroPad2d([old_mask_res_padding,old_mask_res_padding,old_mask_res_padding,old_mask_res_padding])\
        #                             ( self.h_parameters ), requires_grad = True )

        # Interpolate with random values:
        noise_mask = ( torch.randn(1,1, self.mask_resolution, self.mask_resolution) + \
                       1j*torch.randn(1,1, self.mask_resolution, self.mask_resolution) ) * perturbation_rate
        noise_mask[0,0,old_mask_res_padding:self.mask_resolution-old_mask_res_padding,\
                   old_mask_res_padding:self.mask_resolution-old_mask_res_padding] = 0.0

        # plt.imshow(noise_mask[0,0,:,:], cmap='gray'); plt.show()
        self.h_parameters = torch.nn.Parameter( \
                torch.nn.ZeroPad2d([old_mask_res_padding,old_mask_res_padding,old_mask_res_padding,old_mask_res_padding]) \
                    ( self.h_parameters ) + noise_mask.to(self.device) , requires_grad = True )
        return

    def forward(self, x, add_noise = False):
        '''
        Apply forward pass in the Optical Encoder layer
        :param x: Geometric imaging (clear image)
        :return: Captured image (in the sensor)
        '''
        if add_noise and self.h_noise_amp_phase_mean_std != ((0,0),(0,0)):
            self.add_h_noise()

        # Interpolation mode: 'nearest' | 'bilinear' | 'area'
        new_x_list = []
        for i in range( len(self.wavelength_alpha_factor_list) ):
            trans_size = self.transform_wavelength_size_list[i]
            pad_trans = self.transform_wavelength_padding_list[i]
            h_wavelength_parameter = pad_trans( \
                torch.nn.functional.interpolate( torch.real(self.h_parameters), \
                                                 size = trans_size, \
                                                 mode = 'bilinear' ) + \
                1j*torch.nn.functional.interpolate( torch.imag(self.h_parameters), \
                                                 size = trans_size, \
                                                 mode = 'bilinear' ) )
            wavelength_parameter = torch.mul( h_wavelength_parameter.conj(), h_wavelength_parameter ).float()
            wavelength_norm = wavelength_parameter.abs().sum()    # Normalize Kernel to keep img in (-1,1)
            new_x_list += [F.conv2d( x[:,i:i+1,:,:], wavelength_parameter, padding = self.pad ) / wavelength_norm]
        x = torch.cat([nx for nx in new_x_list], 1)
        return x

    def get_filter_score(self):
        '''
        Calculates the score of the CTF mask in terms of power-transmission
        :return: Transmission Score
        '''

        return (torch.mul( self.h_parameters.conj(), self.h_parameters ).float()/(self.mask_resolution**2) ).sum()

    def normalize_psf(self):
        '''
        Normalizing the PSF function
        :return:
        '''
        # print('Normalizing PSF')
        with torch.no_grad():
            # Normalization method - new: Check
            # self.h_parameters /= self.h_parameters.abs().sum()
            self.h_parameters = torch.nn.Parameter( torch.divide( self.h_parameters, self.h_parameters.abs() ), requires_grad=True)
        return

    def normalize_ctf(self):
        '''
        Normalizing the CTF function (via the PSF)
        :return:
        '''
        # print('Normalizing CTF')
        with torch.no_grad():
            ctf = self.get_ctf()
            ctf = np.divide( ctf, np.abs( ctf ) )[np.newaxis, np.newaxis, ...]
            self.h_parameters = torch.nn.Parameter( \
                    torch.tensor( (self.mask_resolution)*np.fft.ifft2( ctf ) ) , requires_grad=True)

        return

    def get_ctf_score(self):
        '''
        Print the Score of the CTF in terms of power-transmission
        :return:
        '''
        ctf = self.get_ctf()
        ctf_score = np.mean( np.abs( ctf )**2 )
        print('Total CTF score (Power transmission ratio from 0 to 1):', ctf_score)
        return

    def add_h_noise(self):
        '''
        Adding noise for h_parameters to get more production robust results
        '''
        with torch.no_grad():
            h_noise_real = self.h_noise_amp_phase_mean_std[0]
            h_noise_imag = self.h_noise_amp_phase_mean_std[0]
            noise_mask = h_noise_real[0] + 1j*h_noise_imag[0] + \
                torch.randn(1, 1, self.mask_resolution, self.mask_resolution) * ( h_noise_real[1] + h_noise_imag[1] )
            self.h_parameters = torch.nn.Parameter( self.h_parameters + noise_mask.to(self.device) , requires_grad=True)

        return

    def set_mask(self, ctf_mask):
        '''
        Given a mask, use it as the CTF
        :param ctf_mask: CTF mask
        '''

        with torch.no_grad():
            self.h_parameters = torch.nn.Parameter( torch.tensor( \
                sfft.fftshift( np.fft.ifft2( (self.mask_resolution) * ctf_mask[np.newaxis, np.newaxis, :, :] ) )  ) , requires_grad=True)
            self.mask_resolution = self.h_parameters.shape[-1]
        return

    def forward_no_grad(self, img_class_input):
        '''
        Perform forward pass with the model with no grad
        :param img_class: ImageClass type Image of the input image
        :return:  ImageClass type Image of the output image
        '''

        img_tensor = self.from_img_converter( img_class_input.img )
        with torch.no_grad():
            img_class_output = optical_encoder.to_ImageClass_converter( self( img_tensor ) )

        return img_class_output

    def from_img_converter(self, img):
        '''
        Convert from ImageClass type to a img type with the corrected dimension and type
        :param img: image frame
        :return: pytorch tensor image with the correct dimension
        '''
        img_tensor = torch.from_numpy(img / 255.).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        return img_tensor

    def to_ImageClass_converter(self, img_tensor):
        '''
        Convert from image tensor type to a ImageClass type with the corrected dimension and type
        :param img_tensor: pytorch tensor image with the correct dimension
        :return: ImageClass type image
        '''
        img = ( np.clip(img_tensor[0,...].permute(1, 2, 0).numpy(), 0.0, 1.0) * 255. ).astype(np.uint8)
        img_class = ImageClass( img )
        return img_class

    def get_coherent_psf(self, show = 0, save = 0):
        with torch.no_grad():
          coherent_psf = self.h_parameters.detach().to('cpu').numpy()[0,0,:,:]
        if show or save:
          # print( np.abs( coherent_psf ).max() )
          plt.close()
          plt.imshow( np.abs( coherent_psf ), cmap = 'gray' )
          plt.title('Coherent PSF (Mask)')
          if save:
            plt.savefig('Figures/coherent_psf.png')
          if show:
            plt.show()
        return coherent_psf

    def get_incoherent_psf(self, show = 0, save = 0):
        with torch.no_grad():
          coherent_psf = self.get_coherent_psf(show = 0)
          incoherent_psf = np.multiply(coherent_psf,coherent_psf)
        if show or save:
          # print( np.abs( incoherent_psf ).max() )
          plt.close()
          plt.imshow( np.abs( incoherent_psf ), cmap = 'gray')
          plt.title('Incoherent PSF (Mask)')
          if save:
            plt.savefig('Figures/incoherent_psf.png')
          if show:
            plt.show()
        return incoherent_psf

    def get_ctf(self, show = 0, save = 0):
        with torch.no_grad():
          coherent_psf = self.get_coherent_psf(show = 0)
          ctf = np.fft.fft2( coherent_psf ) / (self.mask_resolution)
          ctf = sfft.fftshift( ctf )
        if show or save:
          # print( np.abs( ctf ).max() )
          # print(ctf)
          plt.close()
          plt.subplot(1,2,1)
          plt.imshow( np.abs(ctf), cmap = 'gray', vmin=0.0) #, vmax=1.1
          plt.subplot(1,2,2)
          plt.imshow( np.angle(ctf), cmap = 'gray')
          plt.title('CTF (Aperture)')
          if save:
            plt.savefig('Figures/ctf.png')
          if show:
            plt.show()
        # print(ctf.min(), ctf.max())
        # print(ctf.mean())
        return ctf

    def get_otf(self, show = 0, save = 0):
        with torch.no_grad():
          incoherent_psf = self.get_incoherent_psf(show = 0)
          otf = np.fft.fft2( incoherent_psf ) / (self.mask_resolution)
          otf = sfft.fftshift( otf )
        if show or save:
          # print( np.abs( otf ).max() )
          # print(otf)
          plt.close()
          plt.imshow( np.abs(otf), cmap = 'gray')
          plt.title('OTF (Aperture)')
          if save:
            plt.savefig('Figures/otf.png')
          if show:
            plt.show()
        return otf


if __name__ == '__main__':
    mask_resolution = 101
    optical_encoder = OpticalEncNet(mask_resolution = mask_resolution)
    # optical_encoder.get_ctf(show = 1)

    x, y = torch.arange(mask_resolution), torch.arange(mask_resolution)
    grid_x, grid_y = torch.meshgrid(x, y)
    a = torch.sin( torch.multiply( grid_x, grid_y )*2*np.pi/10.0 )
    optical_encoder.set_mask( a )
    # optical_encoder.get_ctf(show = 1)

    # b = torch.exp( 1j*grid_x*2*np.pi/3.0 )
    # optical_encoder.set_mask( b )
    # optical_encoder.get_ctf(show = 1)

    c = ImageClass('../../img2.jpg').resize_image(640, 480)
    # c.show()
    d = optical_encoder.forward_no_grad(c)
    d.show()