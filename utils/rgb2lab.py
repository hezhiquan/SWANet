"""

this file is to decompose the low-light RGB to LAB color space(lightness and color components)

"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import numpy as np


class LAB(nn.Module):
    def __init__(self):
        super().__init__()
        self.illuminants = \
    {"A": {'2': (1.098466069456375, 1, 0.3558228003436005),
           '10': (1.111420406956693, 1, 0.3519978321919493)},
     "D50": {'2': (0.9642119944211994, 1, 0.8251882845188288),
             '10': (0.9672062750333777, 1, 0.8142801513128616)},
     "D55": {'2': (0.956797052643698, 1, 0.9214805860173273),
             '10': (0.9579665682254781, 1, 0.9092525159847462)},
     "D65": {'2': (0.95047, 1., 1.08883),   # This was: `lab_ref_white`
             '10': (0.94809667673716, 1, 1.0730513595166162)},
     "D75": {'2': (0.9497220898840717, 1, 1.226393520724154),
             '10': (0.9441713925645873, 1, 1.2064272211720228)},
     "E": {'2': (1.0, 1.0, 1.0),
           '10': (1.0, 1.0, 1.0)}}
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def _get_xyz_coords(self,illuminant,observer):
        """ Get the XYZ coordinates from illuminant and observer

        Parameters
        ==========
        illuminant : {"A","D50","D65","D75","E"}
        observer : {"2","10"}

        Returns
        ==========

        XYZ coordinate Tensor Float

        """
        try:
            return torch.tensor(self.illuminants[illuminant][observer]).float()
        except KeyError:
            raise ValueError("Unknown illuminat:'{}'/observer:'{}' combination".format(illuminant,observer))
    
    def _check_shape(self,tensor):
        if tensor.shape[0] != 3:
            raise ValueError("Input array must have (batch, 3,height,width)")

    def xyz2rgb(self,xyz_tensor,show_results=False):
        """XYZ to RGB color space conversion.

        Parameters
        ==========

        xyz_tensor : shape -> (3,height,width) Tensor
        show_results : whether to display the resulting rgb image

        Returns
        ==========

        rgb_tensor : shape -> (3,height,width) Tensor

        """
        xyz_tensor = xyz_tensor.permute(1,2,0)
        xyz_from_rgb = torch.tensor([[0.412453, 0.357580, 0.180423],
                                     [0.212671, 0.715160, 0.072169],
                                     [0.019334, 0.119193, 0.950227]]).to(self.device)
        rgb_from_xyz = torch.inverse(xyz_from_rgb)
        rgb = torch.matmul(xyz_tensor,torch.t(rgb_from_xyz))
        mask = rgb > 0.0031308
        rgb[mask] = 1.055 * torch.pow(rgb[mask],1/2.4) - 0.055
        rgb[~mask] *= 12.92
        rgb = torch.clamp(rgb,0,1)
        rgb = rgb.permute(2,0,1)
        if show_results:
            rgb_numpy = rgb.cpu().detach().numpy().transpose(1,2,0)
            plt.imshow(rgb_numpy)
            plt.show()
        return rgb

    def lab2xyz(self,lab_tensor,show_results=False,illuminant='D65',observer='2'):
        """LAB to XYZ color space conversion.

        Parameters
        ==========

        lab_tensor : shape -> (3,height,width) Tensor
        show_results : whether to display the resulting xyz image

        Returns
        ==========

        xyz_tensor : shape -> (3,height,width) Tensor

        """
        l,a,b = lab_tensor[0],lab_tensor[1],lab_tensor[2]
        y = (l+16.)/116.
        x = (a / 500.) + y
        z = y - (b / 200.)

        xyz = torch.stack([x,y,z],dim=0)
        mask = xyz > 0.2068966
        xyz[mask] = torch.pow(xyz[mask],3.)
        xyz[~mask] = (xyz[~mask] - 16. / 116.) / 7.787

        xyz_ref_white = self._get_xyz_coords(illuminant,observer).to(self.device)
        xyz = xyz.permute(1,2,0)
        xyz *= xyz_ref_white
        xyz = xyz.permute(2,0,1)
        if show_results:
            xyz_numpy = xyz.cpu().detach().numpy().transpose(1,2,0)
            plt.imshow(xyz_numpy)
            plt.show()
        return xyz


    def lab2rgb(self,lab_tensor,show_results_xyz=False,show_results_rgb=False):
        """LAB to RGB color space conversion.

        Parameters
        ==========

        lab_tensor : shape -> (3,height,width) Tensor
        show_results_xyz : whether to display the resulting xyz image
        show_results_rgb : whether to display the resulting rgb image

        Returns
        ==========

        rgb_tensor : shape -> (3,height,width) Tensor

        """
        results = []
        for i in range(lab_tensor.shape[0]):
            xyz = self.lab2xyz(lab_tensor[i],show_results_xyz)
            rgb = self.xyz2rgb(xyz,show_results_rgb)
            results.append(rgb)
        results = torch.cat(results).reshape(len(results),*results[0].shape)
        return results



    def rgb2xyz(self,rgb_tensor,show_results=False):
        """RGB to XYZ color space conversion.

        Parameters
        ==========

        rgb_tensor : shape -> (3,height,width) Tensor
        show_results : whether to display the resulting xyz image

        Returns
        ==========
        
        xyz_tensor : shape -> (3,height,width) Tensor

        what is xyz_tensor?
        -------------------
            -> https://www.dic-color.com/knowledge/xyz.html 

        """
        self._check_shape(rgb_tensor) #must have input shape {3,height,width}
        rgb_tensor = rgb_tensor.permute(1,2,0)
        mask = rgb_tensor > 0.04045
        rgb_tensor[mask] = torch.pow((rgb_tensor[mask] + 0.055)/1.055,2.4)
        rgb_tensor[~mask] /= 12.92
        xyz_from_rgb = torch.tensor([[0.412453, 0.357580, 0.180423],
                                     [0.212671, 0.715160, 0.072169],
                                     [0.019334, 0.119193, 0.950227]]).to(self.device)
        xyz = torch.matmul(rgb_tensor,torch.t(xyz_from_rgb))
        if show_results: # show matplotlib
            xyz_numpy = xyz.cpu().detach().numpy()
            plt.imshow(xyz_numpy)
            plt.show()

        xyz = xyz.permute(2,0,1)
        return xyz

    def xyz2lab(self,xyz_tensor,show_results=False,illuminant='D65',observer='2'):
        """XYZ to CIE-LAB color space conversion.

        Parameters
        ==========

        xyz_tensor : shape -> (3,height,width) Tensor
        show_results : whether to display the resulting lab image
        
        Returns
        ==========
        
        lab_tensor : shape -> (3,height,width) Tensor

        what is lab_tensor?
        -------------------
            -> http://rysys.co.jp/dpex/help_laboutput.html 


        """
        xyz_tensor = xyz_tensor.permute(1,2,0)

        xyz_ref_white = self._get_xyz_coords(illuminant,observer).to(self.device)
        xyz_tensor = xyz_tensor / xyz_ref_white

        mask = xyz_tensor > 0.008856
        xyz_tensor[mask] = torch.pow(xyz_tensor[mask],1/3)
        xyz_tensor[~mask] = 7.787 * xyz_tensor[~mask] + 16. / 116.
        x,y,z = xyz_tensor[...,0],xyz_tensor[...,1],xyz_tensor[...,2]
        L = (116. * y) - 16.
        a = 500. * (x - y)
        b = 200. * (y - z)
        lab = torch.cat([L.unsqueeze(-1),a.unsqueeze(-1),b.unsqueeze(-1)],dim=-1)
        if show_results:
            lab_numpy = lab.cpu().detach().numpy()
            plt.imshow(lab_numpy)
            plt.show()

        lab = lab.permute(2,0,1)
        return lab

    def forward(self,rgb_tensor,show_xyz_results=False,show_lab_results=False):
        results = []
        for i in range(rgb_tensor.shape[0]):
            xyz = self.rgb2xyz(rgb_tensor[i],show_xyz_results)
            lab = self.xyz2lab(xyz,show_lab_results)
            results.append(lab)
        results = torch.cat(results).reshape(len(results),*results[0].shape)
        return results

if __name__ == "__main__":
    print("Hello,rgb2lab.py!!!")
    
    _IMAGE_PATH = "../../sample_images/silhouette-3038483_1920-1280x640.jpg"
    _DEFAULT_SIZE = 512
    im = cv2.imread(_IMAGE_PATH)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)#BGR->RGB
    height = im.shape[0]
    width = im.shape[1]
    print("image_height -> {}".format(height))
    print("image_width -> {}".format(width))
    im = im[height//2-_DEFAULT_SIZE//2:height//2+_DEFAULT_SIZE//2,width//2-_DEFAULT_SIZE//2:width//2+_DEFAULT_SIZE//2]
    lab = LAB()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    rgb_image = torch.from_numpy(im.transpose(2,0,1)).float().to(device)
    rgb_image = rgb_image.unsqueeze(0)
    rgb_image /= 255.
    lab_output = lab(rgb_image,False,False)
    rgb_output = lab.lab2rgb(lab_output,False,False)
    plt.imshow(rgb_output[0].cpu().detach().numpy().transpose(1,2,0))
    plt.show()
