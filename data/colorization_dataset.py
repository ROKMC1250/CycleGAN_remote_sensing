import os
from data.base_dataset import BaseDataset, get_transform,get_tiff_transform
from data.image_folder import make_dataset
from skimage import color  # require skimage
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from osgeo import gdal
import cv2

class ColorizationDataset(BaseDataset):
    """This dataset class can load a set of natural images in RGB, and convert RGB format into (L, ab) pairs in Lab color space.

    This dataset is required by pix2pix-based colorization model ('--model colorization')
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        By default, the number of channels for input image  is 1 (L) and
        the number of channels for output image is 2 (ab). The direction is from A to B
        """
        parser.set_defaults(input_nc=1, output_nc=2, direction='AtoB')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir, opt.max_dataset_size))
        assert(opt.input_nc == 1 and opt.output_nc == 2 and opt.direction == 'AtoB')
        self.transform = get_transform(self.opt, convert=False)
        self.tif_transform = get_tiff_transform(self.opt, convert=False)
        self.Lband_use = opt.Lband_use
        self.max_value = opt.max_value

    # def __getitem__(self, index):
    #     """Return a data point and its metadata information.

    #     Parameters:
    #         index - - a random integer for data indexing

    #     Returns a dictionary that contains A, B, A_paths and B_paths
    #         A (tensor) - - the L channel of an image
    #         B (tensor) - - the ab channels of the same image
    #         A_paths (str) - - image paths
    #         B_paths (str) - - image paths (same as A_paths)
    #     """
    #     path = self.AB_paths[index]
    #     Lband_use = self.Lband_use

    #     ###################open tif image from gdal################################
    #     if path.split('.')[-1].lower() == 'tif' :

    #         im = self.gdal_tif_open(path) # gdal tif
    #         print(im.dtype)
    #         # im = Image.fromarray(im.astype(np.uint8))
    #         # im = self.transform(im)
    #         im = np.array(im)
    #         if im.shape[2] == 1:
    #             im = np.repeat(im, 3, axis=-1)

    #     elif path.split('.')[-1].lower() == 'jpg':

    #         im = Image.open(path).convert('RGB')
    #         im = self.transform(im)
    #         im = np.array(im)
        
    #     if Lband_use ==1 :            
    #         #############only needed if input tif image is in grayscale##########
    #         im = np.multiply(im,(100/255))
    #         ###############################################################
    #         img_tensored = transforms.ToTensor()(im).float()
    #         A = img_tensored[[0], ...]/50.0 -1.0

    #     elif Lband_use == 0:

    #         lab = color.rgb2lab(im).astype(np.float32) 
    #         lab_t = transforms.ToTensor()(lab)    
    #         A = lab_t[[0], ...] / 50.0 - 1.0    
    #         B = lab_t[[1, 2], ...] / 110.0
        
    #     # return {'A': A, 'B': B, 'A_paths': path, 'B_paths': path}
    #     return {'A': A, 'A_paths': path}

    def __getitem__(self,index):

        path = self.AB_paths[index]
        Lband_use = self.Lband_use

        ###################open tif image from gdal################################
        if path.split('.')[-1].lower() == 'tif' :

            im = self.gdal_tif_open(path) # gdal tif
            # im = self.tif_transform(im)
            im = np.array(im)

        elif path.split('.')[-1].lower() == 'jpg':

            im = Image.open(path).convert('RGB')
            im = self.transform(im)
            im = np.array(im)
        
        if Lband_use ==1 :            
            #############only needed if input tif image is in grayscale##########
            im = np.multiply(im,(100/255))
            ###############################################################
            img_tensored = transforms.ToTensor()(im).float()
            A = img_tensored[[0], ...]/50.0 -1.0

        elif Lband_use == 0:

            lab = cv2.cvtColor(im,cv2.COLOR_RGB2Lab)
            # print(lab[200,200,:])
            lab_t = transforms.ToTensor()(lab)    
            A = lab_t[[0], ...] / 50.0 - 1.0    
            # print(A[:,200,200])
            B = lab_t[[1, 2], ...] / 110.0
            # print(B[:,200,200])
        # return {'A': A, 'B': B, 'A_paths': path, 'B_paths': path}
        return {'A': A, 'A_paths': path}





    def gdal_tif_open(self,imgPath):
        image = gdal.Open(imgPath)
        num_bands = image.RasterCount

        if num_bands == 1 :
            band1 = np.expand_dims(np.array(image.GetRasterBand(1).ReadAsArray())/self.max_value, axis=2)
            band2 = np.expand_dims(np.array(image.GetRasterBand(1).ReadAsArray())/self.max_value, axis=2)
            band3 = np.expand_dims(np.array(image.GetRasterBand(1).ReadAsArray())/self.max_value, axis=2)
            img_array = np.concatenate([band1, band2, band3], axis=2)

            # img_array = band1
        elif num_bands == 3:
            band1 = np.expand_dims(np.array(image.GetRasterBand(1).ReadAsArray()) /self.max_value, axis=2)
            band2 = np.expand_dims(np.array(image.GetRasterBand(2).ReadAsArray()) /self.max_value, axis=2)
            band3 = np.expand_dims(np.array(image.GetRasterBand(3).ReadAsArray()) /self.max_value, axis=2)
            img_array = np.concatenate([band1, band2, band3], axis=2)
        else:
            raise ValueError('This function only supprots images with 1 or 3 bands ')

        self.original_path = imgPath
        self.height, self.width, self.bands = img_array.shape
        return img_array.astype(np.float32)


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
