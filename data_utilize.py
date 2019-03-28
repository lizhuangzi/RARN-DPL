from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Scale,RandomHorizontalFlip,RandomVerticalFlip,RandomRotation,Resize,ColorJitter
import skimage.io
import random
import torchvision.transforms.functional as F

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG','bmp'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Scale(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Scale(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir1,dataset_dir2):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = dataset_dir1
        self.l = len(listdir(dataset_dir1))
        self.image_filenames2 = dataset_dir2
        self.train_hr_transform = train_hr_transform(0)


    def randomHflip(self,img):
        return F.hflip(img)

    def randomVflip(self,img):
        return F.vflip(img)

    def __getitem__(self, index):
        str = "%d.jpg" %index
        lowimage =Image.open(join( self.image_filenames,str))
        labelimage = Image.open(join( self.image_filenames2,str))

        hfp = random.random()
        if hfp < 0.5:
            labelimage = self.randomHflip(labelimage)
            lowimage = self.randomHflip(lowimage)

        vfp = random.random()
        if vfp <0.5:
            labelimage = self.randomVflip(labelimage)
            lowimage = self.randomVflip(lowimage)

        return ToTensor()(lowimage), ToTensor()(labelimage)

    def __len__(self):
        return self.l//6


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir1,dataset_dir2):
        super(ValDatasetFromFolder, self).__init__()
        self.image_filenames = dataset_dir1
        self.l = len(listdir(dataset_dir1))
        self.image_filenames2 = dataset_dir2


    def __getitem__(self, index):
        str = "%d.jpg" %index
        lowimage =Image.open(join( self.image_filenames,str))
        labelimage = Image.open(join( self.image_filenames2,str))

        return ToTensor()(lowimage), ToTensor()(labelimage)

    def __len__(self):
        return 400



class TrainDatasetFromFolderY(Dataset):
    def __init__(self, dataset_dir1,dataset_dir2):
        super(TrainDatasetFromFolderY, self).__init__()
        self.image_filenames = dataset_dir1
        self.l = len(listdir(dataset_dir1))
        self.image_filenames2 = dataset_dir2


    def __getitem__(self, index):
        str = "%d.jpg" %index
        lowimage =Image.open(join( self.image_filenames,str)).convert('YCbCr')
        labelimage = Image.open(join( self.image_filenames2,str)).convert('YCbCr')

        return ToTensor()(lowimage), ToTensor()(labelimage)

    def __len__(self):
        return self.l//4


class ValDatasetFromFolderY(Dataset):
    def __init__(self, dataset_dir1,dataset_dir2):
        super(ValDatasetFromFolderY, self).__init__()
        self.image_filenames = dataset_dir1
        self.l = len(listdir(dataset_dir1))
        self.image_filenames2 = dataset_dir2


    def __getitem__(self, index):
        str = "%d.jpg" %index
        lowimage =Image.open(join( self.image_filenames,str))
        labelimage = Image.open(join( self.image_filenames2,str))

        return ToTensor()(lowimage), ToTensor()(labelimage)

    def __len__(self):
        return 10


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_imgg = skimage.io.imread(self.image_filenames[index])
        hr_image = Image.open(self.image_filenames[index])
        if len(hr_imgg.shape) !=3 :
            hr_image = hr_image.convert('RGB')
        w, h = hr_image.size
        crop_size1 = calculate_valid_crop_size(w, 4)
        crop_size2 = calculate_valid_crop_size(h, 4)

        lr_scale = Resize((crop_size2 // 4, crop_size1 // 4), interpolation=Image.BICUBIC)
        hr_scale = Resize((crop_size2, crop_size1), interpolation=Image.BICUBIC)

        hr_image = CenterCrop((crop_size2, crop_size1))(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)

        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder2(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder2, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        lr_noiseimg1 = skimage.io.imread(self.image_filenames[index])
        lr_noiseimg = Image.open(self.image_filenames[index])
        if len(lr_noiseimg1.shape) !=3 :
            lr_noiseimg = lr_noiseimg.convert('RGB')
        w, h = lr_noiseimg.size

        hr_scale = Resize((4*h, 4*w), interpolation=Image.BICUBIC)


        hr_restore_img = hr_scale(lr_noiseimg)

        return ToTensor()(lr_noiseimg), ToTensor()(hr_restore_img)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder3(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder3, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_imgg = skimage.io.imread(self.image_filenames[index])
        hr_image = Image.open(self.image_filenames[index])
        if len(hr_imgg.shape) !=3 :
            hr_image = hr_image.convert('RGB')
        w, h = hr_image.size

        return ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)