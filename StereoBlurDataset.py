import os
import argparse
import shutil

parser = argparse.ArgumentParser(description='StereoBlurDataset arguments')
parser.add_argument('--datapath', type=str, default='E:\\yewei_workspace\\deblur\\deblur_dataset\\Stereo Blur Dataset\\train\\')
args = parser.parse_args()

class StereoBlurDataset(object):
    def __init__(self, args):
        self.datapath = args.datapath
        self.txtfile = os.path.join(self.datapath, 'train_test_split.txt')
        self.targzfile_train = []
        self.targzfile_test = []
        self.train_filepath = os.path.join(self.datapath, 'train')
        self.test_filepath = os.path.join(self.datapath, 'test')

    #split targz files
    def split_dataset(self):
        label = self.readText(self.txtfile)
        train = []
        val = []
        for i, x in enumerate(label):
            if x == '0\n':
                train.append(i)
            elif x == '1\n':
                val.append(i)
            else:
                print('null')
        targzfile = self.targzfilename(self.datapath)
        for i in train:
            self.targzfile_train.append(os.path.join(self.datapath, targzfile[i]))
        for i in val:
            self.targzfile_test.append(os.path.join(self.datapath, targzfile[i]))
        #return self.targzfile_train, self.targzfile_test

    def move_targz(self):
        #self.train_filepath = os.path.join(self.datapath, 'train')
        self.split_dataset()
        if not os.path.exists(self.train_filepath):
            os.makedirs(self.train_filepath)
        if not os.path.exists(self.test_filepath):
            os.makedirs(self.test_filepath)
        for targz in self.targzfile_train:
            shutil.move(targz, self.train_filepath)
        for targz in self.targzfile_test:
            shutil.move(targz, self.test_filepath)

    def readText(self, full_file_name):
        f = open(full_file_name)
        return f.readlines()

    #get all sharp&blur filepaths
    def sharp_blur_path(self, file_dir):
        #file_name = []
        sharp = []
        blur = []
        for root, dirs, files in os.walk(file_dir):
            if root.split('_')[-1] == 'ga':
                blur.append(root)
            elif root.split('\\')[-1] == 'image_left' or root.split('\\')[-1] == 'image_right':
                sharp.append(root)
        return sharp, blur

    #get targz files in path
    def targzfilename(self, datapath):
        targzfile_list = []
        files = os.listdir(self.datapath)
        for f in files:
            #a = os.path.splitext(f)
            if os.path.splitext(f)[-1] == '.gz':
                targzfile_list.append(f)
        return sorted(targzfile_list)

    #get all sharp&blur filenames
    def get_sharp_blur_images(self):
        #datapath = 'E:\\yewei_workspace\\deblur\\Stereo Blur Dataset\\'
        sharp_paths, blur_paths = self.sharp_blur_path(self.datapath)

        sharp_imgs = []
        blur_imgs = []
        for path in sharp_paths:

            # sharp_imgs += (os.listdir(i) + i)
            for image in os.listdir(path):
                sharp_imgs.append(os.path.join(path, image))

        for path in blur_paths:
            for image in os.listdir(path):
                blur_imgs.append(os.path.join(path, image))

        l = len(sharp_imgs)
        return sharp_imgs, blur_imgs


dataset = StereoBlurDataset(args)
sharp_imgs, blur_imgs = dataset.get_sharp_blur_images()
#targzfile_train, targzfile_test = dataset.split_dataset()
#a = len(targzfile_train)
a = len(sharp_imgs)