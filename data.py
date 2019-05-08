import os

input_path = 'E:\\yewei_workspace\\deblur\\GOPRO_Large\\train\\GOPR0372_07_00\\blur'
imgsName = sorted(os.listdir(input_path))#文件名，非全路径
num = len(imgsName)