import os
import argparse
import tensorflow as tf
# import models.model_gray as model
# import models.model_color as model
import models.model as model


def parse_args():
    parser = argparse.ArgumentParser(description='deblur arguments')
    parser.add_argument('--phase', type=str, default='', help='determine whether train or test')
    parser.add_argument('--datalist', type=str, default='./datalist_gopro.txt', help='training datalist')
    parser.add_argument('--datapath', type=str, default='E:\\yewei_workspace\\deblur\\GOPRO_Large\\train\\', help='train datasets path')
    parser.add_argument("--pre_trained_model", type=str, default="./model_dmphn")
    parser.add_argument("--fine_tuning", type=bool, default=True)
    parser.add_argument('--model', type=str, default='color', help='model type: [lstm | gray | color]')
    parser.add_argument('--batch_size', help='training batch size', type=int, default=16)
    parser.add_argument('--epoch', help='training epoch number', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=1e-4, dest='learning_rate', help='initial learning rate')
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0', help='use gpu or cpu')
    parser.add_argument('--net', type=str, default='DAVANet_dw')
    parser.add_argument('--height', type=int, default=720,
                        help='height for the tensorflow placeholder, should be multiples of 16')
    parser.add_argument('--width', type=int, default=1280,
                        help='width for the tensorflow placeholder, should be multiple of 16 for 3 scales')
    parser.add_argument('--n_feat', type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--num_resb', type=int, default=1)

    parser.add_argument('--save_model_period', type=int, default=400, help='epochs to save model')
    parser.add_argument('--input_path', type=str, default='./testing_set',
                        help='input path for testing images')
    parser.add_argument('--output_path', type=str, default='./testing_res',
                        help='output path for testing images')
    parser.add_argument('--vgg_path', type=str, default='./vgg19/vgg19.npy')
    args = parser.parse_args()
    return args


def main(_):
    args = parse_args()

    # set gpu/cpu mode
    if int(args.gpu_id) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # set up deblur models
    deblur = model.DEBLUR(args)
    deblur.para_flops_count()
    if args.phase == 'test':
        deblur.test(args.height, args.width, args.input_path)
    elif args.phase == 'train':
        deblur.train()
    else:
        print('phase should be set to either test or train')


if __name__ == '__main__':
    tf.app.run()