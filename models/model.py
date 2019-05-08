from __future__ import print_function
import os
import time
import random
import datetime
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
from util.util import *
from vgg19 import Vgg19
from util.BasicConvLSTMCell import *
#from config import parse_args
#from models.config import parse_args
from skimage.measure import compare_psnr, compare_ssim


class DEBLUR(object):
    def __init__(self, args):
        self.args = args
        self.n_levels = 1#3
        self.scale = 0.5
        self.chns = 3 if self.args.model == 'color' else 1  # input / output channels

        # if args.phase == 'train':
        self.crop_size = 256
        self.data_list = open(args.datalist, 'rt').read().splitlines()
        self.data_list = list(map(lambda x: x.split(' '), self.data_list))
        random.shuffle(self.data_list)
        self.datapath = args.datapath
        self.train_dir = os.path.join('./checkpoints', args.model)
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.data_size = (len(self.data_list)) // self.batch_size
        self.max_steps = int(self.epoch * self.data_size)
        self.learning_rate = args.learning_rate
        self.n_feat = args.n_feat
        self.kernel_size = args.kernel_size
        self.num_resb = args.num_resb
        self.model = args.net
        self.save_period = args.save_model_period
        self.output_path = args.output_path
        self.fine_tuning = args.fine_tuning
        self.pre_trained_model = args.pre_trained_model

    def input_producer(self, batch_size=10):
        def read_data():
            img_a = tf.image.decode_image(tf.read_file(tf.string_join([self.datapath, self.data_queue[0]])),
                                          channels=3)
            img_b = tf.image.decode_image(tf.read_file(tf.string_join([self.datapath, self.data_queue[1]])),
                                          channels=3)
            img_a, img_b = preprocessing([img_a, img_b])
            return img_a, img_b

        def preprocessing(imgs):
            imgs = [tf.cast(img, tf.float32) / 255.0 for img in imgs]
            if self.args.model != 'color':
                imgs = [tf.image.rgb_to_grayscale(img) for img in imgs]
            img_crop = tf.unstack(tf.random_crop(tf.stack(imgs, axis=0), [2, self.crop_size, self.crop_size, self.chns]),
                                  axis=0)
            return img_crop

        with tf.variable_scope('input'):
            List_all = tf.convert_to_tensor(self.data_list, dtype=tf.string)
            gt_list = List_all[:, 0]
            in_list = List_all[:, 1]

            self.data_queue = tf.train.slice_input_producer([in_list, gt_list], capacity=20)
            image_in, image_gt = read_data()
            batch_in, batch_gt = tf.train.batch([image_in, image_gt], batch_size=batch_size, num_threads=8, capacity=20)

        return batch_in, batch_gt

    def generator(self, inputs,  reuse=False):
        n, h, w, c = inputs.get_shape().as_list()
        n_feat = self.n_feat
        kernel_size = self.kernel_size
        scope = self.model

        x_unwrap = []
        if self.args.model == 'lstm':
            with tf.variable_scope('LSTM'):
                cell = BasicConvLSTMCell([h / 4, w / 4], [3, 3], 128)
                rnn_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)


        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                biases_initializer=tf.constant_initializer(0.0)):

                inp_pred = inputs
                if self.model == 'SRN':
                    #x_unwrap = []
                    for i in xrange(self.n_levels):
                        scale = self.scale ** (self.n_levels - i - 1)
                        hi = int(round(h * scale))
                        wi = int(round(w * scale))
                        inp_blur = tf.image.resize_images(inputs, [hi, wi], method=0)
                        inp_pred = tf.stop_gradient(tf.image.resize_images(inp_pred, [hi, wi], method=0))
                        inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')
                        if self.args.model == 'lstm':
                            rnn_state = tf.image.resize_images(rnn_state, [hi // 4, wi // 4], method=0)

                        eb1 = InBlock(inp_all, n_feat, kernel_size, num_resb=self.num_resb, scope='InBlock')

                        eb2 = EBlock(eb1, n_feat*2, kernel_size, num_resb=self.num_resb, scope='eb2')

                        eb3 = EBlock(eb2, n_feat*4, kernel_size, num_resb=self.num_resb, scope='eb3')

                        if self.args.model == 'lstm':
                        #deconv3_4, rnn_state = cell(conv3_4, rnn_state)
                            deconv3_4, rnn_state = cell(eb3, rnn_state)
                        else:
                        #deconv3_4 = conv3_4
                            deconv3_4 = eb3

                        db1 = DBlock(eb3, n_feat*4, kernel_size, scope='db1')
                        cat2 = db1 + eb2

                        db2 = DBlock(cat2, n_feat*2, kernel_size, scope='db2')
                        cat1 = db2 + eb1

                        inp_pred = OutBlock(cat1, n_feat, kernel_size)

                        if i >= 0:
                            x_unwrap.append(inp_pred)
                        if i == 0:
                            tf.get_variable_scope().reuse_variables()

                    return x_unwrap
                elif self.model == 'raw':
                    inp_pred = inputs
                    #x_unwrap = []
                    for i in xrange(self.n_levels):
                        scale = self.scale ** (self.n_levels - i - 1)
                        hi = int(round(h * scale))
                        wi = int(round(w * scale))
                        inp_blur = tf.image.resize_images(inputs, [hi, wi], method=0)
                        inp_pred = tf.stop_gradient(tf.image.resize_images(inp_pred, [hi, wi], method=0))
                        inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')
                        if self.args.model == 'lstm':
                            rnn_state = tf.image.resize_images(rnn_state, [hi // 4, wi // 4], method=0)

                        # encoder
                        conv1_1 = slim.conv2d(inp_all, 32, [5, 5], scope='enc1_1')
                        conv1_2 = ResnetBlock(conv1_1, 32, 5, scope='enc1_2')
                        conv1_3 = ResnetBlock(conv1_2, 32, 5, scope='enc1_3')
                        conv1_4 = ResnetBlock(conv1_3, 32, 5, scope='enc1_4')
                        conv2_1 = slim.conv2d(conv1_4, 64, [5, 5], stride=2, scope='enc2_1')
                        conv2_2 = ResnetBlock(conv2_1, 64, 5, scope='enc2_2')
                        conv2_3 = ResnetBlock(conv2_2, 64, 5, scope='enc2_3')
                        conv2_4 = ResnetBlock(conv2_3, 64, 5, scope='enc2_4')
                        conv3_1 = slim.conv2d(conv2_4, 128, [5, 5], stride=2, scope='enc3_1')
                        conv3_2 = ResnetBlock(conv3_1, 128, 5, scope='enc3_2')
                        conv3_3 = ResnetBlock(conv3_2, 128, 5, scope='enc3_3')
                        conv3_4 = ResnetBlock(conv3_3, 128, 5, scope='enc3_4')

                        if self.args.model == 'lstm':
                            deconv3_4, rnn_state = cell(conv3_4, rnn_state)
                        else:
                            deconv3_4 = conv3_4

                        # decoder
                        deconv3_3 = ResnetBlock(deconv3_4, 128, 5, scope='dec3_3')
                        deconv3_2 = ResnetBlock(deconv3_3, 128, 5, scope='dec3_2')
                        deconv3_1 = ResnetBlock(deconv3_2, 128, 5, scope='dec3_1')
                        deconv2_4 = slim.conv2d_transpose(deconv3_1, 64, [4, 4], stride=2, scope='dec2_4')
                        cat2 = deconv2_4 + conv2_4
                        deconv2_3 = ResnetBlock(cat2, 64, 5, scope='dec2_3')
                        deconv2_2 = ResnetBlock(deconv2_3, 64, 5, scope='dec2_2')
                        deconv2_1 = ResnetBlock(deconv2_2, 64, 5, scope='dec2_1')
                        deconv1_4 = slim.conv2d_transpose(deconv2_1, 32, [4, 4], stride=2, scope='dec1_4')
                        cat1 = deconv1_4 + conv1_4
                        deconv1_3 = ResnetBlock(cat1, 32, 5, scope='dec1_3')
                        deconv1_2 = ResnetBlock(deconv1_3, 32, 5, scope='dec1_2')
                        deconv1_1 = ResnetBlock(deconv1_2, 32, 5, scope='dec1_1')
                        inp_pred = slim.conv2d(deconv1_1, self.chns, [5, 5], activation_fn=None, scope='dec1_0')

                        if i >= 0:
                            x_unwrap.append(inp_pred)
                        if i == 0:
                            tf.get_variable_scope().reuse_variables()

                    return x_unwrap

                elif self.model == 'DAVANet':
                    #x_unwrap = []
                    conv1_1 = Conv(inputs, 8, ksize=[3, 3], scope='conv1_1')
                    conv1_2 = resnet_block(conv1_1, 8, ksize=3, scope='conv1_2')

                    #downsample
                    conv2_1 = Conv(conv1_2, 16, ksize=[3,3], stride=2, scope='conv2_1')
                    conv2_2 = resnet_block(conv2_1, 16, ksize=3, scope='conv2_2')

                    #downsample
                    conv3_1 = Conv(conv2_2, 32, ksize=[3,3], stride=2, scope='conv3_1')
                    conv3_2 = resnet_block(conv3_1, 32, ksize=3, scope='conv3_2')

                    conv4_1 = Conv(conv3_2)


                    dilation = [1, 2, 3, 4]
                    convd_1 = resnet_block(conv3_2, 32, ksize=3, dilation=[2, 1], scope='convd_1')
                    convd_2 = resnet_block(convd_1, 32, ksize=3, dilation=[3, 1], scope='convd_2')
                    convd_3 = ms_dilate_block(convd_2, 32, dilation=dilation, scope='convd_3')

                    #decode
                    upconv3_2 = Conv(convd_3, 32, ksize=[3, 3], scope='upconv3_4')
                    upconv3_1 = resnet_block(upconv3_2, 32, ksize=3, scope='upconv3_3')

                    #upsample
                    upconv2_u = upconv(upconv3_1, 16, scope='upconv2_u')
                    cat1 = tf.concat((upconv2_u, conv2_2), axis=3)
                    upconv2_4 = Conv(cat1, 16, ksize=[3,3], scope='upconv2_4')
                    upconv2_3 = resnet_block(upconv2_4, 16, ksize=3, scope='upconv2_3')

                    #upsample
                    upconv1_u = upconv(upconv2_3, 8, scope='upconv1_u')
                    cat0 = tf.concat((upconv1_u, conv1_2), axis=3)
                    upconv1_2 = Conv(cat0, 8, ksize=[3,3], scope='upconv1_2')
                    upconv1_1 = resnet_block(upconv1_2, 8, ksize=3, scope='upconv1_1')

                    inp_pred = Conv(upconv1_1, 3, ksize=[3,3], scope='output')

                    return x_unwrap.append(inp_pred + inputs)#inp_pred + inputs

                elif self.model == 'unet':

                    conv1_1 = slim.conv2d(inputs, 8, [kernel_size, kernel_size], scope='enc1_1')
                    #conv1_4 = ResnetBlock(conv1_1, 8, kernel_size, scope='enc1_4')
                    conv1_4 = InvertedResidualBlock(conv1_1, 8, expansion=2, scope='enc1_4')


                    #conv2_1 = slim.conv2d(conv1_4, 16, [kernel_size, kernel_size], stride=2, scope='enc2_1')
                    conv2_1 = DepthwiseSeparableConvBlock(conv1_4, 16, stride=2, scope='enc2_1')
                    #conv2_4 = ResnetBlock(conv2_1, 16, kernel_size, scope='enc2_4')
                    conv2_4 = InvertedResidualBlock(conv2_1, 16, expansion=2, scope='enc2_4')


                    #conv3_1 = slim.conv2d(conv2_4, 32, [kernel_size, kernel_size], stride=2, scope='enc3_1')
                    conv3_1 = DepthwiseSeparableConvBlock(conv2_4, 32, stride=2, scope='enc3_1')
                    #conv3_4 = ResnetBlock(conv3_1, 32, kernel_size, scope='enc3_4')
                    conv3_4 = InvertedResidualBlock(conv3_1, 32, expansion=4, scope='enc3_4')


                    #conv4_1 = slim.conv2d(conv3_4, 48, [kernel_size, kernel_size], stride=2, scope='conv4_1')
                    conv4_1 = DepthwiseSeparableConvBlock(conv3_4, 48, stride=2, scope='enc4_1')
                    #conv4_4 = ResnetBlock(conv4_1, 48, kernel_size, scope='conv4_4')
                    conv4_4 = InvertedResidualBlock(conv4_1, 48, expansion=4, scope='enc4_4')

                    #conv5_1 = slim.conv2d(conv4_4, 64, [kernel_size, kernel_size], stride=2, scope='conv5_1')
                    #conv5_4 = ResnetBlock(conv5_1, 64, kernel_size, scope='conv5_4')
                    conv5_1 = DepthwiseSeparableConvBlock(conv4_4, 64, stride=2, scope='enc5_1')
                    conv5_4 = InvertedResidualBlock(conv5_1, 64, expansion=4, scope='enc5_4')

                    deconv5_4 = conv5_4
                    #                # decoder
                    #deconv5_3 = InvertedResidualBlock(deconv5_4, 64, expansion=4, scope='deconv5_3')
                    deconv5_0 = slim.conv2d_transpose(deconv5_4, 48, [4, 4], stride=2, scope='deconv5_0')
                    cat4 = deconv5_0 + conv4_4
                    deconv4_3 = InvertedResidualBlock(cat4, 48, expansion=4, scope='deconv4_3')
                    deconv4_0 = slim.conv2d_transpose(deconv4_3, 32, [4, 4], stride=2, scope='deconv4_0')
                    cat3 = deconv4_0 + conv3_4
                    deconv3_3 = InvertedResidualBlock(cat3, 32, expansion=4, scope='deconv3_3')
                    deconv3_0 = slim.conv2d_transpose(deconv3_3, 16, [4, 4], stride=2, scope='deconv3_0')
                    cat2 = deconv3_0 + conv2_4
                    deconv2_3 = InvertedResidualBlock(cat2, 16, expansion=2, scope='deconv2_3')
                    deconv2_0 = slim.conv2d_transpose(deconv2_3, 8, [4, 4], stride=2, scope='deconv2_0')
                    cat1 = deconv2_0 + conv1_4
                    deconv1_3 = InvertedResidualBlock(cat1, 8, expansion=2, scope='dec1_3')
                    inp_pred = slim.conv2d(deconv1_3, 3, [kernel_size, kernel_size], activation_fn=slim.nn.sigmoid, scope='output')
                    return x_unwrap.append(inp_pred)


                elif self.model == 'DMPHN':
                    #x_unwrap = []
                    net = slim.conv2d(inputs, n_feat, [3, 3], activation_fn=None, scope='ec_conv1')
                    net = ResidualLinkBlock(net, n_feat, ksize=3, scope='ec_rlb1')
                    net = ResidualLinkBlock(net, n_feat, ksize=3, scope='ec_rlb2')

                    net = slim.conv2d(net, n_feat*2, [3, 3], stride=2, activation_fn=None, scope='ec_conv2')
                    net = ResidualLinkBlock(net, n_feat*2, ksize=3, scope='ec_rlb3')
                    net = ResidualLinkBlock(net, n_feat*2, ksize=3, scope='ec_rlb4')

                    net = slim.conv2d(net, n_feat*4, [3, 3], stride=2, activation_fn=None, scope='ec_conv3')
                    net = ResidualLinkBlock(net, n_feat*4, ksize=3, scope='ec_rlb5')
                    net = ResidualLinkBlock(net, n_feat*4, ksize=3, scope='ec_rlb6')

                    net = ResidualLinkBlock(net, n_feat * 4, ksize=3, scope='dc_rlb1')
                    net = ResidualLinkBlock(net, n_feat * 4, ksize=3, scope='dc_rlb2')

                    net = slim.conv2d_transpose(net, n_feat*2, [4, 4], stride=2, activation_fn=None, scope='dc_deconv1')
                    net = ResidualLinkBlock(net, n_feat*2, ksize=3, scope='dc_rlb3')
                    net = ResidualLinkBlock(net, n_feat*2, ksize=3, scope='dc_flb4')

                    net = slim.conv2d_transpose(net, n_feat, [4, 4], stride=2, activation_fn=None, scope='dc_deconv2')
                    net = ResidualLinkBlock(net, n_feat, ksize=3, scope='dc_rlb5')
                    net = ResidualLinkBlock(net, n_feat, ksize=3, scope='dc_flb6')

                    net = slim.conv2d(net, 3, [3, 3], activation_fn=None, scope='dc_conv1')

                    return x_unwrap.append(net)#net

                elif self.model == 'DAVANet_light':
                    eb1 = InBlock(inputs, n_feat, kernel_size, num_resb=1, scope='InBlock')

                    eb2 = EBlock(eb1, n_feat * 2, kernel_size, num_resb=1, scope='eb1')

                    eb3 = EBlock(eb2, n_feat * 4, kernel_size, num_resb=1, scope='eb2')

                    context = ContextModule_lite(eb3, n_feat * 4)

                    db1 = DBlock(context, n_feat * 4, kernel_size, num_resb=1, scope='db1')
                    cat2 = db1 + eb2

                    db2 = DBlock(cat2, n_feat * 2, kernel_size, num_resb=1, scope='db2')
                    cat1 = db2 + eb1

                    inp_pred = OutBlock(cat1, n_feat, kernel_size, num_resb=1, scope='OutBlock')

                    return x_unwrap.append(inp_pred + inputs)

                elif self.model == 'DAVANet_dw':

                    eb1 = InBlock_dw(inputs, n_feat, num_resb=self.num_resb, expansion=2, scope='InBlock')

                    eb2 = EBlock_dw(eb1, n_feat * 2, num_resb=self.num_resb, expansion=4, scope='eb1')

                    eb3 = EBlock_dw(eb2, n_feat * 4, num_resb=self.num_resb, expansion=4, scope='eb2')

                    context = ContextModule_dwlite(eb3, n_feat * 4)

                    db1 = DBlock_dw(context, n_feat * 4, num_resb=self.num_resb, expansion=4, scope='db1')
                    cat2 = db1 + eb2

                    db2 = DBlock_dw(cat2, n_feat * 2, num_resb=self.num_resb, expansion=4, scope='db2')
                    cat1 = db2 + eb1

                    inp_pred = OutBlock_dw(cat1, n_feat, num_resb=self.num_resb, expansion=2, scope='OutBlock')

                    return x_unwrap.append(inp_pred + inputs)

                elif self.model == 'DFANet':
                    conv1 = slim.conv2d(inputs, 8, kernel_size=[3, 3], stride=2, scope='conv1')



                elif self.model == 'Deblur_lite':
                    conv1 = slim.conv2d(inputs, 8, [3, 3], scope='conv1')




    def build_model(self):
        img_in, img_gt = self.input_producer(self.batch_size)

        #tf.summary.image('img_in', im2uint8(img_in))
        #tf.summary.image('img_gt', im2uint8(img_gt))
        print('img_in, img_gt', img_in.get_shape(), img_gt.get_shape())

        # generator
        x_unwrap = self.generator(img_in, reuse=False, scope='g_net')
        # calculate multi-scale loss
        self.loss_total = 0
        for i in xrange(self.n_levels):
            _, hi, wi, _ = x_unwrap[i].get_shape().as_list()
            gt_i = tf.image.resize_images(img_gt, [hi, wi], method=0)
            loss = tf.reduce_mean((gt_i - x_unwrap[i]) ** 2)#MSE loss
            #perceptual_loss
            vgg_net = Vgg19(self.vgg_path)
            vgg_net.build(tf.concat([gt_i, x_unwrap[i]], axis=0))
            perceptual_loss = tf.reduce_mean(tf.reduce_sum(tf.square(vgg_net.relu3_3[self.batch_size:] - vgg_net.relu3_3[:self.batch_size]), axis=3))
            self.loss_total += (loss + perceptual_loss*0.01)

            tf.summary.image('out_' + str(i), im2uint8(x_unwrap[i]))
            tf.summary.scalar('loss_' + str(i), loss)

        # losses
        tf.summary.scalar('loss_total', self.loss_total)


        self.PSNR = tf.reduce_mean(tf.image.psnr(((x_unwrap[0] + 1.0) / 2.0), ((img_gt + 1.0) / 2.0), max_val=1.0))
        self.ssim = tf.reduce_mean(tf.image.ssim(((x_unwrap[0] + 1.0) / 2.0), ((img_gt + 1.0) / 2.0), max_val=1.0))
        tf.summary.scalar(name='PSNR', tensor=self.PSNR)
        tf.summary.scalar(name='SSIM', tensor=self.ssim)

        self.output = (x_unwrap[0] + 1.0) * 255.0 / 2.0
        self.output = tf.round(self.output)
        self.output = tf.cast(self.output, tf.uint8)

        # training vars
        all_vars = tf.trainable_variables()
        self.all_vars = all_vars
        self.g_vars = [var for var in all_vars if 'g_net' in var.name]
        self.lstm_vars = [var for var in all_vars if 'LSTM' in var.name]
        for var in all_vars:
            print(var.name)

    def train(self):
        def get_optimizer(loss, global_step=None, var_list=None, is_gradient_clip=False):
            train_op = tf.train.AdamOptimizer(self.lr)
            if is_gradient_clip:
                grads_and_vars = train_op.compute_gradients(loss, var_list=var_list)
                unchanged_gvs = [(grad, var) for grad, var in grads_and_vars if not 'LSTM' in var.name]
                rnn_grad = [grad for grad, var in grads_and_vars if 'LSTM' in var.name]
                rnn_var = [var for grad, var in grads_and_vars if 'LSTM' in var.name]
                capped_grad, _ = tf.clip_by_global_norm(rnn_grad, clip_norm=3)
                capped_gvs = list(zip(capped_grad, rnn_var))
                train_op = train_op.apply_gradients(grads_and_vars=capped_gvs + unchanged_gvs, global_step=global_step)
            else:
                train_op = train_op.minimize(loss, global_step, var_list)
            return train_op

        global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
        self.global_step = global_step

        # build model
        self.build_model()

        # learning rate decay
        self.lr = tf.train.polynomial_decay(self.learning_rate, global_step, self.max_steps, end_learning_rate=0.0,
                                            power=0.3)
        tf.summary.scalar('learning_rate', self.lr)

        # training operators
        train_gnet = get_optimizer(self.loss_total, global_step, self.all_vars)

        # session and thread
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = sess
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        if self.fine_tuning:
            checkpoints_name = 'checkpoints_' + self.model + '_' + str(self.n_feat) + '_' + str(
                self.kernel_size) + '_' + str(self.num_resb)
            checkpoint_path = os.path.join(self.train_dir, checkpoints_name)
            self.saver.restore(sess, checkpoint_path)
            print("saved model is loaded for fine-tuning!")
            print("model path is %s" % (checkpoint_path))
        # training summary
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph, flush_secs=30)

        for step in xrange(sess.run(global_step), self.max_steps + 1):

            start_time = time.time()

            # update G network
            _, loss_total_val = sess.run([train_gnet, self.loss_total])

            duration = time.time() - start_time
            # print loss_value
            assert not np.isnan(loss_total_val), 'Model diverged with loss = NaN'

            if step % 5 == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = (%.5f; %.5f, %.5f)(%.1f data/s; %.3f s/bch)')
                print(format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, loss_total_val, 0.0,
                                    0.0, examples_per_sec, sec_per_batch))

            if step % 20 == 0:
                # summary_str = sess.run(summary_op, feed_dict={inputs:batch_input, gt:batch_gt})
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

            # Save the model checkpoint periodically.
            if step % (self.data_size * self.save_period) == 0 or step == self.max_steps:
                checkpoints_name = 'checkpoints_' + self.model + '_' + str(self.n_feat) + '_' + str(self.kernel_size) + '_' + str(self.num_resb)
                checkpoint_path = os.path.join(self.train_dir, checkpoints_name)
                self.save(sess, checkpoint_path, step)

    def save(self, sess, checkpoint_dir, step):
        model_name = "deblur.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None):
        print(" [*] Reading checkpoints...")
        model_name = "deblur.model"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if step is not None:
            ckpt_name = model_name + '-' + str(step)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading intermediate checkpoints... Success")
            return str(step)
        elif ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_iter = ckpt_name.split('-')[1]
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading updated checkpoints... Success")
            return ckpt_iter
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False

    def val(self, height, width, input_path, label_path, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        imgsName = sorted(os.listdir(input_path))

    def test(self, height, width, input_path):
        output_path = self.output_path + '_' + self.model + '_' + str(self.n_feat) + '_' + str(self.kernel_size) + '_' + str(self.num_resb)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        imgsName = sorted(os.listdir(input_path))

        H, W = height, width
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3
        inputs = tf.placeholder(shape=[self.batch_size, H, W, inp_chns], dtype=tf.float32)
        outputs = self.generator(inputs, reuse=False)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.saver = tf.train.Saver()
        self.load(sess, self.train_dir, step=523000)

        for imgName in imgsName:
            blur = scipy.misc.imread(os.path.join(input_path, imgName))
            h, w, c = blur.shape
            # make sure the width is larger than the height
            rot = False
            if h > w:
                blur = np.transpose(blur, [1, 0, 2])
                rot = True
            h = int(blur.shape[0])
            w = int(blur.shape[1])
            resize = False
            if h > H or w > W:
                scale = min(1.0 * H / h, 1.0 * W / w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                blur = scipy.misc.imresize(blur, [new_h, new_w], 'bicubic')
                resize = True
                blurPad = np.pad(blur, ((0, H - new_h), (0, W - new_w), (0, 0)), 'edge')
            else:
                blurPad = np.pad(blur, ((0, H - h), (0, W - w), (0, 0)), 'edge')
            blurPad = np.expand_dims(blurPad, 0)
            if self.args.model != 'color':
                blurPad = np.transpose(blurPad, (3, 1, 2, 0))

            start = time.time()
            deblur = sess.run(outputs, feed_dict={inputs: blurPad / 255.0})
            duration = time.time() - start


            print('Saving results: %s ... %4.3fs' % (os.path.join(output_path, imgName), duration))
            res = deblur[-1]
            if self.args.model != 'color':
                res = np.transpose(res, (3, 1, 2, 0))
            res = im2uint8(res[0, :, :, :])
            # crop the image into original size
            if resize:
                res = res[:new_h, :new_w, :]
                res = scipy.misc.imresize(res, [h, w], 'bicubic')
            else:
                res = res[:h, :w, :]

            if rot:
                res = np.transpose(res, [1, 0, 2])
            scipy.misc.imsave(os.path.join(output_path, imgName), res)


    def para_flops_count(self):

        def em(sess, run_meta):
            """
            https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/profile_model_architecture.md
            """

            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='scope', options=opts)

            opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
            params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='scope', options=opts)

            print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))

        run_meta = tf.RunMetadata()
        with tf.Session(graph=tf.Graph()) as sess:
            input = tf.placeholder('float32', shape=(1, 1280, 720, 3))
            # net = model.generator(x=input)
            #net = self.model(input)
            net = self.generator(input)
            em(sess, run_meta)


#args = parse_args()
#deblur = DEBLUR(args)
#deblur.para_flops_count()