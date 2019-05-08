import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

if sys.version_info.major == 3:
    xrange = range


def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)


def ResnetBlock(x, dim, ksize, scope='rb'):
    with tf.variable_scope(scope):
        net = slim.conv2d(x, dim, [ksize, ksize], scope='conv1')
        net = slim.conv2d(net, dim, [ksize, ksize], activation_fn=None, scope='conv2')
        return net + x

"""------------------------------------------------------------------------------------------------"""
def EBlock(x, dim, ksize, num_resb=3, scope='Eblock'):
    with tf.variable_scope(scope):
        net = slim.conv2d(x, dim, [ksize, ksize], stride=2, scope='conv')
        for i in range(num_resb):
            net = ResnetBlock(net, dim, ksize, scope='rb_'+str(i+1))
        return net

def DBlock(x, dim, ksize, num_resb=3, scope='Dblock'):
    with tf.variable_scope(scope):
        for i in range(num_resb):
            x = ResnetBlock(x, dim, ksize, scope='rb_'+str(i+1))
        x = slim.conv2d_transpose(x, dim//2, [4, 4], stride=2, scope='deconv')
        return x

def OutBlock(x, dim, ksize, num_resb=3, scope='OutBlock'):
    with tf.variable_scope(scope):
        for i in range(num_resb):
            x = ResnetBlock(x, dim, ksize, scope='rb_'+str(i+1))
        x = slim.conv2d(x, 3, [ksize, ksize], activation_fn=None, scope='conv')
        return x

def InBlock(x, dim, ksize, num_resb=3, scope='InBlock'):
    with tf.variable_scope(scope):
        x = slim.conv2d(x, dim, [ksize, ksize], scope='conv')  # 32, 5
        for i in range(num_resb):
            x = ResnetBlock(x, dim, ksize, scope='rb_'+str(i+1))
        return x


def ResidualLinkBlock(x, n_filters, ksize, scope='rlb'):
    """
    Deep Stacked Hierarchical Multi-patch Network for Image Deblurring
    https://arxiv.org/pdf/1904.03468v1.pdf
    """
    with tf.variable_scope(scope):
        net = slim.conv2d(x, n_filters, [ksize, ksize], scope='conv1')
        net = slim.conv2d(net, n_filters, [ksize, ksize], activation_fn=None, scope='conv2')
        return net + x


def AtrousResBlock(x, n_filters, ksize, scope='arb'):
    """
    DAVANet: Stereo Deblurring with View Aggregation
    https://arxiv.org/pdf/1904.05065.pdf
    """
    with tf.variable_scope(scope):
        net = slim.conv2d(x, n_filters, [ksize, ksize], scope='conv1')
        net = slim.conv2d(net, n_filters, [ksize, ksize], rate=4, activation_fn=None, scope='conv2')#rate?
        return net + x

def ContextModule(inputs, depth=256, scope='ContextModule'):
    """
    DAVANet: Stereo Deblurring with View Aggregation
    https://arxiv.org/pdf/1904.05065.pdf
    """
    with tf.variable_scope(scope):
        atrous_pool_block_1 = slim.conv2d(inputs, depth, [3, 3], activation_fn=None)

        atrous_pool_block_2 = slim.conv2d(inputs, depth, [3, 3], rate=2, activation_fn=None)

        atrous_pool_block_3 = slim.conv2d(inputs, depth, [3, 3], rate=3, activation_fn=None)

        atrous_pool_block_4 = slim.conv2d(inputs, depth, [3, 3], rate=4, activation_fn=None)

        net = tf.concat((atrous_pool_block_1, atrous_pool_block_2, atrous_pool_block_3, atrous_pool_block_4), axis=3)
        net = slim.conv2d(inputs, depth, [1, 1], activation_fn=None)

        return net + inputs

"""------------------------------speed up inference. by yewei----------------------------------------"""

def dwsconv(inputs, n_filters, kernel_size=[3, 3], rate=1, stride=1, scope='separable_conv'):

    with tf.variable_scope(scope):
        x = slim.separable_conv2d(inputs, num_outputs=None, depth_multiplier=1, kernel_size=kernel_size, rate=rate, stride=stride)
        x = slim.conv2d(x, n_filters, kernel_size=[1, 1])
        return x

def InvertedResBlock(net, input_filters, output_filters, expansion, stride, scope='InvertedResidualBlock'):

    with tf.variable_scope(scope):
        res_block = net
        res_block = slim.conv2d(inputs=res_block, num_outputs=input_filters * expansion, kernel_size=[1, 1])
        res_block = slim.separable_conv2d(inputs=res_block, num_outputs=None, kernel_size=[3, 3], stride=stride)
        res_block = slim.conv2d(inputs=res_block, num_outputs=output_filters, kernel_size=[1, 1], activation_fn=None)
        if stride == 2:
            return res_block
        else:
            if input_filters != output_filters:
                net = slim.conv2d(inputs=net, num_outputs=output_filters, kernel_size=[1, 1], activation_fn=None)
            return tf.add(res_block, net)

def InBlock_lite(x, dim, ksize, num_resb=3, scope='InBlock_lite'):
    with tf.variable_scope(scope):
        x = slim.conv2d(x, dim, [ksize, ksize], scope='conv1')  # 32, 5
        for i in range(num_resb):
            x = ResnetBlock(x, dim, ksize, scope='rb_'+str(i+1))
        return x



def ContextModule_lite(inputs, depth=256, scope='ContextModule'):
    """
    DAVANet: Stereo Deblurring with View Aggregation
    https://arxiv.org/pdf/1904.05065.pdf
    """
    with tf.variable_scope(scope):

        atrous_pool_block_1 = slim.conv2d(inputs, depth//4, [3, 3], activation_fn=None)

        atrous_pool_block_2 = slim.conv2d(inputs, depth//4, [3, 3], rate=2, activation_fn=None)

        atrous_pool_block_3 = slim.conv2d(inputs, depth//4, [3, 3], rate=3, activation_fn=None)

        atrous_pool_block_4 = slim.conv2d(inputs, depth//4, [3, 3], rate=4, activation_fn=None)

        net = tf.concat((atrous_pool_block_1, atrous_pool_block_2, atrous_pool_block_3, atrous_pool_block_4), axis=3)
        #net = slim.conv2d(inputs, depth, [1, 1], activation_fn=None)

        return net + inputs



def DepthwiseSeparableConvBlock(inputs, n_filters, rate=1, stride=1, scope="dw"):
    """DepthwiseConv"""
    with tf.variable_scope(scope):
        net = slim.separable_conv2d(inputs, num_outputs=None, depth_multiplier=1, kernel_size=[3, 3], rate=rate, stride=stride, normalizer_fn=slim.instance_norm, activation_fn=tf.nn.relu6)
        #net = slim.instance_norm(net)
        #net = tf.nn.relu6(net)
        net = slim.conv2d(net, n_filters, kernel_size=[1, 1], activation_fn=None)
        return net





def ContextModule_dwlite(inputs, depth=256, scope='ContextModule'):

    with tf.variable_scope(scope):
        atrous_pool_block_1 = slim.separable_conv2d(inputs, num_outputs=depth//4, kernel_size=[3,3], depth_multiplier=1, rate=1)#DepthwiseSeparableConvBlock(inputs, depth//4, rate=1, scope='dw1')
        atrous_pool_block_2 = slim.separable_conv2d(inputs, num_outputs=depth//4, kernel_size=[3,3], depth_multiplier=1, rate=2)#DepthwiseSeparableConvBlock(inputs, depth//4, rate=2, scope='dw2')
        atrous_pool_block_3 = slim.separable_conv2d(inputs, num_outputs=depth//4, kernel_size=[3,3], depth_multiplier=1, rate=3)#DepthwiseSeparableConvBlock(inputs, depth//4, rate=4, scope='dw3')
        atrous_pool_block_4 = slim.separable_conv2d(inputs, num_outputs=depth//4, kernel_size=[3,3], depth_multiplier=1, rate=4)#DepthwiseSeparableConvBlock(inputs, depth//4, rate=8, scope='dw4')
        net = tf.concat((atrous_pool_block_1, atrous_pool_block_2, atrous_pool_block_3, atrous_pool_block_4), axis=3)
        return net + inputs

def InvertedResidualBlock(net, n_filters, expansion, stride=1, scope='InvertedResidualBlock'):

    with tf.variable_scope(scope):
        res_block = net
        res_block = slim.conv2d(inputs=res_block, num_outputs=n_filters * expansion, kernel_size=[1, 1], activation_fn=None)
        res_block = tf.nn.leaky_relu(res_block, alpha=0.1)
        res_block = slim.separable_conv2d(inputs=res_block, num_outputs=None, kernel_size=[3, 3], depth_multiplier=1, stride=stride, activation_fn=None)
        res_block = tf.nn.leaky_relu(res_block, alpha=0.1)
        res_block = slim.conv2d(inputs=res_block, num_outputs=n_filters, kernel_size=[1, 1], activation_fn=None)
        if stride == 2:
            return res_block
        else:
            return tf.add(res_block, net)


def InBlock_dw(x, n_filters,  expansion=6, num_resb=3, scope='InBlock'):
    with tf.variable_scope(scope):
        x = slim.conv2d(x, n_filters, [3, 3], scope='conv1')  # 32, 5
        for i in range(num_resb):
            #x = ResnetBlock(x, dim, ksize, scope='rb_'+str(i+1))
            x = InvertedResidualBlock(x, n_filters, expansion, stride=1, scope='irb_'+str(i+1))
        return x

def OutBlock_dw(x, n_filters,  expansion=6, num_resb=3, scope='OutBlock'):
    with tf.variable_scope(scope):
        for i in range(num_resb):
            #x = ResnetBlock(x, dim, ksize, scope='rb_'+str(i+1))
            x = InvertedResidualBlock(x, n_filters, expansion, scope='irb_'+str(i+1))
        x = slim.conv2d(x, 3, [3, 3], activation_fn=None, scope='conv')
        return x

def EBlock_dw(x, n_filters,  expansion=6, num_resb=3, scope='Eblock'):

    with tf.variable_scope(scope):
        #net = slim.conv2d(x, dim, [ksize, ksize], stride=2, scope='conv')
        net = DepthwiseSeparableConvBlock(x, n_filters, stride=2, scope='dw')
        for i in range(num_resb):
            #net = ResnetBlock(net, dim, ksize, scope='rb_' + str(i + 1))
            net = InvertedResidualBlock(net, n_filters, expansion, scope='irb_'+str(i+1))
        return net

def DBlock_dw(x, n_filters,  expansion=6, num_resb=3, scope='Dblock'):
    with tf.variable_scope(scope):
        for i in range(num_resb):
            #x = ResnetBlock(x, dim, ksize, scope='rb_'+str(i+1))
            x = InvertedResidualBlock(x, n_filters, expansion, scope='irb_' + str(i + 1))
        x = slim.conv2d_transpose(x, n_filters//2, [4, 4], stride=2, scope='deconv')
        return x

"""-----------------------------------DAVANet-DeblurNet--------------------------------------------"""

def Conv(inputs, n_filters, ksize=[3,3], rate=1, stride=1, scope="conv"):

    with tf.variable_scope(scope):
        net = slim.conv2d(inputs, n_filters, kernel_size=ksize, rate=rate, stride=stride, scope=scope, activation_fn=None)
        net = tf.nn.leaky_relu(net, 0.1)
        return net

def resnet_block(x, dim, ksize, dilation=[1,1], scope='rb'):

    with tf.variable_scope(scope):
        net = slim.conv2d(x, dim, [ksize, ksize], rate=dilation[0], activation_fn=None, scope='conv1')
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.conv2d(net, dim, [ksize, ksize], rate=dilation[1], activation_fn=None, scope='conv2')
        return net + x

def ms_dilate_block(x, dim, dilation=[1,1,1,1], scope='ms_dilate_block'):

    with tf.variable_scope(scope):

        atrous_pool_block_1 = slim.conv2d(x, dim, [3, 3], rate=dilation[0], activation_fn=None)
        atrous_pool_block_2 = slim.conv2d(x, dim, [3, 3], rate=dilation[1], activation_fn=None)
        atrous_pool_block_3 = slim.conv2d(x, dim, [3, 3], rate=dilation[2], activation_fn=None)
        atrous_pool_block_4 = slim.conv2d(x, dim, [3, 3], rate=dilation[3], activation_fn=None)
        net = tf.concat((atrous_pool_block_1, atrous_pool_block_2, atrous_pool_block_3, atrous_pool_block_4), axis=3)
        net = slim.conv2d(net, dim, [3, 3], activation_fn=None)

        return net + x

def upconv(inputs, n_filters,  scope="upconv"):
    with tf.variable_scope(scope):
        x = slim.conv2d_transpose(inputs, n_filters, [4, 4], stride=2, activation_fn=None,scope='deconv')
        x = tf.nn.leaky_relu(x, alpha=0.1)
        return x