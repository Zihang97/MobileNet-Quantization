import numpy as np
import time
import threading
import tensorflow as tf
from fp import fp as fixpoint
from .. import suppress_stdout_stderr as block
import scipy.io
import os
from .global_var import global_var as gv
#basedir = '/home/tiandong/tvm/example/tests/vgg19/output/'
#if os.path.exists(basedir+'fracLen.txt'): os.remove(basedir+'fracLen.txt')
global_var = gv()

def dump2file(filename, string):
    with open(filename, 'a') as f:
        print(string, file=f)
    name = string.split('=')[0].split('_')[-1]
    fl = string.split('=')[1]
    if name=='fm':
        global_var.fracLenDict['fm'].append(int(fl))
    elif name=='weight':
        global_var.fracLenDict['weight'].append(int(fl))
    elif name=='bias':
        global_var.fracLenDict['bias'].append(int(fl))

def image_resize(target, ifm, size, method):
    method_dict = {'BILINEAR':0,'NEAREST_NEIGHBOR':1,'BICUBIC':2,'AREA':3}
    out = tf.image.resize_images(images=ifm, size=size, method=method_dict[method])
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out        

def strided_slice(target, ifm, begin, end, axis=3):
    out = tf.strided_slice(ifm, begin, end)
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out        
        
def upsampling(target, ifm, scale, method='nearest'):
    #import ipdb
    #ipdb.set_trace()
    if ifm.shape[1]==ifm.shape[2]:
        size = [x*scale for x in ifm.shape[1:3]]
        out = tf.image.resize_images(ifm, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False, preserve_aspect_ratio=False)
        #out = tf.transpose(out,perm=[0,3,1,2])
    else:
        size = [x*scale for x in ifm.shape[2:]]
        ifm = tf.transpose(ifm,perm=[0,2,3,1])
        out = tf.image.resize_images(ifm, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False, preserve_aspect_ratio=False)
        #out = tf.transpose(out,perm=[0,3,1,2])
    if target=='sw':
        return out
    elif target=='opu':
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
        return out

def dense(target, ifm, w, layerId, basedir):
    if target=='sw':
        out = tf.matmul(ifm.astype(np.float32),w.transpose(1,0))
        return out
    elif target=='opu':
        weight_fp = fixpoint.fp(w, wordLen=8,opt=True)
        weight_fp = fixpoint.fp(w, fracLen=weight_fp._fl)
        weight = weight_fp._d_fp
        print('<>kernel_fracLen=',weight_fp._fl)
        scipy.io.savemat(basedir+'weight_'+str(layerId)+'.mat',{'value':weight})
        dump2file(basedir+'fracLen.txt',str(layerId)+'_weight='+str(weight_fp._fl))
        if not weight.dtype==ifm.dtype:
            weight = weight.astype(ifm.dtype)
        out = tf.matmul(ifm,weight.transpose(1,0))
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
        return out.astype(np.float32)
        
def reshape(target, ifm, out_shape):
    out = tf.reshape(ifm,out_shape)
    if target=='opu':
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out

def mean(target, ifm, out_shape):
    mean_dims = []
    for i in range(len(ifm.shape)):
        if not ifm.shape[i]==out_shape[i]:
            mean_dims.append(i)
    out = tf.reduce_mean(ifm, mean_dims, keepdims=True)
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out

def residualAdd(target, operand_0, operand_1):
    out = tf.add(operand_0, operand_1)
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out
        
def multiply(target, operand_0, operand_1):
    out = tf.multiply(operand_0, operand_1)
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out

def subtract(target, operand_0, operand_1):
    out = tf.subtract(operand_0, operand_1)
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out
    
def transpose(ifm, out_shape):
    out = ifm.transpose(0,3,1,2)
    assert out.shape==tuple(out_shape)
    return out

def tfpad(ifm, pad_width):
    paddings = tf.constant([[pad_width[i][0].value, pad_width[i][1].value] for i in range(len(pad_width))])
    ofm = tf.pad(ifm, paddings, 'CONSTANT')
    with block.suppress_stdout_stderr():
        with tf.Session() as sess:
            ofm = sess.run(ofm)
    return ofm

def relu(target, ifm):
    if target=="sw":
        return tf.nn.relu(ifm)
    elif target=="opu":
        out = tf.nn.relu(ifm)
        with block.suppress_stdout_stderr():
             with tf.Session() as sess:
                 out = sess.run(out)
        return out

def tanh(target, ifm):
    if target=="sw":
        return tf.nn.tanh(ifm)
    elif target=="opu":
        out = tf.nn.tanh(ifm)
        with block.suppress_stdout_stderr():
             with tf.Session() as sess:
                 out = sess.run(out)
        return out        
        
def clip(target, ifm, min, max):
    if target=="sw":
        return tf.clip_by_value(ifm, min, max)
    elif target=="opu":
        out = tf.clip_by_value(ifm, min, max)
        with block.suppress_stdout_stderr():
             with tf.Session() as sess:
                 out = sess.run(out)
        return out        
        
        
def concat(target, ifms):
    if target=="sw":
        return tf.concat(ifms,3)
    elif target=="opu":
        out = tf.concat(ifms,3)
        with block.suppress_stdout_stderr():
             with tf.Session() as sess:
                 out = sess.run(out)
        return out

def conv2d(target, ifm, weight, strides, padding, data_format, kernel_format, groups, inpFracLen, layerId, cutposLen, basedir):
    if groups is not None and groups>1:
        return conv2d_depthwise(target, ifm, weight, strides, padding, data_format, groups, inpFracLen, layerId, cutposLen, basedir)
    if data_format=='NCHW':# ->NHWC
        if target=='opu':
            ifm = ifm.transpose(0,2,3,1)
            #weight = weight.transpose(0,2,3,1)
        elif target=='sw':
            ifm = tf.transpose(ifm, perm=[0,2,3,1])
            #weight = tf.transpose(weight, perm=[0,2,3,1])
    if kernel_format=='OIHW':# ->HWIO
        if target=='opu':
            weight = weight.transpose(2,3,1,0)
        elif target=='sw':
            weight = tf.transpose(weight, perm=[2,3,1,0])
    elif kernel_format=='OHWI':
        if target=='opu':
            weight = weight.transpose(1,2,3,0)
        elif target=='sw':
            weight = tf.transpose(weight, perm=[1,2,3,0])
    if target=="opu":
        out = conv2d_opu(ifm, weight, strides[1], padding, inpFracLen, layerId, cutposLen, basedir)
    elif target=="sw":
        out = tf.nn.conv2d(ifm, weight, strides=strides, padding=padding)
    else:
        assert 0,"unknown target"
    '''if data_format=='NCHW':
        if target=='opu':
            out = out.transpose(0,3,1,2)
        elif target=='sw':
            out = tf.transpose(out,perm=[0,3,1,2])'''
    return out

def conv2d_depthwise(target, ifm, weight, strides, padding, data_format, groups, inpFracLen, layerId, cutposLen, basedir):
    if target=='sw':
        out = tf.nn.depthwise_conv2d(input=ifm, filter=weight, strides=strides, padding=padding, data_format=data_format)
    elif target=='opu':
        out = conv2d_depthwise_opu(ifm, weight, strides[1], padding, inpFracLen, layerId, cutposLen, basedir)
    else:
        assert 0,"unknown target"
    return out

def conv2d_depthwise_opu(ifm, w, stride, padding, inpFracLen, layerId, cutposLen, basedir):
    weight_fp = fixpoint.fp(w, wordLen=8,opt=True)
    weight_fp = fixpoint.fp(w, fracLen=weight_fp._fl)
    weight = weight_fp._d_fp
    print('<>kernel_fracLen=',weight_fp._fl)
    scipy.io.savemat(basedir+'weight_'+str(layerId)+'.mat',{'value':weight})
    dump2file(basedir+'fracLen.txt',str(layerId)+'_weight='+str(weight_fp._fl))
    prod_fracLen = weight_fp._fl+inpFracLen
    cutposLen = 15 - (1+7-cutposLen) 
    print('<>*',prod_fracLen,'->',cutposLen)
    #
    global_var.fracLenDict['cutposLen'].append(cutposLen)      
    fm_size = [int(ifm.shape[1]),int(ifm.shape[2])]
    depth = 1
    ker_size = [int(w.shape[0]),int(w.shape[1])]
    ker_num = int(w.shape[2]) # different from conv2d
    if padding=='SAME':
        ofm_size = [x//stride for x in fm_size]
    else:
        temp = [fm_size[0]-ker_size[0]+1,fm_size[1]-ker_size[1]+1]
        ofm_size = [(x+stride-1)//stride for x in temp]
    ofm = np.zeros([1,int(ofm_size[0]), int(ofm_size[1]), ker_num])
    with tf.Session() as sess:
        for kx in range(ker_size[0]):
            for ky in range(ker_size[1]):
                tmp = tf.nn.depthwise_conv2d(ifm[:,kx:int(fm_size[0])-ker_size[0]+kx+1,ky:int(fm_size[1])-ker_size[1]+ky+1,:],weight[kx:kx+1,ky:ky+1,:,:],strides=[1,stride,stride,1],padding='SAME')
                tmp = sess.run(tmp)
                ofm+=tmp
                ofm = fixpoint.fp(ofm,wordLen=16,fracLen=cutposLen, roundMethod='floor')._d_fp
    return ofm
    
    
def biasAdd(target, ifm, bias, layerId, basedir):
    if target=="sw":
        return tf.nn.bias_add(ifm, bias)
    elif target=="opu":
        return biasAdd_opu(ifm, bias, layerId, basedir)     
        

def leakyRelu(target, ifm, alpha):
    if target=="sw":
        return tf.nn.leaky_relu(ifm, alpha)
    elif target=="opu":
        out = tf.nn.leaky_relu(ifm, alpha)
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
        return out

def maxPool(target, ifm, kz, strides, pad_mode, data_format):
    if target=="sw":
        return tf.nn.max_pool(ifm, ksize=kz, strides=strides, padding=pad_mode, data_format=data_format)
    elif target=="opu":
        out = tf.nn.max_pool(ifm, ksize=kz, strides=strides, padding=pad_mode, data_format=data_format)
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
        return out

def avgPool(target, ifm, kz, strides, pad_mode, data_format):
    if target=="sw":
        return tf.nn.avg_pool(ifm, ksize=kz, strides=strides, padding=pad_mode, data_format=data_format)
    elif target=="opu":
        out = tf.nn.avg_pool(ifm, ksize=kz, strides=strides, padding=pad_mode, data_format=data_format)
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
        return out

def biasAdd_opu(fm, bias, layerId, basedir):
    bias_fp = fixpoint.fp(bias,wordLen=16,opt=True)
    bias_fp = fixpoint.fp(bias,wordLen=16,fracLen=bias_fp._fl)
    b0 = bias
    bias = bias_fp._d_fp
    print('<>bias_fracLen=',bias_fp._fl)
    #import ipdb
    #ipdb.set_trace()
    scipy.io.savemat(basedir+'bias_'+str(layerId)+'.mat',{'value':bias})
    dump2file(basedir+'fracLen.txt',str(layerId)+'_bias='+str(bias_fp._fl))
    out = tf.nn.bias_add(fm, bias)
    with block.suppress_stdout_stderr():
        with tf.Session() as sess:
            out = sess.run(out)
    #out = fixpoint.fp(out,wordLen=16,opt=True)._d_fp
    return out
    

def conv2d_opu(fm, w, stride, padding, inpFracLen, layerId, cutposLen, basedir):
    assert fm.shape[3] == w.shape[2]
    fm_size = [int(fm.shape[1]),int(fm.shape[2])]
    depth = int(fm.shape[3])
    ker_size = [int(w.shape[0]),int(w.shape[1])]
    ker_num = int(w.shape[3])
    if padding=='SAME':
        ofm_size = [x//stride for x in fm_size]
    else:
        temp = [fm_size[0]-ker_size[0]+1,fm_size[1]-ker_size[1]+1]
        ofm_size = [(x+stride-1)//stride for x in temp]
    # compute padding size
    pad_num = [int(max(stride*(ofm_size[0]-1)+ker_size[0]-fm_size[0],0)),int(max(stride*(ofm_size[1]-1)+ker_size[1]-fm_size[1],0))]
    pad_size_0 = [int(x//2) for x in pad_num]
    pad_size_1 = [int(pad_num[0] - pad_size_0[0]),int(pad_num[1] - pad_size_0[1])]
    # padding 0s
    ifm_size = [int(pad_num[0]+fm_size[0]),int(pad_num[1]+fm_size[1])]
    ifm = np.ndarray([1,ifm_size[0],ifm_size[1],depth])
    for d in range(depth):
        for i in range(ifm_size[0]):
            for j in range(pad_size_0[1]):
                ifm[0][i][j][d] = 0
        for i in range(pad_size_0[0]):
            for j in range(ifm_size[1]):
                ifm[0][i][j][d] = 0
        for i in range(fm_size[0]):
            for j in range(fm_size[1]):
                ifm[0][pad_size_0[0]+i][pad_size_0[1]+j][d]=fm[0][i][j][d]
        for i in range(pad_size_0[0]+fm_size[0], ifm_size[0]):
            for j in range(ifm_size[1]):
                ifm[0][i][j][d] = 0
        for i in range(ifm_size[0]):
            for j in range(pad_size_0[1]+fm_size[1], ifm_size[1]):
                ifm[0][i][j][d] = 0
    ofm = np.zeros([1,int(ofm_size[0]), int(ofm_size[1]), ker_num])
    ofm0 = np.zeros([1,int(ofm_size[0]), int(ofm_size[1]), ker_num])
    weight_fp = fixpoint.fp(w, wordLen=8,opt=True)
    weight_fp = fixpoint.fp(w, fracLen=weight_fp._fl)
    weight = weight_fp._d_fp
    print('<>kernel_fracLen=',weight_fp._fl)
    import ipdb
    #ipdb.set_trace()
    scipy.io.savemat(basedir+'weight_'+str(layerId)+'.mat',{'value':weight})
    dump2file(basedir+'fracLen.txt',str(layerId)+'_weight='+str(weight_fp._fl))
    prod_fracLen = weight_fp._fl+inpFracLen
    cutposLen = 15 - (1+7-cutposLen) 
    print('<>*',prod_fracLen,'->',cutposLen)
    #cutposLen = min(prod_fracLen, cutposLen)
    fl_local = [cutposLen]
    
    '''global_var.fracLenDict['cutposLen'].append(cutposLen) 
    ofm = tf.nn.conv2d(ifm, weight, strides=[1,stride,stride,1], padding=padding)
    with tf.Session() as sess:
        ofm = sess.run(ofm)
    return ofm'''
    
    with tf.Session() as sess:
        for kx in range(ker_size[0]):
            for ky in range(ker_size[1]):
                tmp = tf.nn.conv2d(ifm[:,kx:int(ifm_size[0])-ker_size[0]+kx+1,ky:int(ifm_size[1])-ker_size[1]+ky+1,:],weight[kx:kx+1,ky:ky+1,:,:],strides=[1,stride,stride,1],padding='SAME')
                tmp = sess.run(tmp)
                fl_t = frange(16, tmp.flatten())
                fl_local.append(fl_t)
    cutposLen = min(fl_local)     
    global_var.fracLenDict['cutposLen'].append(cutposLen)       
    #with block.suppress_stdout_stderr():
    with tf.Session() as sess:
        for kx in range(ker_size[0]):
            for ky in range(ker_size[1]):
                #import ipdb
                #ipdb.set_trace()
                tmp = tf.nn.conv2d(ifm[:,kx:int(ifm_size[0])-ker_size[0]+kx+1,ky:int(ifm_size[1])-ker_size[1]+ky+1,:],weight[kx:kx+1,ky:ky+1,:,:],strides=[1,stride,stride,1],padding='SAME')
                tmp = sess.run(tmp)
                ofm+=tmp
                ofm0 = ofm
                ofm = fixpoint.fp(ofm,wordLen=16,fracLen=cutposLen, roundMethod='floor')._d_fp
                '''if layerId==1:
                    print(np.max(ofm),np.max(ofm0))
                    import ipdb
                    ipdb.set_trace()
                    print()'''
    #import ipdb
    #ipdb.set_trace()
    return ofm
    
def frange(wl,value_f):
    vmin,vmax = np.min(value_f),np.max(value_f)
    il = 0
    while 2**il<vmax or -2**il>vmin:
        il += 1
    return wl-il