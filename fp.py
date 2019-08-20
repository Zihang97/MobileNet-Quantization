import numpy as np
import time
import threading
import tensorflow as tf
import tvm
from .op import * 
import math

class fp:
    def __init__(self,value,fracLen=2,wordLen=8,signed=True,opt=False,ref=None, roundMethod='round'):
        self._len = wordLen-1 if signed==True else wordLen
        shape = value.shape
        self._d = value.flatten()  
        self._d_prev = self._d      
        self.wl = wordLen
        self.ref_value = ref.flatten() if ref is not None else value.flatten()
        self.mymin = lambda x,y: x if x<y else y
        self.mymax = lambda x,y: x if x>y else y
        if opt==True:
            #fl_opt = self.func(value.flatten())
            fl_opt = self.find_opt_fracLen_by_trial_tf(value)#.flatten())
            #fl_opt = self.find_opt_fracLen_by_range(self._d)
            #print('opt_fracLen:',fl_opt)
            self._fl = fl_opt#fracLen
            self._d_fp = self._d_prev.reshape(shape) 
        else:
            '''#import ipdb
            #ipdb.set_trace()
            self._fl = fracLen
            self._il = self._len - self._fl
            gap = 2**(-self._fl)
            minVal = -2**self._len*gap
            maxSteps = 2**wordLen
            if roundMethod=='floor':
                nSteps = [math.floor(x) for x in (value.flatten()-minVal)/gap]
            else:
                nSteps = [round(x) for x in (value.flatten()-minVal)/gap]
            nSteps = [self.mymax(0,self.mymin(maxSteps,x)) for x in nSteps]
            val_fp = minVal+np.array(nSteps)*gap
            self._d = val_fp
            #err = self.ems(value)
            #print(err)
            self._d_fp = self._d.reshape(shape)'''
            self._fl = fracLen
            if roundMethod=='floor':
                self._d = self.tf_float2fx_floor(value, wordLen, fracLen)
            else:
                self._d = self.tf_float2fx_round(value, wordLen, fracLen)
            self._d_fp = self._d
    
    def ems(self, value):
        errors = 0
        value_f = value.flatten()
        for i in range(len(self._d)):
            errors += value_f[i] - self._d[i]
        #print('e:',errors/len(self._d))
        return errors/len(self._d)
    
    def quantize_multithreads(self, levels, threadNum=4):
        nlen = len(self._d)
        base = nlen//threadNum
        pos = [x*base for x in range(threadNum)]
        if pos[-1]>nlen: pos[-1]=nlen
        elif pos[-1]<nlen: pos.append(nlen)
        levels = np.array(levels)
        _st = time.time()
        '''i=0
        self._d[i] = levels[np.argmin(abs(levels-self._d[i]))]#self._d[i] = min(levels, key=lambda x : abs(x-self._d[i]))
        print('time elapsed:',time.time()-_st)
        import ipdb
        ipdb.set_trace()'''
        thread_pool = []
        for i in range(len(pos)-1):
            t = threading.Thread(target=self.quantize, args=(levels,range(pos[i],pos[i+1]),))
            t.start()
            thread_pool.append(t) 
        for t in thread_pool:
            t.join()
        #print('time elapsed:',time.time()-_st)
        #t1= threading.Thread(target=self.quantize, args=(levels,[0,len(self._d)//2],)) 
        #t2= threading.Thread(target=self.quantize, args=(levels,[len(self._d)//2,len(self._d)-1],)) 
        '''t1= threading.Thread(target=self.quantize, args=(levels,range(0,len(self._d)//2),)) 
        t2= threading.Thread(target=self.quantize, args=(levels,range(len(self._d)//2,len(self._d)),)) 
        t1.start()
        t2.start()
        t1.join()
        t2.join()'''
    
    def func(self, value):
        tn = 4
        self.errs = np.zeros([tn])
        tpool = []
        _st = time.time()
        for fl in range(tn):
            t = threading.Thread(target=self.quant, args=(fl,value))
            t.start()
            tpool.append(t)
        for t in tpool:
            t.join()
        print("finish",time.time()-_st)
        opt = np.argmin(self.errs)
        for i in range(tn):
            print(i,self.errs[i])
        print('[]',opt)
        return opt
    
    def quantize(self, levels, _range):
        #print(_range)
        #_st = time.time()
        for i in _range:
            self._d[i] = levels[np.argmin(abs(levels-self._d[i]))]
            #min(levels[2**self._len:2**(self._len+1)-1], key=lambda x : abs(x-self._d[i])) if self._d[i]>=0 else min(levels[0:2**self._len-1], key=lambda x : abs(x-self._d[i])) 
        #print(_range,"finish",time.time()-_st)
         
    def quant(self, fl, value_flatten):
        _st = time.time()
        num = len(value_flatten)
        gap = 2**(-fl)
        levels = np.array([x * gap for x in range(-2**self._len,2**self._len,1)])
        temp = np.zeros([num])
        _range = range(num)
        for i in _range:
            temp[i] = levels[np.argmin(abs(levels-self._d[i]))]
        self.errs[fl] = abs(np.sum(value_flatten-temp)/num)
        print("singlefinish",time.time()-_st)
        print('\tfl:',fl,'err:',self.errs[fl])     
            
    def find_opt_fracLen_by_range(self, value_f):
        vmin,vmax = np.min(value_f),np.max(value_f)
        il = 0
        while 2**il<vmax or -2**il>vmin:
            il += 1
        return self._len-il
    
    def find_opt_fracLen_by_trial(self, value):
        init_fl = -3
        fl=init_fl
        errs = []
        while True:
            _st = time.time()
            gap = 2**(-fl)
            minVal = -2**self._len*gap
            #nSteps = (value-minVal)//gap
            maxSteps = 2**self.wl
            nSteps = [x for x in (value.flatten()-minVal)/gap]
            nSteps = np.round(nSteps)
            nSteps = [self.mymax(0,self.mymin(maxSteps,x)) for x in nSteps]
            val_fp = minVal+np.array(nSteps)*gap
            self._d_prev = self._d
            self._d = val_fp
            print("finish:",time.time()-_st,len(nSteps))
            err = np.sum((self._d-self.ref_value)**2)/len(self._d)
            print('\tfl:',fl,'err:',err)
            if len(errs)>0 and err>errs[-1] and fl>0:
                least_err_idx = int(np.argmin(np.array(errs)))
                return least_err_idx+init_fl
                #return fl-1
            errs.append(err)
            fl+=1
            self._d = value
        return fl
    
    def find_opt_fracLen_by_trial_tf(self, value):
        init_fl = -14
        fl=init_fl
        errs = []
        while fl<14:#True:
            _st = time.time()
            val_fp = self.tf_float2fx_round(value, self.wl, fl).flatten()
            print("finish:",time.time()-_st,value.size)
            self._d_prev = self._d
            self._d = val_fp
            err = np.sum((self._d-self.ref_value)**2)/len(self._d)
            print('\tfl:',fl,'err:',err)
            if len(errs)>0 and err>errs[-1] and fl>0:
                least_err_idx = int(np.argmin(np.array(errs)))
                return least_err_idx+init_fl
                #return fl-1
            errs.append(err)
            fl+=1
            self._d = value
        return fl
    
    
    def tf_float2fx_round(self, value, wl, fl):
        with tf.Graph().as_default():
            gap = tf.constant(2.0**(-fl))
            minVal = tf.constant(-2.0**(wl-1))*gap
            maxSteps = 2.0**wl
            nSteps = tf.clip_by_value(tf.round(tf.divide((value-minVal),gap)),0.0, maxSteps)
            val_fp = minVal+nSteps*gap
            with block.suppress_stdout_stderr():
                with tf.Session() as sess:
                    val_fp = sess.run(val_fp)
            return val_fp    
        
    def tf_float2fx_floor(self, value, wl, fl):
        with tf.Graph().as_default():
            gap = tf.constant(2.0**(-fl))
            minVal = tf.constant(-2.0**(wl-1))*gap
            maxSteps = 2.0**wl
            nSteps = tf.clip_by_value(tf.math.floordiv((value-minVal),gap),0.0,maxSteps)
            val_fp = minVal+nSteps*gap
            with block.suppress_stdout_stderr():
                with tf.Session() as sess:
                    val_fp = sess.run(val_fp)
            return val_fp
