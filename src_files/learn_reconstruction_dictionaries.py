from builtins import input
import argparse

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np
import os, tqdm

from sporco.dictlrn import prlcnscdl
from sporco import util
from sporco import signal
from sporco import plot
from PIL import Image
from sporco.cupy import (cupy_enabled, np2cp, cp2np, select_device_by_load, gpu_info)
from sporco.cupy.dictlrn import onlinecdl
from multiprocessing import Pool

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Reconstruction Dictionaries Creation.')
    parser.add_argument('--gpu', dest='gpu', help='GPU device id to use', nargs='+', default=[0, 1], type=int)
    parser.add_argument('--trainning_data_dir', dest='trainning_data_dir', help='Directory path for trainning data.', default='../datasets/nsfw/train_balanced/', type=str)
    parser.add_argument('--save_path', dest='dictionary_path', help='Directory path to save dictionaries.', default='../clean_dictionaries/', type=str)
    args = parser.parse_args()
    return args

def func(x):
    npd = 16
    fltlmbd = 0
    return signal.tikhonov_filter(x, fltlmbd, npd)

def load_images(path):
    all_images = []
    for folder in sorted(os.listdir(path)):
        images = []
        full_path = os.path.join(path, folder)
        img_list = os.listdir(full_path)
        for img in tqdm.tqdm(img_list):
            img_path = os.path.join(full_path, img)
            images.append(np.array(Image.open(img_path).convert('RGB').resize((224,224))))
        all_images.append(images)
    return all_images

def main(args):

    path_folder_SMCC = args.dictionary_path
    number_clusters_per_class = 1
    n_classes = 2
    class_images = load_images(args.trainning_data_dir)
    idx = 0
    for imgs_class_i in class_images:

        npd = 16
        fltlmbd = 0
        print('Applying Tikhonov Filters for class:', idx)
        p = Pool(32)
        a = p.map(func, imgs_class_i)
        sl = []
        sh = []
        for x in a:
            if x[1].ndim==3:
                sh.append(x[1].reshape((3,224,224)))
            if x[0].ndim==3:
                sl.append(x[0].reshape((3,224,224)))
        sl = np.array(sl).reshape((224,224,3,-1))
        sh = np.array(sh).reshape((224,224,3,-1))
        print("SL: ", sl.shape)
        np.random.seed(12345)
        D0 = np.random.randn(8, 8, 3, 64)
        lmbda = 0.2
        print('Calculating Dictionary for Cluster:', idx)
        opt = onlinecdl.OnlineConvBPDNDictLearn.Options({
                    'Verbose': True, 'ZeroMean': False, 'eta_a': 10.0,
                    'eta_b': 20.0, 'DataType': np.float32,
                    'CBPDN': {'rho': 5.0, 'AutoRho': {'Enabled': True},
                        'RelaxParam': 1.8, 'RelStopTol': 1e-7, 'MaxMainIter': 1500,
                        'FastSolve': False, 'DataType': np.float32}})
        if not cupy_enabled():
            print('CuPy/GPU device not available: running without GPU acceleration\n')
        else:
            id = select_device_by_load()
            info = gpu_info()
            if info:
                print('Running on GPU %d (%s)\n' % (id, info[id].name))
        d = onlinecdl.OnlineConvBPDNDictLearn(np2cp(D0), lmbda, opt)
        iter = 2500
        d.display_start()
        for it in range(iter):
            img_index = np.random.randint(0, sh.shape[-1])
            d.solve(np2cp(sh[..., [img_index]]))

        d.display_end()
        D1 = cp2np(d.getdict())
        print("OnlineConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))
        path = path_folder_SMCC+ 'dictionary_class_'+str(idx)+'.npy'
        idx = idx+1
        with open(path, 'wb') as f:
            np.save(f, D1)
        print('Saved dictionary to:', path)

if __name__ == '__main__':
    args = parse_args()
    main(args)