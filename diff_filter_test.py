import numpy as np
import cv2

from diff_filter import DiffFilter

df = DiffFilter()

np.set_printoptions(precision=5, linewidth=256)

def sum_minmax_test():
    print('test sum minmax')
    
    for kernels in [df.kernels_3x3, df.kernels_5x5, df.kernels_7x7, df.kernels_9x9]:
        for kernel in kernels:
            print(kernel)
            print(np.sum(kernel))
            print(np.sum(np.maximum(0, kernel)))
            print(np.sum(np.minimum(0, kernel)))
            print('\n')

            assert np.abs(np.sum(kernel)) < 1e-6
            assert np.abs(np.sum(np.maximum(0, kernel)) - 1) < 1e-6
            assert np.abs(np.sum(np.minimum(0, kernel)) + 1) < 1e-6

def error_test():
    print('test error')

    for i in range(2):
        x = np.zeros((3, 3))
        x[0, i] = +1
        x[1, 1] = -1
        
        print(x)
        print(df.kernels_3x3[i])
        print('\n')

        assert np.sum(np.abs(df.kernels_3x3[i] - x)) < 1e-6

    for i in range(3):
        x = np.zeros((6, 6))
        x[0:2, i:i+2] = +1
        x[2:4, 2:4] = -1
        x = cv2.resize(x, (5, 5), interpolation=cv2.INTER_AREA)
        x /= np.sum(np.maximum(0, x))

        print(x)
        print(df.kernels_5x5[i])
        print('\n')

        assert np.sum(np.abs(df.kernels_5x5[i] - x)) < 1e-6

    for i in range(4):
        x = np.zeros((9, 9))
        x[0:3, i:i+3] = +1
        x[3:6, 3:6] = -1
        x = cv2.resize(x, (7, 7), interpolation=cv2.INTER_AREA)
        x /= np.sum(np.maximum(0, x))

        print(x)
        print(df.kernels_7x7[i])
        print('\n')

        assert np.sum(np.abs(df.kernels_7x7[i] - x)) < 1e-6

def image_test():
    print('test image')

    imgnames = ['test_white', 'test_grey', 'test_black']

    for imgname in imgnames:
        img_src = cv2.imread('imgs/src/%s.png' % imgname, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        for ksize, kernels in zip(['3x3', '5x5', '7x7', '9x9'], [df.kernels_3x3, df.kernels_5x5, df.kernels_7x7, df.kernels_9x9]):
            for i, kernel in enumerate(kernels):
                img_dst = cv2.filter2D(img_src, ddepth = -1, kernel=kernel, borderType = cv2.BORDER_REPLICATE)

                cv2.imwrite('imgs/dst/%s_%s_%d.png' % (imgname, ksize, i), img_dst)

def duplicate_test():
    print('test duplicate')

    for kernels in [df.kernels_3x3, df.kernels_5x5, df.kernels_7x7, df.kernels_9x9]:
        for i, kernel_a in enumerate(kernels):
            for kernel_b in kernels[i+1:]:

                print(kernel_a)
                print(kernel_b)
                print('\n')

                assert np.sum(np.abs(kernel_a - kernel_b)) > 1e-6

if __name__ == '__main__':
    sum_minmax_test()
    error_test()
    image_test()
    duplicate_test()

    print('all tests passed')