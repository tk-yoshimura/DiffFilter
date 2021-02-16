import numpy as np

class DiffFilter():
    def __init__(self):
        self.__kernel_3x3_base = [
            np.array([[1,  0, 0], 
                      [0, -1, 0], 
                      [0,  0, 0]], dtype=np.float32),
            np.array([[0,  1, 0], 
                      [0, -1, 0], 
                      [0,  0, 0]], dtype=np.float32),
        ]

        self.__kernel_5x5_base = [
            np.array([[9,  6,  0,  0, 0], 
                      [6,  3, -3, -1, 0], 
                      [0, -3, -9, -3, 0], 
                      [0, -1, -3, -1, 0], 
                      [0,  0,  0,  0, 0]], dtype=np.float32) / 24,
            np.array([[3, 18,   9,  0, 0], 
                      [2, 10,   0, -2, 0], 
                      [0, -6, -18, -6, 0], 
                      [0, -2,  -6, -2, 0], 
                      [0,  0,   0,  0, 0]], dtype=np.float32) / 42,
            np.array([[0,  3,  9,  3, 0], 
                      [0,  1,  3,  1, 0], 
                      [0, -3, -9, -3, 0], 
                      [0, -1, -3, -1, 0], 
                      [0,  0,  0,  0, 0]], dtype=np.float32) / 20
        ]

        self.__kernel_7x7_base = [
            np.array([[9, 9,  3,  0,  0, 0, 0], 
                      [9, 9,  3,  0,  0, 0, 0], 
                      [3, 3, -3, -6, -4, 0, 0], 
                      [0, 0, -6, -9, -6, 0, 0],
                      [0, 0, -4, -6, -4, 0, 0],
                      [0, 0,  0,  0,  0, 0, 0],
                      [0, 0,  0,  0,  0, 0, 0]], dtype=np.float32) / 48,
            np.array([[6, 27,  27,   3,   0, 0, 0], 
                      [6, 27,  27,   3,   0, 0, 0], 
                      [2,  9,  -3, -17, -12, 0, 0], 
                      [0,  0, -18, -27, -18, 0, 0],
                      [0,  0, -12, -18, -12, 0, 0],
                      [0,  0,   0,   0,   0, 0, 0],
                      [0,  0,   0,   0,   0, 0, 0]], dtype=np.float32) / 137,
            np.array([[0, 12,  27,  24,   0, 0, 0], 
                      [0, 12,  27,  24,   0, 0, 0], 
                      [0,  4,  -3, -10, -12, 0, 0], 
                      [0,  0, -18, -27, -18, 0, 0],
                      [0,  0, -12, -18, -12, 0, 0],
                      [0,  0,   0,   0,   0, 0, 0],
                      [0,  0,   0,   0,   0, 0, 0]], dtype=np.float32) / 130,
            np.array([[0, 0,  6,  9,  6, 0, 0], 
                      [0, 0,  6,  9,  6, 0, 0], 
                      [0, 0, -2, -3, -2, 0, 0], 
                      [0, 0, -6, -9, -6, 0, 0],
                      [0, 0, -4, -6, -4, 0, 0],
                      [0, 0,  0,  0,  0, 0, 0],
                      [0, 0,  0,  0,  0, 0, 0]], dtype=np.float32) / 42,
        ]

        self.__kernel_9x9_base = [
            np.array([[1, 1, 1,  0,  0,  0, 0, 0, 0], 
                      [1, 1, 1,  0,  0,  0, 0, 0, 0], 
                      [1, 1, 1,  0,  0,  0, 0, 0, 0], 
                      [0, 0, 0, -1, -1, -1, 0, 0, 0],
                      [0, 0, 0, -1, -1, -1, 0, 0, 0],
                      [0, 0, 0, -1, -1, -1, 0, 0, 0],
                      [0, 0, 0,  0,  0,  0, 0, 0, 0],
                      [0, 0, 0,  0,  0,  0, 0, 0, 0],
                      [0, 0, 0,  0,  0,  0, 0, 0, 0],], dtype=np.float32) / 9,
            np.array([[0, 1, 2,  2,  1,  0, 0, 0, 0], 
                      [0, 1, 2,  2,  1,  0, 0, 0, 0], 
                      [0, 1, 2,  2,  1,  0, 0, 0, 0], 
                      [0, 0, 0, -2, -2, -2, 0, 0, 0],
                      [0, 0, 0, -2, -2, -2, 0, 0, 0],
                      [0, 0, 0, -2, -2, -2, 0, 0, 0],
                      [0, 0, 0,  0,  0,  0, 0, 0, 0],
                      [0, 0, 0,  0,  0,  0, 0, 0, 0],
                      [0, 0, 0,  0,  0,  0, 0, 0, 0],], dtype=np.float32) / 18,
            np.array([[0, 0, 0,  1,  1,  1, 0, 0, 0], 
                      [0, 0, 0,  1,  1,  1, 0, 0, 0], 
                      [0, 0, 0,  1,  1,  1, 0, 0, 0], 
                      [0, 0, 0, -1, -1, -1, 0, 0, 0],
                      [0, 0, 0, -1, -1, -1, 0, 0, 0],
                      [0, 0, 0, -1, -1, -1, 0, 0, 0],
                      [0, 0, 0,  0,  0,  0, 0, 0, 0],
                      [0, 0, 0,  0,  0,  0, 0, 0, 0],
                      [0, 0, 0,  0,  0,  0, 0, 0, 0],], dtype=np.float32) / 9,
        ]

    @staticmethod
    def symmetric_8(kernel_base):
        assert len(kernel_base) == 2

        kernels = [
            kernel_base[0].copy(),
            kernel_base[1].copy(),
            np.flip(kernel_base[0], axis = 1),
            np.transpose(kernel_base[1]),
            np.flip(np.transpose(kernel_base[1]), axis = 1),
            np.flip(kernel_base[0], axis = 0),
            np.flip(kernel_base[1], axis = 0),
            np.flip(np.flip(kernel_base[0], axis = 1), axis = 0),
        ]

        return kernels

    @staticmethod
    def symmetric_16(kernel_base):
        assert len(kernel_base) == 3

        kernels = [
            kernel_base[0].copy(),
            kernel_base[1].copy(),
            kernel_base[2].copy(),
            np.flip(kernel_base[1], axis = 1),
            np.flip(kernel_base[0], axis = 1),
            np.transpose(kernel_base[1]),
            np.flip(np.transpose(kernel_base[1]), axis = 1),
            np.transpose(kernel_base[2]),
            np.flip(np.transpose(kernel_base[2]), axis = 1),
            np.flip(np.transpose(kernel_base[1]), axis = 0),
            np.flip(np.flip(np.transpose(kernel_base[1]), axis = 1), axis = 0),
            np.flip(kernel_base[0], axis = 0),
            np.flip(kernel_base[1], axis = 0),
            np.flip(kernel_base[2], axis = 0),
            np.flip(np.flip(kernel_base[1], axis = 1), axis = 0),
            np.flip(np.flip(kernel_base[0], axis = 1), axis = 0),
        ]

        return kernels

    @staticmethod
    def symmetric_24(kernel_base):
        assert len(kernel_base) == 4

        kernels = [
            kernel_base[0].copy(),
            kernel_base[1].copy(),
            kernel_base[2].copy(),
            kernel_base[3].copy(),
            np.flip(kernel_base[2], axis = 1),
            np.flip(kernel_base[1], axis = 1),
            np.flip(kernel_base[0], axis = 1),
            np.transpose(kernel_base[1]),
            np.flip(np.transpose(kernel_base[1]), axis = 1),
            np.transpose(kernel_base[2]),
            np.flip(np.transpose(kernel_base[2]), axis = 1),
            np.transpose(kernel_base[3]),
            np.flip(np.transpose(kernel_base[3]), axis = 1),
            np.flip(np.transpose(kernel_base[2]), axis = 0),
            np.flip(np.flip(np.transpose(kernel_base[2]), axis = 1), axis = 0),
            np.flip(np.transpose(kernel_base[1]), axis = 0),
            np.flip(np.flip(np.transpose(kernel_base[1]), axis = 1), axis = 0),
            np.flip(kernel_base[0], axis = 0),
            np.flip(kernel_base[1], axis = 0),
            np.flip(kernel_base[2], axis = 0),
            np.flip(kernel_base[3], axis = 0),
            np.flip(np.flip(kernel_base[2], axis = 1), axis = 0),
            np.flip(np.flip(kernel_base[1], axis = 1), axis = 0),
            np.flip(np.flip(kernel_base[0], axis = 1), axis = 0),
        ]

        return kernels

    @property
    def kernels_3x3(self):
        return DiffFilter.symmetric_8(self.__kernel_3x3_base)

    @property
    def kernels_5x5(self):
        return DiffFilter.symmetric_16(self.__kernel_5x5_base)

    @property
    def kernels_7x7(self):
        return DiffFilter.symmetric_24(self.__kernel_7x7_base)

    @property
    def kernels_9x9(self):
        return DiffFilter.symmetric_16(self.__kernel_9x9_base)