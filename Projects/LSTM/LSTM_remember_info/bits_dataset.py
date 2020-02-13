import numpy as np
from saveload import save_obj, load_obj

class BitsDataset:
    @staticmethod
    def get_dataset(size,num=4):
        val_arr = np.random.randint(0,2**(num-1),size=(size,1,2), dtype=np.uint8)
        sum_arr = val_arr[:,:,0]+val_arr[:,:,1]
        bits_x_arr = np.unpackbits(val_arr, axis=1)
        bits_x_arr = bits_x_arr[:,-num:,:]
        bits_x_arr = np.flip(bits_x_arr,axis=1)
        bits_y_arr = np.unpackbits(sum_arr, axis=1)
        bits_y_arr = bits_y_arr[:,-num:]
        bits_y_arr = np.flip(bits_y_arr,axis=1)

        def flip_bits(x):
            if x > 0.5:
                return 0
            else:
                return 1

        vec_flip = np.vectorize(flip_bits)
        bits_y_arr_flip = vec_flip(bits_y_arr)
        bits_y_arr = np.stack((bits_y_arr_flip, bits_y_arr), axis=2)
        return bits_x_arr, bits_y_arr


if __name__ == '__main__':

    train_X, train_y = BitsDataset.get_dataset(256,4)

    print(train_X.shape, train_y.shape)
    save_obj(train_X, '../data/bits/train_X')
    save_obj(train_y, '../data/bits/train_y')

    test_X, test_y = BitsDataset.get_dataset(256,4)
    save_obj(test_X, '../data/bits/test_X')
    save_obj(test_y, '../data/bits/test_y')
    i = 0
    print(test_X[i:i+1])
    print(test_y[i:i+1,:,-1:])


