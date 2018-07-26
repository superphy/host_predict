import numpy as np
from keras.utils import to_categorical

def remove_symbols(data):
    arry = np.array([], dtype = 'i4')
    for i in data:
        temp = int(float((i.decode('utf-8')).split("=")[-1]))
        # 0=1, 1=2, 2=4, 3=8, 4=16, 5=32
        # CHANGE ALL OF THIS CODE TO AN ENCODER
        if(temp ==1):
            temp = 0
        elif(temp ==2):
            temp = 1
        elif(temp ==4):
            temp = 2
        elif(temp ==8):
            temp = 3
        elif(temp ==16):
            temp = 4
        elif(temp ==32):
            temp = 5
        else:
            print("woahhhhh there buddy, check line 32 ish of neural_net.py cause shit went down")

        arry = np.append(arry,temp)
    return arry

if __name__ == "__main__":
    x_train = np.load('amr_data/AMP/train_data.npy')
    x_test = np.load('amr_data/AMP/test_data.npy')	
    y_train = np.load('amr_data/AMP/train_names.npy')
    y_test = np.load('amr_data/AMP/test_names.npy')

    num_classes = 6
    y_train  = remove_symbols(y_train)
    y_train  = to_categorical(y_train, num_classes)
    y_test  = remove_symbols(y_test)
    y_test  = to_categorical(y_test, num_classes)

    np.save('amr_data/AMP/train_data.npy', x_train)
    np.save('amr_data/AMP/test_data.npy', x_test)	
    np.save('amr_data/AMP/train_names.npy', y_train)
    np.save('amr_data/AMP/test_names.npy', y_test)