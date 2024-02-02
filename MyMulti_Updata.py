from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Input, add
from keras.models import Model
from keras.layers import Input, Dense, Add
from keras import optimizers
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import time
import warnings

warnings.filterwarnings("ignore")

# The LUCAS topsoil dataset URL：http://esdac.jrc.ec.europa.eu/
# Python version：3.7.3
# keras version：2.3.1

# Create a new .txt file to save accuracy and other information
start = time.time()
txt = open('ans1.24_my.txt', 'w')
print(time.strftime(f"%Y-%m-%d %H:%M:%S"), file=txt)

# Read data
x_num = 4200  # Number of wavelengths
y_max = 4208  # Maximum number of columns required to read
pH = 4203  # The number of columns where the pH property is located
OC = 4204  # The number of columns where the OC property is located
CaCO3 = 4205  # The number of columns where the CaCO3 property is located
N = 4206  # The number of columns where the N property is located
P = 4207  # The number of columns where the P property is located
K = 4208  # The number of columns where the K property is located
CEC = 4209  # The number of columns where the CEC property is located

df = pd.read_csv(r'Data_SG_50_1_SNV.csv')  # Read data that has been randomly shuffled

df = df.sample(frac=1.0, random_state=7)  #shuffle
print(df)
# data = df.iloc[:, 1:y_max].values
data = df.iloc[:, 0:y_max].values
print(data.shape)
print(data)
print(data[0, 0])     # spc 400
print(data[0, 4199])  # spc 2499.5
print(data[0, 4207])  # cec 7.2
pH_max = np.amax(data[:, pH - 2])
OC_max = np.amax(data[:, OC - 2])
CaCO3_max = np.amax(data[:, CaCO3 - 2])
N_max = np.amax(data[:, N - 2])
P_max = np.amax(data[:, P - 2])
K_max = np.amax(data[:, K - 2])
CEC_max = np.amax(data[:, CEC - 2])
print(f"pH_max:{pH_max}", file=txt)
print(f"OC_max:{OC_max}", file=txt)
print(f"CaCO3_max:{CaCO3_max}", file=txt)
print(f"N_max:{N_max}", file=txt)
print(f"P_max:{P_max}", file=txt)
print(f"K_max:{K_max}", file=txt)
print(f"CEC_max:{CEC_max}", file=txt)

# Divide training set and testing set
ratio = int(0.75 * len(data))
train = data[:ratio]
test = data[ratio:]
train_x = train[:, :x_num]
train_x = train_x.reshape(train_x.shape[0], 1, x_num, 1)
# print("输入数据有问题吗")  # 没问题
# print(train_x[0, 0, 0, 0])
# print(train_x[0, 0, 4199, 0])
test_x = test[:, :x_num]
test_x = test_x.reshape(test_x.shape[0], 1, x_num, 1)
print(train_x.shape, file=txt)
print(test_x.shape, file=txt)

# Normalized soil properties values
train_pH = train[:, pH - 2]
train_pH_normalization = np.divide(train_pH, pH_max)
train_OC = train[:, OC - 2]
train_OC_normalization = np.divide(train_OC, OC_max)
train_CaCO3 = train[:, CaCO3 - 2]
train_CaCO3_normalization = np.divide(train_CaCO3, CaCO3_max)
train_N = train[:, N - 2]
train_N_normalization = np.divide(train_N, N_max)
train_P = train[:, P - 2]
train_P_normalization = np.divide(train_P, P_max)
train_K = train[:, K - 2]
train_K_normalization = np.divide(train_K, K_max)
train_CEC = train[:, CEC - 2]
train_CEC_normalization = np.divide(train_CEC, CEC_max)

test_pH = test[:, pH - 2]
test_pH_normalization = np.divide(test_pH, pH_max)
test_OC = test[:, OC - 2]
test_OC_normalization = np.divide(test_OC, OC_max)
test_CaCO3 = test[:, CaCO3 - 2]
test_CaCO3_normalization = np.divide(test_CaCO3, CaCO3_max)
test_N = test[:, N - 2]
test_N_normalization = np.divide(test_N, N_max)
test_P = test[:, P - 2]
test_P_normalization = np.divide(test_P, P_max)
test_K = test[:, K - 2]
test_K_normalization = np.divide(test_K, K_max)
test_CEC = test[:, CEC - 2]
test_CEC_normalization = np.divide(test_CEC, CEC_max)

input_shape = (1, x_num, 1)  # input_shape = (1, 4200, 1)


# Build model
def Conv2d_BN1(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        conv_name = name + '_conv'
    else:
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='tanh', name=conv_name)(x)
    return x


def Conv2d_BN2(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        conv_name = name + '_conv'
    else:
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='tanh', name=conv_name)(x)
    return x


def Conv2d_BN3(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        conv_name = name + '_conv'
    else:
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='tanh', name=conv_name)(x)
    return x


# ResidualBlock
def Conv_Block1(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN1(inpt, nb_filter=nb_filter[0], kernel_size=(1, 1), strides=strides, padding='same')
    x = Conv2d_BN1(x, nb_filter=nb_filter[1], kernel_size=(1, 3), padding='same')
    x = Conv2d_BN1(x, nb_filter=nb_filter[2], kernel_size=(1, 1), padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN1(inpt, nb_filter=nb_filter[2], strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


# ResidualBlock
def Conv_Block2(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN2(inpt, nb_filter=nb_filter[0], kernel_size=(1, 1), strides=strides, padding='same')
    x = Conv2d_BN2(x, nb_filter=nb_filter[1], kernel_size=(1, 3), padding='same')
    x = Conv2d_BN2(x, nb_filter=nb_filter[2], kernel_size=(1, 1), padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN2(inpt, nb_filter=nb_filter[2], strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


# ResidualBlock
def Conv_Block3(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN3(inpt, nb_filter=nb_filter[0], kernel_size=(1, 1), strides=strides, padding='same')
    x = Conv2d_BN3(x, nb_filter=nb_filter[1], kernel_size=(1, 3), padding='same')
    x = Conv2d_BN3(x, nb_filter=nb_filter[2], kernel_size=(1, 1), padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN3(inpt, nb_filter=nb_filter[2], strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


# def multi_LucasResNet_16():
# inpt = Input(shape=input_shape)
# # shared layer PH CaCO3 K(OC-N 0.92 , N-CEC 0.59, OC-CEC 0.48 )
# x = Conv2d_BN1(inpt, nb_filter=6, kernel_size=(1, 7), strides=(1, 2), padding='same')
# x = MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='same')(x)
#
# x = Conv_Block1(x, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1), with_conv_shortcut=True)
# x = Conv_Block1(x, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1))
#
# x = Conv_Block1(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 2), with_conv_shortcut=True)
# x = Conv_Block1(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 1))
#
# x = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(x)
# x_share = Flatten()(x)
#
# x_pH = Dense(200, activation='tanh')(x_share)
# x_pH = Dropout(0.3)(x_pH)
# x_pH = Dense(100, activation='tanh')(x_pH)
# x_pH = Dropout(0.3)(x_pH)
# x_pH = Dense(1, activation='sigmoid', name='pH_output')(x_pH)
#
# x_CaCO3 = Dense(200, activation='tanh')(x_share)
# x_CaCO3 = Dropout(0.3)(x_CaCO3)
# x_CaCO3 = Dense(100, activation='tanh')(x_CaCO3)
# x_CaCO3 = Dropout(0.3)(x_CaCO3)
# x_CaCO3 = Dense(1, activation='sigmoid', name='CaCO3_output')(x_CaCO3)
#
# # share layer OC N CEC
# x = Conv2d_BN2(inpt, nb_filter=6, kernel_size=(1, 7), strides=(1, 2), padding='same')
# x = MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='same')(x)
#
# x = Conv_Block2(x, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1), with_conv_shortcut=True)
# x = Conv_Block2(x, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1))
#
# x = Conv_Block2(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 2), with_conv_shortcut=True)
# x = Conv_Block2(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 1))
#
# x = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(x)
# x_share_ONC = Flatten()(x)
#
# x_OC = Dense(200, activation='tanh')(x_share_ONC)
# x_OC = Dropout(0.3)(x_OC)
# x_OC = Dense(100, activation='tanh')(x_OC)
# x_OC = Dropout(0.3)(x_OC)
# x_OC = Dense(1, activation='sigmoid', name='OC_output')(x_OC)
#
# x_N = Dense(200, activation='tanh')(x_share_ONC)
# x_N = Dropout(0.3)(x_N)
# x_N = Dense(100, activation='tanh')(x_N)
# x_N = Dropout(0.3)(x_N)
# x_N = Dense(1, activation='sigmoid', name='N_output')(x_N)
#
# x_CEC = Dense(200, activation='tanh')(x_share_ONC)
# x_CEC = Dropout(0.3)(x_CEC)
# x_CEC = Dense(100, activation='tanh')(x_CEC)
# x_CEC = Dropout(0.3)(x_CEC)
# x_CEC = Dense(1, activation='sigmoid', name='CEC_output')(x_CEC)
#
# # share layer PK
# x = Conv2d_BN3(inpt, nb_filter=6, kernel_size=(1, 7), strides=(1, 2), padding='same')
# x = MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='same')(x)
#
# x = Conv_Block3(x, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1), with_conv_shortcut=True)
# x = Conv_Block3(x, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1))
#
# x = Conv_Block3(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 2), with_conv_shortcut=True)
# x = Conv_Block3(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 1))
#
# x = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(x)
# x_share_PK = Flatten()(x)
#
# x_K = Dense(200, activation='tanh')(x_share_PK)
# x_K = Dropout(0.3)(x_K)
# x_K = Dense(100, activation='tanh')(x_K)
# x_K = Dropout(0.3)(x_K)
# x_K = Dense(1, activation='sigmoid', name='K_output')(x_K)
#
# x_P = Dense(200, activation='tanh')(x_share_PK)
# x_P = Dropout(0.3)(x_P)
# x_P = Dense(100, activation='tanh')(x_P)
# x_P = Dropout(0.3)(x_P)
# x_P = Dense(1, activation='sigmoid', name='P_output')(x_P)
#
# model = Model(inputs=inpt, outputs=[x_pH, x_OC, x_CaCO3, x_N, x_P, x_K, x_CEC])
# return model


def PS_MTL_Lucas():
    input = Input(shape=input_shape)


    # layer ph
    x = Conv2d_BN2(input, nb_filter=6, kernel_size=(1, 7), strides=(1, 2), padding='same')
    pool_ph = MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='same')(x)

    x = Conv_Block2(pool_ph, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1), with_conv_shortcut=True)
    x = Conv_Block2(x, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1))

    x = Conv_Block2(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 2), with_conv_shortcut=True)
    x_ph = Conv_Block2(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 1))

    x = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(x_ph)
    x_share_ph = Flatten()(x)

    x_pH = Dense(200, activation='tanh')(x_share_ph)
    x_pH = Dropout(0.3)(x_pH)
    x_pH = Dense(100, activation='tanh')(x_pH)
    x_pH = Dropout(0.3)(x_pH)
    x_pH = Dense(1, activation='sigmoid', name='pH_output')(x_pH)


    # layer CaCO3
    x = Conv2d_BN3(input, nb_filter=6, kernel_size=(1, 7), strides=(1, 2), padding='same')
    pool_ca = MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='same')(x)

    x = Conv_Block3(pool_ca, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1), with_conv_shortcut=True)
    x = Conv_Block3(x, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1))

    x = Conv_Block3(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 2), with_conv_shortcut=True)
    x_ca = Conv_Block3(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 1))

    x = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(x_ca)
    x_share_CaCO3 = Flatten()(x)

    x_CaCO3 = Dense(200, activation='tanh')(x_share_CaCO3)
    x_CaCO3 = Dropout(0.3)(x_CaCO3)
    x_CaCO3 = Dense(100, activation='tanh')(x_CaCO3)
    x_CaCO3 = Dropout(0.3)(x_CaCO3)
    x_CaCO3 = Dense(1, activation='sigmoid', name='CaCO3_output')(x_CaCO3)


    # layer OC
    x = Conv2d_BN2(input, nb_filter=6, kernel_size=(1, 7), strides=(1, 2), padding='same')
    pool_oc = MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='same')(x)

    x = Conv_Block2(pool_oc, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1), with_conv_shortcut=True)
    x = Conv_Block2(x, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1))

    x = Conv_Block2(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 2), with_conv_shortcut=True)
    x_oc = Conv_Block2(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 1))

    x = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(x_oc)
    x_share_OC = Flatten()(x)

    x_OC = Dense(200, activation='tanh')(x_share_OC)
    x_OC = Dropout(0.3)(x_OC)
    x_OC = Dense(100, activation='tanh')(x_OC)
    x_OC = Dropout(0.3)(x_OC)
    x_OC = Dense(1, activation='sigmoid', name='OC_output')(x_OC)


    # layer N
    x = Conv2d_BN2(input, nb_filter=6, kernel_size=(1, 7), strides=(1, 2), padding='same')
    pool_n = MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='same')(x)

    x = Conv_Block2(pool_n, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1), with_conv_shortcut=True)
    x = Conv_Block2(x, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1))

    x = Conv_Block2(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 2), with_conv_shortcut=True)
    x_n = Conv_Block2(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 1))

    x = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(x_n)
    x_share_N = Flatten()(x)

    x_N = Dense(200, activation='tanh')(x_share_N)
    x_N = Dropout(0.3)(x_N)
    x_N = Dense(100, activation='tanh')(x_N)
    x_N = Dropout(0.3)(x_N)
    x_N = Dense(1, activation='sigmoid', name='N_output')(x_N)


    # layer CEC
    x = Conv2d_BN2(input, nb_filter=6, kernel_size=(1, 7), strides=(1, 2), padding='same')
    pool_cec = MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='same')(x)

    x = Conv_Block2(pool_cec, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1), with_conv_shortcut=True)
    x = Conv_Block2(x, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1))

    x = Conv_Block2(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 2), with_conv_shortcut=True)
    x_cec = Conv_Block2(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 1))

    x = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(x_cec)
    x_share_CEC = Flatten()(x)

    x_CEC = Dense(200, activation='tanh')(x_share_CEC)
    x_CEC = Dropout(0.3)(x_CEC)
    x_CEC = Dense(100, activation='tanh')(x_CEC)
    x_CEC = Dropout(0.3)(x_CEC)
    x_CEC = Dense(1, activation='sigmoid', name='CEC_output')(x_CEC)


    # layer P
    x = Conv2d_BN2(input, nb_filter=6, kernel_size=(1, 7), strides=(1, 2), padding='same')
    pool_p = MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='same')(x)

    x = Conv_Block2(pool_p, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1), with_conv_shortcut=True)
    x = Conv_Block2(x, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1))

    x = Conv_Block2(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 2), with_conv_shortcut=True)
    x_p = Conv_Block2(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 1))

    x = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(x_p)
    x_share_P = Flatten()(x)

    x_P = Dense(200, activation='tanh')(x_share_P)
    x_P = Dropout(0.3)(x_P)
    x_P = Dense(100, activation='tanh')(x_P)
    x_P = Dropout(0.3)(x_P)
    x_P = Dense(1, activation='sigmoid', name='P_output')(x_P)


    # layer k
    x = Conv2d_BN3(input, nb_filter=6, kernel_size=(1, 7), strides=(1, 2), padding='same')
    pool_k = MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='same')(x)

    x = Conv_Block3(pool_k, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1), with_conv_shortcut=True)
    x = Conv_Block3(x, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1))

    x = Conv_Block3(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 2), with_conv_shortcut=True)
    x_k = Conv_Block3(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 1))

    x = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(x_k)
    x_share_K = Flatten()(x)

    x_K = Dense(200, activation='tanh')(x_share_K)
    x_K = Dropout(0.3)(x_K)
    x_K = Dense(100, activation='tanh')(x_K)
    x_K = Dropout(0.3)(x_K)
    x_K = Dense(1, activation='sigmoid', name='K_output')(x_K)


    # shared layer PH CaCO3 OC N CEC(OC-N 0.92 , N-CEC 0.59, OC-CEC 0.48 )
    x = Conv2d_BN1(input, nb_filter=6, kernel_size=(1, 7), strides=(1, 2), padding='same')
    x = MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='same')(x)

    # ADD
    combined = Add()([x,pool_k, pool_ph, pool_cec, pool_p, pool_n, pool_oc, pool_ca])

    x = Conv_Block1(combined, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1), with_conv_shortcut=True)
    x = Conv_Block1(x, nb_filter=[6, 6, 12], kernel_size=(1, 3), strides=(1, 1))

    x = Conv_Block1(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 2), with_conv_shortcut=True)
    x = Conv_Block1(x, nb_filter=[12, 12, 24], kernel_size=(1, 3), strides=(1, 1))

    combined_x = Add()([x, x_k, x_ph, x_cec, x_p, x_n, x_oc, x_ca])

    x = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(combined_x)
    x_share = Flatten()(x)


    model = Model(inputs=input, outputs=[x_pH, x_OC, x_CaCO3, x_N, x_P, x_K, x_CEC])
    return model


model = PS_MTL_Lucas()
model.summary()

# Setting parameters, compile and fit
nadam = optimizers.Nadam(lr=0.0001, epsilon=1e-08)
early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
model.compile(optimizer=nadam,
              loss={'pH_output': 'mse', 'OC_output': 'mse', 'CaCO3_output': 'mse', 'N_output': 'mse', 'P_output': 'mse',
                    'K_output': 'mse', 'CEC_output': 'mse'})
hist = model.fit(train_x,
                 [train_pH_normalization, train_OC_normalization, train_CaCO3_normalization, train_N_normalization,
                  train_P_normalization, train_K_normalization, train_CEC_normalization],
                 epochs=1000, batch_size=32, validation_split=0.2, verbose=2, shuffle=True, callbacks=[early_stopping])


# predict and save

model = load_model('multi_LucasResNet_16.h5')

predict = model.predict(test_x)
predict_values = np.squeeze(predict)
predict_pH = predict_values[0] * pH_max
predict_OC = predict_values[1] * OC_max
predict_CaCO3 = predict_values[2] * CaCO3_max
predict_N = predict_values[3] * N_max
predict_P = predict_values[4] * P_max
predict_K = predict_values[5] * K_max
predict_CEC = predict_values[6] * CEC_max

# 输出预测
predict_all = np.vstack((test_pH,predict_pH, test_OC,predict_OC, test_CaCO3,predict_CaCO3, test_N,predict_N,
                         test_N,predict_P, test_K,predict_K,test_CEC, predict_CEC))
predict_all_transpose = np.transpose(predict_all)
pre = pd.DataFrame(data=predict_all_transpose)
pre.to_csv('predict_values1.24.csv')


# Define R2, RMSE, RPD
def R2(measured_values, predicted_values):
    SS_res = sum((measured_values - predicted_values) ** 2)
    SS_tot = sum((measured_values - np.mean(measured_values)) ** 2)
    return 1 - SS_res / SS_tot


def RMSE(measured_values, predicted_values):
    return (np.mean((predicted_values - measured_values) ** 2)) ** 0.5


def RPD(measured_values, predicted_values):
    return ((np.mean((measured_values - np.mean(measured_values)) ** 2)) ** 0.5) / (
            (np.mean((predicted_values - measured_values) ** 2)) ** 0.5)


# Test accuracy
R2_pH = R2(test_pH, predict_pH)
RMSE_pH = RMSE(test_pH, predict_pH)
RPD_pH = RPD(test_pH, predict_pH)
print(f"Test_R2_pH:{R2_pH}", file=txt)
print(f"Test_RMSE_pH:{RMSE_pH}", file=txt)
print(f"Test_RPD_pH:{RPD_pH}", file=txt)

R2_OC = R2(test_OC, predict_OC)
RMSE_OC = RMSE(test_OC, predict_OC)
RPD_OC = RPD(test_OC, predict_OC)
print(f"Test_R2_OC:{R2_OC}", file=txt)
print(f"Test_RMSE_OC:{RMSE_OC}", file=txt)
print(f"Test_RPD_OC:{RPD_OC}", file=txt)

R2_CaCO3 = R2(test_CaCO3, predict_CaCO3)
RMSE_CaCO3 = RMSE(test_CaCO3, predict_CaCO3)
RPD_CaCO3 = RPD(test_CaCO3, predict_CaCO3)
print(f"Test_R2_CaCO3:{R2_CaCO3}", file=txt)
print(f"Test_RMSE_CaCO3:{RMSE_CaCO3}", file=txt)
print(f"Test_RPD_CaCO3:{RPD_CaCO3}", file=txt)

R2_N = R2(test_N, predict_N)
RMSE_N = RMSE(test_N, predict_N)
RPD_N = RPD(test_N, predict_N)
print(f"Test_R2_N:{R2_N}", file=txt)
print(f"Test_RMSE_N:{RMSE_N}", file=txt)
print(f"Test_RPD_N:{RPD_N}", file=txt)

R2_P = R2(test_P, predict_P)
RMSE_P = RMSE(test_P, predict_P)
RPD_P = RPD(test_P, predict_P)
print(f"Test_R2_P:{R2_P}", file=txt)
print(f"Test_RMSE_P:{RMSE_P}", file=txt)
print(f"Test_RPD_P:{RPD_P}", file=txt)

R2_K = R2(test_K, predict_K)
RMSE_K = RMSE(test_K, predict_K)
RPD_K = RPD(test_K, predict_K)
print(f"Test_R2_K:{R2_K}", file=txt)
print(f"Test_RMSE_K:{RMSE_K}", file=txt)
print(f"Test_RPD_K:{RPD_K}", file=txt)

R2_CEC = R2(test_CEC, predict_CEC)
RMSE_CEC = RMSE(test_CEC, predict_CEC)
RPD_CEC = RPD(test_CEC, predict_CEC)
print(f"Test_R2_CEC:{R2_CEC}", file=txt)
print(f"Test_RMSE_CEC:{RMSE_CEC}", file=txt)
print(f"Test_RPD_CEC:{RPD_CEC}", file=txt)



# Graph of loss change during training
plt.figure()
plt.plot(np.arange(0, len(hist.history["loss"])), hist.history["loss"], label="Calibration")
plt.plot(np.arange(0, len(hist.history["loss"])), hist.history["val_loss"], label="Validation")
plt.title("Calibration/Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.savefig("multi_LucasResNet_16_loss.jpg", dpi=300)


# Save the model
model.save('multi_LucasResNet_16_1.24.h5')


# Running time
end = time.time()
print(time.strftime(f"%Y-%m-%d %H:%M:%S"), file=txt)
print("Running time:%.2fseconds" % (end - start), file=txt)
txt.close()
