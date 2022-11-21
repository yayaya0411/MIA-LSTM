# import tensorflow.keras.layer impot Layer as tf
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from utility.attention import CustomAttention
from utility.MultiHeadAttention import MultiHeadAttention
from utility.transformer import Time2Vector,TransformerEncoder
import numpy as np

# batch_size = 32

"""
Original Multi Input Network
"""
def MultiInput(input_x):
    # define sets of inputs
    
    target = np.array(input_x['target_history']).shape
    pos = np.array(input_x['pos_history']).shape
    neg = np.array(input_x['neg_history']).shape
    index = np.array(input_x['index_history']).shape

    target = Input(shape=(target[1],target[2]), name="target_history")
    pos_history = Input(shape=(pos[1],pos[2]), name="pos_history")
    neg_history = Input(shape=(neg[1],neg[2]), name="neg_history")
    index_history = Input(shape=(index[1],index[2]), name="index_history")

    t = LSTM(64, return_sequences=True,activation="relu", name="target_lstm1")(target)
    p = LSTM(64, return_sequences=True, activation="relu", name="pos_lstm1")(pos_history)
    n = LSTM(64, return_sequences=True, activation="relu", name="neg_lstm1")(neg_history)
    i = LSTM(64, return_sequences=True, activation="relu", name="index_lstm1")(index_history)

    # combine the output of the branches
    combined = concatenate([t, p, n, i],1)
    z = LSTM(64, return_sequences=True, activation="relu")(combined)

    z, slf_attn = MultiHeadAttention(n_head=3, d_model=300, d_k=64, d_v=64, dropout=0.1)(combined, combined, combined)
    z = Flatten()(z)
    z = Dense(64, activation="relu")(z)
    z = Dense(64, activation="relu")(z)
    z = Dense(64, activation="relu")(z)
    z = Dense(64, activation="relu")(z)
    z = Dense(64, activation="relu")(z)
    outputs = Dense(1, name='output')(z)

    model = Model(inputs=[target,pos_history,neg_history, index_history], outputs=[outputs] , name = 'MultiInput')
    model.compile(loss="mse", optimizer='adam')
    print(model.summary())

    tf.keras.utils.plot_model(
        model, 
        'img/Ori-MIlstm.png', 
        show_shapes=True,
        expand_nested=True,
        )

    return model

"""
Single Attention Multi Input Network
"""
def SingleAttentionLSTM(input_x):
    # define sets of inputs
    
    target = np.array(input_x['target_history']).shape
    pos = np.array(input_x['pos_history']).shape
    neg = np.array(input_x['neg_history']).shape
    index = np.array(input_x['index_history']).shape

    target = Input(shape=(target[1],target[2]), name="target_history")
    pos_history = Input(shape=(pos[1],pos[2]), name="pos_history")
    neg_history = Input(shape=(neg[1],neg[2]), name="neg_history")
    index_history = Input(shape=(index[1],index[2]), name="index_history")
    # target_price = Input(shape=(n_obs[1],n_obs[2]), name="target_price")

    t = LSTM(128, return_sequences=True,activation="relu", name="target_lstm1")(target)

    p = LSTM(128, return_sequences=True, activation="relu", name="pos_lstm1")(pos_history)
    # p = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(p)
    
    n = LSTM(128, return_sequences=True, activation="relu", name="neg_lstm1")(neg_history)
    # n = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(n)
    
    i = LSTM(128, return_sequences=True, activation="relu", name="index_lstm1")(index_history)

    # combine the output of the branches
    combined = concatenate([t, p, n, i],1)
    z = CustomAttention(combined.shape[1])(combined)
    z = Dense(256, activation="relu")(z)
    outputs = Dense(1, name='output')(z)

    model = Model(inputs=[target,pos_history,neg_history, index_history], outputs=[outputs] , name = 'MultiInput')
    model.compile(loss="mse", optimizer='adam')
    print(model.summary())

    tf.keras.utils.plot_model(
        model, 
        'img/Single-Attention.png', 
        show_shapes=True,
        expand_nested=True,
        )

    return model

"""
MultiHeadAttention Multi Input Network
"""
def MultiHeadAttentionLSTM(input_x):
    # define sets of inputs
    
    target = np.array(input_x['target_history']).shape
    pos = np.array(input_x['pos_history']).shape
    neg = np.array(input_x['neg_history']).shape
    index = np.array(input_x['index_history']).shape

    target = Input(shape=(target[1],target[2]), name="target_history")
    pos_history = Input(shape=(pos[1],pos[2]), name="pos_history")
    neg_history = Input(shape=(neg[1],neg[2]), name="neg_history")
    index_history = Input(shape=(index[1],index[2]), name="index_history")

    t = LSTM(128, return_sequences=True,activation="relu", name="target_lstm1")(target)
    p = LSTM(128, return_sequences=True, activation="relu", name="pos_lstm1")(pos_history)
    n = LSTM(128, return_sequences=True, activation="relu", name="neg_lstm1")(neg_history)
    i = LSTM(128, return_sequences=True, activation="relu", name="index_lstm1")(index_history)

    # combine the output of the branches
    combined = concatenate([t, p, n, i],1)
    z, slf_attn = MultiHeadAttention(n_head=3, d_model=300, d_k=64, d_v=64, dropout=0.1)(combined, combined, combined)
    # z = Flatten()(z)
    z = LSTM(128, activation="relu")(z)
    outputs = Dense(1, name='output')(z)

    model = Model(inputs=[target,pos_history,neg_history, index_history], outputs=[outputs] , name = 'MultiInput')
    model.compile(loss="mse", optimizer='adam')
    print(model.summary())

    tf.keras.utils.plot_model(
        model, 
        'img/MultiHead-Attention.png', 
        show_shapes=True,
        expand_nested=True,
        )

    return model

'''
dnn 
'''
def dnn(n_obs):
    """ A multi-layer perceptron """
    model = Sequential()
    model.add(Dense(units=128, input_shape=[n_obs[1]], activation="relu"))
    model.add(Dense(units=256, input_shape=[n_obs[1]], activation="relu"))
    # model.add(Dropout(0.3))
    model.add(BatchNormalization())
    # model.add(Dense(units=256, activation="relu"))
    # model.add(Dense(units=128, activation="relu"))
    # model.add(Dropout(0.3))
    # model.add(BatchNormalization())
    model.add(Dense(1))
    model.compile(loss="mse", optimizer='adam')
    print(model.summary())
    return model

'''
Conv1d 
'''
def conv1d(n_obs):
    kernel_size=2
    strides=1
    padding = 'same'
    model = Sequential()
    model.add(Conv1D(filters = 128, kernel_size=kernel_size, strides=strides, padding=padding, activation = 'relu',input_shape=(n_obs[1],n_obs[2])))
    model.add(Conv1D(filters = 128, kernel_size=kernel_size, strides=strides, padding=padding, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    # model.add(Conv1D(filters = 512, kernel_size=kernel_size, strides=strides, padding=padding, activation = 'relu'))
    model.add(Flatten())
    # model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer='adam')
    print(model.summary())
    return model

'''
Conv2d 
'''
def conv2d(n_obs):
    kernel_size=(2,2)
    # strides=(1,1)
    padding = 'same'
    model = Sequential()
    model.add(Conv2D(filters = 128, kernel_size=kernel_size,  padding=padding, activation = 'relu',input_shape=(n_obs[1],n_obs[2],1)))
    model.add(Conv2D(filters = 128, kernel_size=kernel_size,  padding=padding, activation = 'relu'))
    # model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    # model.add(Conv2D(filters = 128, kernel_size=kernel_size,  padding=padding, activation = 'linear'))
    model.add(Flatten())
    # model.add(Dropout(0.3))
    model.add(Dense(1))
    # model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['mae', 'mape'])
    model.compile(loss="mse", optimizer='adam')
    print(model.summary())
    return model

'''
LSTM 
'''
def lstm(n_obs):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation = 'relu',input_shape=(n_obs[1],n_obs[2])))
    model.add(LSTM(128, return_sequences=True, activation = 'relu'))
    # model.add(LSTM(128, dropout=0.2, return_sequences=True))
    # model.add(Dropout(0.3))
    # model.add(BatchNormalization())
    model.add(LSTM(256, return_sequences=True,activation = 'relu'))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    # model.add(BatchNormalization())
    # model.add(Flatten())
    # model.add(Dropout(0.3))
    model.add(Dense(1))
    # model.compile(loss="mse", optimizer='adam', metrics=['mae', 'mape'])
    model.compile(loss="mean_squared_error", optimizer='adam')
    print(model.summary())
    return model

'''
Transformer 
'''

seq_len = 20
d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256
def transformer(n_obs):
    '''Initialize time and transformer layers'''
    time_embedding = Time2Vector(seq_len)
    attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

    '''Construct model'''
    in_seq = Input(shape=(n_obs[1],n_obs[2]))
    x = time_embedding(in_seq)
    x = Concatenate(axis=-1)([in_seq, x])
    x = attn_layer1((x, x, x))
    x = attn_layer2((x, x, x))
    x = attn_layer3((x, x, x))
    x = GlobalAveragePooling1D(data_format='channels_first')(x)
    x = Dropout(0.3)(x)
    # x = Dense(128, activation='linear')(x)
    out = Dense(1)(x)

    model = Model(inputs=in_seq, outputs=out)
    model.compile(loss='mse', optimizer='adam')
    # model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['mae', 'mape'])
    print(model.summary())
    return model



