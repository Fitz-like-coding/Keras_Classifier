from keras.layers import *
from keras.models import *


def build_model_textcnn(max_length, emb, max_words, class_num):
    word_input = Input(shape=(max_length,), dtype='int32', name='word_input')
    embed = Embedding(output_dim=emb, dtype='float32', input_dim=max_words+1, input_length=max_length)(word_input)

    cnn1_1 = Conv1D(filter=256, kernel_size=1, padding='same', strides=1)(embed)
    cnn1_1_bn = BatchNormalization()(cnn1_1)
    cnn1_1_at = Activation(activation='relu')(cnn1_1_bn)
    cnn1_2 = Conv1D(filter=256, kernel_size=1, padding='same', strides=1)(cnn1_1_at)
    cnn1_2_bn = BatchNormalization()(cnn1_2)
    cnn1_2_at = Activation(activation='relu')(cnn1_2_bn)
    cnn1 = GlobalMaxPooling1D()(cnn1_2_at)

    cnn2_1 = Conv1D(filter=256, kernel_size=2, padding='same', strides=1)(embed)
    cnn2_1_bn = BatchNormalization()(cnn2_1)
    cnn2_1_at = Activation(activation='relu')(cnn2_1_bn)
    cnn2_2 = Conv1D(filter=256, kernel_size=2, padding='same', strides=1)(cnn2_1_at)
    cnn2_2_bn = BatchNormalization()(cnn2_2)
    cnn2_2_at = Activation(activation='relu')(cnn2_2_bn)
    cnn2 = GlobalMaxPooling1D()(cnn2_2_at)

    cnn3_1 = Conv1D(filter=256, kernel_size=4, padding='same', strides=1)(embed)
    cnn3_1_bn = BatchNormalization()(cnn3_1)
    cnn3_1_at = Activation(activation='relu')(cnn3_1_bn)
    cnn3_2 = Conv1D(filter=256, kernel_size=4, padding='same', strides=1)(cnn3_1_at)
    cnn3_2_bn = BatchNormalization()(cnn3_2)
    cnn3_2_at = Activation(activation='relu')(cnn3_2_bn)
    cnn3 = GlobalMaxPooling1D()(cnn3_2_at)

    con = concatenate([cnn1, cnn2, cnn3], axis=-1)
    drop_out_word = Dropout(0.5, name='dropout')(con)
    output = Dense(class_num, activation='softmax', name='output')(drop_out_word)
    model = Model(inputs=[word_input], output=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print (model.summary())
    return model