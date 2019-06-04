from keras.layers import *
from keras.models import *


def build_model_textrnn(max_length, emb, max_words, class_num):
    word_input = Input(shape=(max_length,), dtype='int32', name='word_input')
    embed = Embedding(output_dim=emb, dtype='float32', input_dim=max_words+1, input_length=max_length)(word_input)

    lstm = Bidirectional(LSTM(units=256, dropout=0.5, return_sequences=True))(embed)
    max_pool = GlobalMaxPooling1D()(lstm)
    output = Dense(class_num, activation='softmax')(max_pool)
    model = Model(inputs=[input], output=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print (model.summary())
    return model
