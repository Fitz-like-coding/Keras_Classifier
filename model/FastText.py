from keras.layers import *
from keras.models import *


def build_model_fasttext(max_length, emb, max_words, classNum):
    word_input = Input(shape=(max_length,), dtype='int32', name='word_input')
    embed = Embedding(output_dim=emb, dtype='float32', input_dim=max_words+1, input_length=max_length)(word_input)
    dropout = Dropout(0.5)(embed)
    ave_pool = GlobalAveragePooling1D()(dropout)
    max_pool = GlobalMaxPooling1D()(dropout)
    con = concatenate([ave_pool, max_pool])
    output = Dense(classNum, activation='softmax')(con)
    model = Model(inputs=[input], output=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print (model.summary())
    return model
