import keras as K
from keras.callbacks import ModelCheckpoint
import time
from model.FastText import *
from model.textRNN import *
from model.textCNN import *


def train_model(hyperparameter, datapath, x_train, y_train, x_dev, y_dev, classNum, class_weight):
    model_name = hyperparameter['model']
    max_length = hyperparameter['max_length']
    max_words = hyperparameter['max_words']
    emb = hyperparameter['emb']
    batch = hyperparameter['batch']
    epochs = hyperparameter['epochs']
    model_output = datapath['config']

    print ('build model...')
    if model_name == 'FastText':
        model = build_model_fasttext(max_length, emb, max_words, classNum)
    elif model_name == 'CNN':
        model = build_model_textcnn(max_length, emb, max_words, classNum)
    elif model_name == 'RNN':
        model = build_model_textrnn(max_length, emb, max_words, classNum)

    print ('Train...')
    print (model_output + '/' + model_name)
    start_time = time.time()
    model.fit([x_train], y_train,
              validation_data=(x_dev, y_dev),
              batch_size=batch, epochs=epochs,class_weight=class_weight,shuffle=False,
              callbacks=[
                  ModelCheckpoint(model_output + '/' + model_name + 'epoch:02d_{val_acc:2f}.h5',
                                  monitor='val_acc', verbose=1, save_best_only=True, period=5)
              ])
    end_time = time.time()
    print ('模型训练用时: %s S' % (end_time - start_time))
    model.save(model_output + '/model_' + model_name + str(int(end_time)) + '.h5')
    print ('model save success!')


def load_model(model_path):
    model = K.models.load_model(model_path)
    return model


def test_model(model, x, id2label, batch=8):
    res = []
    temp = model.predict([x], batch=batch)
    for i in temp:
        res.append(id2label[str(np.argmax(i,axis=0))])
    return res
