from flask import Flask, app, render_template,request

from keras.models import load_model
import numpy as np


from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence
import cv2
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from tensorflow.keras.applications import ResNet50
res_model = ResNet50(include_top = True)
res_model.summary()

from keras.models import Model
req_output = res_model.layers[-2].output
new_res_model = Model(inputs = res_model.input, outputs = req_output)
new_res_model.summary()

from keras.models import Model
req_output = res_model.layers[-2].output
new_res_model = Model(inputs = res_model.input, outputs = req_output)
new_res_model.summary()

vocab = np.load('vocab.npy', allow_pickle=True)

vocab = vocab.item()

inv_vocab = {v:k for k,v in vocab.items()}

embedding_size = 128
max_len = 34
vocab_size = len(vocab)

image_model = Sequential()
image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))

image_model.summary()

language_model = Sequential()

language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))

language_model.summary()

concat = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(concat)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs = out)

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model.summary()

model=load_model(r'C:\Users\asus\Desktop\project\imagemodel.h5')


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/',methods=['POST', 'GET'])
def index():
   return render_template('index.html')

@app.route('/after',methods = ['GET','POST'])

def after():
    global model, vocav, inv_vocab
    file = request.files['hello']

    file.save('static/file.jpg')
    
    

    test_img = cv2.imread('static/file.jpg')
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    test_img = cv2.resize(test_img, (224,224))

    test_img = np.reshape(test_img, (1,224,224,3))
    
    incept = new_res_model.predict(test_img).reshape(1,2048)

    text_in = ['sos']
    final=' '

    print("="*50)
    print("GETTING CAPTIONS")

    count = 0
    while tqdm(count < 25):
        count += 1

        encoded = []
        for i in text_in:
            encoded.append(vocab[i])
        encoded = [encoded]

        encoded = pad_sequences(encoded, padding='post', truncating='post', maxlen=max_len).reshape(1,max_len)

        

        sampled_index = np.argmax(model.predict([incept, encoded]))

        sampled_word = inv_vocab[sampled_index]

        if sampled_word != 'eos':
            final = final + ' ' + sampled_word

        text_in.append(sampled_word)
    return render_template('after.html', data=final)    

if __name__ == "__main__":
   app.run(debug = True)