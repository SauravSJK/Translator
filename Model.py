import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from Utilities import extract

def main():
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	os.system("rm log.csv loss.png accuracy.png model.h5")
	bdir = ""
	filelendf = pd.read_csv(bdir + 'Dataset/key_data.dat', engine = 'python', sep = ':', index_col = 0)

	train_dataset = extract('Train')
	validation_dataset = extract('validation')

	encoder_input = Input(shape = (filelendf.loc['max_source']['Length'], filelendf.loc['eng_word']['Length']))
	_, state_h, state_c = LSTM(256, return_state = True)(encoder_input)
	encoder_states = [state_h, state_c]
	decoder_output, _, _ = LSTM(256, return_sequences = True, return_state = True)(decoder_input, initial_state = encoder_states)
	decoder_output = Dense(filelendf.loc['mal_words']['Length'], activation = "softmax")(decoder_output)
	model = Model([encoder_input, decoder_input], decoder_output)
	model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
	es = EarlyStopping(monitor = 'accuracy', mode = 'max', verbose = 1, patience = 50)
	cl = CSVLogger(bdir + 'log.csv', append = True, separator = ',')
	mc = ModelCheckpoint(bdir + 'model.h5', monitor = 'val_accuracy', verbose = 1, save_best_only = True)

	history = model.fit(train_dataset, epochs = 1000, steps_per_epoch = (filelendf.loc['train_data']['Length'] // 128), callbacks = [cl, es, mc], validation_data = validation_dataset, validation_steps = (filelendf.loc['val_data']['Length']// 128))
	print ("--> Plotting Loss")
	print(history.history.keys())

	f1 = plt.figure()
	f2 = plt.figure()
	ax1 = f1.add_subplot(111)
	ax1.plot(history.history['loss'])
	ax1.plot(history.history['val_loss'])
	ax1.set_title('model loss')
	ax1.set_ylabel('loss')
	ax1.set_xlabel('epoch')
	ax1.legend(['train', 'validation'], loc = 'upper left')
	f1.savefig(bdir + 'loss.png', bbox_inches = 'tight')

	ax2 = f2.add_subplot(111)
	ax2.plot(history.history['accuracy'])
	ax2.plot(history.history['val_accuracy'])
	ax2.set_title('model accuracy')
	ax2.set_ylabel('accuracy')
	ax2.set_xlabel('epoch')
	ax2.legend(['train', 'validation'], loc = 'upper left')
	f2.savefig(bdir + 'accuracy.png', bbox_inches = 'tight')

if __name__ == "__main__":
	main()