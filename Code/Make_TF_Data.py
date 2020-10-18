import pandas as pd
import os, re, codecs
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

def main():
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	bdir = ""
	eng_files = [f for f in os.listdir(bdir + "Dataset/") if re.match(r".*\.en.*", f)]
	mal_files = [f for f in os.listdir(bdir + "Dataset/") if re.match(r".*\.ml$", f)]
	files = []
	find = re.compile(r"^(.*?)\..*")

	for eng_file in eng_files:
		for mal_file in mal_files:
			if re.search(find, eng_file).group(1) == re.search(find, mal_file).group(1):
				files.append((eng_file, mal_file))
	English = []
	Malayalam = []
	for file in files:
		eng_lines = codecs.open(bdir + "Dataset/" + file[0], encoding = "utf-8").readlines()
		mal_lines = codecs.open(bdir + "Dataset/" + file[1], encoding = "utf-8").readlines()
		English.extend(eng_lines)
		Malayalam.extend(mal_lines)

	Data = pd.DataFrame({"English": English, "Malayalam": Malayalam})
	Data.English = Data.English.apply(lambda x: x.lower().strip())
	Data.Malayalam = Data.Malayalam.apply(lambda x: "START_ " + x.lower().strip() + " _END")

	eng_words = set()
	mal_words = set()
	max_source_len = 0
	max_tar_len = 0

	for eng in Data.English:
		words = eng.split()
		max_source_len = max(max_source_len, len(words))
		eng_words.update(set(words))

	for mal in Data.Malayalam:
		words = mal.split()
		max_tar_len = max(max_tar_len, len(words))
		mal_words.update(set(words))

	eng_words = sorted(list(eng_words))
	eng_words.append("_END")
	mal_words = sorted(list(mal_words))

	input_token_index = dict([(word, i) for i, word in enumerate(eng_words)])
	output_token_index = dict([(word, i) for i, word in enumerate(mal_words)])
	input_index_token = dict([(i, word) for i, word in enumerate(eng_words)])
	output_index_token = dict([(i, word) for i, word in enumerate(mal_words)])

	"""print ("max_source_len = ", max_source_len)
	print ("max_tar_len = ", max_tar_len)
	print ("Number of input tokens = ", len(eng_words))
	print ("Number of output tokens = ", len(mal_words))"""

	train_data, val_data = train_test_split(Data, test_size = 0.2, random_state = 12)
	os.system("mkdir TfRecords")
	train_writer = tf.io.TFRecordWriter(bdir + "TfRecords/Train.tfrecords")

	for j in range(len(train_data)):
		print (j, end = " ")
		encoder_input = np.zeros((1, max_source_len, len(eng_words)))
		decoder_input = np.zeros((1, max_tar_len, len(mal_words)))
		decoder_output = np.zeros((1, max_tar_len, len(mal_words)))
		for i, word in enumerate(train_data.iloc[j].English.split()):
			encoder_input[0, i, input_token_index[word]] = 1.0
		for i, word in enumerate(train_data.iloc[j].Malayalam.split()):
			decoder_input[0, i, output_token_index[word]] = 1.0
			if i > 0:
				decoder_output[0, i-1, output_token_index[word]] = 1.0
		feature = {"Encoder_Input" : tf.train.Feature(float_list = tf.train.FloatList(value = encoder_input[0].reshape(-1))), "Decoder_Input": tf.train.Feature(float_list = tf.train.FloatList(value = decoder_input[0].reshape(-1))), "Decoder_Output": tf.train.Feature(float_list = tf.train.FloatList(value = decoder_output[0].reshape(-1)))}
		example = tf.train.Example(features = tf.train.Features(feature = feature))
		train_writer.write(example.SerializeToString())

	#print (len(train_data))

	validation_writer = tf.io.TFRecordWriter(bdir + "TfRecords/Validation.tfrecords")

	for j in range(len(val_data)):
		print (j, end = " ")
		encoder_input = np.zeros((1, max_source_len, len(eng_words)))
		decoder_input = np.zeros((1, max_tar_len, len(mal_words)))
		decoder_output = np.zeros((1, max_tar_len, len(mal_words)))
		for i, word in enumerate(val_data.iloc[j].English.split()):
			encoder_input[0, i, input_token_index[word]] = 1.0
		for i, word in enumerate(val_data.iloc[j].Malayalam.split()):
			decoder_input[0, i, output_token_index[word]] = 1.0
			if i > 0:
				decoder_output[0, i-1, output_token_index[word]] = 1.0
		feature = {"Encoder_Input" : tf.train.Feature(float_list = tf.train.FloatList(value = encoder_input[0].reshape(-1))), "Decoder_Input": tf.train.Feature(float_list = tf.train.FloatList(value = decoder_input[0].reshape(-1))), "Decoder_Output": tf.train.Feature(float_list = tf.train.FloatList(value = decoder_output[0].reshape(-1)))}
		example = tf.train.Example(features = tf.train.Features(feature = feature))
		validation_writer.write(example.SerializeToString())

	#print (len(val_data))

	key_data = {"max_source": max_source_len, "max_target": max_tar_len, "eng_words": len(eng_words), "mal_words": len(mal_words), "train_data": len(train_data), "val_data": len(val_data)}
	pd.DataFrame.from_dict(key_data, orient = "index").to_csv(bdir + 'Dataset/key_data.dat', sep = ':', header = ['Length'])

if __name__ == "__main__":
	main()