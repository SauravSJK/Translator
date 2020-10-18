import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from Utilities import extract_fn

def main():
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	bdir = ""
	filelendf = pd.read_csv(bdir + 'Dataset/key_data.dat', engine = 'python', sep = ':', index_col = 0)

	validation_dataset = tf.data.TFRecordDataset([bdir + "TfRecords/Validation.tfrecords"])
	validation_dataset = validation_dataset.map(extract_fn)
	validation_dataset = validation_dataset.batch(1)

	model = load_model(bdir + "model.h5")

	encoder_inputs = model.input[0]
	encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output
	encoder_states = [state_h_enc, state_c_enc]
	encoder_model = Model(encoder_inputs, encoder_states)

	decoder_inputs = model.input[1]
	decoder_state_input_h = Input(shape = (256,), name = "input_3")
	decoder_state_input_c = Input(shape = (256,), name = "input_4")
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_lstm = model.layers[3]
	decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state = decoder_states_inputs)
	decoder_states = [state_h_dec, state_c_dec]
	decoder_dense = model.layers[4]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

	def decode_sequence(input_seq, expected_out_seq):
		states_value = encoder_model.predict(input_seq)
		target_seq = np.zeros((1, 1, filelendf.loc['mal_words']['Length']))
		target_seq[0, 0, output_token_index["START_"]] = 1.0
		stop_condition = False
		decoded_sentence = ""
		input_sentence = ""
		expected_output_sentence = ""
		while not stop_condition:
			output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
			sampled_index = np.argmax(output_tokens[0, -1, :])
			sampled_char = output_index_token[sampled_index]
			decoded_sentence += sampled_char + " "
			if sampled_char == "_END" or len(decoded_sentence) > filelendf.loc['max_target']['Length']:
				stop_condition = True
			target_seq = np.zeros((1, 1, filelendf.loc['mal_words']['Length']))
			target_seq[0, 0, sampled_index] = 1.0
			states_value = [h, c]
		for i in range(len(expected_out_seq[0])):
			sampled_index = np.argmax(expected_out_seq[0, i, :])
			sampled_char = output_index_token[sampled_index]
			expected_output_sentence += sampled_char + " "
			if sampled_char == "_END":
				break
		for i in range(len(input_seq[0])):
			sampled_index = np.argmax(input_seq[0, i, :])
			sampled_char = input_index_token[sampled_index]
			input_sentence += sampled_char + " "
			if sampled_char == "_END":
				break
		return input_sentence, expected_output_sentence, decoded_sentence

	for i in validation_dataset.take(10):
		input_seq = i[0][0]
		expected_out_seq = i[1]
		input_sentence, expected_output_sentence, decoded_sentence = decode_sequence(input_seq, expected_out_seq)
		print("Input sentence:", input_sentence.strip())
		print("Expected output sentence: ", expected_output_sentence.strip())
		print("Decoded sentence:", decoded_sentence.strip())

if __name__ == "__main__":
	main()