import tensorflow as tf
import pandas as pd

def extract_fn(data_record):
	filelendf = pd.read_csv(bdir + 'Dataset/key_data.dat', engine = 'python', sep = ':', index_col = 0)
	features = {'Encoder_Input': tf.io.FixedLenFeature([filelendf.loc['max_source']['Length']*filelendf.loc['eng_word']['Length']], tf.float32), 'Decoder_Input': tf.io.FixedLenFeature([filelendf.loc['max_target']['Length']*filelendf.loc['mal_words']['Length']], tf.float32), 'Decoder_Output': tf.io.FixedLenFeature([filelendf.loc['max_target']['Length']*filelendf.loc['mal_words']['Length']], tf.float32)}
	sample = tf.io.parse_single_example(data_record, features)
	English = tf.reshape(sample['Encoder_Input'], (filelendf.loc['max_source']['Length'], filelendf.loc['eng_word']['Length']))
	Malayalam_1 = tf.reshape(sample['Decoder_Input'], (filelendf.loc['max_target']['Length'], filelendf.loc['mal_words']['Length']))
	Malayalam_2 = tf.reshape(sample['Decoder_Output'], (filelendf.loc['max_target']['Length'], filelendf.loc['mal_words']['Length']))
	return (English, Malayalam_1), Malayalam_2

def extract(file):
	dataset = tf.data.TFRecordDataset([bdir + "TfRecords/" + file + ".tfrecords"])
	dataset = dataset.map(extract_fn)
	dataset = dataset.shuffle(256)
	dataset = dataset.repeat()
	dataset = dataset.batch(128)