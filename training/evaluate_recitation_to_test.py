"""
Contains the main method used to do inference with recitation2text models and save relevant outputs.
Examples of running this from the command line:
> python3 infer_recitation_to_text.py -f "1551586533.5614707"
"""
import tensorflow as tf
from utils import decode_sequence, get_one_hot_encodings, get_seq2seq_data
from argparse import ArgumentParser
import os
import numpy as np


parser = ArgumentParser(description='Tarteel Train Sequence-Sequence Model')
parser.add_argument('-o', '--output_dir', type=str, help="directory with saved model", default='.outputs/')
parser.add_argument('-f', '--filename_prefix', type=str, help="prefix of the saved models")
parser.add_argument('-p', '--num_predict', type=int, default=-1)
args = parser.parse_args()


def infer_recitation_to_text(args):
    prefix = args.filename_prefix
    model = tf.keras.models.load_model(os.path.join(args.output_dir, f'model-{prefix}.h5'))
    print("Model loaded")

    encoder_input_data, decoder_input_data, decoder_target_data, filenames = get_seq2seq_data(
        n=args.num_predict, return_filenames=True)

    num_decoder_tokens = decoder_input_data.shape[-1]
    print("Data loaded")

    one_hot_obj = get_one_hot_encodings()
    reverse_target_char_index = one_hot_obj['int_to_char']
    reverse_target_char_index[num_decoder_tokens - 2] = '->'
    reverse_target_char_index[num_decoder_tokens - 1] = '<-'

    for i in range(args.num_predict):
        print(filenames[i])
        loss = model.evaluate([encoder_input_data[i: i+1],
                               decoder_input_data[i: i+1]],
                              decoder_target_data[i: i+1])
        print('True loss: ' + str(loss))

    for i in range(args.num_predict):
        print(filenames[i])
        loss = model.evaluate([encoder_input_data[i: i+1],
                               decoder_input_data[i: i+1]],
                              decoder_target_data[0: 1])
        print('Shuffle loss: ' + str(loss))

    for i in range(args.num_predict):
        print(filenames[i])
        loss = model.evaluate([encoder_input_data[i: i+1],
                               decoder_input_data[i: i+1]],
                              np.random.random(decoder_target_data[0: 1].shape))
        print('Rand loss: ' + str(loss))

    for i in range(args.num_predict):
        print(filenames[i])
        loss = model.evaluate([encoder_input_data[i: i+1],
                               decoder_input_data[i: i+1]],
                              np.zeros(decoder_target_data[0: 1].shape))
        print('Zero loss: ' + str(loss))


if __name__ == "__main__":
    infer_recitation_to_text(args)
