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
parser.add_argument('-p', '--num_predict', type=int, default=100)
args = parser.parse_args()


def infer_recitation_to_text(args):
    prefix = args.filename_prefix
    decoder_model = tf.keras.models.load_model(os.path.join(args.output_dir, f'decoder-model-{prefix}.h5'))
    encoder_model = tf.keras.models.load_model(os.path.join(args.output_dir, f'encoder_model-{prefix}.h5'))
    print("Models loaded")

    encoder_input_data, decoder_input_data, decoder_target_data = get_seq2seq_data(n=args.num_predict)
    print("Data loaded")

    max_decoder_seq_length = decoder_input_data.shape[1]
    num_decoder_tokens = decoder_input_data.shape[-1]

    one_hot_obj = get_one_hot_encodings()
    reverse_target_char_index = one_hot_obj['int_to_char']
    reverse_target_char_index[num_decoder_tokens - 2] = '->'
    reverse_target_char_index[num_decoder_tokens - 1] = '<-'

    # Perform inference on some of the audio files
    with open(os.path.join(args.output_dir, f'inference-{prefix}.txt'), 'w') as f:
        num_predict = args.num_predict
        for seq_index in range(num_predict):
            print(seq_index, end=' ')
            input_seq = encoder_input_data[seq_index: seq_index + 1]
            decoded_sentence = decode_sequence(
                input_seq, num_decoder_tokens, encoder_model, decoder_model, max_decoder_seq_length)

            true_array = decoder_target_data[seq_index]
            true_sentence = ''
            for pos in range(true_array.shape[0]):
                sampled_token_index = np.argmax(true_array[pos])
                sampled_char = reverse_target_char_index[sampled_token_index]
                true_sentence += sampled_char
            f.write(true_sentence + ',' + decoded_sentence + '\n')


if __name__ == "__main__":
    infer_recitation_to_text(args)
