"""
Contains the main method used to train recitation2text models and save relevant outputs.
Examples of running this from the command line:
> python3 train_recitation_to_text.py -e 50 -p 10
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from architectures.seq2seq import *
from utils import *
import os
import time
import mlflow
import mlflow.keras

parser = ArgumentParser(description='Tarteel Train Sequence-Sequence Model')
parser.add_argument('-b', '--batch_size', type=int, help="batch size for training", default=20)
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-o', '--output_dir', type=str, default='.outputs/')
parser.add_argument('-l', '--learning_rate', type=float, default=0.01)
parser.add_argument('-n', '--num_train', type=int, default=100)
parser.add_argument('-p', '--num_predict', type=int, default=5)
parser.add_argument('-d', '--latent_dim', type=int, default=20)
args = parser.parse_args()


def train_recitation_to_text(args):
    """
    Trains a recitation-to-text keras model and saves history and predictions to file
    :param args:
    :return:
    """
    time_of_experiment = time.time()  # Saves the time experiment started, useful for timestamping output files.
    latent_dim = args.latent_dim  # Latent dimensionality of the encoding space.
    encoder_input_data, decoder_input_data, decoder_target_data = get_seq2seq_data(n=args.num_train)

    max_decoder_seq_length = decoder_input_data.shape[1]
    num_encoder_tokens = encoder_input_data.shape[-1]
    num_decoder_tokens = decoder_input_data.shape[-1]

    print("Audio data shape:", encoder_input_data.shape)
    print("Text data shape:", decoder_target_data.shape)

    # Get the model for training
    encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense, model = \
        lstm_encoder_decoder_with_teacher_forcing_training(
            latent_dim=latent_dim, num_encoder_tokens=num_encoder_tokens, num_decoder_tokens=num_decoder_tokens)
    adam = tf.keras.optimizers.Adam(lr=args.learning_rate)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    model.summary()

    # Train the model
    with mlflow.start_run():
        history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                            batch_size=args.batch_size,
                            epochs=args.epochs,
                            validation_split=0.2)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_metric("loss", history.history['loss'][-1])
        mlflow.log_metric("val_loss", history.history['val_loss'][-1])
        mlflow.keras.log_model(model, "model")

    # Save training history
    plt.plot(range(args.epochs), history.history['loss'], label='Training')
    plt.plot(range(args.epochs), history.history['val_loss'], label='Validation')
    plt.legend()
    plt.savefig(f'.outputs/seq2seq-loss-{time_of_experiment}.png')

    # Get inference model
    encoder_model, decoder_model = lstm_encoder_decoder_with_teacher_forcing_inference(
        latent_dim, encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense)

    # Save models
    os.makedirs(args.output_dir, exist_ok=True)
    model.save(os.path.join(args.output_dir, f'model-{time_of_experiment}.h5'))
    decoder_model.save(os.path.join(args.output_dir, f'decoder-model-{time_of_experiment}.h5'))
    encoder_model.save(os.path.join(args.output_dir, f'encoder_model-{time_of_experiment}.h5'))

    # Perform inference on some of the audio files
    with open(os.path.join(args.output_dir, f'predictions-{time_of_experiment}.txt'), 'w') as f:
        for seq_index in range(args.num_predict):
            input_seq = encoder_input_data[seq_index: seq_index + 1]
            decoded_sentence = decode_sequence(
                input_seq, num_decoder_tokens, encoder_model, decoder_model, max_decoder_seq_length)
            f.write(decoded_sentence + '\n')


if __name__ == "__main__":
    train_recitation_to_text(args)

