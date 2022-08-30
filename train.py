import glob
import numpy as np
import os
import tensorflow as tf
import random
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import CuDNNLSTM, LSTM, Bidirectional
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def get_sequences(notes, unique_notes):
  """ Prepare the sequences used by the Neural Network """
  sequence_length = 100
  # getting pitch names
  pitchnames = sorted(set(item for item in notes))
  # mapping pitches to integers
  note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
  # creating sequences
  input, output = [], []
  for i in range(0, len(notes) - sequence_length, 1):
    input.append([note_to_int[char] for char in notes[i:i + sequence_length]]) # input sequence
    output.append(note_to_int[notes[i + sequence_length]]) # output sequence based on input

  # reshaping input into compatible format for LSTM
  input = np.reshape(input, (len(input), sequence_length, 1))
  # normalizing input between 0 and 1
  input = input / float(unique_notes)
  output = np_utils.to_categorical(output)
  return input, output

def get_notes():
    notes = []
    """The commented code below parsed the MIDI files for the notes. Instead of re-parsing each time we run the application, I saved the notes to notes.txt and get_notes simply extracts the notes from the text file"""
    # for file in glob.glob('Classical MIDIS/*.mid'):
    # midi = converter.parse(file)
    # print("Parsing %s" % file)
    # try: # instrument parts
    #     partitioned = instrument.partitionByInstrument(midi)
    #     parsed = partitioned.parts[0].recurse()
    # except: # flat notes
    #     parsed = midi.flat.notes
    # for element in parsed:
    #   if isinstance(element, note.Note):
    #       notes.append(str(element.pitch))
    #   elif isinstance(element, chord.Chord):
    #       notes.append('.'.join(str(n) for n in element.normalOrder))
    with open("notes.txt", "r") as file:
      for line in file:
        notes.append(line[:-1])
    return notes

def create_network(input, unique_notes):
    """ create the model for neural network """
    # Using CuDNNLSTM to speed up training with GPU
    model = Sequential()
    model.add(CuDNNLSTM(512,input_shape=(input.shape[1], input.shape[2]),return_sequences=True))
    model.add(Dropout(0.3)) # dropout helps reduce overfitting
    model.add(Bidirectional(CuDNNLSTM(512, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(CuDNNLSTM(512)))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(unique_notes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def train_model(input, output, unique_notes):
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    model = create_network(input, unique_notes)
    # model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    # model.build(tf.TensorShape([1, None]))
    model.fit(input, output, callbacks=[checkpoint_callback], epochs=200, batch_size=64)
    model.save("200Epochs.h5")

if __name__ == "__main__":
    notes = get_notes()
    unique_notes = len(set(notes))
    input, output = get_sequences(notes, unique_notes)
    train_model(input, output, unique_notes)
