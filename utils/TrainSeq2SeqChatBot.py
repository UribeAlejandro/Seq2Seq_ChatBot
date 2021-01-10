import os
import warnings
from pathlib import Path
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

warnings.filterwarnings('ignore')


class TrainSeq2SeqChatBot:
    def __init__(self, input_vocab_size, output_vocab_size, encoder_input_data, decoder_input_data, decoder_target_data,
                 epochs=100, batch_size=64, num_neurons=256):

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_neurons = num_neurons
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.encoder_input_data = encoder_input_data
        self.decoder_input_data = decoder_input_data
        self.decoder_target_data = decoder_target_data

        self.model_parent_dir = Path("./model/")
        self.model_file_dir = Path("./model/chatBot.hdf5")
        self.models = dict()

    def train_model(self) -> None:
        """
        Train encoder and decoder if the model has not been trained yet.
        Saves the trained model within "model" folder.

        :return: None
        """
        if not os.path.isdir(self.model_parent_dir):
            os.mkdir(self.model_parent_dir)
        else:
            pass

        if not os.path.isfile(self.model_file_dir):

            encoder_inputs = Input(shape=(None, self.input_vocab_size), name="Input_Encoder")
            encoder = LSTM(self.num_neurons, return_state=True, name="LSTM_Encoder_Layer")
            encoder_outputs, state_h, state_c = encoder(encoder_inputs)
            encoder_states = [state_h, state_c]

            decoder_inputs = Input(shape=(None, self.output_vocab_size), name="Decoder_Input")
            decoder_lstm = LSTM(self.num_neurons, return_sequences=True, return_state=True, name="LSTM_Decoder_Layer")
            decoder_dense = Dense(self.output_vocab_size, activation='softmax', name="Dense_Decoder_Layer")

            decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
            decoder_outputs = decoder_dense(decoder_outputs)

            model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            print(model.summary())

            plot_model(model, to_file="./model/model.png",
                       show_shapes=True, show_dtype=True,
                       show_layer_names=True, rankdir="TB",
                       expand_nested=True, dpi=96)

            callbacks = [ModelCheckpoint(filepath=self.model_file_dir,
                                         monitor="loss",
                                         save_best_only=True,
                                         verbose=1),

                         ReduceLROnPlateau(monitor='loss',
                                           min_lr=0.0001,
                                           factor=0.2,
                                           patience=1,
                                           verbose=1)]

            model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                      batch_size=self.batch_size, epochs=self.epochs, validation_split=0.1, callbacks=callbacks)

            # model.save(self.model_file_dir)

        else:
            pass

    def assemble_model_seq_gen(self) -> None:
        """
        Loads model stored in model folder, assembles
        :return: None.
        """
        model = load_model(self.model_file_dir)
        print(model.summary())

        encoder_inputs = model.input[0]
        encoder_outputs, state_h_enc, state_c_enc = model.get_layer("LSTM_Encoder_Layer").output
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = Model(encoder_inputs, encoder_states)

        thought_input = [Input(shape=(self.num_neurons,)), Input(shape=(self.num_neurons,))]

        decoder_inputs = model.input[1]
        decoder_lstm = model.get_layer("LSTM_Decoder_Layer")
        decoder_dense = model.get_layer("Dense_Decoder_Layer")

        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=thought_input)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + thought_input, [decoder_outputs] + decoder_states)

        self.models = {
            "encoder_model": encoder_model,
            "decoder_model": decoder_model
        }

    def train_assemble(self) -> dict:

        self.train_model()
        self.assemble_model_seq_gen()

        return self.models
