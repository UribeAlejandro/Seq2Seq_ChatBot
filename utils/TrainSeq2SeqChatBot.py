from keras.models import Model
from keras.layers import Input, LSTM, Dense


class TrainSeq2SeqChatBot:
    def __init__(self, input_vocab_size, output_vocab_size, encoder_input_data, decoder_input_data, decoder_target_data,
                 epochs=100, batch_size=64, num_neurons=256):
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        self.encoder_input_data = encoder_input_data
        self.decoder_input_data = decoder_input_data
        self.decoder_target_data = decoder_target_data

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_neurons = num_neurons

        self.encoder_inputs = Input(shape=(None, self.input_vocab_size))
        self.decoder_inputs = Input(shape=(None, self.output_vocab_size))

        self.encoder_states = []

    def train_model(self) -> dict:
        encoder = LSTM(self.num_neurons, return_state=True)
        encoder_outputs, state_h, state_c = encoder(self.encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_lstm = LSTM(self.num_neurons, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(self.decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.output_vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                  batch_size=self.batch_size,
                  epochs=self.epochs, validation_split=0.1)

        cache = {
            "decoder_lstm": decoder_lstm,
            "decoder_dense": decoder_dense
        }

        return cache

    def assemble_model_seq_gen(self, decoder_lstm, decoder_dense) -> dict:
        encoder_model = Model(self.encoder_inputs, self.encoder_states)
        thought_input = [Input(shape=(self.num_neurons,)), Input(shape=(self.num_neurons,))]

        decoder_outputs, state_h, state_c = decoder_lstm(self.decoder_inputs, initial_state=thought_input)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)

        decoder_model = Model(inputs=[self.decoder_inputs] + thought_input, output=[decoder_outputs] + decoder_states)

        cache = {
            "encoder_model": encoder_model,
            "decoder_model": decoder_model
        }

        return cache
