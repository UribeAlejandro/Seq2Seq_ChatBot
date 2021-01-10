import warnings
import numpy as np

warnings.filterwarnings('ignore')


class ChatterBot:
    def __init__(self, models: dict, input_vocab_size: int, output_vocab_size: int, target_token_index: dict,
                 stop_token: str, reverse_target_char_index: dict, max_encoder_seq_length: int,
                 max_decoder_seq_length: int, input_token_index: dict):
        self.models = models
        self.stop_token = stop_token

        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.input_token_index = input_token_index
        self.target_token_index = target_token_index

        self.max_encoder_seq_length = max_encoder_seq_length
        self.max_decoder_seq_length = max_decoder_seq_length

        self.reverse_target_char_index = reverse_target_char_index

        self.negative_commands = ("no", "nope", "nah", "naw", "not a chance", "sorry")
        self.exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")

    def decode_sequence(self, input_seq):
        encoder_model = self.models['encoder_model']
        decoder_model = self.models['decoder_model']

        thought = encoder_model.predict(input_seq)

        target_seq = np.zeros((1, 1, self.output_vocab_size))
        target_seq[0, 0, self.target_token_index[self.stop_token]] = 1.

        stop_condition = False
        generated_sequence = ''

        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + thought)

            generated_token_idx = np.argmax(output_tokens[0, -1, :])
            generated_char = self.reverse_target_char_index[generated_token_idx]
            generated_sequence += generated_char

            if ((generated_char == self.stop_token) or (len(generated_sequence) > self.max_decoder_seq_length)):
                stop_condition = True

            target_seq = np.zeros((1, 1, self.output_vocab_size))
            target_seq[0, 0, generated_token_idx] = 1.
            thought = [h, c]

        generated_sequence = generated_sequence.replace("\n", "")
        generated_sequence = generated_sequence.replace("\t", "")

        return generated_sequence

    def response(self, input_text: str) -> str:
        input_text = input_text.lower()
        input_seq = np.zeros((1, self.max_encoder_seq_length, self.input_vocab_size), dtype='float32')

        for t, char in enumerate(input_text):
            input_seq[0, t, self.input_token_index[char]] = 1.

        decoded_sentence = self.decode_sequence(input_seq)

        return decoded_sentence

    def start_chat(self):
        print("Bot: Hi, would like to chat with me?")
        user_response = input("User: ")
        if self.make_exit(user_response.lower()):
            print("Goodbye")
        else:
            self.chat()

    def chat(self) -> None:
        user_response = input("\nUser: ")
        if self.make_exit(user_response.lower()):
            print("Goodbye")
        else:
            print("Bot reply: " + self.response(user_response))
            self.chat()

    def make_exit(self, user_response) -> bool:
        if (user_response in self.negative_commands) or (user_response in self.exit_commands):
            exit_bool = True
        else:
            exit_bool = False

        return exit_bool
