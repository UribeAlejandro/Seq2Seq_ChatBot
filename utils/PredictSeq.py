import numpy as np


class PredictSeq:
    def __init__(self):
        pass

    def decode_sequence(self, input_seq):
        thought = encoder_model.predict(input_seq)

        target_seq = np.zeros((1, 1, output_vocab_size))
        target_seq[0, 0, target_token_index[stop_token]] = 1.

        stop_condition = False
        generated_sequence = ""

        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + thought)

            generated_token_idx = np.argmax(output_tokens[0, -1, :])
            generated_char = reverse_target_char_index[generated_token_idx]
            generated_sequence += generated_char

            if generated_char == stop_token or len(generated_sequence) > max_decoder_seq_length:
                stop_condition = True

                target_seq = np.zeros((1, 1, output_vocab_size))
                target_seq[0, 0, generated_token_idx] = 1.
                thought = [h, c]

        return generated_sequence

    def response(self, input_text):
        input_seq = np.zeros((1, max_encoder_seq_length, input_vocab_size), dtype='float32')

        for t, char in enumerate(input_text):
            input_seq[0, t, input_token_index[char]] = 1.

        decoded_sentence = decode_sequence(input_seq)

        print(f"Bot Reply (Decoded sentence): {decoded_sentence}")
