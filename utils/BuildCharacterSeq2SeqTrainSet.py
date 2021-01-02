import numpy as np
from nlpia.loaders import get_data


class BuildCharacterSeq2SeqTrainSet:

    def __init__(self):
        self.input_texts, self.target_texts = [], []
        self.input_vocabulary, self.output_vocabulary = set(), set()
        self.input_vocab_size, self.output_vocab_size = 0, 0
        self.max_encoder_seq_length, self.max_decoder_seq_length = 0, 0
        self.input_token_index, self.target_token_index = dict(), dict()

        self.encoder_input_data, self.decoder_input_data, self.decoder_target_data = None, None, None

    def corpus_training(self) -> None:
        """
        Loads the corpus and generate the training sets.
        :return: None
        """

        df = get_data("moviedialog")

        start_token = '\t'
        stop_token = '\n'
        max_training_examples = min(25000, len(df) - 1)

        for input_text, target_text in zip(self.input_texts, self.target_texts):
            target_text = start_token + target_text + stop_token
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            for char in input_text:
                if char not in self.input_vocabulary:
                    self.input_vocabulary.add(char)
            for char in target_text:
                self.output_vocabulary.add(char)

    def build_char_dict(self) -> None:
        """

        :return: None
        """

        self.input_vocabulary = sorted(self.input_vocabulary)
        self.output_vocabulary = sorted(self.output_vocabulary)

        self.input_vocab_size = len(self.input_vocabulary)
        self.output_vocab_size = len(self.output_vocabulary)

        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])

        self.input_token_index = dict([(char, i) for i, char in enumerate(self.input_vocabulary)])
        self.target_token_index = dict([(char, i) for i, char in enumerate(self.output_vocabulary)])

        reverse_input_char_index = dict((i, char) for char, i in self.input_token_index.items())
        reverse_target_char_index = dict((i, char) for char, i in self.target_token_index.items())

    def gen_one_hot_encoded_train_sets(self) -> None:
        """

        :return: None
        """
        self.encoder_input_data = np.zeros((len(self.input_texts), self.max_encoder_seq_length, self.input_vocab_size), dtype='float32')
        self.decoder_input_data = np.zeros((len(self.target_texts), self.max_decoder_seq_length, self.output_vocab_size), dtype='float32')
        self.decoder_target_data = np.zeros((len(self.target_texts), self.max_decoder_seq_length, self.output_vocab_size), dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            for t, char in enumerate(input_text):
                self.encoder_input_data[i, t, self.input_token_index[char]] = 1.

            for t, char in enumerate(target_text):
                self.decoder_input_data[i, t, self.target_token_index[char]] = 1.

                if t > 0:
                    self.decoder_target_data[i, t-1, self.target_token_index[char]] = 1

    def fit_gen_train_sets(self) -> dict:
        self.corpus_training()
        self.build_char_dict()
        self.gen_one_hot_encoded_train_sets()

        cache = {
            "input_vocab_size": self.input_vocab_size,
            "output_vocab_size": self.output_vocab_size,
            "encoder_input_data": self.encoder_input_data,
            "decoder_input_data": self.decoder_input_data,
            "decoder_target_data": self.decoder_target_data
        }

        return cache
