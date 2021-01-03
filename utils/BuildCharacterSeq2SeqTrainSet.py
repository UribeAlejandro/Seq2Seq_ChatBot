import warnings
import numpy as np
from nlpia.loaders import get_data

warnings.filterwarnings('ignore')


class BuildCharacterSeq2SeqTrainSet:

    def __init__(self):
        self.input_texts, self.target_texts = [], []
        self.input_vocabulary, self.output_vocabulary = set(), set()
        self.input_vocab_size, self.output_vocab_size = 0, 0
        self.max_encoder_seq_length, self.max_decoder_seq_length = 0, 0
        self.input_token_index, self.target_token_index = dict(), dict()

        self.encoder_input_data, self.decoder_input_data, self.decoder_target_data = None, None, None

        self.reverse_input_char_index, self.reverse_target_char_index = dict(), dict()

        self.start_token = '\t'
        self.stop_token = '\n'

    def corpus_training(self) -> None:
        """
        Loads the corpus and generate the training sets.
        :return: None
        """

        df = get_data("moviedialog")

        # max_training_examples = min(25000, len(df) - 1)

        for input_text, target_text in zip(df.statement, df.reply):
            target_text = self.start_token + target_text + self.stop_token
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            for char in input_text:
                if char not in self.input_vocabulary:
                    self.input_vocabulary.add(char)
            for char in target_text:
                self.output_vocabulary.add(char)

        print(f"Number of samples: {len(self.input_texts)}")
        print(f"Number of unique Input Tokens: {len(self.input_vocabulary)}")
        print(f"Number of unique Output Tokens: {len(self.output_vocabulary)}")

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

        self.reverse_input_char_index = dict((i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict((i, char) for char, i in self.target_token_index.items())

        print(f"Maximum sequence length for Inputs: {self.max_encoder_seq_length}")
        print(f"Maximum sequence length for Outputs: {self.max_decoder_seq_length}")

    def gen_one_hot_encoded_train_sets(self) -> None:
        """

        :return: None
        """
        self.encoder_input_data = np.zeros((len(self.input_texts), self.max_encoder_seq_length, self.input_vocab_size),
                                           dtype='float32')
        self.decoder_input_data = np.zeros((len(self.input_texts), self.max_decoder_seq_length, self.output_vocab_size),
                                           dtype='float32')
        self.decoder_target_data = np.zeros(
            (len(self.input_texts), self.max_decoder_seq_length, self.output_vocab_size), dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            for t, char in enumerate(input_text):
                self.encoder_input_data[i, t, self.input_token_index[char]] = 1.

            for t, char in enumerate(target_text):
                self.decoder_input_data[i, t, self.target_token_index[char]] = 1.

                if t > 0:
                    self.decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.

    def fit_gen_train_sets(self) -> dict:
        """
        Loads the corpus, builds the character dictionary and generates the one hot encoded training set.

        :return: Dictionary with the following objects
        """
        self.corpus_training()
        self.build_char_dict()
        self.gen_one_hot_encoded_train_sets()

        cache = {
            "input_vocab_size": self.input_vocab_size,
            "output_vocab_size": self.output_vocab_size,
            "encoder_input_data": self.encoder_input_data,
            "decoder_input_data": self.decoder_input_data,
            "decoder_target_data": self.decoder_target_data,
            "reverse_input_char_index": self.reverse_input_char_index,
            "reverse_target_char_index": self.reverse_target_char_index,
            "max_encoder_seq_length": self.max_encoder_seq_length,
            "max_decoder_seq_length": self.max_decoder_seq_length,
            "input_token_index": self.input_token_index,
            "target_token_index": self.target_token_index,
            "stop_token": self.stop_token
        }

        return cache
