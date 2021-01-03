import warnings
from utils.ChatterBot import ChatterBot
from utils.TrainSeq2SeqChatBot import TrainSeq2SeqChatBot
from utils.BuildCharacterSeq2SeqTrainSet import BuildCharacterSeq2SeqTrainSet

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    build_seq2seq = BuildCharacterSeq2SeqTrainSet()
    build_seq2seq_cache = build_seq2seq.fit_gen_train_sets()

    train_seq2seq = TrainSeq2SeqChatBot(build_seq2seq.input_vocab_size, build_seq2seq.output_vocab_size,
                                        build_seq2seq.encoder_input_data, build_seq2seq.decoder_input_data,
                                        build_seq2seq.decoder_target_data, batch_size=32, epochs=1)
    models = train_seq2seq.train_assemble()

    chatterbot = ChatterBot(models, build_seq2seq.input_vocab_size, build_seq2seq.output_vocab_size,
                            build_seq2seq.target_token_index, build_seq2seq.stop_token,
                            build_seq2seq.reverse_target_char_index,
                            build_seq2seq.max_encoder_seq_length, build_seq2seq.max_decoder_seq_length,
                            build_seq2seq.input_token_index)
    chatterbot.start_chat()
