import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, Input, Dropout
from tensorflow.keras.models import Model

# These constants must match what's used in train.py
VOCAB_SIZE = 30 # Example: adjust based on your character set
EMBEDDING_DIM = 64
LSTM_UNITS = 128
MAX_SEQUENCE_LENGTH = 20 # Example: adjust based on your max sentence length

def create_model():
    """Creates and returns a simple character-level LSTM sequence-to-sequence model."""
    # Encoder
    encoder_inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
    encoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(encoder_inputs)
    encoder_embedding = Dropout(0.3)(encoder_embedding)
    encoder_lstm = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_lstm2 = LSTM(LSTM_UNITS, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm2(encoder_outputs)


    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
    decoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(decoder_inputs)
    decoder_embedding = Dropout(0.3)(decoder_embedding)
    decoder_lstm = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_outputs = Dropout(0.3)(decoder_outputs)
    decoder_dense = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))
    output = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], output)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'] # Accuracy here refers to token prediction accuracy
    )

    return model