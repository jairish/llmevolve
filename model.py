import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, Input, Dropout
from tensorflow.keras.models import Model

# These constants must match what's used in train.py
VOCAB_SIZE = 20000 # Increased vocabulary size for larger dataset (adjust based on data)
EMBEDDING_DIM = 256 # Increased embedding dimension
LSTM_UNITS = 512    # Increased LSTM units
MAX_SEQUENCE_LENGTH = 100 # Increased max sentence length (adjust based on data)

def create_model():
    """Creates and returns a simple word-level LSTM sequence-to-sequence model."""
    # Encoder
    encoder_inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
    encoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(encoder_inputs) # mask_zero for padding
    encoder_embedding = Dropout(0.3)(encoder_embedding)
    encoder_lstm1 = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)
    encoder_outputs1, state_h1, state_c1 = encoder_lstm1(encoder_embedding)
    encoder_lstm1_dropout = Dropout(0.3)(encoder_outputs1)
    encoder_lstm2 = LSTM(LSTM_UNITS, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm2(encoder_lstm1_dropout)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
    decoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(decoder_inputs) # mask_zero for padding
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