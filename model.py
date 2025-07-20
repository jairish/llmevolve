import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, Input
from tensorflow.keras.models import Model

# These constants are now smaller to make the model simpler (less complex).
# They must still match what's used in train.py's TextVectorization layer.
VOCAB_SIZE = 5000 # Example: A more realistic vocabulary size for word-level
EMBEDDING_DIM = 64 # Reduced embedding dimension for simplicity
LSTM_UNITS = 128    # Reduced LSTM units for simplicity
MAX_SEQUENCE_LENGTH = 30 # Example: Longer max sentence length

def create_model():
    """Creates and returns a simple word-level LSTM sequence-to-sequence model."""
    # Encoder
    encoder_inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
    encoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(encoder_inputs) # mask_zero for padding
    encoder_lstm = LSTM(LSTM_UNITS, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
    decoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(decoder_inputs) # mask_zero for padding
    decoder_lstm = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))
    output = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], output)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'] # Accuracy here refers to token prediction accuracy
    )
    
    return model