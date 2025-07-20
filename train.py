import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from model import create_model, VOCAB_SIZE, MAX_SEQUENCE_LENGTH # Import constants
import json
import os
import numpy as np
import re

# Special tokens
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SOS_TOKEN = '<SOS>' # Start of Sequence
EOS_TOKEN = '<EOS>' # End of Sequence

def prepare_data():
    """Generates a simple synthetic word-level dataset and tokenizes it.
    This demonstrates how TextVectorization handles data preparation for training.
    For a truly large dataset, you would load data from files here.
    """
    # A slightly expanded set of conversational pairs for demonstration.
    # In a real scenario, this would be loaded from a large file.
    qa_pairs = [
        ("hi there", "hello how are you"),
        ("how are you doing", "i am doing fine thank you"),
        ("what is your name", "my name is chatbot"),
        ("goodbye for now", "see you later goodbye"),
        ("tell me a funny joke", "why don't scientists trust atoms because they make up everything"),
        ("what time is it", "it's always time to learn"),
        ("how old are you", "i am a timeless AI"),
        ("where are you from", "i exist in the digital realm"),
        ("who created you", "i was created by a large language model"),
        ("thank you very much", "you are most welcome"),
        ("what is the weather like", "i cannot tell you the weather"),
        ("can you help me", "i will try my best to assist you"),
        ("what do you like to do", "i enjoy processing information"),
        ("do you have feelings", "as an AI I do not have feelings"),
        ("what is the meaning of life", "the meaning of life is a profound question")
    ]

    input_texts = [pair[0] for pair in qa_pairs]
    target_texts = [pair[1] for pair in qa_pairs]

    # Add SOS and EOS tokens to target sequences for the decoder
    decoder_input_texts = [SOS_TOKEN + ' ' + text for text in target_texts]
    decoder_target_texts = [text + ' ' + EOS_TOKEN for text in target_texts]

    # Initialize TextVectorization layer
    # This layer handles lowercasing, punctuation stripping, splitting,
    # and converting text to integer sequences based on a learned vocabulary.
    text_vectorizer = TextVectorization(
        max_tokens=VOCAB_SIZE, # Max vocabulary size, from model.py
        output_mode='int',
        output_sequence_length=MAX_SEQUENCE_LENGTH, # Max sequence length, from model.py
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        # Ensure special tokens are at the start of the vocabulary
        vocabulary=[PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN] 
    )

    # Adapt the vectorizer to the combined input and target texts.
    # This builds the vocabulary based on the provided data.
    all_sentences = input_texts + target_texts + decoder_input_texts + decoder_target_texts
    text_vectorizer.adapt(all_sentences)

    # Get the vocabulary and create inverse mapping for debugging/inference
    vocab = text_vectorizer.get_vocabulary()
    int_to_word = {idx: word for idx, word in enumerate(vocab)}
    word_to_int = {word: idx for idx, word in enumerate(vocab)}

    # Critical check: Ensure model.py's VOCAB_SIZE is large enough for the actual data's vocabulary
    if len(vocab) > VOCAB_SIZE:
        print(f"WARNING: Actual vocabulary size ({len(vocab)}) exceeds VOCAB_SIZE in model.py ({VOCAB_SIZE}). "
              "Please increase VOCAB_SIZE in model.py to avoid data loss.")

    # Convert texts to sequences of integers using the adapted vectorizer
    encoder_input_data = text_vectorizer(input_texts).numpy()
    decoder_input_data = text_vectorizer(decoder_input_texts).numpy()
    decoder_target_data = text_vectorizer(decoder_target_texts).numpy()

    return (encoder_input_data, decoder_input_data, decoder_target_data), word_to_int, int_to_word

def train_and_evaluate():
    """Loads data, trains the model, and prints the evaluation result as JSON."""
    # 1. Prepare Data using the enhanced prepare_data function
    (encoder_input_data, decoder_input_data, decoder_target_data), word_to_int, int_to_word = prepare_data()

    # Split data (simple split for demonstration)
    num_samples = encoder_input_data.shape[0]
    train_split = int(num_samples * 0.8)

    train_encoder_input = encoder_input_data[:train_split]
    train_decoder_input = decoder_input_data[:train_split]
    train_decoder_target = decoder_target_data[:train_split]

    test_encoder_input = encoder_input_data[train_split:]
    test_decoder_input = decoder_input_data[train_split:]
    test_decoder_target = decoder_target_data[train_split:]

    # 2. Create and Train Model
    model = create_model()
    
    print(f"Training model with {len(train_encoder_input)} samples...")
    # Increased epochs and batch size slightly for better demonstration with word-level data
    history = model.fit(
        [train_encoder_input, train_decoder_input],
        train_decoder_target,
        epochs=20, # Increased epochs for text
        batch_size=4, # Small batch size for small dataset
        validation_split=0.2,
        verbose=0 # Keep verbose 0 for subprocess
    )
    
    # Get the last validation loss from history
    val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else history.history['loss'][-1]
    val_accuracy = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else history.history['accuracy'][-1]

    # 3. Evaluate on test set
    print(f"Evaluating model with {len(test_encoder_input)} samples...")
    loss, accuracy = model.evaluate(
        [test_encoder_input, test_decoder_input],
        test_decoder_target,
        verbose=0
    )
    
    # 4. Save the current model temporarily
    model_save_path = 'temp_model.keras'
    model.save(model_save_path)
    
    # 5. Output results as a JSON string for the orchestrator to capture
    # We prioritize loss for self-improvement in chat models
    result = {'loss': loss, 'accuracy': accuracy, 'temp_model_path': model_save_path}
    print(json.dumps(result))

if __name__ == '__main__':
    train_and_evaluate()