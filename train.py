import tensorflow as tf
from model import create_model, VOCAB_SIZE, MAX_SEQUENCE_LENGTH # Import constants
import json
import os
import numpy as np

def prepare_data():
    """Generates a simple synthetic character-level dataset."""
    # Simple conversational pairs
    qa_pairs = [
        ("hi", "hello"),
        ("how are you", "i am fine"),
        ("what is your name", "i am a bot"),
        ("bye", "goodbye"),
        ("tell me a joke", "why did the scarecrow win an award because he was outstanding in his field"),
        ("what time is it", "it's time to learn"),
        ("how old are you", "i am ageless"),
        ("where are you from", "i live in the cloud"),
        ("who created you", "i was created by google"),
        ("thank you", "you're welcome")
    ]

    all_chars = sorted(list(set("".join([q + a for q, a in qa_pairs]))))
    char_to_int = {char: i for i, char in enumerate(all_chars)}
    int_to_char = {i: char for i, char in enumerate(all_chars)}
    
    # Add padding and OOV (Out-Of-Vocabulary) tokens if necessary
    # For simplicity, we'll assume all chars are in vocab and pad with 0
    if '' not in char_to_int:
        char_to_int['<PAD>'] = 0
        int_to_char[0] = '<PAD>'
        all_chars.insert(0, '<PAD>')
    
    # Update VOCAB_SIZE to match actual characters + padding
    # This is a critical point for consistency between model.py and train.py
    # In a real scenario, VOCAB_SIZE would be passed or derived from a shared tokenizer
    # For this self-upgrading demo, we'll ensure it's consistent.
    # The model.py will use a hardcoded VOCAB_SIZE, so we need to ensure our data fits.
    # For now, let's ensure our data's vocab size is <= model's VOCAB_SIZE
    # And if model.py's VOCAB_SIZE is larger, it just means some embeddings are unused.

    # Ensure MAX_SEQUENCE_LENGTH is sufficient
    # MAX_SEQUENCE_LENGTH is also hardcoded in model.py
    # We need to ensure our padding aligns with it.

    encoder_input_data = np.zeros((len(qa_pairs), MAX_SEQUENCE_LENGTH), dtype='int32')
    decoder_input_data = np.zeros((len(qa_pairs), MAX_SEQUENCE_LENGTH), dtype='int32')
    decoder_target_data = np.zeros((len(qa_pairs), MAX_SEQUENCE_LENGTH), dtype='int32')

    for i, (input_text, target_text) in enumerate(qa_pairs):
        # Encoder input
        for t, char in enumerate(input_text):
            if t < MAX_SEQUENCE_LENGTH:
                encoder_input_data[i, t] = char_to_int.get(char, 0) # Use 0 for unknown/padding

        # Decoder input (shifted by one, includes start token if applicable, but here just shifted target)
        # For character-level, we just shift the target.
        # Start token is implicitly handled by the first char of target_text
        for t, char in enumerate(target_text):
            if t < MAX_SEQUENCE_LENGTH:
                decoder_input_data[i, t] = char_to_int.get(char, 0)

        # Decoder target (one-hot encoded for each time step)
        for t, char in enumerate(target_text):
            if t < MAX_SEQUENCE_LENGTH:
                decoder_target_data[i, t] = char_to_int.get(char, 0) # No +1 for next char, it's the char itself

    return (encoder_input_data, decoder_input_data, decoder_target_data), char_to_int, int_to_char

def train_and_evaluate():
    """Loads data, trains the model, and prints the evaluation result as JSON."""
    # 1. Prepare Data
    (encoder_input_data, decoder_input_data, decoder_target_data), char_to_int, int_to_char = prepare_data()

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
    # Use a small subset for faster cycles for actual training
    # For a real chat model, you'd need a much larger dataset and more epochs
    history = model.fit(
        [train_encoder_input, train_decoder_input],
        train_decoder_target,
        epochs=10, # Increased epochs for text
        batch_size=2, # Small batch size for small dataset
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