import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from model import create_model, VOCAB_SIZE, MAX_SEQUENCE_LENGTH # Import constants
import json
import os
import numpy as np
import re
import io
import requests # Added for downloading the dataset
import subprocess # Added for subprocess calls

# Special tokens
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SOS_TOKEN = '<SOS>' # Start of Sequence
EOS_TOKEN = '' # End of Sequence

def download_cornell_dialogs():
    """Downloads and extracts the Cornell Movie-Dialogs Corpus."""
    url = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
    zip_path = "cornell_movie_dialogs_corpus.zip"
    extract_path = "cornell_movie_dialogs_corpus"
    lines_filepath = os.path.join(extract_path, 'movie_lines.txt') # Define expected file path

    if not os.path.exists(extract_path):
        print(f"Downloading {url}...")
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status() # Raise an exception for bad status codes
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {zip_path} successfully.")

            print(f"Extracting {zip_path}...")
            # Use -o to overwrite without prompting, -d to specify directory
            subprocess.run(['unzip', '-o', zip_path, '-d', extract_path], check=True, capture_output=True, text=True)
            print(f"Extraction complete to {extract_path}.")

            # Verify if the expected file exists after extraction
            if not os.path.exists(lines_filepath):
                print(f"ERROR: Expected file '{lines_filepath}' not found after extraction.")
                # You might want to inspect the contents of extract_path here if it exists
                if os.path.exists(extract_path):
                     print(f"Contents of {extract_path}: {os.listdir(extract_path)}")
                return None # Indicate failure
            else:
                print(f"Verification successful: '{lines_filepath}' found.")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading dataset: {e}")
            return None
        except subprocess.CalledProcessError as e:
             print(f"Error extracting dataset: {e}")
             print(f"Extraction Stderr: {e.stderr}")
             return None
        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path) # Clean up the zip file
                print(f"Cleaned up zip file: {zip_path}")
    else:
        print(f"Extraction directory '{extract_path}' already exists. Skipping download/extraction.")
        # Verify if the expected file exists even if directory exists (e.g., partial previous run)
        if not os.path.exists(lines_filepath):
             print(f"Warning: Extraction directory exists but expected file '{lines_filepath}' not found.")
             # Consider re-downloading/extracting or reporting error depending on desired behavior
             # For now, just warn and return None
             return None
        else:
             print(f"Verification successful: '{lines_filepath}' found in existing directory.")


    return extract_path

def load_cornell_dialogs(corpus_path):
    """Loads conversational pairs from the Cornell Movie-Dialogs Corpus."""
    lines_filepath = os.path.join(corpus_path, 'movie_lines.txt')
    conv_filepath = os.path.join(corpus_path, 'movie_conversations.txt')

    # Check if files exist before trying to open
    if not os.path.exists(lines_filepath):
        print(f"ERROR: Lines file not found at {lines_filepath}")
        return None

    if not os.path.exists(conv_filepath):
        print(f"ERROR: Conversations file not found at {conv_filepath}")
        return None


    # Load lines
    id2line = {}
    with open(lines_filepath, errors='ignore') as f:
        for line in f:
            parts = line.strip().split(' +++$+++ ')
            if len(parts) == 5:
                id2line[parts[0]] = parts[4]

    # Load conversations and extract pairs
    qa_pairs = []
    with open(conv_filepath, errors='ignore') as f:
        for line in f:
            parts = line.strip().split(' +++$+++ ')
            if len(parts) == 4:
                line_ids = parts[3][1:-1].replace("'", "").split(", ")
                for i in range(len(line_ids) - 1):
                    # Get question and answer lines
                    question = id2line.get(line_ids[i], '')
                    answer = id2line.get(line_ids[i+1], '')

                    # Simple cleaning
                    question = re.sub(r'[^a-zA-Z0-9\s]', '', question).strip().lower()
                    answer = re.sub(r'[^a-zA-Z0-9\s]', '', answer).strip().lower()

                    if question and answer: # Only keep non-empty pairs
                        qa_pairs.append((question, answer))

    return qa_pairs

def prepare_data():
    """Downloads, loads, and preprocesses the Cornell dataset."""
    corpus_path = download_cornell_dialogs()
    if not corpus_path:
        print("Download or extraction failed.")
        return None, None, None, None # Return None if download/extraction failed

    qa_pairs = load_cornell_dialogs(corpus_path)

    if qa_pairs is None or not qa_pairs: # Check if load failed or no pairs found
        print("No conversational pairs loaded from the dataset.")
        return None, None, None, None

    print(f"Loaded {len(qa_pairs)} conversational pairs.")

    # Use a subset for faster iteration during development
    # In a real scenario, you'd use the full dataset or a large subset.
    # For demonstrating the flow, we'll use a small fraction.
    subset_size = min(10000, len(qa_pairs)) # Use up to 10000 pairs or the total number if smaller
    qa_pairs_subset = qa_pairs[:subset_size]
    print(f"Using a subset of {len(qa_pairs_subset)} pairs.")

    questions = [q for q, a in qa_pairs_subset]
    answers = [a for q, a in qa_pairs_subset]

    # Add SOS and EOS tokens to answers for sequence generation
    answers_input = [SOS_TOKEN + ' ' + a for a in answers]
    answers_target = [a + ' ' + EOS_TOKEN for a in answers]

    # Text Vectorization
    # The TextVectorization layer handles tokenization, lowercasing, and vocabulary creation.
    # It also pads/truncates sequences to MAX_SEQUENCE_LENGTH.

    # Create vectorizer for questions
    question_vectorizer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_sequence_length=MAX_SEQUENCE_LENGTH,
        standardize='lower_and_strip_punctuation',
        split='whitespace',
        # We don't need to explicitly add PAD or UNK here, TextVectorization handles them.
        # index=0 is reserved for padding by default.
    )
    question_vectorizer.adapt(questions)

    # Create vectorizer for answers
    # Include special tokens in the vocabulary explicitly
    answer_vectorizer = TextVectorization(
        max_tokens=VOCAB_SIZE, # Use the same vocabulary size
        output_sequence_length=MAX_SEQUENCE_LENGTH,
        standardize='lower_and_strip_punctuation',
        split='whitespace',
        vocabulary=[PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN] + sorted(list(set(" ".join(answers_input + answers_target).split())))[:VOCAB_SIZE-4]
    )
    answer_vectorizer.adapt(answers_input + answers_target)


    # Convert text data to sequences of integers
    encoder_input_data = question_vectorizer(questions).numpy()
    decoder_input_data = answer_vectorizer(answers_input).numpy()
    decoder_target_data = answer_vectorizer(answers_target).numpy()

    # Ensure shapes are consistent (should be handled by TextVectorization)
    # print("Encoder input shape:", encoder_input_data.shape)
    # print("Decoder input shape:", decoder_input_data.shape)
    # print("Decoder target shape:", decoder_target_data.shape)

    # Get vocabulary and inverse vocabulary for later use
    vocab = answer_vectorizer.get_vocabulary()
    int_to_char = {i: char for i, char in enumerate(vocab)} # Renaming for consistency with previous char model

    return (encoder_input_data, decoder_input_data, decoder_target_data), vocab, int_to_char

def train_and_evaluate():
    """Loads data, trains the model, and prints the evaluation result as JSON."""
    # 1. Prepare Data
    (encoder_input_data, decoder_input_data, decoder_target_data), vocab, int_to_char = prepare_data()

    if encoder_input_data is None:
        print("Data preparation failed. Skipping training.")
        result = {'loss': float('inf'), 'accuracy': 0.0, 'temp_model_path': ''}
        print(json.dumps(result))
        return # Exit if data prep failed


    # Split data (simple split for demonstration)
    num_samples = encoder_input_data.shape[0]
    if num_samples < 10: # Need at least some samples for split
        print(f"Not enough data ({num_samples} samples) to perform train/test split. Skipping training.")
        result = {'loss': float('inf'), 'accuracy': 0.0, 'temp_model_path': ''}
        print(json.dumps(result))
        return


    train_split = int(num_samples * 0.8)

    train_encoder_input = encoder_input_data[:train_split]
    train_decoder_input = decoder_input_data[:train_split]
    train_decoder_target = decoder_target_data[:train_split]

    test_split = num_samples - train_split
    if test_split < 1:
         print(f"Not enough data ({num_samples} samples) to create a test set. Skipping training.")
         result = {'loss': float('inf'), 'accuracy': 0.0, 'temp_model_path': ''}
         print(json.dumps(result))
         return


    test_encoder_input = encoder_input_data[train_split:]
    test_decoder_input = decoder_input_data[train_split:]
    test_decoder_target = decoder_target_data[train_split:]

    print(f"Train samples: {len(train_encoder_input)}, Test samples: {len(test_encoder_input)}")


    # 2. Create and Train Model
    model = create_model()

    print(f"Training model...")
    # Use a small subset for faster cycles for actual training
    # For a real chat model, you'd need a much larger dataset and more epochs
    history = model.fit(
        [train_encoder_input, train_decoder_input],
        train_decoder_target,
        epochs=5, # Reduced epochs for faster cycles
        batch_size=64, # Increased batch size for larger data
        validation_split=0.2,
        verbose=0 # Keep verbose 0 for subprocess
    )

    # Get the last validation loss from history
    val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else history.history['loss'][-1]
    val_accuracy = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else history.history['accuracy'][-1]

    # 3. Evaluate on test set
    print(f"Evaluating model...")
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