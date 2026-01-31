import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets
import numpy as np
import sys
import os

# Try to import train_test_split from sklearn, if it fails, use our own implementation
try:
    from sklearn.model_selection import train_test_split
    USE_SKLEARN = True
except ImportError as e:
    print(f"Warning: sklearn not available ({e}). Using custom train_test_split implementation.")
    USE_SKLEARN = False
    
    def train_test_split(df, test_size=0.2, random_state=42, stratify=None):
        """
        Custom train_test_split implementation that doesn't require sklearn.
        """
        np.random.seed(random_state)
        
        # Shuffle the dataframe
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Calculate split index
        n_samples = len(df_shuffled)
        n_test = int(n_samples * test_size)
        
        # If stratify is requested, try to maintain class distribution
        if stratify is not None and stratify in df.columns:
            # Group by the stratify column
            train_list = []
            test_list = []
            
            for label_value in df_shuffled[stratify].unique():
                label_df = df_shuffled[df_shuffled[stratify] == label_value]
                label_n_test = int(len(label_df) * test_size)
                
                if label_n_test > 0:
                    test_df = label_df.iloc[:label_n_test]
                    train_df = label_df.iloc[label_n_test:]
                else:
                    test_df = pd.DataFrame()
                    train_df = label_df
                
                train_list.append(train_df)
                test_list.append(test_df)
            
            train_df = pd.concat(train_list, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
            test_df = pd.concat(test_list, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
        else:
            # Simple random split
            test_df = df_shuffled.iloc[:n_test]
            train_df = df_shuffled.iloc[n_test:]
        
        return train_df, test_df

# Configuration: Choose 'multilingual' or 'single_language'
# This can be set via command line argument or changed here
MODE = None
if len(sys.argv) > 1:
    MODE = sys.argv[1].lower()
else:
    # Default behavior: can be set here or will prompt
    MODE = None  # Will be set based on user choice or default to 'multilingual'

def preprocess_data(mode='multilingual'):
    """
    Preprocess the Amazon Reviews dataset.
    
    Args:
          mode: 'multilingual' for 200k reviews across languages, 
              'single_language' for 200k reviews from one language
    """
    # Load the dataset from CSV files (train.csv, validation.csv, test.csv)
    print("Loading Amazon Reviews Multi dataset...")
    df = None
    
    # Try to load from train.csv (main dataset file)
    train_csv_paths = [
        "./train.csv",
        "/data01/aiello/train.csv",
        "./data/train.csv",
        "train.csv"
    ]
    
    for csv_path in train_csv_paths:
        if os.path.exists(csv_path):
            print(f"Loading dataset from CSV file: {csv_path}")
            try:
                # Read in chunks if file is very large
                print("Reading CSV file (this may take a while for large files)...")
                df = pd.read_csv(csv_path, low_memory=False)
                print(f"Successfully loaded {len(df)} rows from CSV")
                break
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                continue
    
    # If train.csv not found, try single CSV file (old format)
    if df is None:
        single_csv_paths = [
            "/data01/aiello/amazon-reviews-multi.csv",
            "./amazon-reviews-multi.csv",
            "./data/amazon-reviews-multi.csv",
            "amazon-reviews-multi.csv"
        ]
        
        for csv_path in single_csv_paths:
            if os.path.exists(csv_path):
                print(f"Loading dataset from CSV file: {csv_path}")
                try:
                    df = pd.read_csv(csv_path, low_memory=False)
                    print(f"Successfully loaded {len(df)} rows from CSV")
                    break
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
                    continue
    
    # If CSV not found, try Hugging Face (may not work as dataset is defunct)
    if df is None:
        print("CSV files not found. Trying Hugging Face dataset...")
        try:
            full_dataset = load_dataset("amazon_reviews_multi", "all_languages", split="train")
            df = full_dataset.to_pandas()
        except Exception as e:
            print(f"Error loading from Hugging Face: {e}")
            print("\n" + "="*60)
            print("ERROR: Dataset not found!")
            print("="*60)
            print("Please ensure train.csv is in the current directory or one of these locations:")
            for path in train_csv_paths:
                print(f"    - {path}")
            raise FileNotFoundError("Amazon Reviews Multi dataset CSV file (train.csv) not found.")

    print(f"Total dataset size: {len(df)}")
    if 'language' in df.columns:
        print(f"Available languages: {df['language'].unique()}")
    else:
        print("Warning: 'language' column not found. Assuming all reviews are in the same language.")
        df['language'] = 'unknown'  # Add a default language column

    # Map labels function
    def map_labels(stars):
        if stars <= 2:
            return 0  # "Bad"
        elif stars == 3:
            return 1  # "Neutral"
        else:
            return 2  # "Good"

    # Apply label mapping
    df['label'] = df['stars'].apply(map_labels)

    target_size = 200000

    if mode == 'multilingual':
        # Get 200k reviews split across different languages
        print(f"\nCreating {target_size} reviews split across different languages...")
        multilang_df = pd.DataFrame()
        languages = df['language'].unique()
        num_languages = len(languages)

        if num_languages == 0:
            raise ValueError("No languages found in dataset")

        samples_per_lang = target_size // num_languages
        remaining = target_size % num_languages

        for i, lang in enumerate(languages):
            lang_df = df[df['language'] == lang]
            sample_size = samples_per_lang + (1 if i < remaining else 0)
            if len(lang_df) >= sample_size:
                sampled = lang_df.sample(n=sample_size, random_state=42)
                multilang_df = pd.concat([multilang_df, sampled], ignore_index=True)
            else:
                # If not enough samples, take all available
                print(f"Warning: Only {len(lang_df)} samples available for {lang}, requested {sample_size}")
                multilang_df = pd.concat([multilang_df, lang_df], ignore_index=True)

        print(f"Multilanguage dataset size: {len(multilang_df)}")
        language_info = {
            "mode": "multilingual",
            "languages": languages.tolist() if hasattr(languages, "tolist") else list(languages),
            "samples_per_language": {
                lang: int(len(multilang_df[multilang_df['language'] == lang])) for lang in languages
            },
        }
        combined_df = multilang_df
        
    elif mode == 'single_language':
        # Get 200k reviews from the same language (using the most common language, typically English)
        print(f"\nCreating {target_size} reviews from same language...")
        most_common_lang = df['language'].mode()[0]
        print(f"Using language: {most_common_lang}")
        same_lang_df = df[df['language'] == most_common_lang]
        if len(same_lang_df) >= target_size:
            same_lang_df = same_lang_df.sample(n=target_size, random_state=42)
        else:
            print(f"Warning: Only {len(same_lang_df)} reviews available for {most_common_lang}, using all available")

        print(f"Same language dataset size: {len(same_lang_df)}")
        language_info = {
            "mode": "single_language",
            "language": most_common_lang,
            "samples": int(len(same_lang_df)),
        }
        combined_df = same_lang_df
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'multilingual' or 'single_language'")

    # Remove duplicates if any
    if 'review_id' in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=['review_id'], keep='first')
    print(f"Final dataset size: {len(combined_df)}")

    # Split into training and validation
    # Try to use validation.csv if available, otherwise split from train data
    print("\nPreparing train and validation sets...")
    
    val_df = None
    # Check if validation.csv exists and use it
    validation_csv_paths = ["./validation.csv", "/data01/aiello/validation.csv", "./data/validation.csv", "validation.csv"]
    for val_path in validation_csv_paths:
        if os.path.exists(val_path):
            print(f"Found validation.csv at {val_path}, using it for validation set...")
            try:
                val_df_raw = pd.read_csv(val_path, low_memory=False)
                # Apply same label mapping
                val_df_raw['label'] = val_df_raw['stars'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))
                # Filter validation to match the same language distribution if needed
                if mode == 'single_language' and 'language' in val_df_raw.columns:
                    most_common_lang = combined_df['language'].mode()[0] if 'language' in combined_df.columns else None
                    if most_common_lang:
                        val_df_raw = val_df_raw[val_df_raw['language'] == most_common_lang]
                        print(f"Filtered validation set to language: {most_common_lang}")
                val_df = val_df_raw
                print(f"Loaded {len(val_df)} validation samples from validation.csv")
                break
            except Exception as e:
                print(f"Error reading validation.csv: {e}, will split from train data instead")
                val_df = None
                break
    
    # If validation.csv not used, split from combined_df
    if val_df is None:
        print("Splitting train data into train and validation sets...")
        # Use stratify only if we have enough samples per class
        try:
            train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42, stratify=combined_df['label'])
        except ValueError as e:
            # If stratification fails (not enough samples per class), split without stratification
            print(f"Warning: Stratified split failed ({e}), using random split instead")
            train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42)
    else:
        # Use combined_df as train_df when validation.csv is used
        train_df = combined_df

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Label distribution in training: {train_df['label'].value_counts().to_dict()}")
    print(f"Label distribution in validation: {val_df['label'].value_counts().to_dict()}")

    # Convert to Hugging Face Dataset format
    columns_to_keep = ['review_title', 'review_body', 'label']
    if 'language' in train_df.columns:
        columns_to_keep.append('language')
    
    train_dataset = Dataset.from_pandas(train_df[columns_to_keep])
    val_dataset = Dataset.from_pandas(val_df[columns_to_keep])

    # Determine model name (tokenizer will be loaded later in training to avoid PIL import issues)
    if mode == 'multilingual':
        model_name = "distilbert-base-multilingual-cased"
    else:
        # For single language, we can still use multilingual model or switch to English-only
        model_name = "distilbert-base-multilingual-cased"  # Keep multilingual for consistency
    
    print(f"\nModel name: {model_name}")
    print("Note: Tokenization will be done during training to avoid library compatibility issues.")

    # Keep datasets with text columns for now (tokenization will happen in training.py)
    print("\nPreprocessing complete!")
    print(f"Train dataset: {train_dataset}")
    print(f"Validation dataset: {val_dataset}")
    print("Datasets contain text columns and will be tokenized during training.")
    
    return train_dataset, val_dataset, model_name, mode, language_info

# Run preprocessing if called directly or if MODE is set
if __name__ == "__main__" or MODE is not None:
    if MODE is None:
        MODE = 'multilingual'  # Default
    
    train_dataset, val_dataset, model_name, mode, language_info = preprocess_data(MODE)
else:
    # If imported, set defaults (will be overridden when preprocess_data is called)
    train_dataset = None
    val_dataset = None
    model_name = "distilbert-base-multilingual-cased"
    mode = None
    language_info = {}
