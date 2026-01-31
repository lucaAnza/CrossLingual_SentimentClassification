from datasets import load_dataset
import os

# Download and load the Amazon Reviews Multi dataset from Hugging Face
# This dataset contains reviews in multiple languages
print("Loading Amazon Reviews Multi dataset...")

try:
    # Try loading the dataset - it may be available under different names
    dataset = load_dataset("amazon_reviews_multi", "all_languages", split="train")
    val_dataset = load_dataset("amazon_reviews_multi", "all_languages", split="validation")
    test_dataset = load_dataset("amazon_reviews_multi", "all_languages", split="test")
except Exception as e:
    print(f"Error loading 'amazon_reviews_multi': {e}")
    print("Trying alternative dataset sources...")
    try:
        # Alternative: try loading without language specification
        dataset = load_dataset("amazon_reviews_multi", split="train")
        val_dataset = load_dataset("amazon_reviews_multi", split="validation")
        test_dataset = load_dataset("amazon_reviews_multi", split="test")
    except Exception as e2:
        print(f"Error with alternative loading: {e2}")
        print("Please check if the dataset is available or use a different source.")
        raise

print(f"Train dataset size: {len(dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Check available languages
if 'language' in dataset.features:
    languages = set(dataset['language']) if hasattr(dataset, '__getitem__') else set()
    print(f"Available languages: {languages}")
else:
    print("Warning: 'language' field not found in dataset")

# Save dataset info
print("\nDataset loaded successfully!")
print("Dataset features:", dataset.features)
