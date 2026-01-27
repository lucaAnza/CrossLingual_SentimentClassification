from datasets import Value


# Filter the dataset + change labels and remove useless + tokenizzation
def preprocessing(db , tokenizer ,  task = 'Classification' , k = None, max_length = None):

    task_list = ['Classification' , 'Regression']

    # Reduce dataset size for faster experimentation 
    if(k):
        train_size = len(db["train"])
        test_size = len(db["test"])
        val_size = len(db["validation"])

        train_n = min(k, train_size)
        eval_n = min(30000, k // 6)
        test_n = min(eval_n, test_size)
        val_n = min(eval_n, val_size)

        db["train"] = db["train"].shuffle(seed=42).select(range(train_n))
        db["test"] = db["test"].shuffle(seed=42).select(range(test_n))
        db["validation"] = db["validation"].shuffle(seed=42).select(range(val_n))

    # Fix labels from # 1–5 → 0–4
    def adjust_label_classification(example):
        example['label'] = example['label'] - 1
        return example

    # Fix labels from # 1–5 → 0–1
    def adjust_label_regression(example):
        example['label'] = (example['label'] - 1) / 4  
        return example

    # Add ids column + mask column (there is for because examples is a batch [usually 64] )
    def preprocess_function(examples):
        titles = examples["review_title"]
        bodies = examples["review_body"]

        titles = [t if t is not None else "" for t in titles]  # Avoid Null titles
        bodies = [b if b is not None else "" for b in bodies]  # Avoid Null bodies

        # In this way the tokenizer insert a special token (such as [SEP]) between titles and bodies
        if max_length:
            return tokenizer(
                titles,
                bodies,
                truncation=True,
                max_length=max_length
            )
        return tokenizer(
            titles,
            bodies,
            truncation=True
        )

    # Tokenization
    db_tokenized = db.map(preprocess_function, batched=True)  # features : [ 'review_body' , 'review_id' , ... , label' , 'ids' , 'mask']

    # Rename columns and remove unnecessary ones
    db_tokenized = db_tokenized.rename_column("stars", "label")
    db_tokenized = db_tokenized.remove_columns(["Unnamed: 0", 'review_body' , 'review_id', 'product_id', 'reviewer_id', 'review_title', 'language', 'product_category'])  # Remove unnecessary index column
    if(task == 'Classification'):
        # Fix labels to start from 0
        db_tokenized = db_tokenized.map(adjust_label_classification)  # features : [ label' , 'ids' , 'mask']
    elif(task == 'Regression'): 
        db_tokenized = db_tokenized.cast_column("label", Value("float32"))  # regression needs continuous labels
        db_tokenized = db_tokenized.map(adjust_label_regression)  # features : [ label' , 'ids' , 'mask']
    else:
        print("Define the task from the following list : " , task_list)
        return None


    
    
    
    print("\n✅ Preprocessing completed. Db struct : " , db_tokenized)
    return db_tokenized
