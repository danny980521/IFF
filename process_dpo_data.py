from datasets import load_dataset, DatasetDict, concatenate_datasets


def get_train_test_dataset(dataset_path=None, dataset_files=None, extension="json"):
    """
    Load the train and test datasets from the specified path and name.
    """
    ds = load_dataset(
        extension,
        data_dir=dataset_path,
        data_files=dataset_files,
        cache_dir="./cache",
        num_proc=32,
        split="train",
    )

    if len(ds) > 10_000:
        ds = ds.shuffle(seed=42).select(range(10_000))

    split_dataset = ds.train_test_split(
        test_size=0.01, shuffle=True, seed=42
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # change column name "question" to "prompt" if it exists
    # https://github.com/huggingface/trl/blob/5a0cebc7869b0435b4e916a104e0b7b14a7f03f3/trl/data_utils.py#L118-L138
    if "question" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("question", "prompt")
        eval_dataset = eval_dataset.rename_column("question", "prompt")

    if "chosen_response" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("chosen_response", "chosen")
        eval_dataset = eval_dataset.rename_column("chosen_response", "chosen")
    if "rejected_response" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("rejected_response", "rejected")
        eval_dataset = eval_dataset.rename_column("rejected_response", "rejected")

    # leave only column names: ['prompt', 'chosen', 'rejected']
    train_dataset = train_dataset.remove_columns(
        [col for col in train_dataset.column_names if col not in ["prompt", "chosen", "rejected"]]
    )
    eval_dataset = eval_dataset.remove_columns(
        [col for col in eval_dataset.column_names if col not in ["prompt", "chosen", "rejected"]]
    )

    return train_dataset, eval_dataset


if __name__ == "__main__":
    total_train_ds = []
    total_test_ds = []
    helpsteer2_dpo_path = "/data/ib-a100-cluster-a-pri-lmt_967/users/wavy/workspace/study/data/HelpSteer2-DPO/data"
    train_ds, test_ds = get_train_test_dataset(
        dataset_path=helpsteer2_dpo_path,
        extension="parquet",
    )
    total_train_ds.append(train_ds)
    total_test_ds.append(test_ds)
    human_like_dpo_path = "/data/ib-a100-cluster-a-pri-lmt_967/users/wavy/workspace/study/data/Human-Like-DPO-Dataset/data.json"
    train_ds, test_ds = get_train_test_dataset(
        dataset_files=human_like_dpo_path,
        extension="json",
    )
    total_train_ds.append(train_ds)
    total_test_ds.append(test_ds)
    ling_coder_dpo = "/data/ib-a100-cluster-a-pri-lmt_967/users/wavy/workspace/study/data/Ling-Coder-DPO/data"
    train_ds, test_ds = get_train_test_dataset(
        dataset_path=ling_coder_dpo,
        extension="parquet",
    )
    total_train_ds.append(train_ds)
    total_test_ds.append(test_ds)

    # concatenate all train datasets and shuffle
    train_dataset = concatenate_datasets(total_train_ds)
    train_dataset = train_dataset.shuffle(seed=42)
    # concatenate all test datasets and shuffle
    test_dataset = concatenate_datasets(total_test_ds)
    test_dataset = test_dataset.shuffle(seed=42)

    # merge train / test
    merged_dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    # save to disk
    merged_dataset.save_to_disk("/data/ib-a100-cluster-a-pri-lmt_967/users/wavy/workspace/study/alignment/IFF/dpo-dataset")

