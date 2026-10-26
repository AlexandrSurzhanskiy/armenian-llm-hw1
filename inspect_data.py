from paths import DATA_DIR, RAW_DIR
from pathlib import Path
import pyarrow.parquet as pq

data_dir = (RAW_DIR / "data")

for split in ["train", "validation", "test_iid", "test_tail"]:
    files = sorted((data_dir / split).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Нет файлов для {split}")

    n_rows = sum(pq.read_metadata(path).num_rows for path in files)
    print(f"{split}: {n_rows:,} документов, {len(files)} файлов")

train_file = sorted((data_dir / "train").glob("*.parquet"))[0]

with pq.ParquetFile(train_file) as parquet:
    print("\nПоля:", parquet.schema_arrow.names)

    batch = next(
        parquet.iter_batches(
            batch_size=3,
            columns=["text", "source", "url"],
        )
    )

for i, row in enumerate(batch.to_pylist(), start=1):
    print(f"\nПример {i}")
    print("Источник:", row["source"])
    print("Ссылка:", row["url"])
    print("Текст:")
    print(row["text"][:1000])
