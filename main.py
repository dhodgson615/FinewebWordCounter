import csv
import signal
from collections import Counter
from logging import (INFO, Formatter, StreamHandler, basicConfig, error,
                     getLogger, info, warning)
from pathlib import Path
from queue import Empty, Queue
from sqlite3 import connect
from sys import stdout
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import sleep, time

import pyarrow.compute
from huggingface_hub import HfApi, hf_hub_download
from pyarrow.parquet import ParquetFile

DATASET_REPO = "HuggingFaceFW/fineweb"
DATA_DIR = Path("data")
DB_FILE = DATA_DIR / "fineweb_counts.db"
LOG_FILE = "fineweb_processor.log"
FINAL_OUTPUT_FILE = "final_word_counts.csv"

PARQUET_BATCH_SIZE = 10_000  # rows per PyArrow batch
MAX_MEMORY_WORDS = 2_000_000  # flush local Counter to temp table when reached

SHUTDOWN_REQUESTED = False


def signal_handler(sig, frame):
    global SHUTDOWN_REQUESTED

    info(
        f"Signal {sig} received. Completing current file transaction, then exiting..."
    )

    SHUTDOWN_REQUESTED = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

basicConfig(
    filename=LOG_FILE,
    level=INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = StreamHandler(stdout)
console.setFormatter(Formatter("%(asctime)s - %(levelname)s - %(message)s"))
getLogger().addHandler(console)


def setup_database():
    """Create tables once. Crash‑safe WAL mode, temp tables in memory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect(DB_FILE)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA cache_size = -131072;")  # 128 MB page cache
    conn.execute("PRAGMA temp_store = MEMORY;")  # temp tables stay in RAM

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS word_counts (
                                                            word TEXT PRIMARY KEY,
                                                            count INTEGER NOT NULL
                 ) WITHOUT ROWID;
                 """)

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS processed_files (
                                                                filename TEXT PRIMARY KEY
                 );
                 """)
    conn.commit()
    return conn


def get_processed_files(conn):
    return {
        row[0] for row in conn.execute("SELECT filename FROM processed_files")
    }


def robust_download(filename, target_dir):
    """Download with retries & backoff. Returns file path or None if shutdown."""
    attempts = 0

    while not SHUTDOWN_REQUESTED:
        try:
            return hf_hub_download(
                repo_id=DATASET_REPO,
                filename=filename,
                repo_type="dataset",
                cache_dir=target_dir,
                local_dir_use_symlinks=False,  # Keep files isolated directly inside SSD temp directory
            )

        except Exception as e:
            attempts += 1
            sleep_time = 5 if attempts <= 3 else 900

            warning(
                f"Download error {filename}: {e}. Retrying in {sleep_time}s"
            )

            for _ in range(sleep_time):
                if SHUTDOWN_REQUESTED:
                    return None

                sleep(1)

    return None


def download_worker(file_list, download_queue):
    for filename in file_list:
        if SHUTDOWN_REQUESTED:
            break

        with TemporaryDirectory(dir=DATA_DIR, prefix="fw_temp_") as tmpdir:
            try:
                filepath = robust_download(filename, tmpdir)

            except KeyboardInterrupt:
                break  # silently exit the worker

            if filepath is None:
                break

            done_event = Event()
            download_queue.put((filename, filepath, done_event))
            done_event.wait()

    download_queue.put((None, None, None))


def _flush_counter_to_temp(conn, counter):
    """Insert (word, count) rows into the temp table (no conflict resolution)."""
    if not counter:
        return

    conn.executemany(
        "INSERT INTO file_counts (word, count) VALUES (?, ?);",
        counter.items(),
    )

    counter.clear()


def process_one_file(filepath, fname, conn):
    """
    Processes a single Parquet file using PyArrow's C++ engine.
    All counts AND file tracking are accumulated within a SINGLE transaction.
    """
    try:
        conn.execute("BEGIN TRANSACTION;")

        # Safely handle temp table lifecycle over the connection lifetime
        conn.execute("""
                     CREATE TEMP TABLE IF NOT EXISTS file_counts (
                                                                     word TEXT,
                                                                     count INTEGER
                     );
                     """)
        conn.execute("DELETE FROM file_counts;")

        pf = ParquetFile(filepath)
        local_counter = Counter()  # accumulates counts for a few batches

        for batch in pf.iter_batches(batch_size=PARQUET_BATCH_SIZE):
            if SHUTDOWN_REQUESTED:
                info("Shutdown requested – rolling back current file.")
                conn.execute("ROLLBACK;")
                return False

            text_array = batch.column("text")
            text_array = pyarrow.compute.fill_null(text_array, "")
            text_array = pyarrow.compute.utf8_lower(text_array)
            word_lists = pyarrow.compute.utf8_split_whitespace(text_array)
            flat_words = pyarrow.compute.list_flatten(word_lists)
            batch_counts = pyarrow.compute.value_counts(flat_words)

            # High-speed native C++ extraction structure
            for record in batch_counts.to_pylist():
                word = record["values"]
                cnt = record["counts"]

                if word:
                    local_counter[word] += cnt

            if len(local_counter) >= MAX_MEMORY_WORDS:
                _flush_counter_to_temp(conn, local_counter)

        _flush_counter_to_temp(conn, local_counter)

        conn.execute("""
                     INSERT INTO word_counts (word, count)
                     SELECT word, SUM(count) FROM file_counts GROUP BY word
                     ON CONFLICT(word) DO UPDATE SET count = word_counts.count + excluded.count;
                     """)

        # Co-locate file logging inside the same transaction to guarantee exact-once atomicity
        conn.execute(
            "INSERT OR REPLACE INTO processed_files (filename) VALUES (?);",
            (fname,),
        )

        conn.commit()
        return True

    except Exception as e:
        error(f"Error processing file: {e}. Rolling back.")

        try:
            conn.execute("ROLLBACK;")

        except Exception:
            pass

        return False


def export_final_counts(conn):
    info(f"Writing final output to {FINAL_OUTPUT_FILE} ...")

    with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "count"])

        for row in conn.execute(
            "SELECT word, count FROM word_counts ORDER BY count DESC"
        ):
            writer.writerow(row)

    info("Export complete.")


def main():
    info("Starting Fineweb word counter (C++ accelerated, atomic, resumable)")
    conn = setup_database()

    try:
        api = HfApi()
        all_files = api.list_repo_files(DATASET_REPO, repo_type="dataset")

        parquet_files = sorted(
            [f for f in all_files if f.endswith(".parquet")]
        )

    except Exception as e:
        error(f"Failed to fetch file list: {e}")
        conn.close()
        return

    processed = get_processed_files(conn)
    to_process = [f for f in parquet_files if f not in processed]
    total_remaining = len(to_process)

    info(
        f"Total files: {len(parquet_files)}, processed: {len(processed)}, remaining: {total_remaining}"
    )

    if not to_process:
        info("All files already processed. Exporting final counts.")
        export_final_counts(conn)
        conn.close()
        return

    download_queue = Queue(maxsize=1)

    downloader = Thread(
        target=download_worker,
        args=(to_process, download_queue),
        daemon=True,
    )

    downloader.start()
    start_time = time()
    files_done_this_run = 0

    while not SHUTDOWN_REQUESTED:
        try:
            fname, fpath, done_event = download_queue.get(timeout=1)

        except Empty:
            continue

        if fname is None:
            break

        info(f"Processing {fname} ...")
        # Pass fname directly into processing transaction logic
        success = process_one_file(fpath, fname, conn)

        if success:
            files_done_this_run += 1
            elapsed = time() - start_time
            avg_time = elapsed / files_done_this_run

            eta_hours = (
                avg_time * (total_remaining - files_done_this_run)
            ) / 3600

            info(
                f"Finished {fname} ({files_done_this_run}/{total_remaining}). "
                f"Avg: {avg_time:.1f}s, ETA: {eta_hours:.2f}h"
            )

        else:
            if SHUTDOWN_REQUESTED:
                done_event.set()
                break

            warning(f"Failed to process {fname} – will retry next run.")
            sleep(10)

        # Signal downloader to delete the temp directory
        done_event.set()

    if SHUTDOWN_REQUESTED:
        info("Waiting for downloader thread to exit...")
        downloader.join(timeout=5)

    else:
        if files_done_this_run == total_remaining:
            info("All files processed. Generating final output.")
            export_final_counts(conn)

    conn.close()
    info("Shutdown complete. Database is safe.")


if __name__ == "__main__":
    main()
