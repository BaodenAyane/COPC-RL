"""Prepare pinned Search-R1 assets with resumable downloads."""
import argparse
import array
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

PLAN = json.loads((Path(__file__).with_name("sources.json")).read_text())
EXPECTED_ROWS = 21015324
INDEX_BYTES = 75656528422
INDEX_FILENAME = "e5_HNSW64.index"
INDEX_DIRECTORY = "downloads/index_hnsw"
INDEX_TYPE = "hnsw64"
CHUNK = 8 * 1024 * 1024

def digest(path, expected):
    if len(expected) == 40:
        h = hashlib.sha1()
        h.update(f"blob {path.stat().st_size}\\0".encode().replace(b"\\0", b"\x00"))
    else:
        h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def download_one(root, source, item, endpoint):
    from huggingface_hub import hf_hub_download
    name, size, expected = item
    local = root / source["directory"] / name
    receipt = local.with_suffix(local.suffix + ".verified.json")
    if local.exists() and local.stat().st_size == size and receipt.exists():
        record = json.loads(receipt.read_text())
        if record.get("digest") == expected and record.get("mtime_ns") == local.stat().st_mtime_ns:
            print("Verified cache:", source["repo"], name, flush=True)
            return
    if not local.exists() or local.stat().st_size != size:
        print("Downloading:", source["repo"], name, size, flush=True)
        hf_hub_download(repo_id=source["repo"], repo_type=source["kind"],
                        filename=name, revision=source["revision"], token=False,
                        local_dir=str(root / source["directory"]), endpoint=endpoint)
    if local.stat().st_size != size or digest(local, expected) != expected:
        raise ValueError(f"Checksum mismatch: {source['repo']}/{name}; file retained for inspection")
    atomic_json(receipt, dict(digest=expected, size=size, mtime_ns=local.stat().st_mtime_ns,
                             repo=source["repo"], revision=source["revision"]))
    print("Verified:", source["repo"], name, flush=True)

def assemble_index(root):
    final = root / INDEX_FILENAME
    receipt = root / "index_hnsw.ready.json"
    if final.exists() and receipt.exists():
        record = json.loads(receipt.read_text())
        if final.stat().st_size == INDEX_BYTES and record.get("mtime_ns") == final.stat().st_mtime_ns:
            return record
    if final.exists():
        raise ValueError(f"Existing unverified {INDEX_FILENAME}: inspect it instead of overwriting")
    partial = root / f"{INDEX_FILENAME}.partial"
    h = hashlib.sha256()
    with partial.open("wb") as output:
        for name in ("part_aa", "part_ab"):
            with (root / INDEX_DIRECTORY / name).open("rb") as stream:
                for block in iter(lambda: stream.read(CHUNK), b""):
                    output.write(block)
                    h.update(block)
            print("Joined:", name, flush=True)
        output.flush()
        os.fsync(output.fileno())
    if partial.stat().st_size != INDEX_BYTES:
        raise ValueError("Wrong merged index size")
    os.replace(partial, final)
    record = dict(size=final.stat().st_size, sha256=h.hexdigest(), mtime_ns=final.stat().st_mtime_ns,
                  dimension=768, ntotal=EXPECTED_ROWS, metric="inner_product",
                  file=INDEX_FILENAME, index_type=INDEX_TYPE)
    atomic_json(receipt, record)
    return record

@contextmanager
def corpus_stream(path):
    """Open either a gzip JSONL or the official gzip-compressed tar corpus."""
    with gzip.open(path, "rb") as compressed:
        header = compressed.read(512)
    if len(header) >= 262 and header[257:262] == b"ustar":
        with tarfile.open(path, mode="r|gz") as archive:
            source = None
            for member in archive:
                if not member.isfile():
                    continue
                if source is not None:
                    raise ValueError("Corpus archive contains multiple regular files")
                if not member.name.endswith(".jsonl"):
                    raise ValueError(f"Unexpected corpus archive member: {member.name}")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Cannot read corpus archive member: {member.name}")
                print(f"Reading archived corpus: {member.name} ({member.size} bytes)", flush=True)
                yield source
            if source is None:
                raise ValueError("Corpus archive contains no JSONL file")
    else:
        with gzip.open(path, "rb") as source:
            yield source

def unpack_corpus(root):
    final, offsets = root / "wiki-18.jsonl", root / "corpus.offsets.u64"
    receipt = root / "corpus.ready.json"
    if final.exists() and offsets.exists() and receipt.exists():
        record = json.loads(receipt.read_text())
        if (record.get("size") == final.stat().st_size and record.get("mtime_ns") == final.stat().st_mtime_ns
                and offsets.stat().st_size == (EXPECTED_ROWS + 1) * 8
                and record.get("offsets_mtime_ns") == offsets.stat().st_mtime_ns):
            return record
    if final.exists() or offsets.exists():
        raise ValueError("Existing unverified corpus/offsets: inspect instead of overwriting")
    partial, offset_partial = root / "wiki-18.jsonl.partial", root / "corpus.offsets.u64.partial"
    position, count = 0, 0
    h = hashlib.sha256()
    offsets_chunk = array.array("Q", [0])
    first_line = None
    last_complete_line = None
    fragment = b""
    next_report = 1000000
    if sys.byteorder != "little":
        raise RuntimeError("Offset writer requires little-endian host")
    archive = root / "downloads/corpus/wiki-18.jsonl.gz"
    with corpus_stream(archive) as source, partial.open("wb") as output, offset_partial.open("wb") as target:
        for block in iter(lambda: source.read(CHUNK), b""):
            output.write(block)
            h.update(block)
            first_newline = block.find(b"\n")
            if first_newline < 0:
                fragment += block
            else:
                if first_line is None:
                    first_line = fragment + block[:first_newline]
                last_newline = block.rfind(b"\n")
                previous_newline = block.rfind(b"\n", 0, last_newline)
                if previous_newline >= 0:
                    last_complete_line = block[previous_newline + 1:last_newline]
                else:
                    last_complete_line = fragment + block[:last_newline]
                fragment = block[last_newline + 1:]
            cursor = 0
            while True:
                newline = block.find(b"\n", cursor)
                if newline < 0:
                    break
                offsets_chunk.append(position + newline + 1)
                count += 1
                cursor = newline + 1
            position += len(block)
            if len(offsets_chunk) >= 100000:
                offsets_chunk.tofile(target)
                offsets_chunk = array.array("Q")
            while count >= next_report:
                print("Corpus rows:", next_report, flush=True)
                next_report += 1000000
        if fragment:
            offsets_chunk.append(position)
            count += 1
            last_line = fragment
        else:
            last_line = last_complete_line
        offsets_chunk.tofile(target)
        output.flush(); os.fsync(output.fileno())
        target.flush(); os.fsync(target.fileno())
    if count != EXPECTED_ROWS:
        raise ValueError(f"Corpus/index mismatch: {count} != {EXPECTED_ROWS}")
    for label, line in (("first", first_line), ("last", last_line)):
        try:
            row = json.loads(line)
        except (TypeError, ValueError, UnicodeDecodeError) as error:
            raise ValueError(f"Malformed {label} corpus row") from error
        if not isinstance(row.get("contents"), str) or not row["contents"].strip():
            raise ValueError(f"Malformed {label} corpus row")
    os.replace(partial, final)
    os.replace(offset_partial, offsets)
    record = dict(rows=count, size=position, sha256=h.hexdigest(), mtime_ns=final.stat().st_mtime_ns,
                  offsets_size=offsets.stat().st_size, offsets_mtime_ns=offsets.stat().st_mtime_ns)
    atomic_json(receipt, record)
    return record

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (args.root / "index_hnsw.ready.json").exists() and shutil.disk_usage(args.root).free < 170 * 1024**3:
            raise RuntimeError("Need approximately 170 GiB free for downloads, assembled index and corpus")
        atomic_json(args.root / "sources.json", PLAN)
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
        items = [(source, item) for source in PLAN for item in source["files"]]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(download_one, args.root, source, item, args.endpoint) for source,item in items]
            for future in futures:
                future.result()
        if args.download_only:
            return
        index = assemble_index(args.root)
        corpus = unpack_corpus(args.root)
        atomic_json(args.root / "assets.ready.json", dict(schema_version=1, sources=PLAN,
                    index=index, corpus=corpus, model="e5-base-v2", created_at=time.time()))
        print("Assets READY:", args.root, flush=True)

if __name__ == "__main__":
    main()
