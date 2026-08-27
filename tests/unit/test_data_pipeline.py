import torch
from data.sources.registry import SourceRegistry, SourceMetadata, get_default_registry
from data.clean.normalize import normalize_text_nfc, verify_vietnamese_diacritics
from data.filter.quality import compute_document_quality_score, compute_exact_hash
from data.pack.pack import pack_documents
from data.shard.shard_writer import BinaryShardWriter
from data.dataset import AstraOfflineDataset


def test_source_registry():
    reg = get_default_registry()
    sources = reg.list_sources()
    assert len(sources) >= 3
    assert reg.get("vietnamese_curated_v1") is not None


def test_unicode_normalization_and_diacritics():
    text = "Mô hình ngôn ngữ Astra tiếng Việt ă â đ ê ô ơ ư"
    norm_text, stats = normalize_text_nfc(text)
    assert stats["null_bytes_count"] == 0
    assert verify_vietnamese_diacritics(norm_text) is True


def test_quality_score_and_hash():
    doc1 = "This is a clean, well-formatted English educational document about neural networks."
    doc2 = "This is a clean, well-formatted English educational document about neural networks."
    h1 = compute_exact_hash(doc1)
    h2 = compute_exact_hash(doc2)
    assert h1 == h2

    score, breakdown = compute_document_quality_score(doc1)
    assert score > 0.5
    assert "q_edu" in breakdown


def test_packing_and_offline_dataset(tmp_path):
    # Create 3 documents
    doc1 = [10, 11, 12]
    doc2 = [20, 21]
    doc3 = [30, 31, 32, 33]

    seq_len = 6
    packed = list(pack_documents([doc1, doc2, doc3], seq_len=seq_len, eos_token_id=2, pad_token_id=3))

    assert len(packed) >= 1
    sample0 = packed[0]
    assert sample0["input_ids"].shape == (seq_len,)
    assert sample0["doc_ids"].shape == (seq_len,)
    assert sample0["position_ids"].shape == (seq_len,)

    # Write to binary shards
    shards_dir = tmp_path / "shards"
    writer = BinaryShardWriter(output_dir=str(shards_dir), seq_len=seq_len, max_samples_per_shard=2)
    for s in packed:
        writer.add_sample(s["input_ids"], s["doc_ids"], s["position_ids"])
    manifest = writer.close()

    assert manifest["total_tokens"] > 0

    # Read back with AstraOfflineDataset
    dataset = AstraOfflineDataset(shards_dir=str(shards_dir))
    assert len(dataset) == len(packed)

    item0 = dataset[0]
    assert isinstance(item0["input_ids"], torch.Tensor)
    assert item0["input_ids"].shape == (seq_len,)
    assert item0["doc_ids"].shape == (seq_len,)
    assert item0["position_ids"].shape == (seq_len,)
