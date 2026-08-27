"""
Astra Sequence Packing Engine (Phase 2)
Packs variable-length documents into fixed-length training sequences of length T,
generating matched doc_ids and position_ids (reset at document boundaries).
"""

from typing import List, Dict, Generator, Any
import numpy as np


def pack_documents(
    documents: List[List[int]],
    seq_len: int = 4096,
    eos_token_id: int = 2,
    pad_token_id: int = 3,
) -> Generator[Dict[str, np.ndarray], None, None]:
    """
    Args:
        documents: List of tokenized documents (each is a list of token IDs)
        seq_len: Target sequence length T (e.g. 4096)
        eos_token_id: Token ID for end-of-sequence delimiter
        pad_token_id: Token ID for padding final leftover shard
        
    Yields:
        Dictionary with:
          - 'input_ids': np.ndarray of shape (seq_len,)
          - 'doc_ids': np.ndarray of shape (seq_len,)
          - 'position_ids': np.ndarray of shape (seq_len,)
    """
    curr_tokens: List[int] = []
    curr_doc_ids: List[int] = []
    curr_positions: List[int] = []

    doc_counter = 1

    for doc in documents:
        # Append EOS if not already ending with it
        doc_tokens = list(doc)
        if len(doc_tokens) == 0 or doc_tokens[-1] != eos_token_id:
            doc_tokens.append(eos_token_id)

        for pos_in_doc, token in enumerate(doc_tokens):
            curr_tokens.append(token)
            curr_doc_ids.append(doc_counter)
            curr_positions.append(pos_in_doc)

            # When reaching target seq_len, yield packed sample
            if len(curr_tokens) == seq_len:
                yield {
                    "input_ids": np.array(curr_tokens, dtype=np.uint32),
                    "doc_ids": np.array(curr_doc_ids, dtype=np.uint32),
                    "position_ids": np.array(curr_positions, dtype=np.uint32),
                }
                curr_tokens = []
                curr_doc_ids = []
                curr_positions = []

        doc_counter += 1

    # Handle final remainder with padding if desired
    if len(curr_tokens) > 0:
        pad_len = seq_len - len(curr_tokens)
        curr_tokens.extend([pad_token_id] * pad_len)
        curr_doc_ids.extend([0] * pad_len)
        curr_positions.extend([0] * pad_len)
        yield {
            "input_ids": np.array(curr_tokens, dtype=np.uint32),
            "doc_ids": np.array(curr_doc_ids, dtype=np.uint32),
            "position_ids": np.array(curr_positions, dtype=np.uint32),
        }
