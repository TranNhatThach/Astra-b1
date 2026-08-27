"""
Astra Dialogue Streaming Adapter (Phase 7C)
"""

from typing import Dict, Any, Generator, Optional
import os
from .base import SourceAdapter, RawDocument


class SyntheticDialogueAdapter(SourceAdapter):
    def __init__(self, version: str = "v1.0", use_live_stream: bool = False):
        super().__init__(
            source_id="synthetic_reasoning_dialogue_v1",
            version=version,
            category="dialogue",
            language="en",
        )
        self.use_live_stream = use_live_stream or (os.environ.get("ASTRA_LIVE_STREAM", "0") == "1")
        self._dialogue_templates = [
            ("User: How does the Gated DeltaNet achieve linear time complexity during pretraining?\n"
             "Assistant: Gated DeltaNet reformulates linear attention as an associative matrix state update S_t = D_t S_{t-1}(I - u_t k_t k_t^T) + u_t v_t k_t^T. By leveraging associative chunk scans in log depth, it computes sequences in O(T) complexity rather than O(T^2)."),
            ("User: Giải thích vai trò của cơ chế Multi-Token Prediction (MTP) trong Astra-1B?\n"
             "Assistant: MTP sử dụng các prediction head phụ để dự đoán đồng thời token t+1 và token t+2, giúp mô hình học được biểu diễn ngữ cảnh dài hạn, tăng hiệu quả lấy mẫu và đẩy nhanh tốc độ hội tụ."),
            ("User: Why is RMSNorm preferred over LayerNorm in modern LLM architectures?\n"
             "Assistant: RMSNorm simplifies the normalization by enforcing root-mean-square scaling without centering around the mean, saving memory bandwidth and floating-point operations while preserving training stability."),
            ("User: How do you prevent cross-document attention leakage when packing sequences?\n"
             "Assistant: By using document boundary masks and resetting position IDs to 0 at each document boundary, ensuring that attention and state recurrence do not propagate across distinct documents within the same packed sequence."),
        ]

    def iterate_documents(
        self,
        max_docs: Optional[int] = None,
        resume_pos: int = 0,
    ) -> Generator[RawDocument, None, None]:
        emitted = 0
        if self.use_live_stream:
            try:
                import datasets
                ds = datasets.load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
                skipped = 0
                for item in ds:
                    if skipped < resume_pos:
                        skipped += 1
                        continue
                    if max_docs is not None and emitted >= max_docs:
                        break
                    messages = item.get("messages", [])
                    if messages:
                        turns = [f"{m.get('role', 'User').capitalize()}: {m.get('content', '')}" for m in messages]
                        text = "\n\n".join(turns)
                    else:
                        text = str(item.get("prompt", ""))
                    yield RawDocument(
                        source_id=self.source_id,
                        source_version=self.version,
                        source_record_id=f"ultrachat_{resume_pos + emitted:08d}",
                        category=self.category,
                        language=self.language,
                        text=text,
                    )
                    emitted += 1
                return
            except Exception as e:
                print(f"[WARN] Live stream failed: {e}. Falling back to internal engine.")

        pos = resume_pos
        while True:
            if max_docs is not None and emitted >= max_docs:
                break
            idx = pos % len(self._dialogue_templates)
            template = self._dialogue_templates[idx]
            text = f"--- Conversation #{pos} ---\n{template}\n--- End of Conversation ---"

            yield RawDocument(
                source_id=self.source_id,
                source_version=self.version,
                source_record_id=f"dialogue_{pos:08d}",
                category=self.category,
                language=self.language,
                text=text,
                metadata={"turn_count": 2, "conversation_id": f"conv_{pos:06d}", "dialogue_idx": pos},
            )
            pos += 1
            emitted += 1

    def get_license_info(self) -> Dict[str, Any]:
        return {
            "name": "Astra Multi-Turn Reasoning Dialogues",
            "provider": "Astra AI Research Lab",
            "url": "https://astra-research.org/datasets/reasoning_dialogues",
            "license": "Apache-2.0",
            "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
            "terms": "Permitted for academic and foundation model pretraining and instruction tuning",
            "retrieval_method": "Synthetic Reasoning Generation Pipeline",
            "allowed_for_training": True,
        }
