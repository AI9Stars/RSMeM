"""Embedding and hybrid retrieval for the RSMeM knowledge base.

torch and transformers are imported lazily (inside the functions that use them)
so that importing this module — and therefore ``rsmem.runner`` and the CLI — does
not require torch in light-weight environments. Only the embedding/retrieval path
needs torch, and only when actually invoked. This mirrors the lazy treatment of
langchain in ``rsmem.agent``.
"""

from __future__ import annotations

from rsmem.kb import parse_summary_to_subtasks


class EmbeddingModel:
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def compute_embeddings(self, texts: list) -> "torch.Tensor":
        import torch
        import torch.nn.functional as F

        hidden = int(getattr(self.model.config, "hidden_size", 768))
        if not texts:
            return torch.empty(0, hidden, dtype=torch.float32, device=self.device)

        texts = [t if isinstance(t, str) and t.strip() else "empty_text" for t in texts]
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            self.model.eval()
            model_output = self.model(**encoded_input)

        sentence_embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])
        return F.normalize(sentence_embeddings, p=2, dim=1)

    @staticmethod
    def mean_pooling(model_output, attention_mask) -> "torch.Tensor":
        import torch

        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


def precompute_kb_embeddings(kb: dict, emb_model) -> dict:
    """Precompute embeddings for all KB subtasks using ``emb_model``."""
    refined_kb: dict = {}

    for domain, tasks in kb.items():
        if not isinstance(tasks, list):
            continue

        processed_sub_tasks = []
        texts_to_embed = []

        for task in tasks:
            if not isinstance(task, dict):
                continue

            summary_text = task.get("summary", "")
            sub_tasks = parse_summary_to_subtasks(summary_text, domain)

            for st in sub_tasks:
                st.update(
                    {
                        "tool_workflow": task.get("tool_workflow", []),
                        "key_words": task.get("key_words", []),
                    }
                )
                processed_sub_tasks.append(st)
                texts_to_embed.append(st["full_text"])

        if texts_to_embed:
            embeddings = emb_model.compute_embeddings(texts_to_embed)
            refined_kb[domain] = {
                "embeddings": embeddings,
                "sub_tasks": processed_sub_tasks,
            }

    return refined_kb


def hybrid_retrieval(
    query: str,
    kb_embeds: dict,
    emb_model,
    top_k: int = 1,
) -> list:
    """Hybrid scoring: cosine similarity plus title / keyword / content bonuses."""
    del top_k  # retained for API compatibility; behavior matches legacy (best match only).

    if not kb_embeds or not isinstance(query, str) or not query.strip():
        return []

    import torch

    query_emb = emb_model.compute_embeddings([query])
    if query_emb.numel() == 0:
        return []

    query_lower = query.lower()
    candidate_records = []

    for _domain, data in kb_embeds.items():
        embeddings = data["embeddings"]
        sub_tasks = data["sub_tasks"]

        if embeddings.shape[0] != len(sub_tasks):
            continue

        similarities = torch.matmul(query_emb, embeddings.transpose(0, 1)).squeeze(0)
        sims_np = similarities.detach().cpu().numpy()

        for i, sub_task in enumerate(sub_tasks):
            base_score = float(sims_np[i])
            if base_score < 0.1:
                continue

            title_lower = sub_task["sub_title"].lower()
            title_hit = 0.2 if any(word in title_lower for word in query_lower.split()) else 0.0

            kw_hits = 0
            key_words = [kw.lower() for kw in sub_task["key_words"] if isinstance(kw, str)]
            for kw in key_words:
                if kw in query_lower or any(word in kw for word in query_lower.split()):
                    kw_hits += 0.08

            content_lower = sub_task["content"].lower()
            content_hit = 0.1 if any(word in content_lower for word in query_lower.split()[:5]) else 0.0

            combined_score = base_score + title_hit + kw_hits + content_hit
            candidate_records.append(
                {
                    "combined_score": combined_score,
                    "tool_workflow": sub_task["tool_workflow"],
                    "summary": sub_task["content"],
                    "sub_title": sub_task["sub_title"],
                    "domain": sub_task["domain"],
                    "base_similarity": base_score,
                }
            )

    final_results: list = []
    if candidate_records:
        candidate_records.sort(key=lambda x: x["combined_score"], reverse=True)
        if candidate_records[0]["combined_score"] >= 0.2:
            final_results = [candidate_records[0]]

    return final_results
