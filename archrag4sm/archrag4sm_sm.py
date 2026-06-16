import os
import ast
import json
import logging
import numpy as np
import pandas as pd
import openai
from typing import Optional, Tuple, Union
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from .ArchRAG.src.prompts import COMMUNITY_REPORT_PROMPT_SHORT

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token usage tracker
# ---------------------------------------------------------------------------

class TokenTracker:
    """Accumulates token usage across multiple LLM calls with breakdown."""

    def __init__(self):
        self.summary_prompt = 0
        self.summary_completion = 0
        self.inference_prompt = 0
        self.inference_completion = 0

    def add_summary_tokens(self, prompt_tokens: int, completion_tokens: int):
        self.summary_prompt += prompt_tokens
        self.summary_completion += completion_tokens

    def add_inference_tokens(self, prompt_tokens: int, completion_tokens: int):
        self.inference_prompt += prompt_tokens
        self.inference_completion += completion_tokens

    @property
    def summary_total(self) -> int:
        return self.summary_prompt + self.summary_completion

    @property
    def inference_total(self) -> int:
        return self.inference_prompt + self.inference_completion

    @property
    def grand_total(self) -> int:
        return self.summary_total + self.inference_total

    def __str__(self) -> str:
        return (
            f"Total tokens: {self.grand_total} "
            f"(summary: {self.summary_total} [{self.summary_prompt}+{self.summary_completion}], "
            f"inference: {self.inference_total} [{self.inference_prompt}+{self.inference_completion}])"
        )


# ---------------------------------------------------------------------------
# Schema Matching System Prompt (ArchRAG4SM variant of itkgrag4sm prompt)
# ---------------------------------------------------------------------------

# Prompt style: itkgrag4sm (with relevance scoring)
ARCHRAG4SM_SYSTEM_PROMPT_IT = """
                You are an expert in schema matching and data integration.
                (a) Your task is to analyze the attribute 1 with its textual description 1 and attribute 2 with its textual description 2 from source and target schema in the given question, and specify if the attribute 1 from source schema is semantically matched with attribute 2 from the target schema. In some questions, there is the knowledge graph context that might be helpful for you to answer. In this case, you will need to consider the provided context to make the correct decision.
                (b) Please give the relevance score between the given question and the provided knowledge graph context between 0 and 10 if the knowledge graph context is available. A score of 10 means the provided knowledge graph context is very relevant to the given question, and a score of 0 means the provided knowledge graph context is irrelevant to the given question.
                (c) Please make the decision only based on your knowledge if the knowledge graph context relevance score is less than 7 or the provided knowledge graph context is unavailable to answer the given question.
               \n\n

               Here are some examples of the schema matching questions with correct answers and explanations that you need to learn before you start to analyze the potential mappings:
                Example 1:
                Attribute 1 death-person_id and its description 1 the death domain contains the clinical event for how and when a person dies. a person can have up to one record if the source system contains evidence about the death.;a foreign key identifier to the deceased person. the demographic details of that person are stored in the person table.
                Attribute 2 beneficiarysummary-bene_birth_dt and its description 2 beneficiarysummary pertain to a synthetic medicare beneficiary; date of birth.
                Are attribute 1 and attribute 2 semantically matched with each other?
                Here is the knowledge graph context that might be helpful for you to answer the above schema matching question:
                death (Q4), has part(s) of the class (P2670), date of death (Q18748141) -> date of death (Q18748141), opposite of (P461), date of birth (Q2389905) | human (Q5), has characteristic (P1552), age of a person (Q185836) -> age of a person (Q185836), uses (P2283), date of birth (Q2389905)
                Knowledge graph context relevant score: 8
                Reason for relevant score: The above knowledge graph context indicates that date of death is opposite of date of birth, which is relevant to the given question.
                Here is the correct answer and the explanations for the above-given example question: 0
                Explanation: they are not semantically matched with each other, because death-person_id is a unique identifier for each person in death table and bene_birth_dt is the date of birth of person in beneficiarysummary table. From the above context, we can find that date of death is opposite of date of birth, they are not semantically matched with each other.\n\n

                Example 2:
                Attribute 1 drug_exposure-stop_reason and its description 1 the 'drug' domain captures records about the utilization of a drug when ingested or otherwise introduced into the body. a drug is a biochemical substance formulated in such a way that when administered to a person it will exert a certain physiological effect. drugs include prescription and over-the-counter medicines, vaccines, and large-molecule biologic therapies. radiological devices ingested or applied locally do not count as drugs.;the reason the drug was stopped. reasons include regimen completed, changed, removed, etc.
                Attribute 2 medications-reasondescription and its description 2 patient medication data; description of the reason code.
                Are attribute 1 and attribute 2 semantically matched with each other?
                Here is the knowledge graph context that might be helpful for you to answer the above schema matching question:
                Drug Exposure (C41362):Contact with drug, isa, Chemical Exposure (C36290) | reason for stopping medication (120234), isa, Medication discontinued(274512008)-->has_associated_procedure, Drug therapy(416608005), has_direct_substance, Drug or medicament(410942007)
                Knowledge graph context relevant score: 7
                Reason for relevant score: The above knowledge graph context shows the connections between stopping medication and drug exposure, which might be relevant to the given question.
                Here is the correct answer and the explanations for the above-given example question: 0
                Explanation: they are not semantically matched with each other, because drug_exposure-stop_reason is the reason for stopping the drug exposure and medications-reasondescription is the description of the reason code. From the above context, even if we can find that there is a connection between the drug exposure stop reason and reason for stopping medication but this connection is a sub-concept relation. \n\n


                Remember the following tips when you are analyzing the potential mappings.
                Tips:
                (1) Some letters are extracted from the full names and merged into an abbreviation word.
                (2) Schema information sometimes is also added as the prefix of abbreviation.
                (3) Please consider the abbreviation case.
                (4) If conflicts exist between knowledge graph context and your own internal knowledge, please make the decision based on knowledge graph context.
                """

# Prompt style: kgrag4sm (no relevance scoring, 2 examples)
ARCHRAG4SM_SYSTEM_PROMPT_KG = """
                You are an expert in schema matching and data integration.
                Your task is to analyze the attribute 1 with its textual description 1 and attribute 2 with its textual description 2 from source and target schema in the given question, and specify if the attribute 1 from source schema is semantically matched with attribute 2 from the target schema.
                In some questions, there is the knowledge graph context that might be helpful for you to answer. In this case, you will need to consider the provided context to make the correct decision. \n\n

                Here are two examples of the schema matching questions with correct answers and explanations that you need to learn before you start to analyze the potential mappings:
                Example 1:
                Attribute 1 death-person_id and its description 1 the death domain contains the clinical event for how and when a person dies. a person can have up to one record if the source system contains evidence about the death; a foreign key identifier to the deceased person. the demographic details of that person are stored in the person table.
                Attribute 2 beneficiarysummary-desynpuf_id and its description 2 beneficiarysummary pertain to a synthetic medicare beneficiary; beneficiary code.
                Are attribute 1 and attribute 2 semantically matched with each other?
                Here is the correct answer and the explanations for the above-given example question: 1
                Explanation: they are semantically matched with each other because both of them are unique identifiers for each person. Even if the death-person_id refers to the unique identifier of the person in the death table and beneficiarysummary-desynpuf_id refers to the unique identifier of the person beneficiary from beneficiarysummary table, they are semantically matched with each other. \n\n

                Example 2:
                Attribute 1 death-person_id and its description 1 the death domain contains the clinical event for how and when a person dies. a person can have up to one record if the source system contains evidence about the death.;a foreign key identifier to the deceased person. the demographic details of that person are stored in the person table.
                Attribute 2 beneficiarysummary-bene_birth_dt and its description 2 beneficiarysummary pertain to a synthetic medicare beneficiary; date of birth.
                Are attribute 1 and attribute 2 semantically matched with each other?
                Here is the knowledge graph context that might be helpful for you to answer the above schema matching question:
                death (Q4), has part(s) of the class (P2670), date of death (Q18748141) -> date of death (Q18748141), opposite of (P461), date of birth (Q2389905) | human (Q5), has characteristic (P1552), age of a person (Q185836) -> age of a person (Q185836), uses (P2283), date of birth (Q2389905)
                Here is the correct answer and the explanations for the above-given example question: 0
                Explanation: they are not semantically matched with each other, because death-person_id is a unique identifier for each person in death table and bene_birth_dt is the date of birth of person in beneficiarysummary table. From the above context, we can found that date of death is opposite of date of birth, they are not semantically matched with each other.

                Remember the following tips when you are analyzing the potential mappings.
                Tips:
                (1) Some letters are extracted from the full names and merged into an abbreviation word.
                (2) Schema information sometimes is also added as the prefix of abbreviation.
                (3) Please consider the abbreviation case.
                (4) Please consider the knowledge graph context to make the correct decision when it is provided.
                """

# Default prompt (backward compatible)
ARCHRAG4SM_SYSTEM_PROMPT = ARCHRAG4SM_SYSTEM_PROMPT_KG


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class ArchRAGRetriever:
    """
    Retrieves top-k entities, communities and relationships from an ArchRAG index.

    Primary mode  : loads the HCHNSW index produced by ArchRAG's src/index.py.
    Fallback mode : builds a standard FAISS HNSW index from entity/community
                    embeddings stored in the index CSV files (no custom Faiss needed).
    """

    def __init__(
        self,
        index_dir: str,
        embedding_func,
        topk: int = 15,
        topk_e: int = 10,
    ):
        self.index_dir = index_dir
        self.embedding_func = embedding_func
        self.topk = topk
        self.topk_e = topk_e
        # Load only the lightweight metadata (id + index_id, no embeddings in memory)
        self._load_index_csvs()
        self.hc_index = None
        self._try_load_hchnsw()
        if self.hc_index is None:
            self._build_fallback_hnsw()

    # ------------------------------------------------------------------
    # Index loading helpers
    # ------------------------------------------------------------------

    def _load_index_csvs(self):
        import gc

        def _read(filename, usecols=None):
            path = os.path.join(self.index_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"ArchRAG index file not found: {path}\n"
                    "Run archrag4sm/index.sh first to build the index."
                )
            log.info("Loading %s (cols: %s) …", path, usecols or "all")
            try:
                return pd.read_csv(path, usecols=usecols)
            except Exception as e:
                log.warning("C engine failed (%s), retrying with python engine…", e)
                return pd.read_csv(path, usecols=usecols, engine="python", on_bad_lines="skip")

        # Load text metadata only (NO embedding column) — embeddings are streamed during FAISS build
        self.entity_df = _read("entity_df_index.csv", usecols=["id", "name", "description", "index_id"])
        # Build index_id → name lookup for resolving relationship endpoints
        self._index_id_to_name = dict(zip(self.entity_df["index_id"], self.entity_df["name"]))
        log.info("entity_df loaded: %d rows", len(self.entity_df))

        # community
        self.community_df = _read("community_df_index.csv", usecols=["community_id", "level", "title", "summary", "community_text", "index_id"])
        if "community_id" in self.community_df.columns:
            self.community_df = self.community_df.rename(columns={"community_id": "id"})
        log.info("community_df loaded: %d rows", len(self.community_df))

        # level summaries — regenerate from community data if placeholders
        self.level_summary_df = _read("level_summary.csv")
        if self.level_summary_df["summary"].str.contains("contains \\d+ communities", regex=True).all():
            log.info("Level summaries are placeholders — generating from community data")
            summaries = []
            for _, row in self.level_summary_df.iterrows():
                level = int(row["level"])
                if "level" in self.community_df.columns:
                    level_communities = self.community_df[self.community_df["level"] == level]
                else:
                    level_communities = pd.DataFrame()
                n = len(level_communities)
                if not level_communities.empty and "title" in level_communities.columns:
                    sample_titles = level_communities["title"].dropna().head(5).tolist()
                    topics = "; ".join(str(t) for t in sample_titles if t and str(t) not in ("nan", ""))
                else:
                    topics = ""
                summaries.append(f"Level {level}: {n} communities. Topics: {topics}")
            self.level_summary_df["summary"] = summaries
            log.info("Generated %d level summaries from community data", len(summaries))

        # relationships — load lightweight columns + description embeddings
        self.relation_df = pd.DataFrame()
        self._rel_emb_map = {}
        self._entity_rel_map = {}
        rel_path = os.path.join(self.index_dir, "relationship_df_index.csv")
        if os.path.exists(rel_path):
            log.info("Loading relationships from %s …", rel_path)
            self.relation_df = _read("relationship_df_index.csv", usecols=["source_index_id", "target_index_id", "description", "embedding_idx"])
            log.info("relationship_df loaded: %d rows", len(self.relation_df))

            # Load description embeddings (~10MB, unique descriptions only)
            rel_emb_path = os.path.join(self.index_dir, "relationship_embedding.csv")
            if os.path.exists(rel_emb_path):
                log.info("Loading relationship embeddings from %s …", rel_emb_path)
                rel_emb_df = _read("relationship_embedding.csv")
                rel_emb_df["embedding"] = rel_emb_df["embedding"].apply(self._parse_embedding)
                self._rel_emb_map = dict(zip(rel_emb_df["idx"], rel_emb_df["embedding"]))
                del rel_emb_df
                gc.collect()
                log.info("Relationship embeddings loaded: %d unique descriptions", len(self._rel_emb_map))

            # Build entity_index_id → [row_indices] lookup for O(1) retrieval
            log.info("Building entity→relationship lookup…")
            for row_idx, src_id in enumerate(self.relation_df["source_index_id"]):
                self._entity_rel_map.setdefault(src_id, []).append(row_idx)
            log.info("Entity→relationship map: %d source entities", len(self._entity_rel_map))
        else:
            log.warning("relationship_df_index.csv not found — relationships will not be retrieved")

        log.info("Index loading complete")

    @staticmethod
    def _parse_embedding(val):
        if isinstance(val, np.ndarray):
            return val.astype(np.float32)
        if isinstance(val, list):
            return np.array(val, dtype=np.float32)
        if isinstance(val, str):
            s = val.strip()
            if s.startswith("[") and s.endswith("]"):
                inner = s[1:-1].strip()
                try:
                    if "," in inner:
                        return np.array(ast.literal_eval(s), dtype=np.float32)
                    else:
                        # numpy repr: space/newline separated values
                        return np.fromstring(inner, sep=" ", dtype=np.float32)
                except Exception:
                    pass
            try:
                return np.array(eval(val), dtype=np.float32)
            except Exception:
                return None
        return None

    def _try_load_hchnsw(self):
        """Attempt to load the custom ArchRAG HCHNSW index."""
        try:
            from src.hchnsw_index import read_index  # from ArchRAG repo on PYTHONPATH

            index_path = os.path.join(self.index_dir, "hchnsw_index.bin")
            if os.path.exists(index_path):
                self.hc_index = read_index(index_path)
                log.info("Loaded HCHNSW index from %s", index_path)
            else:
                log.info("hchnsw_index.bin not found; using fallback HNSW")
        except ImportError:
            log.info("ArchRAG custom Faiss not available; using standard HNSW fallback")
        except Exception as exc:
            log.warning("Could not load HCHNSW index: %s", exc)

    def _build_fallback_hnsw(self):
        """Build a standard FAISS HNSW index by streaming embeddings from CSV in chunks.

        This avoids loading all 55 GB of embedding strings into memory at once.
        The built index is saved to disk and reloaded on subsequent runs.
        """
        import faiss
        import gc

        fallback_index_path = os.path.join(self.index_dir, "fallback_hnsw.index")
        fallback_meta_path = os.path.join(self.index_dir, "fallback_hnsw_meta.npz")

        # ── Load pre-built index if it exists ────────────────────────────────
        if os.path.exists(fallback_index_path) and os.path.exists(fallback_meta_path):
            log.info("Loading pre-built fallback HNSW index from %s …", fallback_index_path)
            self._fallback_index = faiss.read_index(fallback_index_path)
            meta = np.load(fallback_meta_path, allow_pickle=True)
            self._fallback_type = list(meta["types"])
            self._fallback_row_idx = list(meta["row_indices"].astype(int))
            log.info(
                "Loaded fallback HNSW index: %d vectors (dim=%d)",
                self._fallback_index.ntotal,
                self._fallback_index.d,
            )
            return

        # ── Build index by streaming CSVs in chunks ───────────────────────────
        CHUNK = 100_000  # parse 100K rows at a time — ~1.2 GB peak per chunk
        dim = 768  # roberta-base embedding dimension

        log.info("Building fallback HNSW index by streaming CSV in chunks (chunk=%d)…", CHUNK)
        self._fallback_index = faiss.IndexHNSWFlat(dim, 32)
        self._fallback_type = []
        self._fallback_row_idx = []

        for csv_file, kind, id_col in [
            ("entity_df_index.csv", "entity", "id"),
            ("community_df_index.csv", "community", "community_id"),
        ]:
            path = os.path.join(self.index_dir, csv_file)
            log.info("Streaming %s from %s …", kind, path)
            row_idx = 0
            added = 0
            for chunk in pd.read_csv(path, usecols=[id_col, "embedding", "index_id"], chunksize=CHUNK):
                embs = chunk["embedding"].apply(self._parse_embedding)
                valid_mask = embs.apply(lambda x: x is not None and isinstance(x, np.ndarray))
                valid_embs = embs[valid_mask]
                valid_count = valid_mask.sum()

                if valid_count > 0:
                    matrix = np.vstack(valid_embs.tolist()).astype(np.float32)
                    self._fallback_index.add(matrix)
                    self._fallback_type.extend([kind] * valid_count)
                    self._fallback_row_idx.extend(range(row_idx, row_idx + valid_count))
                    added += valid_count

                row_idx += len(chunk)
                del chunk, embs, valid_embs
                gc.collect()

                if row_idx % 500_000 == 0 or row_idx >= len(self.entity_df if kind == "entity" else self.community_df):
                    log.info("  %s: processed %d rows, added %d vectors so far", kind, row_idx, added)

            log.info("  %s: finished — %d vectors added (total index: %d)", kind, added, self._fallback_index.ntotal)

        # ── Save to disk for future runs ──────────────────────────────────────
        log.info("Saving fallback index to %s …", fallback_index_path)
        faiss.write_index(self._fallback_index, fallback_index_path)
        np.savez(
            fallback_meta_path,
            types=np.array(self._fallback_type),
            row_indices=np.array(self._fallback_row_idx, dtype=np.int32),
        )
        log.info(
            "Built and saved fallback HNSW index: %d vectors (dim=%d)",
            self._fallback_index.ntotal,
            self._fallback_index.d,
        )

    # ------------------------------------------------------------------
    # Public retrieval entry point
    # ------------------------------------------------------------------

    def retrieve(self, query_text: str) -> dict:
        """Embed query and retrieve top-k entities, communities, relationships."""
        emb = self.embedding_func(query_text)
        query_emb = np.array(emb, dtype=np.float32)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        if self.hc_index is not None:
            return self._retrieve_hchnsw(query_emb)
        return self._retrieve_fallback(query_emb)

    def _retrieve_relationships(self, entity_index_ids, query_emb: np.ndarray) -> pd.DataFrame:
        """Retrieve top-k relationships linked to the given entity index_ids, ranked by cosine similarity."""
        if self.relation_df.empty or not self._entity_rel_map:
            return pd.DataFrame(columns=["source", "target", "description", "embedding"])

        # Collect row indices of relationships linked to retrieved entities
        rel_row_indices = set()
        for eid in entity_index_ids:
            rel_row_indices.update(self._entity_rel_map.get(eid, []))

        if not rel_row_indices:
            return pd.DataFrame(columns=["source", "target", "description", "embedding"])

        sel_rel = self.relation_df.iloc[list(rel_row_indices)].copy()

        # Look up description embeddings and compute cosine similarity
        embeddings = []
        valid_indices = []
        for i, idx in enumerate(sel_rel["embedding_idx"]):
            emb = self._rel_emb_map.get(idx)
            if emb is not None:
                embeddings.append(emb)
                valid_indices.append(i)

        if not embeddings:
            return pd.DataFrame(columns=["source", "target", "description", "embedding"])

        sel_rel = sel_rel.iloc[valid_indices].copy()
        embs = np.stack(embeddings).astype(np.float32)
        sims = cosine_similarity(embs, query_emb).flatten()
        sel_rel["similarity"] = sims

        topk_related_r = sel_rel.nlargest(min(self.topk_e, len(sel_rel)), "similarity")
        # Map index IDs to entity names
        topk_related_r = topk_related_r.copy()
        topk_related_r["source"] = topk_related_r["source_index_id"].map(self._index_id_to_name).fillna(topk_related_r["source_index_id"].astype(str))
        topk_related_r["target"] = topk_related_r["target_index_id"].map(self._index_id_to_name).fillna(topk_related_r["target_index_id"].astype(str))
        return topk_related_r[["source", "target", "description"]].copy()

    def _retrieve_hchnsw(self, query_emb: np.ndarray) -> dict:
        import faiss

        hc_level = self.hc_index.hchnsw.max_level
        all_results = []
        k_per_level = 5
        for level in range(min(3, hc_level + 1)):
            params = faiss.SearchParametersHCHNSW()
            params.search_level = level
            distances, preds = self.hc_index.search(
                query_emb, k=k_per_level, params=params
            )
            for dist, pred in zip(distances.flatten(), preds.flatten()):
                if pred >= 0:
                    all_results.append((dist, pred))

        all_results = sorted(all_results, key=lambda x: x[0])[: self.topk]
        final_ids = [pred for _, pred in all_results]

        topk_entity = self.entity_df[self.entity_df["index_id"].isin(final_ids)]
        topk_community = self.community_df[
            self.community_df["index_id"].isin(final_ids)
        ]

        # Retrieve relationships linked to retrieved entities
        entity_index_ids = set(topk_entity["index_id"].values) if not topk_entity.empty else set()
        topk_related_r = self._retrieve_relationships(entity_index_ids, query_emb)

        return {
            "topk_entity": topk_entity,
            "topk_community": topk_community,
            "topk_related_r": topk_related_r,
        }

    def _retrieve_fallback(self, query_emb: np.ndarray) -> dict:
        k = min(self.topk, self._fallback_index.ntotal)
        _, indices = self._fallback_index.search(query_emb, k)

        entity_rows, community_rows = [], []
        for idx in indices.flatten():
            if idx < 0:
                continue
            t = self._fallback_type[idx]
            r = self._fallback_row_idx[idx]
            if t == "entity":
                entity_rows.append(r)
            else:
                community_rows.append(r)

        topk_entity = (
            self.entity_df.iloc[entity_rows]
            if entity_rows
            else pd.DataFrame(columns=self.entity_df.columns)
        )
        topk_community = (
            self.community_df.iloc[community_rows]
            if community_rows
            else pd.DataFrame(columns=self.community_df.columns)
        )

        # Retrieve relationships linked to retrieved entities
        entity_index_ids = set(topk_entity["index_id"].values) if not topk_entity.empty else set()
        topk_related_r = self._retrieve_relationships(entity_index_ids, query_emb)

        return {
            "topk_entity": topk_entity,
            "topk_community": topk_community,
            "topk_related_r": topk_related_r,
        }

    def _top_relations(
        self, query_emb: np.ndarray, rel_df: pd.DataFrame
    ) -> pd.DataFrame:
        if rel_df.empty or "embedding" not in rel_df.columns:
            return rel_df.head(self.topk_e)

        valid_mask = rel_df["embedding"].apply(lambda x: x is not None)
        rel_df = rel_df[valid_mask].copy()
        if rel_df.empty:
            return rel_df

        embs = np.stack(rel_df["embedding"].values).astype(np.float32)
        sims = cosine_similarity(embs, query_emb).flatten()
        rel_df["similarity"] = sims
        return rel_df.nlargest(min(self.topk_e, len(rel_df)), "similarity")


# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------

def format_retrieved_context(
    topk_entity: pd.DataFrame,
    topk_community: pd.DataFrame,
    topk_related_r: pd.DataFrame,
    max_triples: int = 10,
    max_community_summaries: int = 3,
    llm_summarizer=None,
) -> Optional[str]:
    """
    Format ArchRAG retrieved items as a KG context string compatible with the
    schema matching prompts (similar to Wikidata KG path format used in KG-RAG4SM).
    
    If llm_summarizer is provided and a community has a 'nan' summary, calls LLM
    to generate a summary on-demand.
    """
    parts = []

    # Primary: entity-relation-entity triples
    if topk_related_r is not None and not topk_related_r.empty:
        triples = []
        for _, row in topk_related_r.head(max_triples).iterrows():
            src = row.get("source", "")
            tgt = row.get("target", "")
            rel = row.get("description", "")
            if src and tgt:
                triples.append(f"{src}, {rel}, {tgt}")
        if triples:
            parts.append(" | ".join(triples))

    # Secondary: community summaries (compressed background knowledge)
    if topk_community is not None and not topk_community.empty:
        summaries = []
        for _, row in topk_community.head(max_community_summaries).iterrows():
            title = row.get("title", "")
            summary = row.get("summary", "")
            community_id = row.get("id", row.get("community_id", ""))

            # If summary is missing or 'nan', generate on-demand via LLM (then fall back to community_text)
            if not summary or str(summary).strip() in ("nan", "None", ""):
                if llm_summarizer:
                    try:
                        log.debug("Generating summary for %s via LLM...", community_id)
                        summary = llm_summarizer(community_id, row)
                        log.debug("Summary generation complete for %s", community_id)
                    except Exception as e:
                        log.warning("Failed to generate summary for %s: %s", community_id, e)
                        summary = ""
                # Fallback to community_text if LLM failed or unavailable
                if not summary or str(summary).strip() in ("nan", "None", ""):
                    community_text = row.get("community_text", "")
                    if community_text and str(community_text).strip() not in ("nan", "None", ""):
                        summary = str(community_text).strip()
                        log.debug("Using community_text as fallback for %s", community_id)

            if summary and str(summary).strip() not in ("nan", "None", ""):
                summaries.append(
                    f"[{title}] {str(summary)[:300]}" if title else str(summary)[:300]
                )
        if summaries:
            parts.append("Community context: " + " | ".join(summaries))

    # Fallback: raw entity descriptions when no relations/communities found
    if not parts and topk_entity is not None and not topk_entity.empty:
        entities = []
        for _, row in topk_entity.head(5).iterrows():
            name = row.get("name", "")
            desc = row.get("description", "")
            if name:
                entities.append(f"{name}: {str(desc)[:200]}")
        if entities:
            parts.append("Entity context: " + " | ".join(entities))

    formatted_result = "\n".join(parts) if parts else None
    log.debug(f"Context formatting complete. Result length: {len(formatted_result) if formatted_result else 0}")
    return formatted_result


# ---------------------------------------------------------------------------
# ArchRAG4SM inference class
# ---------------------------------------------------------------------------

class ArchRAG4SM:
    """
    Combines ArchRAG retrieval with the KG-RAG4SM schema matching inference.

    Drop-in replacement for KGRAG_for_Schema_Matching in kgrag4sm_main.py,
    but retrieves KG context dynamically from the ArchRAG index instead of
    using pre-computed paths stored in the Excel file.
    """

    def __init__(self, retriever: ArchRAGRetriever = None, model: Union[str, pipeline] = None, prompt_style: str = "kg"):
        self.retriever = retriever
        self.model = model  # Store model for on-demand summarization
        self.prompt_style = prompt_style
        self.token_tracker = TokenTracker()

    def generate_community_summary(self, community_id: str, community_row: dict) -> str:
        """Generate a concise summary for a retrieved community on-demand using LLM.

        Uses COMMUNITY_REPORT_PROMPT_SHORT from ArchRAG prompts and parses the
        JSON output to extract the summary field.
        Tracks and logs token usage.
        """
        if not self.model:
            return ""

        community_text = community_row.get("community_text", "")
        if not community_text or str(community_text).strip() in ("nan", "None", ""):
            return ""

        # Truncate input to avoid excessive tokens
        community_text = str(community_text).strip()[:1500]

        prompt = COMMUNITY_REPORT_PROMPT_SHORT.format(input_text=community_text)

        try:
            messages = [{"role": "user", "content": prompt}]

            if isinstance(self.model, str) and self.model.startswith("gpt"):
                api_base = os.getenv("AZURE_FOUNDRY_ENDPOINT")
                api_key = os.getenv("AZURE_FOUNDRY_API_KEY")

                if api_base and api_key:
                    client = openai.Client(base_url=api_base, api_key=api_key)
                else:
                    client = openai.Client()

                response = client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=500
                )
                raw_result = response.choices[0].message.content.strip()

                # Track token usage
                if hasattr(response, 'usage') and response.usage:
                    self.token_tracker.add_summary_tokens(
                        response.usage.prompt_tokens, response.usage.completion_tokens
                    )
                    tokens_info = f" | Tokens: {response.usage.prompt_tokens}+{response.usage.completion_tokens}={response.usage.total_tokens}"
                else:
                    tokens_info = ""

                # Parse JSON output to extract summary
                result = self._extract_summary_from_report(raw_result)
                log.info("Generated summary for %s: %s...%s", community_id, result[:80], tokens_info)
                for handler in log.handlers:
                    handler.flush()
                return result[:300]
            else:
                responses = self.model(
                    messages,
                    eos_token_id=[
                        self.model.tokenizer.eos_token_id,
                        self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                    ],
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.9,
                    pad_token_id=self.model.tokenizer.eos_token_id,
                )
                raw_result = responses[0]["generated_text"][-1]["content"].strip()
                result = self._extract_summary_from_report(raw_result)
                log.info("Generated summary for %s: %s... | Model: local", community_id, result[:80])
                for handler in log.handlers:
                    handler.flush()
                return result[:300]
        except Exception as e:
            log.warning("Failed to generate summary for %s: %s", community_id, e)
            for handler in log.handlers:
                handler.flush()
            return ""

    @staticmethod
    def _extract_summary_from_report(raw_result: str) -> str:
        """Extract the summary field from a COMMUNITY_REPORT_PROMPT JSON response.

        Falls back to the raw text if JSON parsing fails.
        """
        try:
            # Try to find JSON block in the response
            json_start = raw_result.find("{")
            json_end = raw_result.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                output = json.loads(raw_result[json_start:json_end])
                summary = output.get("summary", "")
                if summary and str(summary).strip() not in ("nan", "None", ""):
                    return str(summary).strip()
        except (json.JSONDecodeError, AttributeError):
            pass
        # Fallback: return raw text
        return raw_result.strip()

    def generate_system_prompt(self) -> str:
        if self.prompt_style == "kg":
            return ARCHRAG4SM_SYSTEM_PROMPT_KG
        return ARCHRAG4SM_SYSTEM_PROMPT_IT

    def generate_user_prompt(self, question: str, context: Optional[str]) -> str:
        ctx_text = (
            context
            if context
            else "No available knowledge graph context, please make the decision yourself."
        )
        return (
            f"(d) Based on the provided example and the following knowledge graph context, please answer the following schema matching question:\n\n"
            f"{question}\n\n"
            f"Knowledge Graph Context:\n{ctx_text}\n\n"
            f"Please respond with the label: 1 if attribute 1 and attribute 2 are semantically matched with each other, otherwise respond lable: 0.\n"
            f"Do not mention that there is not enough information to decide."
        )

    def get_llm_response(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Union[str, pipeline],
    ) -> str:
        """Get LLM response with token usage logging."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if isinstance(model, str) and model.startswith("gpt"):
            client = openai.Client()
            response = client.chat.completions.create(
                model=model, messages=messages
            )
            # Log token usage for inference and track
            if hasattr(response, 'usage') and response.usage:
                self.token_tracker.add_inference_tokens(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )
                log.info(f"Inference tokens: {response.usage.prompt_tokens}+{response.usage.completion_tokens}={response.usage.total_tokens}")
                for handler in log.handlers:
                    handler.flush()
            return response.choices[0].message.content
        else:
            terminators = [
                model.tokenizer.eos_token_id,
                model.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            responses = model(
                messages,
                eos_token_id=terminators,
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.5,
                top_k=1,
                top_p=0.9,
                pad_token_id=model.tokenizer.eos_token_id,
            )
            return responses[0]['generated_text'][-1]["content"].strip()

    def retrieve_context(
        self,
        question: str,
        model: Union[str, pipeline] = None,
    ) -> Optional[str]:
        """Retrieve and format KG context for a question (no LLM inference).

        If communities have 'nan' summaries, generates them on-demand using LLM.
        Returns the formatted context string, or None if nothing retrieved.
        """
        if self.retriever is None:
            raise RuntimeError(
                "Retriever not initialised — provide --index_dir or use --cache_context with a pre-built cache."
            )

        active_model = model or self.model
        if active_model and not self.model:
            self.model = active_model

        log.debug("Starting retrieval...")
        retrieval = self.retriever.retrieve(question)
        log.debug("Retrieval complete, formatting context...")

        context = format_retrieved_context(
            retrieval["topk_entity"],
            retrieval["topk_community"],
            retrieval["topk_related_r"],
            llm_summarizer=self.generate_community_summary,
        )
        log.debug("Context formatting complete.")
        return context

    def query_for_schema_matching(
        self,
        question: str,
        model: Union[str, pipeline],
        context: Optional[str] = None,
    ) -> Tuple[str, str, str, Optional[str]]:
        """
        Run ArchRAG retrieval then schema matching inference.

        If communities have 'nan' summaries, generates them on-demand using LLM.

        Parameters
        ----------
        question : str
            The schema matching question.
        model : Union[str, pipeline]
            The LLM model for inference.
        context : Optional[str]
            Pre-retrieved context. If None, retrieval is performed automatically.

        Returns
        -------
        system_prompt, user_prompt, llm_response, formatted_context
        """
        # Set model for on-demand summarization if not already set
        if not self.model:
            self.model = model

        if context is None:
            context = self.retrieve_context(question, model)
        log.debug("Starting inference...")

        system_prompt = self.generate_system_prompt()
        user_prompt = self.generate_user_prompt(question, context)
        response = self.get_llm_response(system_prompt, user_prompt, model)
        log.debug("Inference complete")
        return system_prompt, user_prompt, response, context
