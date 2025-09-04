import os
import time
import hashlib
from datetime import datetime
from typing import List, Dict

import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer


class RagTool:
    def __init__(self):
        self.embedding_model = None
        self.client = None
        self.collection = None
        self.collection_name = "rag-poc-collection"
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.chroma_api_key = os.getenv("CHROMA_API_KEY")
        self.chroma_tenant = os.getenv("CHROMA_TENANT")
        self.chroma_database = os.getenv("CHROMA_DATABASE")

    @st.cache_resource
    def load_embedding_model(_self):
        try:
            _self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
            return True
        except Exception as e:
            st.error(f"Embedding model yükleme hatası: {e}")
            return False

    def initialize_chromadb(self):
        try:
            self.client = chromadb.HttpClient(
                ssl=True,
                host='api.trychroma.com',
                tenant=self.chroma_tenant,
                database=self.chroma_database,
                headers={'x-chroma-token': self.chroma_api_key}
            )
            embedding_function = SentenceTransformerEmbeddingFunction(
                model_name='intfloat/multilingual-e5-large'
            )
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=embedding_function
                )
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=embedding_function
                )
            return True
        except Exception as e:
            st.error(f"ChromaDB Cloud bağlantı hatası: {e}")
            return False

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = start + chunk_size
            if end > text_length:
                end = text_length
            chunk = text[start:end]
            chunks.append(chunk.strip())
            if end == text_length:
                break
            start = end - overlap
        return [chunk for chunk in chunks if len(chunk.strip()) > 10]

    def generate_id(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def upsert_documents(self, documents: List[str], metadata_list: List[Dict] = None) -> bool:
        if not self.collection:
            return False
        try:
            all_ids, all_documents, all_metadatas = [], [], []
            with st.spinner("Dökümanlar işleniyor ve vektörleştiriliyor..."):
                for i, doc in enumerate(documents):
                    chunks = self.chunk_text(doc)
                    for j, chunk in enumerate(chunks):
                        chunk_id = f"doc_{i}_chunk_{j}_{self.generate_id(chunk)[:8]}"
                        metadata = {
                            "doc_id": i,
                            "chunk_id": j,
                            "timestamp": datetime.now().isoformat(),
                            "length": len(chunk)
                        }
                        if metadata_list and i < len(metadata_list):
                            metadata.update(metadata_list[i])
                        all_ids.append(chunk_id)
                        all_documents.append(chunk)
                        all_metadatas.append(metadata)
            self.collection.add(ids=all_ids, documents=all_documents, metadatas=all_metadatas)
            return True
        except Exception as e:
            st.error(f"Döküman ekleme hatası: {e}")
            return False

    def search_similar(self, query: str, top_k: int = 5, score_threshold: float = 0.5, search_type: str = "semantic") -> List[Dict]:
        if not self.collection:
            return []
        try:
            if search_type == "semantic":
                results = self.collection.query(query_texts=[query], n_results=top_k, include=["documents", "metadatas", "distances"])
            elif search_type == "keyword":
                results = self.collection.query(query_texts=[query], n_results=top_k, where={"$or": [{"text": {"$contains": word}} for word in query.lower().split() if len(word) > 2]}, include=["documents", "metadatas", "distances"]) 
            elif search_type == "hybrid":
                semantic_results = self.collection.query(query_texts=[query], n_results=top_k * 2, include=["documents", "metadatas", "distances"]) 
                keyword_results = self.collection.query(query_texts=[query], n_results=top_k * 2, where={"$or": [{"text": {"$contains": word}} for word in query.lower().split() if len(word) > 2]}, include=["documents", "metadatas", "distances"]) 
                all_results = []
                if semantic_results['documents'] and semantic_results['documents'][0]:
                    for doc, metadata, distance in zip(semantic_results['documents'][0], semantic_results['metadatas'][0], semantic_results['distances'][0]):
                        similarity_score = max(0.0, 1 - distance)
                        all_results.append({"text": doc, "score": similarity_score * 0.5, "metadata": metadata, "search_type": "semantic"})
                if keyword_results['documents'] and keyword_results['documents'][0]:
                    for doc, metadata, distance in zip(keyword_results['documents'][0], keyword_results['metadatas'][0], keyword_results['distances'][0]):
                        similarity_score = 1 - distance
                        all_results.append({"text": doc, "score": similarity_score * 0.3, "metadata": metadata, "search_type": "keyword"})
                unique_results = {}
                for result in all_results:
                    text_key = result['text'][:100]
                    if text_key not in unique_results:
                        unique_results[text_key] = result
                    else:
                        if result['score'] > unique_results[text_key]['score']:
                            unique_results[text_key] = result
                sorted_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)
                return [r for r in sorted_results[:top_k] if r['score'] >= score_threshold]
            else:
                results = self.collection.query(query_texts=[query], n_results=top_k, include=["documents", "metadatas", "distances"]) 
            filtered_results = []
            if results['documents'] and results['documents'][0]:
                for doc, metadata, distance in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                    similarity_score = 1 - distance
                    if similarity_score >= score_threshold:
                        filtered_results.append({"text": doc, "score": similarity_score, "metadata": metadata, "search_type": search_type})
            return filtered_results
        except Exception as e:
            st.error(f"Arama hatası: {e}")
            return []

    def call_openrouter_llm(self, prompt: str, model: str = "microsoft/wizardlm-2-8x22b") -> str:
        if not self.openrouter_api_key:
            return "❌ OpenRouter API key bulunamadı. Lütfen .env dosyasında OPENROUTER_API_KEY'i ayarlayın."
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://streamlit.io",
                "X-Title": "RAG PoC System"
            }
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Sen yardımcı bir asistansın. Verilen bilgileri kullanarak doğru, detaylı ve Türkçe yanıtlar ver."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"❌ API Hatası: {response.status_code} - {response.text}"
        except Exception as e:
            return f"❌ LLM çağrı hatası: {e}"

    def generate_rag_response(self, query: str, search_results: List[Dict], model: str) -> str:
        if not search_results:
            return "İlgili bilgi bulunamadı."
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"Kaynak {i}: {result['text']}")
        context = "\n\n".join(context_parts)
        prompt = f"""Aşağıdaki kaynaklardaki bilgileri kullanarak soruyu yanıtla:

KAYNAKLAR:
{context}

SORU: {query}

YANIT: Yukarıdaki kaynaklardaki bilgileri kullanarak soruyu detaylı bir şekilde yanıtla. Hangi kaynakları kullandığını belirt."""
        return self.call_openrouter_llm(prompt, model)

    def get_collection_stats(self) -> Dict:
        if not self.collection:
            return {}
        try:
            count = self.collection.count()
            return {"total_documents": count, "collection_name": self.collection_name}
        except Exception as e:
            return {"error": str(e)}


