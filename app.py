import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib
import time
import json
from typing import List, Dict, Any, Tuple
import requests
import os
from datetime import datetime
from dotenv import load_dotenv
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM
import pandas as pd
from pandasai import SmartDataframe

# .env dosyasını yükle
load_dotenv()

# pandasai konfigürasyonu
def setup_pandasai():
    """pandasai için LiteLLM konfigürasyonu"""
    try:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_api_key:
            # OpenRouter API key'i environment variable olarak ayarla
            os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
            
            # LiteLLM ile LLM oluştur
            llm = LiteLLM(model="openrouter/mistralai/mistral-small-3.1-24b-instruct:free")
            
            # pandasai konfigürasyonu - verilen örnekteki gibi
            pai.config.set({
                "llm": llm
            })
            
            return True
        else:
            st.warning("⚠️ pandasai için OPENROUTER_API_KEY bulunamadı")
            return False
    except Exception as e:
        st.error(f"pandasai konfigürasyon hatası: {e}")
        return False

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="TrizRAG - AI-Powered Document Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri
st.markdown("""
<style>
    .main-header {
        background: #ff4a4a;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    .feature-card {
        background: #ff4a4a;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-connected { background-color: #4CAF50; }
    .status-disconnected { background-color: #f44336; }
    .status-loading { background-color: #ff9800; }
    
    .metric-card {
        background: #ff4a4a;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .help-tooltip {
        
        border-left: 4px solid #ff4a4a;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-message {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .tab-content {
        padding: 1rem 0;
    }
    
    .upload-area {
        border: 1px solid;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #ff4a4a;
        background-color: #ff4a4a;
        transition: all 0.5s ease;
    }
    }
</style>
""", unsafe_allow_html=True)


class AdvancedRAGSystem:
    def __init__(self):
        self.embedding_model = None
        self.client = None
        self.collection = None
        self.collection_name = "rag-poc-collection"
        # OpenRouter API key'i .env'den otomatik al
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.dimension = 384  # all-MiniLM-L6-v2 embedding boyutu
        self.chroma_api_key = os.getenv("CHROMA_API_KEY")
        self.chroma_tenant = os.getenv("CHROMA_TENANT")
        self.chroma_database = os.getenv("CHROMA_DATABASE")
        # pandasai için DataFrame'ler
        self.dataframes = {}
        self.pandasai_configured = False

    def initialize_chromadb(self):
        """ChromaDB Cloud'u başlat"""
        try:
            # ChromaDB Cloud client'ı başlat
            self.client = chromadb.HttpClient(
                ssl=True,
                host='api.trychroma.com',
                tenant=self.chroma_tenant,
                database=self.chroma_database,
                headers={
                    'x-chroma-token': self.chroma_api_key
                }
            )
            
            # Embedding function oluştur
            embedding_function = SentenceTransformerEmbeddingFunction(
                model_name='all-MiniLM-L6-v2'
            )
            
            # Collection'ı oluştur veya al
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=embedding_function
                )
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=embedding_function
                )
            
            return True

        except Exception as e:
            st.error(f"ChromaDB Cloud bağlantı hatası: {e}")
            return False

    def initialize_pandasai(self):
        """pandasai'ı başlat"""
        try:
            self.pandasai_configured = setup_pandasai()
            return self.pandasai_configured
        except Exception as e:
            st.error(f"pandasai başlatma hatası: {e}")
            return False

    def add_dataframe(self, name: str, df: pd.DataFrame) -> bool:
        """DataFrame ekle"""
        try:
            # pandasai DataFrame'i olarak oluştur
            # Önce CSV olarak kaydet, sonra pandasai ile oku
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                df.to_csv(tmp_file.name, index=False)
                tmp_path = tmp_file.name
            
            # pandasai ile CSV'yi oku
            pai_df = pai.read_csv(tmp_path)
            
            # Geçici dosyayı sil
            os.unlink(tmp_path)
            
            # pandasai DataFrame'i sakla
            self.dataframes[name] = pai_df
            return True
        except Exception as e:
            st.error(f"DataFrame ekleme hatası: {e}")
            return False

    def get_dataframe(self, name: str):
        """DataFrame getir"""
        return self.dataframes.get(name)

    def list_dataframes(self) -> List[str]:
        """Mevcut DataFrame'lerin listesini getir"""
        return list(self.dataframes.keys())

    def chat_with_dataframe(self, df_name: str, query: str) -> str:
        """DataFrame ile sohbet et"""
        if not self.pandasai_configured:
            return "❌ pandasai henüz yapılandırılmadı!"
        
        if df_name not in self.dataframes:
            return f"❌ '{df_name}' adında DataFrame bulunamadı!"
        
        try:
            df = self.dataframes[df_name]
            # Verilen örnekteki gibi basit pandasai kullanımı
            response = df.chat(query)
            return str(response)
        except Exception as e:
            return f"❌ DataFrame sohbet hatası: {e}"

    def analyze_dataframe(self, df_name: str) -> Dict:
        """DataFrame analizi"""
        if df_name not in self.dataframes:
            return {"error": f"'{df_name}' adında DataFrame bulunamadı!"}
        
        try:
            pai_df = self.dataframes[df_name]
            # pandasai DataFrame'den orijinal pandas DataFrame'i al
            df = pai_df._df if hasattr(pai_df, '_df') else pai_df
            
            analysis = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
                "sample_data": df.head(5).to_dict('records')
            }
            return analysis
        except Exception as e:
            return {"error": f"Analiz hatası: {e}"}

    @st.cache_resource
    def load_embedding_model(_self):
        """Embedding modelini yükle"""
        try:
            _self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            return True
        except Exception as e:
            st.error(f"Embedding model yükleme hatası: {e}")
            return False

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Metni parçalara böl"""
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
        """Metin için benzersiz ID üret"""
        return hashlib.md5(text.encode()).hexdigest()

    def upsert_documents(self, documents: List[str], metadata_list: List[Dict] = None) -> bool:
        """Dökümanları ChromaDB'ye ekle"""
        if not self.collection:
            return False

        try:
            all_ids = []
            all_documents = []
            all_metadatas = []

            with st.spinner("Dökümanlar işleniyor ve vektörleştiriliyor..."):
                for i, doc in enumerate(documents):
                    # Metni parçalara böl
                    chunks = self.chunk_text(doc)

                    for j, chunk in enumerate(chunks):
                        # ID oluştur
                        chunk_id = f"doc_{i}_chunk_{j}_{self.generate_id(chunk)[:8]}"

                        # Metadata hazırla
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

            # ChromaDB'ye ekle
            self.collection.add(
                ids=all_ids,
                documents=all_documents,
                metadatas=all_metadatas
            )

            return True

        except Exception as e:
            st.error(f"Döküman ekleme hatası: {e}")
            return False

    def search_similar(self, query: str, top_k: int = 5, score_threshold: float = 0.5, search_type: str = "semantic") -> List[Dict]:
        """Benzer dökümanları ara"""
        if not self.collection:
            return []

        try:
            if search_type == "semantic":
                # Semantic search - embedding tabanlı
                results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
            elif search_type == "keyword":
                # Keyword search - where clause ile
                results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    where={"$or": [{"text": {"$contains": word}} for word in query.lower().split() if len(word) > 2]},
                    include=["documents", "metadatas", "distances"]
                )
            elif search_type == "hybrid":
                # Hybrid search - hem semantic hem keyword
                semantic_results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k * 2,  # Daha fazla sonuç al
                    include=["documents", "metadatas", "distances"]
                )
                
                keyword_results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k * 2,
                    where={"$or": [{"text": {"$contains": word}} for word in query.lower().split() if len(word) > 2]},
                    include=["documents", "metadatas", "distances"]
                )
                
                # Sonuçları birleştir ve sırala
                all_results = []
                
                # Semantic sonuçları ekle
                if semantic_results['documents'] and semantic_results['documents'][0]:
                    for i, (doc, metadata, distance) in enumerate(zip(
                        semantic_results['documents'][0], 
                        semantic_results['metadatas'][0], 
                        semantic_results['distances'][0]
                    )):
                        similarity_score = 1 - distance
                        all_results.append({
                            "text": doc,
                            "score": similarity_score * 0.5,  # Semantic ağırlığı
                            "metadata": metadata,
                            "search_type": "semantic"
                        })
                
                # Keyword sonuçları ekle
                if keyword_results['documents'] and keyword_results['documents'][0]:
                    for i, (doc, metadata, distance) in enumerate(zip(
                        keyword_results['documents'][0], 
                        keyword_results['metadatas'][0], 
                        keyword_results['distances'][0]
                    )):
                        similarity_score = 1 - distance
                        all_results.append({
                            "text": doc,
                            "score": similarity_score * 0.3,  # Keyword ağırlığı
                            "metadata": metadata,
                            "search_type": "keyword"
                        })
                
                # Tekrar eden sonuçları birleştir ve sırala
                unique_results = {}
                for result in all_results:
                    text_key = result['text'][:100]  # İlk 100 karakteri key olarak kullan
                    if text_key not in unique_results:
                        unique_results[text_key] = result
                    else:
                        # Daha yüksek skoru al
                        if result['score'] > unique_results[text_key]['score']:
                            unique_results[text_key] = result
                
                # Skora göre sırala ve top_k kadar al
                sorted_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)
                return [r for r in sorted_results[:top_k] if r['score'] >= score_threshold]
            else:
                # Default semantic search
                results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )

            # Sonuçları filtrele ve düzenle
            filtered_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    # Distance'ı similarity score'a çevir (ChromaDB cosine distance kullanır)
                    similarity_score = 1 - distance  # Cosine distance'ı similarity'e çevir
                    
                    if similarity_score >= score_threshold:
                        filtered_results.append({
                            "text": doc,
                            "score": similarity_score,
                            "metadata": metadata,
                            "search_type": search_type
                        })

            return filtered_results

        except Exception as e:
            st.error(f"Arama hatası: {e}")
            return []

    def call_openrouter_llm(self, prompt: str, model: str = "microsoft/wizardlm-2-8x22b") -> str:
        """OpenRouter API ile LLM çağrısı"""
        if not self.openrouter_api_key:
            return "❌ OpenRouter API key bulunamadı. Lütfen .env dosyasında OPENROUTER_API_KEY'i ayarlayın."

        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://streamlit.io",
                "X-Title": "RAG PoC System"
            }

            data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Sen yardımcı bir asistansın. Verilen bilgileri kullanarak doğru, detaylı ve Türkçe yanıtlar ver."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"❌ API Hatası: {response.status_code} - {response.text}"

        except Exception as e:
            return f"❌ LLM çağrı hatası: {e}"

    def generate_rag_response(self, query: str, search_results: List[Dict], model: str) -> str:
        """RAG ile yanıt üret"""
        if not search_results:
            return "İlgili bilgi bulunamadı."

        # Context oluştur
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"Kaynak {i}: {result['text']}")

        context = "\n\n".join(context_parts)

        # Prompt oluştur
        prompt = f"""Aşağıdaki kaynaklardaki bilgileri kullanarak soruyu yanıtla:

KAYNAKLAR:
{context}

SORU: {query}

YANIT: Yukarıdaki kaynaklardaki bilgileri kullanarak soruyu detaylı bir şekilde yanıtla. Hangi kaynakları kullandığını belirt."""

        return self.call_openrouter_llm(prompt, model)

    def get_collection_stats(self) -> Dict:
        """Collection istatistiklerini al"""
        if not self.collection:
            return {}

        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            return {"error": str(e)}


# Ana uygulama
def main():
    # Modern Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 TrizRAG</h1>
        <p>AI-Powered Document Intelligence & Data Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # RAG sistemi başlat
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = AdvancedRAGSystem()

    rag_system = st.session_state.rag_system

    # Sidebar - Ayarlar
    with st.sidebar:
        # Logo ve başlık
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #ff4a4a; margin-bottom: 0;">🚀 TrizRAG</h2>
            <p style="color: #666; font-size: 0.9rem; margin: 0;">Control Panel</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Sistem başlat
        st.header("⚙️ System Setup")
        if st.button("🔄 Initialize System", type="primary", use_container_width=True):
            with st.spinner("🚀 Initializing TrizRAG..."):
                # Embedding model yükle
                embedding_success = rag_system.load_embedding_model()
                # ChromaDB başlat
                chromadb_success = rag_system.initialize_chromadb()
                # pandasai başlat
                pandasai_success = rag_system.initialize_pandasai()

                if embedding_success and chromadb_success and pandasai_success:
                    st.success("✅ TrizRAG successfully initialized!")
                    st.rerun()
                else:
                    st.error("❌ System initialization failed!")

        st.divider()
        
        # Model seçimi
        st.header("🤖 AI Model Settings")
        available_models = {
            "🚀 WizardLM-2 8x22B (Free)": "microsoft/wizardlm-2-8x22b",
            "🦙 Meta-Llama 3 8B (Free)": "meta-llama/llama-3.2-3b-instruct:free",
            "🌟 Google: Gemini 2.5 Pro (Free)": "google/gemini-2.5-pro-exp-03-25",
            "🔍 DeepSeek R1 (Free)": "deepseek/deepseek-r1:free"
        }

        selected_model_name = st.selectbox(
            "Select LLM Model:",
            list(available_models.keys())
        )
        selected_model = available_models[selected_model_name]

        # Arama ayarları
        st.subheader("🔍 Search Configuration")
        
        # Arama tipi seçimi
        search_type = st.selectbox(
            "Search Type:",
            ["semantic", "keyword", "hybrid"],
            format_func=lambda x: {
                "semantic": "🔍 Semantic (AI-Powered)",
                "keyword": "🔤 Keyword (Text Matching)",
                "hybrid": "🔄 Hybrid (Best of Both)"
            }[x],
            help="Semantic: AI-powered understanding, Keyword: Exact text matching, Hybrid: Combines both approaches"
        )
        
        search_k = st.slider("Results Count:", 1, 10, 5)
        score_threshold = st.slider("Similarity Threshold:", 0.0, 1.0, 0.5, 0.1)

        st.divider()

        # Sistem durumu
        st.header("📊 System Status")

        # Bağlantı durumları
        chromadb_status = "🟢 Connected" if rag_system.collection else "🔴 Disconnected"
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
            <span class="status-indicator {'status-connected' if rag_system.collection else 'status-disconnected'}"></span>
            <span><strong>ChromaDB Cloud:</strong> {chromadb_status}</span>
        </div>
        """, unsafe_allow_html=True)

        embedding_status = "🟢 Loaded" if rag_system.embedding_model else "🔴 Not Loaded"
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
            <span class="status-indicator {'status-connected' if rag_system.embedding_model else 'status-disconnected'}"></span>
            <span><strong>Embedding Model:</strong> {embedding_status}</span>
        </div>
        """, unsafe_allow_html=True)

        # pandasai durumu
        pandasai_status = "🟢 Configured" if rag_system.pandasai_configured else "🔴 Not Configured"
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
            <span class="status-indicator {'status-connected' if rag_system.pandasai_configured else 'status-disconnected'}"></span>
            <span><strong>pandasai:</strong> {pandasai_status}</span>
        </div>
        """, unsafe_allow_html=True)

        # ChromaDB Cloud bilgileri


        # Collection istatistikleri
        if rag_system.collection:
            stats = rag_system.get_collection_stats()
            if "total_documents" in stats:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: 1.5rem;">{stats['total_documents']}</h3>
                    <p style="margin: 0; opacity: 0.9;">Total Documents</p>
                </div>
                """, unsafe_allow_html=True)
                

        st.divider()
        
        # Help section
        st.header("❓ Quick Help")
        with st.expander("How to use TrizRAG"):
            st.markdown("""
            **🚀 Getting Started:**
            1. Click "Initialize System" to start
            2. Upload documents in RAG tab
            3. Ask questions about your documents
            4. Upload data in pandasai tab for analysis
            
            **💡 Tips:**
            - Use semantic search for best results
            - Try different AI models for varied responses
            - Upload CSV files for data analysis
            """)

    # Ana içerik - Sekmeli yapı
    tab1, tab2 = st.tabs(["📚 Document Intelligence", "📊 Data Analytics"])
    
    # Tab 1: Document Intelligence (RAG)
    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        
        # Help tooltip
        st.markdown("""
        <div class="help-tooltip">
            <strong>💡 Document Intelligence:</strong> Upload documents and ask AI-powered questions. 
            TrizRAG will search through your documents and provide intelligent answers based on the content.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📄 Document Management")
            
            # Modern upload area
            st.markdown("""
            <div class="upload-area">
                <h4>📁 Upload Documents</h4>
                <p>Drag & drop or click to upload TXT files</p>
            </div>
            """, unsafe_allow_html=True)

            # Dosya yükleme
            uploaded_files = st.file_uploader(
                "Choose files:",
                type=['txt'],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )

            # Manuel döküman girişi
            st.subheader("✍️ Manual Document Input")
            
            # Örnek dökümanlar
            default_docs = """Python, yorumlanabilir, etkileşimli ve nesne yönelimli bir programlama dilidir. Guido van Rossum tarafından 1991 yılında geliştirilmiştir. Basit sözdizimi ve güçlü kütüphaneleri sayesinde veri bilimi, web geliştirme, yapay zeka ve otomasyon projelerinde yaygın olarak kullanılmaktadır.

Machine Learning (Makine Öğrenmesi), bilgisayarların verilerden öğrenmesini sağlayan yapay zeka dalıdır. Supervised learning (denetimli öğrenme), unsupervised learning (denetimsiz öğrenme) ve reinforcement learning (pekiştirmeli öğrenme) olmak üzere üç ana kategoriye ayrılır. Scikit-learn, TensorFlow ve PyTorch gibi kütüphaneler ML projelerinde sıkça kullanılır.

RAG (Retrieval-Augmented Generation), büyük dil modellerinin performansını artırmak için kullanılan bir tekniktir. Önce ilgili bilgiyi bir vektör veritabanından alır, sonra bu bilgiyi context olarak kullanarak LLM ile yanıt üretir. Bu sayede modelin bilgi güncelliği ve doğruluğu artar.

Pinecone, yüksek performanslı vektör veritabanı hizmetidir. Similarity search ve recommendation sistemlerinde kullanılır. Serverless mimarisi sayesinde ölçeklenebilir ve yönetimi kolaydır. Ücretsiz tier ile başlayıp, ihtiyaca göre ücretli planlara geçilebilir.

OpenRouter, farklı AI modellerine tek bir API üzerinden erişim sağlayan platformdur. GPT, Claude, Llama gibi çeşitli modelleri destekler. Ücretsiz tier ile deneme imkanı sunar ve pay-per-use modeliyle maliyet kontrolü sağlar."""

            documents_text = st.text_area(
                "Enter your documents (each paragraph as a separate document):",
                value=default_docs,
                height=300,
                placeholder="Paste your documents here..."
            )

            # Döküman ekleme
            if st.button("📚 Add Documents", type="primary", use_container_width=True):
                if not rag_system.collection:
                    st.warning("⚠️ Please initialize the system first!")
                else:
                    documents = []
                    metadata_list = []

                    # Yüklenen dosyalardan dökümanları al
                    if uploaded_files:
                        for file in uploaded_files:
                            content = str(file.read(), "utf-8")
                            documents.append(content)
                            metadata_list.append({
                                "source": file.name,
                                "type": "uploaded_file"
                            })

                    # Manuel girilen dökümanları al
                    if documents_text.strip():
                        manual_docs = [doc.strip() for doc in documents_text.split('\n\n') if doc.strip()]
                        documents.extend(manual_docs)
                        for i, doc in enumerate(manual_docs):
                            metadata_list.append({
                                "source": f"manual_input_{i + 1}",
                                "type": "manual_text"
                            })

                    if documents:
                        success = rag_system.upsert_documents(documents, metadata_list)
                        if success:
                            st.success(f"✅ {len(documents)} documents successfully added!")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("❌ Document addition failed!")
                    else:
                        st.warning("⚠️ No documents to add!")

        with col2:
            st.header("💬 AI-Powered Q&A")

            # Soru girişi
            query = st.text_input(
                "Ask your question:",
                placeholder="Example: What libraries are used for machine learning in Python?",
                key="rag_query"
            )

            if st.button("🔍 Get Intelligent Answer", type="primary", use_container_width=True) and query:
                if not rag_system.collection:
                    st.warning("⚠️ Please initialize the system first!")
                elif not rag_system.openrouter_api_key:
                    st.error("❌ OpenRouter API key not found! Please set OPENROUTER_API_KEY in .env file.")
                else:
                    # Benzer dökümanları ara
                    with st.spinner(f"🔍 Searching for relevant information... ({search_type})"):
                        search_results = rag_system.search_similar(
                            query,
                            top_k=search_k,
                            score_threshold=score_threshold,
                            search_type=search_type
                        )

                    if search_results:
                        # Arama performans metrikleri
                        avg_score = sum(r['score'] for r in search_results) / len(search_results)
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="margin: 0; font-size: 1.5rem;">{len(search_results)}</h3>
                            <p style="margin: 0; opacity: 0.9;">Results Found</p>
                            <small>Avg Score: {avg_score:.3f}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # RAG yanıtı üret
                        with st.spinner("🤖 Generating intelligent response..."):
                            response = rag_system.generate_rag_response(
                                query, search_results, selected_model
                            )

                        # Sonuçları göster
                        st.subheader("🎯 AI Response")
                        st.markdown(f"""
                        <div class="success-message">
                            {response}
                        </div>
                        """, unsafe_allow_html=True)

                        # Kaynak bilgileri
                        with st.expander(f"📚 Sources Used ({len(search_results)} documents)"):
                            for i, result in enumerate(search_results, 1):
                                # Arama tipi badge'i
                                search_type_badge = {
                                    "semantic": "🔍 Semantic",
                                    "keyword": "🔤 Keyword", 
                                    "hybrid": "🔄 Hybrid"
                                }.get(result.get('search_type', 'semantic'), '🔍')
                                
                                st.write(f"**Source {i}** {search_type_badge} (Score: {result['score']:.3f})")
                                st.write(result['text'])

                                # Metadata bilgileri
                                metadata = result.get('metadata', {})
                                if metadata:
                                    st.caption(f"📍 Source: {metadata.get('source', 'Unknown')} | "
                                               f"Type: {metadata.get('type', 'Unknown')}")
                                st.divider()
                    else:
                        st.warning("⚠️ No relevant information found. Try lowering the similarity threshold.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 2: Data Analytics (pandasai)
    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        
        # Help tooltip
        st.markdown("""
        <div class="help-tooltip">
            <strong>💡 Data Analytics:</strong> Upload CSV files or enter data manually, then ask AI-powered questions about your data. 
            TrizRAG will analyze your data and provide intelligent insights.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📁 Data Upload")
            
            # Modern upload area
            st.markdown("""
            <div class="upload-area">
                <h4>📊 Upload Data</h4>
                <p>Drag & drop or click to upload CSV files</p>
            </div>
            """, unsafe_allow_html=True)
            
            # CSV dosya yükleme
            csv_file = st.file_uploader(
                "Choose CSV file:",
                type=['csv'],
                key="csv_uploader",
                label_visibility="collapsed"
            )

            if csv_file:
                try:
                    df = pd.read_csv(csv_file)
                    df_name = csv_file.name.replace('.csv', '')
                    
                    if st.button(f"📊 Add '{df_name}' Dataset", type="primary", use_container_width=True):
                        success = rag_system.add_dataframe(df_name, df)
                        if success:
                            st.success(f"✅ '{df_name}' dataset successfully added!")
                            st.rerun()
                        else:
                            st.error("❌ Dataset addition failed!")
                    
                    # DataFrame önizleme
                    st.subheader("👀 Data Preview")
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin: 0; font-size: 1.5rem;">{df.shape[0]} × {df.shape[1]}</h3>
                        <p style="margin: 0; opacity: 0.9;">Rows × Columns</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(df.head(10))
                    
                except Exception as e:
                    st.error(f"CSV reading error: {e}")

            # Manuel veri girişi
            st.subheader("✍️ Manual Data Input")
            
            # Örnek veri
            sample_data = """name,age,city,salary
Ahmet,25,İstanbul,5000
Ayşe,30,Ankara,6000
Mehmet,35,İzmir,7000
Fatma,28,Bursa,5500
Ali,32,Antalya,6500"""
            
            manual_data = st.text_area(
                "Enter data in CSV format:",
                value=sample_data,
                height=150,
                placeholder="Enter your data here..."
            )
            
            if st.button("📊 Convert to Dataset", type="primary", use_container_width=True):
                try:
                    from io import StringIO
                    df = pd.read_csv(StringIO(manual_data))
                    df_name = "manual_data"
                    
                    success = rag_system.add_dataframe(df_name, df)
                    if success:
                        st.success(f"✅ '{df_name}' dataset successfully added!")
                        st.rerun()
                    else:
                        st.error("❌ Dataset addition failed!")
                        
                except Exception as e:
                    st.error(f"Data reading error: {e}")

        with col2:
            st.header("🗂️ Available Datasets")
            
            if rag_system.dataframes:
                for df_name in rag_system.list_dataframes():
                    pai_df = rag_system.get_dataframe(df_name)
                    # pandasai DataFrame'den orijinal pandas DataFrame'i al
                    df = pai_df._df if hasattr(pai_df, '_df') else pai_df
                    
                    with st.expander(f"📊 {df_name} ({df.shape[0]}×{df.shape[1]})"):
                        st.dataframe(df.head(5))
                        
                        # DataFrame analizi
                        if st.button(f"🔍 Analyze {df_name}", key=f"analyze_{df_name}"):
                            analysis = rag_system.analyze_dataframe(df_name)
                            if "error" not in analysis:
                                st.write("**📈 Data Analysis:**")
                                st.write(f"**Shape:** {analysis['shape']}")
                                st.write(f"**Columns:** {', '.join(analysis['columns'])}")
                                st.write(f"**Numeric Columns:** {', '.join(analysis['numeric_columns'])}")
                                st.write(f"**Categorical Columns:** {', '.join(analysis['categorical_columns'])}")
                                
                                # Eksik veri analizi
                                missing_data = {k: v for k, v in analysis['missing_values'].items() if v > 0}
                                if missing_data:
                                    st.write(f"**⚠️ Missing Values:** {missing_data}")
                            else:
                                st.error(analysis['error'])
            else:
                st.info("ℹ️ No datasets uploaded yet. Upload data from the left side!")

        # pandasai Sohbet - Tam genişlik
        if rag_system.pandasai_configured and rag_system.dataframes:
            st.divider()
            st.header("💬 AI Data Analysis")
            
            col3, col4 = st.columns([1, 1])
            
            with col3:
                # DataFrame seçimi
                selected_df = st.selectbox(
                    "Select dataset:",
                    rag_system.list_dataframes()
                )
                
                # Sohbet sorusu
                chat_query = st.text_input(
                    "Ask about your data:",
                    placeholder="Example: What is the average age? Who has the highest salary?",
                    key="pandasai_chat"
                )
                
                if st.button("🤖 Get AI Analysis", type="primary", use_container_width=True) and chat_query:
                    with st.spinner("🤖 AI is analyzing your data..."):
                        response = rag_system.chat_with_dataframe(selected_df, chat_query)
                    
                    st.subheader("🎯 AI Analysis")
                    st.markdown(f"""
                    <div class="success-message">
                        {response}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="help-tooltip">
                    <strong>💡 AI Data Analysis Tips:</strong>
                    
                    **📊 Basic Questions:**
                    • "What is the average age?"
                    • "Who has the highest salary?"
                    
                    **🔍 Advanced Analysis:**
                    • "Show me salary distribution by city"
                    • "What's the correlation between age and salary?"
                    • "Create a summary of the data"
                    
                    **📈 Visualization Requests:**
                    • "Plot age vs salary"
                    • "Show city distribution"
                </div>
                """, unsafe_allow_html=True)
        
        elif not rag_system.pandasai_configured:
            st.warning("⚠️ pandasai not configured yet!")
        elif not rag_system.dataframes:
            st.info("ℹ️ Please upload a dataset first!")

        st.markdown('</div>', unsafe_allow_html=True)

    # Alt bilgi
    st.markdown("---")
    
    # Modern footer
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("""
        <div style="text-align: center;">
            <h3 style="color: #ff4a4a; margin: 0;">🚀 TrizRAG</h3>
            <p style="color: #666; font-size: 0.9rem; margin: 0;">AI-Powered Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <p style="color: #666; margin: 0;">
                <strong>Powered by:</strong> ChromaDB Cloud • OpenRouter LLM • pandasai
            </p>
            <p style="color: #999; font-size: 0.8rem; margin: 0.5rem 0 0 0;">
                Transform your documents and data into intelligent insights
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center;">
            <p style="color: #666; font-size: 0.9rem; margin: 0;">Version 1.0</p>
            <p style="color: #999; font-size: 0.8rem; margin: 0;">© 2025 TrizRAG</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()