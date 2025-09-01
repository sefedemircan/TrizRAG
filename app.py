import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib
import time
from typing import List, Dict
import requests
import os
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM


# .env dosyasını yükle
load_dotenv()

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
        self.dimension = 1024  # all-MiniLM-L6-v2 embedding boyutu
        self.chroma_api_key = os.getenv("CHROMA_API_KEY")
        self.chroma_tenant = os.getenv("CHROMA_TENANT")
        self.chroma_database = os.getenv("CHROMA_DATABASE")
        

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
                model_name='intfloat/multilingual-e5-large'
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

    

    @st.cache_resource
    def load_embedding_model(_self):
        """Embedding modelini yükle"""
        try:
            _self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
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
                        similarity_score = max(0.0, 1 - distance)
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

    def analyze_data_with_pandasai(self, df: pd.DataFrame, query: str) -> Dict:
        """PandasAI ile veri analizi yap"""
        try:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                return {
                    "success": False,
                    "error": "OpenRouter API key bulunamadı. Lütfen .env dosyasında OPENROUTER_API_KEY'i ayarlayın."
                }

            # OpenRouter API key'i environment variable olarak ayarla
            os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
            
            # LiteLLM ile LLM oluştur
            llm = LiteLLM(model="openrouter/mistralai/mistral-small-3.1-24b-instruct:free")
            
            # pandasai konfigürasyonu
            pai.config.set({
                "llm": llm
            })

            # pandas DataFrame'i pandasai DataFrame'e çevir
            pai_df = pai.DataFrame(df)

            # Veri analizi yap - pai.chat kullan
            with st.spinner("🤖 AI analyzing your data..."):
                response = pai_df.chat(query)

            return {
                "success": True,    
                "response": response,
                "dataframe_info": {
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.to_dict(),
                    "missing_values": df.isnull().sum().to_dict()
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Veri analizi hatası: {str(e)}"
            }

    def create_visualization(self, df: pd.DataFrame, chart_type: str, x_col: str, y_col: str, 
                           title: str = "", color_col: str = None) -> go.Figure:
        """Veri görselleştirme oluştur"""
        try:
            if chart_type == "bar":
                fig = px.bar(df, x=x_col, y=y_col, title=title, color=color_col)
            elif chart_type == "line":
                fig = px.line(df, x=x_col, y=y_col, title=title, color=color_col)
            elif chart_type == "scatter":
                fig = px.scatter(df, x=x_col, y=y_col, title=title, color=color_col)
            elif chart_type == "histogram":
                fig = px.histogram(df, x=x_col, title=title, color=color_col)
            elif chart_type == "box":
                fig = px.box(df, x=x_col, y=y_col, title=title, color=color_col)
            elif chart_type == "pie":
                fig = px.pie(df, values=y_col, names=x_col, title=title)
            else:
                fig = px.bar(df, x=x_col, y=y_col, title=title)

            fig.update_layout(
                template="plotly_white",
                title_x=0.5,
                height=500
            )
            return fig

        except Exception as e:
            st.error(f"Görselleştirme hatası: {e}")
            return None

    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Veri özeti oluştur"""
        try:
            summary = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
                "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            }

            # Sayısal sütunlar için istatistikler
            if summary["numeric_columns"]:
                summary["numeric_stats"] = df[summary["numeric_columns"]].describe().to_dict()

            # Kategorik sütunlar için unique değer sayıları
            if summary["categorical_columns"]:
                summary["categorical_stats"] = {
                    col: df[col].nunique() for col in summary["categorical_columns"]
                }

            return summary

        except Exception as e:
            return {"error": f"Özet oluşturma hatası: {str(e)}"}

    def suggest_visualizations(self, df: pd.DataFrame) -> List[Dict]:
        """Veri tipine göre görselleştirme önerileri"""
        suggestions = []
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Sayısal sütunlar için öneriler
            if len(numeric_cols) >= 2:
                suggestions.append({
                    "type": "scatter",
                    "title": f"{numeric_cols[0]} vs {numeric_cols[1]} İlişkisi",
                    "x_col": numeric_cols[0],
                    "y_col": numeric_cols[1],
                    "description": "İki sayısal değişken arasındaki ilişkiyi gösterir"
                })
                
                suggestions.append({
                    "type": "line",
                    "title": f"{numeric_cols[0]} Trendi",
                    "x_col": df.index.name if df.index.name else "Index",
                    "y_col": numeric_cols[0],
                    "description": "Zaman serisi analizi için uygun"
                })
            
            # Kategorik sütunlar için öneriler
            if categorical_cols and numeric_cols:
                suggestions.append({
                    "type": "bar",
                    "title": f"{categorical_cols[0]} Kategorilerine Göre {numeric_cols[0]}",
                    "x_col": categorical_cols[0],
                    "y_col": numeric_cols[0],
                    "description": "Kategorik değişkenlere göre sayısal değerlerin karşılaştırması"
                })
            
            # Histogram önerileri
            if numeric_cols:
                suggestions.append({
                    "type": "histogram",
                    "title": f"{numeric_cols[0]} Dağılımı",
                    "x_col": numeric_cols[0],
                    "description": "Sayısal değişkenin dağılımını gösterir"
                })
            
            # Box plot önerileri
            if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                suggestions.append({
                    "type": "box",
                    "title": f"{categorical_cols[0]} Kategorilerine Göre {numeric_cols[0]} Dağılımı",
                    "x_col": categorical_cols[0],
                    "y_col": numeric_cols[0],
                    "description": "Kategorilere göre sayısal değişkenin dağılımını gösterir"
                })
                
        except Exception as e:
            st.error(f"Öneri oluşturma hatası: {e}")
            
        return suggestions


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
                

                if embedding_success and chromadb_success:
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

    # Tab 2: Data Analytics
    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        
        # Help tooltip
        st.markdown("""
        <div class="help-tooltip">
            <strong>💡 Data Analytics:</strong> Upload CSV/Excel files and use natural language to analyze your data. 
            AI will help you understand patterns, create visualizations, and generate insights.
        </div>
        """, unsafe_allow_html=True)
        
        # Session state'de veri saklama
        if "uploaded_data" not in st.session_state:
            st.session_state.uploaded_data = None
        if "data_analysis_history" not in st.session_state:
            st.session_state.data_analysis_history = []
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📁 Data Upload")
            
            # Modern upload area
            st.markdown("""
            <div class="upload-area">
                <h4>📊 Upload Data Files</h4>
                <p>Drag & drop or click to upload CSV, Excel files</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Dosya yükleme
            uploaded_data_files = st.file_uploader(
                "Choose data files:",
                type=['csv', 'xlsx', 'xls'],
                accept_multiple_files=False,
                key="data_uploader"
            )
            
            # Örnek veri oluşturma
            st.subheader("🎲 Sample Data Generator")
            if st.button("📊 Generate Sample Dataset", use_container_width=True):
                # Örnek satış verisi oluştur
                np.random.seed(42)
                dates = pd.date_range('2024-01-01', periods=100, freq='D')
                categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
                
                sample_data = pd.DataFrame({
                    'Date': dates,
                    'Category': np.random.choice(categories, 100),
                    'Sales_Amount': np.random.normal(1000, 300, 100),
                    'Units_Sold': np.random.poisson(50, 100),
                    'Customer_Rating': np.random.uniform(1, 5, 100).round(1),
                    'Region': np.random.choice(['North', 'South', 'East', 'West'], 100)
                })
                
                st.session_state.uploaded_data = sample_data
                st.success("✅ Sample dataset generated successfully!")
                st.rerun()
            
            # Veri yükleme işlemi
            if uploaded_data_files:
                try:
                    if uploaded_data_files.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_data_files)
                    else:
                        df = pd.read_excel(uploaded_data_files)
                    
                    st.session_state.uploaded_data = df
                    st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                    
                except Exception as e:
                    st.error(f"❌ Error loading data: {e}")
            
            # Yüklenen veri özeti
            if st.session_state.uploaded_data is not None:
                df = st.session_state.uploaded_data
                st.subheader("📋 Data Overview")
                
                # Veri özeti
                summary = rag_system.get_data_summary(df)
                if "error" not in summary:
                    col1_metric, col2_metric, col3_metric = st.columns(3)
                    
                    with col1_metric:
                        st.metric("Rows", summary["shape"][0])
                    with col2_metric:
                        st.metric("Columns", summary["shape"][1])
                    with col3_metric:
                        st.metric("Memory (MB)", f"{summary['memory_usage']:.2f}")
                    
                    # Veri önizleme
                    with st.expander("👀 Preview Data"):
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    # Veri tipleri
                    with st.expander("🔍 Data Types & Missing Values"):
                        col1_type, col2_missing = st.columns(2)
                        
                        with col1_type:
                            st.write("**Data Types:**")
                            for col, dtype in summary["dtypes"].items():
                                st.write(f"• {col}: {dtype}")
                        
                        with col2_missing:
                            st.write("**Missing Values:**")
                            for col, missing in summary["missing_values"].items():
                                if missing > 0:
                                    st.write(f"• {col}: {missing}")
                                else:
                                    st.write(f"• {col}: ✅ Complete")
        
        with col2:
            st.header("🤖 AI-Powered Analysis")
            
            if st.session_state.uploaded_data is not None:
                df = st.session_state.uploaded_data
                
                # Doğal dil ile analiz
                st.subheader("💬 Ask Questions About Your Data")
                
                # Örnek sorular
                example_questions = [
                    "Bu veri setinde kaç satır var?",
                    "En yüksek satış miktarı nedir?",
                    "Kategorilere göre ortalama satış miktarını göster",
                    "Bölgelere göre müşteri puanlarını karşılaştır",
                    "Satış miktarı ile müşteri puanı arasında korelasyon var mı?"
                ]
                
                # Örnek soru seçimi
                selected_example = st.selectbox(
                    "💡 Example Questions:",
                    ["Custom Question"] + example_questions,
                    help="Choose an example question or write your own"
                )
                
                if selected_example != "Custom Question":
                    analysis_query = selected_example
                else:
                    analysis_query = st.text_input(
                        "Your analysis question:",
                        placeholder="Example: What is the average sales amount by category?",
                        key="analysis_query"
                    )
                
                if st.button("🔍 Analyze with AI", type="primary", use_container_width=True) and analysis_query:
                    #  PI key kontrolü
                    if not os.getenv("OPENROUTER_API_KEY"):
                        st.error("❌API key bulunamadı. Lütfen .env dosyasında OPENROUTER_API_KEY'i ayarlayın.")
                    else:
                        # AI analizi yap
                        analysis_result = rag_system.analyze_data_with_pandasai(df, analysis_query)
                        
                        if analysis_result["success"]:
                            # Sonucu göster
                            st.subheader("🎯 AI Analysis Result")
                            st.markdown(f"""
                            <div class="success-message">
                                {analysis_result["response"]}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Analiz geçmişine ekle
                            st.session_state.data_analysis_history.append({
                                "query": analysis_query,
                                "response": analysis_result["response"],
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            
                        else:
                            st.error(f"❌ Analysis failed: {analysis_result['error']}")
                
                # Görselleştirme önerileri
                st.subheader("📊 Visualization Suggestions")
                suggestions = rag_system.suggest_visualizations(df)
                
                if suggestions:
                    for i, suggestion in enumerate(suggestions[:3]):  # İlk 3 öneriyi göster
                        with st.expander(f"💡 {suggestion['title']}"):
                            st.write(suggestion['description'])
                            
                            if st.button(f"Create {suggestion['type'].title()} Chart", key=f"create_{i}"):
                                fig = rag_system.create_visualization(
                                    df, suggestion['type'], suggestion['x_col'], 
                                    suggestion.get('y_col', ''), suggestion['title']
                                )
                                if fig:
                                    st.session_state.current_chart = fig
            
            else:
                st.info("📁 Please upload a data file or generate sample data to start analysis.")
        
        # Alt kısım - Gelişmiş görselleştirme
        if st.session_state.uploaded_data is not None:
            st.markdown("---")
            st.header("🎨 Advanced Visualization")
            
            df = st.session_state.uploaded_data
            
            col1_viz, col2_viz = st.columns(2)
            
            with col1_viz:
                st.subheader("📈 Chart Configuration")
                
                # Grafik tipi seçimi
                chart_types = {
                    "Bar Chart": "bar",
                    "Line Chart": "line", 
                    "Scatter Plot": "scatter",
                    "Histogram": "histogram",
                    "Box Plot": "box",
                    "Pie Chart": "pie"
                }
                
                selected_chart = st.selectbox("Chart Type:", list(chart_types.keys()))
                chart_type = chart_types[selected_chart]
                
                # Sütun seçimi
                available_columns = df.columns.tolist()
                
                if chart_type in ["pie"]:
                    x_col = st.selectbox("Values Column:", available_columns)
                    y_col = None
                else:
                    x_col = st.selectbox("X-Axis Column:", available_columns)
                    y_col = st.selectbox("Y-Axis Column:", available_columns) if chart_type != "histogram" else None
                
                # Renk sütunu (opsiyonel)
                color_col = st.selectbox("Color Column (Optional):", ["None"] + available_columns)
                color_col = None if color_col == "None" else color_col
                
                # Başlık
                chart_title = st.text_input("Chart Title:", value=f"{selected_chart} of {x_col}")
                
                if st.button("🎨 Create Chart", type="primary"):
                    fig = rag_system.create_visualization(
                        df, chart_type, x_col, y_col, chart_title, color_col
                    )
                    if fig:
                        st.session_state.current_chart = fig
            
            with col2_viz:
                st.subheader("📊 Chart Display")
                
                if "current_chart" in st.session_state:
                    st.plotly_chart(st.session_state.current_chart, use_container_width=True)
                    
                    # Grafik indirme
                    chart_bytes = st.session_state.current_chart.to_image(format="png")
                    st.download_button(
                        label="📥 Download Chart as PNG",
                        data=chart_bytes,
                        file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )
                else:
                    st.info("🎨 Configure and create a chart to display it here.")
        
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