import streamlit as st
import numpy as np
import time
import os
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tools import RagTool, AnalyticsTool, Neo4jTool


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

# Ana uygulama
def main():
    # Modern Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 TrizRAG</h1>
        <p>AI-Powered Document Intelligence & Data Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Araçları başlat
    if "rag_tool" not in st.session_state:
        st.session_state.rag_tool = RagTool()
    if "analytics_tool" not in st.session_state:
        st.session_state.analytics_tool = AnalyticsTool()
    if "neo4j_tool" not in st.session_state:
        st.session_state.neo4j_tool = Neo4jTool()

    rag_tool = st.session_state.rag_tool
    analytics_tool = st.session_state.analytics_tool
    neo4j_tool = st.session_state.neo4j_tool

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
                embedding_success = rag_tool.load_embedding_model()
                chromadb_success = rag_tool.initialize_chromadb()
                

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
        chromadb_status = "🟢 Connected" if rag_tool.collection else "🔴 Disconnected"
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
            <span class="status-indicator {'status-connected' if rag_tool.collection else 'status-disconnected'}"></span>
            <span><strong>ChromaDB Cloud:</strong> {chromadb_status}</span>
        </div>
        """, unsafe_allow_html=True)

        embedding_status = "🟢 Loaded" if rag_tool.embedding_model else "🔴 Not Loaded"
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
            <span class="status-indicator {'status-connected' if rag_tool.embedding_model else 'status-disconnected'}"></span>
            <span><strong>Embedding Model:</strong> {embedding_status}</span>
        </div>
        """, unsafe_allow_html=True)

        

        # ChromaDB Cloud bilgileri


        # Collection istatistikleri
        if rag_tool.collection:
            stats = rag_tool.get_collection_stats()
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
    tab1, tab2, tab3 = st.tabs(["📚 Document Intelligence", "📊 Data Analytics", "🕸️ Graph Chat (Neo4j)"])
    
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
                if not rag_tool.collection:
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
                        success = rag_tool.upsert_documents(documents, metadata_list)
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
                if not rag_tool.collection:
                    st.warning("⚠️ Please initialize the system first!")
                elif not rag_tool.openrouter_api_key:
                    st.error("❌ OpenRouter API key not found! Please set OPENROUTER_API_KEY in .env file.")
                else:
                    # Benzer dökümanları ara
                    with st.spinner(f"🔍 Searching for relevant information... ({search_type})"):
                        search_results = rag_tool.search_similar(
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
                            response = rag_tool.generate_rag_response(
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
                summary = analytics_tool.get_data_summary(df)
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
                        analysis_result = analytics_tool.analyze_data_with_pandasai(df, analysis_query)
                        
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
                suggestions = analytics_tool.suggest_visualizations(df)
                
                if suggestions:
                    for i, suggestion in enumerate(suggestions[:3]):  # İlk 3 öneriyi göster
                        with st.expander(f"💡 {suggestion['title']}"):
                            st.write(suggestion['description'])
                            
                            if st.button(f"Create {suggestion['type'].title()} Chart", key=f"create_{i}"):
                                fig = analytics_tool.create_visualization(
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
                    fig = analytics_tool.create_visualization(
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

    # Tab 3: Neo4j Graph Chat
    with tab3:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.header("🕸️ Neo4j Graph Chat")
        st.markdown("""
        <div class="help-tooltip">
            <strong>💡 Neo4j Graph Chat:</strong> Doğal dilde soru sorun, sistem soruyu Cypher'a çevirip graf üzerinde çalıştırır.
        </div>
        """, unsafe_allow_html=True)

        # Bağlantı ve şema
        colA, colB = st.columns([1, 2])
        with colA:
            if st.button("🔌 Initialize Neo4j", use_container_width=True):
                ok = neo4j_tool.initialize_neo4j()
                if ok:
                    st.success("✅ Neo4j connected!")
                    st.rerun()
                else:
                    st.error("❌ Neo4j initialization failed!")

            neo_status = "🟢 Connected" if neo4j_tool.neo4j_driver else "🔴 Disconnected"
            st.markdown(f"**Connection:** {neo_status}")

            if neo4j_tool.neo4j_driver:
                schema = neo4j_tool.get_basic_schema()
                with st.expander("📚 Basic Schema"):
                    st.write("Labels:", schema.get("labels", []))
                    st.write("Relationships:", schema.get("relationships", []))

        with colB:
            st.subheader("💬 Ask the Graph")
            if "neo4j_messages" not in st.session_state:
                st.session_state.neo4j_messages = []

            user_q = st.text_input(
                "Sorunuz:",
                placeholder="Örn: En çok ilişkisi olan ilk 5 düğüm kimler?"
            )

            if st.button("🧠 Ask Graph", type="primary", use_container_width=True) and user_q:
                if not neo4j_tool.neo4j_driver:
                    st.warning("⚠️ Önce Neo4j bağlantısını başlatın.")
                elif not neo4j_tool.openrouter_api_key:
                    st.error("❌ OpenRouter API anahtarı bulunamadı.")
                else:
                    with st.spinner("🔧 Generating Cypher with AI..."):
                        schema_hint = " , ".join((neo4j_tool.get_basic_schema().get("labels") or [])[:10])
                        gen = neo4j_tool.llm_generate_cypher(user_q, schema_hint=schema_hint, model=selected_model)
                    if not gen.get("success") or not gen.get("cypher"):
                        st.error(f"Cypher üretim hatası: {gen.get('error', 'empty cypher')}")
                    else:
                        cypher = gen["cypher"]
                        st.code(cypher, language="cypher")
                        with st.spinner("🚀 Running Cypher on Neo4j..."):
                            res = neo4j_tool.run_cypher(cypher)
                        if not res.get("success"):
                            st.error(f"Sorgu hatası: {res.get('error')}")
                        else:
                            records = res.get("records", [])
                            st.success(f"✅ {len(records)} kayıt döndü")
                            if records:
                                try:
                                    st.dataframe(pd.DataFrame(records))
                                except Exception:
                                    st.write(records)

                            # LLM ile sonuçları özetle
                            try:
                                summary_prompt = (
                                    "Aşağıdaki Cypher sonucu verilerini Türkçe kısa ve net bir şekilde özetle. "
                                    "Gerekirse madde işaretleri kullan. İşte JSON veriler: "
                                    f"{records[:50]}"
                                )
                                answer = neo4j_tool.call_openrouter_llm(summary_prompt, model=selected_model)
                            except Exception as _:
                                answer = ""

                            st.subheader("🎯 AI Summary")
                            st.markdown(f"""
                            <div class="success-message">{answer}</div>
                            """, unsafe_allow_html=True)

                            # Sohbet geçmişi
                            st.session_state.neo4j_messages.append({
                                "question": user_q,
                                "cypher": cypher,
                                "answer": answer,
                                "records_count": len(records),
                                "time": datetime.now().strftime("%H:%M:%S")
                            })

            # Geçmiş
            if st.session_state.neo4j_messages:
                with st.expander(f"🕑 Conversation History ({len(st.session_state.neo4j_messages)})"):
                    for i, m in enumerate(reversed(st.session_state.neo4j_messages), 1):
                        st.write(f"{i}. {m['time']} — {m['question']}")
                        st.code(m["cypher"], language="cypher")
                        st.caption(f"Records: {m['records_count']}")

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