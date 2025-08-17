# 🚀 TrizRAG - AI-Powered Document Intelligence Platform

**Transform your documents and data into intelligent insights with the power of AI**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0078D4?style=for-the-badge&logo=chromadb&logoColor=white)](https://chromadb.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)

## ✨ Features

### 🔍 **Document Intelligence**
- **Smart Document Upload**: Support for TXT files and manual text input
- **AI-Powered Search**: Semantic, keyword, and hybrid search capabilities
- **Intelligent Q&A**: Ask questions about your documents and get AI-generated answers
- **Vector Database**: ChromaDB Cloud integration for efficient document storage and retrieval

### 📊 **Data Analytics**
- **CSV Data Import**: Upload and analyze CSV datasets
- **AI Data Analysis**: Ask natural language questions about your data
- **Smart Insights**: Get intelligent analysis and visualizations
- **pandasai Integration**: Advanced AI-powered data manipulation

### 🚀 **Advanced Capabilities**
- **Multiple AI Models**: Support for various LLM providers via OpenRouter
- **Real-time Processing**: Instant document indexing and search
- **Scalable Architecture**: Cloud-ready vector database solution
- **Modern UI/UX**: Beautiful, responsive web interface

## 🛠️ Technologies

- **Frontend**: Streamlit (Modern Web UI)
- **Vector Database**: ChromaDB Cloud
- **AI Models**: OpenRouter (Multiple LLM providers)
- **Data Analysis**: pandasai + pandas
- **Embeddings**: SentenceTransformers
- **Language**: Python 3.8+

## 📋 Requirements

- Python 3.8 or higher
- OpenAI API key (via OpenRouter)
- ChromaDB Cloud account
- Internet connection for AI model access

## 🚀 Quick Start

### 1. **Clone Repository**
```bash
git clone <repository-url>
cd trizrag
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Environment Setup**
Create a `.env` file in the project root:
```env
# OpenRouter API Key (Get from: https://openrouter.ai/)
OPENROUTER_API_KEY=your_api_key_here

# ChromaDB Cloud Credentials
CHROMA_API_KEY=your_chroma_api_key
CHROMA_TENANT=your_tenant_name
CHROMA_DATABASE=your_database_name
```

### 4. **Launch Application**
```bash
streamlit run app.py
```

### 5. **Access TrizRAG**
Open your browser and navigate to: `http://localhost:8501`

## 🎯 Usage Guide

### **Document Intelligence**
1. **Initialize System**: Click "Initialize System" in the sidebar
2. **Upload Documents**: Add TXT files or paste text manually
3. **Ask Questions**: Use natural language to query your documents
4. **Get AI Answers**: Receive intelligent responses based on your content

### **Data Analytics**
1. **Upload Data**: Import CSV files or enter data manually
2. **Select Dataset**: Choose from available datasets
3. **Ask Questions**: Query your data in natural language
4. **AI Analysis**: Get intelligent insights and visualizations

## 🤖 Supported AI Models

- **🚀 WizardLM-2 8x22B** (Free tier)
- **🦙 Meta-Llama 3 8B** (Free tier)
- **🌟 Google Gemini 2.5 Pro** (Free tier)
- **🔍 DeepSeek R1** (Free tier)

## 🔑 API Keys

### **OpenRouter**
- Visit [OpenRouter.ai](https://openrouter.ai/)
- Create account and get API key
- Add to `.env` file as `OPENROUTER_API_KEY`

### **ChromaDB Cloud**
- Visit [ChromaDB Cloud](https://cloud.chromadb.com/)
- Create account and database
- Get API key, tenant, and database name
- Add to `.env` file

## 📁 Project Structure

```
trizrag/
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── README.md            # This file
└── chroma_db/           # Local ChromaDB storage (if used)
```

## 🎨 UI Features

- **Modern Design**: Beautiful gradient themes and modern components
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Interactive Elements**: Hover effects, loading states, and smooth transitions
- **Professional Branding**: TrizRAG identity throughout the interface

## 🔧 Configuration

### **Search Settings**
- **Search Type**: Choose between semantic, keyword, or hybrid search
- **Results Count**: Configure number of results returned
- **Similarity Threshold**: Set minimum similarity score for results

### **AI Model Selection**
- Switch between different LLM providers
- Adjust model parameters for optimal performance
- Free tier models available for testing

## 📊 Performance

- **Fast Indexing**: Documents processed in real-time
- **Efficient Search**: Vector similarity search with configurable thresholds
- **Scalable Storage**: ChromaDB Cloud handles large document collections
- **AI Response Time**: Typically 2-5 seconds depending on model and query complexity

## 🚀 Deployment

### **Local Development**
```bash
streamlit run app.py
```

### **Production Deployment**
- Deploy to Streamlit Cloud
- Use cloud-based ChromaDB
- Configure environment variables
- Scale based on usage requirements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: Report bugs via GitHub Issues
- **Community**: Join our community discussions

## 🎉 Acknowledgments

- **Streamlit**: For the amazing web framework
- **ChromaDB**: For vector database technology
- **OpenRouter**: For AI model access
- **pandasai**: For intelligent data analysis

---

**🚀 TrizRAG** - Transforming documents and data into intelligent insights since 2024 