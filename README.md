# 🚀 TrizRAG - AI-Powered Document Intelligence & Data Analytics Platform

TrizRAG, belgelerinizi ve verilerinizi yapay zeka ile analiz eden, doğal dil ile soru-cevap yapabilen ve veri görselleştirme özellikleri sunan kapsamlı bir platformdur.

## ✨ Özellikler

### 📚 Document Intelligence (RAG)
- **Belge Yükleme**: TXT dosyaları yükleyin veya manuel metin girişi yapın
- **AI-Powered Search**: Semantic, keyword ve hybrid arama seçenekleri
- **Intelligent Q&A**: Belgeleriniz hakkında doğal dil ile soru sorun
- **Vector Database**: ChromaDB Cloud ile güçlü vektör arama

### 📊 Data Analytics
- **Veri Yükleme**: CSV, Excel dosyalarını kolayca yükleyin
- **AI-Powered Analysis**: Doğal dil ile veri analizi yapın
- **Smart Visualizations**: Otomatik grafik önerileri ve görselleştirme
- **Interactive Charts**: Plotly tabanlı interaktif grafikler
- **Data Insights**: PandasAI ile akıllı veri analizi

### 🤖 AI Models
- **Multiple LLMs**: WizardLM, Llama, Gemini, DeepSeek desteği
- **OpenRouter Integration**: Farklı AI modellerine tek API üzerinden erişim
- **PandasAI**: Veri analizi için özel AI asistanı

## 🚀 Kurulum

### 1. Gereksinimler
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
`.env` dosyası oluşturun ve gerekli API key'leri ekleyin:

```env
# OpenAI API Key (PandasAI için gerekli)
OPENAI_API_KEY=your_openai_api_key_here

# OpenRouter API Key (LLM çağrıları için)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# ChromaDB Cloud API Key
CHROMA_API_KEY=your_chroma_api_key_here

# ChromaDB Cloud Tenant
CHROMA_TENANT=your_chroma_tenant_here

# ChromaDB Cloud Database
CHROMA_DATABASE=your_chroma_database_here
```

### 3. Uygulamayı Çalıştırın
```bash
streamlit run app.py
```

## 📖 Kullanım

### Document Intelligence
1. **Sistemi Başlatın**: Sidebar'dan "Initialize System" butonuna tıklayın
2. **Belge Yükleyin**: TXT dosyaları yükleyin veya metin girişi yapın
3. **Soru Sorun**: Belgeleriniz hakkında doğal dil ile soru sorun
4. **AI Yanıtları**: AI, belgelerinizden bilgi çıkararak yanıt verir

### Data Analytics
1. **Veri Yükleyin**: CSV/Excel dosyası yükleyin veya örnek veri oluşturun
2. **AI Analizi**: "Bu veri setinde kaç satır var?" gibi sorular sorun
3. **Görselleştirme**: Otomatik grafik önerilerini kullanın
4. **Özel Grafikler**: İstediğiniz grafik tipini ve sütunları seçin

## 🎯 Örnek Kullanım Senaryoları

### Belge Analizi
- **Soru**: "Python'da hangi kütüphaneler makine öğrenmesi için kullanılır?"
- **AI Yanıt**: Belgelerinizden scikit-learn, TensorFlow, PyTorch gibi kütüphaneleri bulur

### Veri Analizi
- **Soru**: "Kategorilere göre ortalama satış miktarını göster"
- **AI Yanıt**: Veriyi analiz eder ve kategorilere göre ortalama satışları hesaplar

## 🔧 Teknik Detaylar

- **Frontend**: Streamlit
- **Vector Database**: ChromaDB Cloud
- **Embeddings**: Sentence Transformers (multilingual-e5-large)
- **LLM Integration**: OpenRouter API
- **Data Analysis**: PandasAI + OpenAI
- **Visualization**: Plotly

## 📊 Desteklenen Dosya Formatları

- **Documents**: TXT
- **Data**: CSV, XLSX, XLS

## 🌟 Özellikler

- **Multilingual Support**: Türkçe ve İngilizce desteği
- **Real-time Processing**: Anlık belge ve veri analizi
- **Cloud Integration**: ChromaDB Cloud ile ölçeklenebilir altyapı
- **Interactive UI**: Modern ve kullanıcı dostu arayüz
- **Export Options**: Grafikleri PNG olarak indirin

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 📞 İletişim

- **Proje**: [GitHub Repository](https://github.com/yourusername/trizrag)
- **Versiyon**: 1.0
- **Güncelleme**: 2025

---

**🚀 TrizRAG ile belgelerinizi ve verilerinizi yapay zeka gücüyle analiz edin!** 