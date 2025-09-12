# 🚀 TrizRAG - AI-Powered Document Intelligence & Data Analytics Platform

TrizRAG, belgelerinizi ve verilerinizi yapay zeka ile analiz eden, doğal dil ile soru-cevap yapabilen, veri görselleştirme ve graf veritabanı sorguları sunan kapsamlı bir platformdur.

## ✨ Özellikler

### 📚 Document Intelligence (RAG)
- **Belge Yükleme**: TXT dosyaları yükleyin veya manuel metin girişi yapın
- **AI-Powered Search**: Semantic, keyword ve hybrid arama seçenekleri
- **Intelligent Q&A**: Belgeleriniz hakkında doğal dil ile soru sorun
- **Vector Database**: ChromaDB Cloud ile güçlü vektör arama
- **Multi-language Support**: Türkçe ve İngilizce belge analizi

### 📊 Data Analytics
- **Veri Yükleme**: CSV, Excel dosyalarını kolayca yükleyin
- **AI-Powered Analysis**: Doğal dil ile veri analizi yapın
- **Smart Visualizations**: Otomatik grafik önerileri ve görselleştirme
- **Interactive Charts**: Plotly tabanlı interaktif grafikler
- **Data Insights**: PandasAI ile akıllı veri analizi
- **Fallback System**: Bağlantı sorunlarında yedek analiz sistemi

### 🕸️ Graph Database (Neo4j)
- **Natural Language Queries**: Doğal dil ile graf sorguları
- **Cypher Generation**: AI ile otomatik Cypher sorgu üretimi
- **Graph Visualization**: Graf yapısını görselleştirme
- **Schema Analysis**: Graf şemasını analiz etme
- **Cloud Integration**: Neo4j Aura Cloud desteği

### 🤖 AI Models
- **Multiple LLMs**: WizardLM, Llama, Gemini, DeepSeek desteği
- **OpenRouter Integration**: Farklı AI modellerine tek API üzerinden erişim
- **PandasAI**: Veri analizi için özel AI asistanı
- **Thread-safe Processing**: Güvenli asenkron işlem yönetimi

## 🚀 Kurulum

### 1. Gereksinimler
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
`.env` dosyası oluşturun ve gerekli API key'leri ekleyin:

```env
# OpenRouter API Key (LLM çağrıları için)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# ChromaDB Cloud Configuration
CHROMA_API_KEY=your_chroma_api_key_here
CHROMA_TENANT=your_chroma_tenant_here
CHROMA_DATABASE=your_chroma_database_here

# Neo4j Configuration
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
NEO4J_DATABASE=neo4j
```

### 3. API Key'leri Alma

#### OpenRouter API Key
1. [OpenRouter.ai](https://openrouter.ai/) hesabı oluşturun
2. API key'inizi alın ve `.env` dosyasına ekleyin

#### ChromaDB Cloud
1. [ChromaDB Cloud](https://www.trychroma.com/) hesabı oluşturun
2. Yeni database oluşturun
3. API key, tenant ve database bilgilerini alın

#### Neo4j Aura
1. [Neo4j Aura Console](https://console.neo4j.io/) hesabı oluşturun
2. Yeni AuraDB instance oluşturun
3. Connection details'den URI, username ve password'ü alın

### 4. Uygulamayı Çalıştırın
```bash
streamlit run app.py
```

## 📖 Kullanım

### Document Intelligence (RAG)
1. **Sistemi Başlatın**: Sidebar'dan "Initialize System" butonuna tıklayın
2. **Belge Yükleyin**: TXT dosyaları yükleyin veya metin girişi yapın
3. **Soru Sorun**: Belgeleriniz hakkında doğal dil ile soru sorun
4. **AI Yanıtları**: AI, belgelerinizden bilgi çıkararak yanıt verir
5. **Arama Türü**: Semantic, keyword veya hybrid arama seçin

### Data Analytics
1. **Veri Yükleyin**: CSV/Excel dosyası yükleyin veya örnek veri oluşturun
2. **AI Analizi**: "Bu veri setinde kaç satır var?" gibi sorular sorun
3. **Görselleştirme**: Otomatik grafik önerilerini kullanın
4. **Özel Grafikler**: İstediğiniz grafik tipini ve sütunları seçin
5. **Fallback System**: Bağlantı sorunlarında otomatik yedek analiz

### Graph Database (Neo4j)
1. **Neo4j Bağlantısı**: "Initialize Neo4j" butonuna tıklayın
2. **Graf Sorguları**: "En çok ilişkisi olan düğümler kimler?" gibi sorular sorun
3. **Cypher Üretimi**: AI otomatik olarak Cypher sorgusu üretir
4. **Sonuç Analizi**: Graf verilerini analiz edin ve görselleştirin
5. **Şema İnceleme**: Graf yapısını ve ilişkilerini keşfedin

## 🎯 Örnek Kullanım Senaryoları

### Belge Analizi (RAG)
- **Soru**: "Python'da hangi kütüphaneler makine öğrenmesi için kullanılır?"
- **AI Yanıt**: Belgelerinizden scikit-learn, TensorFlow, PyTorch gibi kütüphaneleri bulur
- **Arama Türü**: Semantic search ile en ilgili belgeleri bulur

### Veri Analizi
- **Soru**: "Kategorilere göre ortalama satış miktarını göster"
- **AI Yanıt**: Veriyi analiz eder ve kategorilere göre ortalama satışları hesaplar
- **Görselleştirme**: Otomatik bar chart oluşturur

### Graf Veritabanı Sorguları
- **Soru**: "En çok ilişkisi olan ilk 5 düğüm kimler?"
- **Cypher**: `MATCH (n) RETURN n, size((n)--()) as degree ORDER BY degree DESC LIMIT 5`
- **AI Yanıt**: Graf yapısını analiz eder ve en bağlantılı düğümleri bulur

### Çoklu Platform Analizi
- **Belge + Veri**: RAG ile belge analizi + CSV veri analizi
- **Graf + Belge**: Neo4j sorguları + belge arama
- **Tüm Platformlar**: Entegre analiz ve raporlama

## 🔧 Teknik Detaylar

### Frontend & UI
- **Frontend**: Streamlit
- **UI Components**: Modern, responsive design
- **Real-time Updates**: Live data processing
- **Multi-tab Interface**: Organized workflow

### Backend & AI
- **Vector Database**: ChromaDB Cloud
- **Embeddings**: Sentence Transformers (multilingual-e5-large)
- **LLM Integration**: OpenRouter API
- **Data Analysis**: PandasAI + Thread-safe processing
- **Graph Database**: Neo4j Aura Cloud
- **Visualization**: Plotly

### Architecture
- **Modular Design**: Separate tools for each functionality
- **Error Handling**: Comprehensive fallback systems
- **Thread Safety**: Asynchronous processing for stability
- **Cloud Integration**: Multiple cloud services support

## 📊 Desteklenen Dosya Formatları

- **Documents**: TXT
- **Data**: CSV, XLSX, XLS

## 🌟 Özellikler

### Dil Desteği
- **Multilingual Support**: Türkçe ve İngilizce desteği
- **Natural Language Processing**: Doğal dil ile sorgu yapma
- **AI-Powered Translation**: Otomatik dil algılama

### Performans & Güvenilirlik
- **Real-time Processing**: Anlık belge ve veri analizi
- **Thread-safe Operations**: Güvenli asenkron işlemler
- **Fallback Systems**: Bağlantı sorunlarında yedek sistemler
- **Error Recovery**: Otomatik hata kurtarma

### Cloud Integration
- **ChromaDB Cloud**: Ölçeklenebilir vektör veritabanı
- **Neo4j Aura**: Graf veritabanı cloud servisi
- **OpenRouter**: Çoklu AI model desteği
- **Scalable Architecture**: Bulut tabanlı ölçeklenebilir mimari

### Kullanıcı Deneyimi
- **Interactive UI**: Modern ve kullanıcı dostu arayüz
- **Export Options**: Grafikleri PNG olarak indirin
- **Progress Tracking**: İşlem durumu takibi
- **Diagnostic Tools**: Bağlantı ve sistem teşhis araçları

## 🔧 Sorun Giderme

### Yaygın Sorunlar

#### Data Analytics Bağlantı Kesilmesi
- **Sorun**: "AI analyzing your data..." sırasında bağlantı kesilir
- **Çözüm**: Sistem otomatik olarak yedek analiz devreye sokar
- **Önlem**: Büyük veri setleri otomatik olarak küçültülür

#### Neo4j Bağlantı Hatası
- **Sorun**: "Cannot resolve address" veya "encryption settings" hatası
- **Çözüm**: URI formatını kontrol edin (`neo4j+s://` kullanın)
- **Teşhis**: "Diagnose Connection" butonunu kullanın

#### ChromaDB Bağlantı Sorunu
- **Sorun**: Vector database bağlantısı kurulamıyor
- **Çözüm**: API key'leri ve tenant bilgilerini kontrol edin
- **Test**: "Initialize System" butonunu kullanın

### Sistem Gereksinimleri
- **Python**: 3.8+
- **RAM**: Minimum 4GB (8GB önerilen)
- **İnternet**: Stabil bağlantı (cloud servisler için)
- **Disk**: 2GB boş alan

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 📞 İletişim & Bilgi

- **Proje**: [GitHub Repository](https://github.com/yourusername/trizrag)
- **Versiyon**: 2.0
- **Son Güncelleme**: Ocak 2025
- **Desteklenen Platformlar**: Windows, macOS, Linux

## 🏆 Özellikler Özeti

✅ **Document Intelligence (RAG)** - Belge analizi ve arama  
✅ **Data Analytics** - AI destekli veri analizi  
✅ **Graph Database** - Neo4j ile graf sorguları  
✅ **Multi-LLM Support** - Çoklu AI model desteği  
✅ **Cloud Integration** - Bulut servis entegrasyonu  
✅ **Thread-safe Processing** - Güvenli asenkron işlemler  
✅ **Fallback Systems** - Yedek sistemler  
✅ **Diagnostic Tools** - Teşhis araçları  

---

**🚀 TrizRAG ile belgelerinizi, verilerinizi ve graf veritabanlarınızı yapay zeka gücüyle analiz edin!** 