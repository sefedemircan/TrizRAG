import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM


class AnalyticsTool:
    def __init__(self):
        pass

    def _safe_pandasai_analysis(self, df: pd.DataFrame, query: str) -> str:
        """Güvenli PandasAI analizi - thread içinde çalışır"""
        try:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
            
            # Model seçimi
            try:
                llm = LiteLLM(model="openrouter/mistralai/mistral-small-3.1-24b-instruct:free")
            except:
                llm = LiteLLM(model="openrouter/microsoft/wizardlm-2-8x22b")
            
            # PandasAI konfigürasyonu
            pai.config.set({
                "llm": llm,
                "use_sandbox": False,
                "enable_cache": False,
                "max_retries": 1,
                "timeout": 25,
            })
            
            pai_df = pai.DataFrame(df)
            response = pai_df.chat(query)
            return response
            
        except Exception as e:
            raise Exception(f"PandasAI Error: {str(e)}")

    def analyze_data_with_openrouter_direct(self, df: pd.DataFrame, query: str) -> str:
        """OpenRouter'ı doğrudan kullanarak analiz yap"""
        try:
            import requests
            
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                return "API key bulunamadı"
            
            # DataFrame'i string'e çevir
            df_info = f"""
            Veri Seti Bilgileri:
            - Satır sayısı: {df.shape[0]}
            - Sütun sayısı: {df.shape[1]}
            - Sütunlar: {', '.join(df.columns.tolist())}
            - Sayısal sütunlar: {', '.join(df.select_dtypes(include=[np.number]).columns.tolist())}
            - Kategorik sütunlar: {', '.join(df.select_dtypes(include=['object']).columns.tolist())}
            
            İlk 5 satır:
            {df.head().to_string()}
            
            Temel istatistikler:
            {df.describe().to_string() if not df.select_dtypes(include=[np.number]).empty else 'Sayısal sütun yok'}
            """
            
            prompt = f"""
            Aşağıdaki veri seti hakkında soruyu yanıtla:
            
            {df_info}
            
            Soru: {query}
            
            Lütfen Türkçe olarak detaylı bir analiz yap. Veri setindeki sayısal değerleri kullan.
            """
            
            headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://streamlit.io",
                "X-Title": "Data Analytics"
            }
            
            data = {
                "model": "microsoft/wizardlm-2-8x22b",
                "messages": [
                    {"role": "system", "content": "Sen bir veri analisti asistanısın. Veri setlerini analiz eder ve Türkçe yanıtlar verirsin."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 800,
                "temperature": 0.3
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
                return f"API Hatası: {response.status_code}"
                
        except Exception as e:
            return f"Analiz hatası: {str(e)}"

    def analyze_data_with_pandasai(self, df: pd.DataFrame, query: str) -> Dict:
        try:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                return {"success": False, "error": "OpenRouter API key bulunamadı. Lütfen .env dosyasında OPENROUTER_API_KEY'i ayarlayın."}
            
            # DataFrame boyutunu kontrol et
            if df.shape[0] > 5000:
                st.warning("⚠️ Büyük veri seti tespit edildi. İlk 1000 satır analiz edilecek.")
                df = df.head(1000)
            
            # Önce doğrudan OpenRouter'ı dene (daha güvenli)
            try:
                response = self.analyze_data_with_openrouter_direct(df, query)
                if response and "API Hatası" not in response and "Analiz hatası" not in response:
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
                st.warning(f"⚠️ Doğrudan analiz başarısız: {str(e)[:50]}...")
            
            # PandasAI'yi thread'de dene
            import threading
            import queue
            
            result_queue = queue.Queue()
            error_queue = queue.Queue()
            
            def run_pandasai():
                try:
                    response = self._safe_pandasai_analysis(df, query)
                    result_queue.put(response)
                except Exception as e:
                    error_queue.put(str(e))
            
            # Thread'i başlat
            thread = threading.Thread(target=run_pandasai)
            thread.daemon = True
            thread.start()
            
            # Thread'in tamamlanmasını bekle (maksimum 20 saniye)
            thread.join(timeout=20)
            
            if thread.is_alive():
                # Timeout - yedek analiz döndür
                return {
                    "success": True,
                    "response": self._generate_fallback_analysis(df, query),
                    "dataframe_info": {
                        "shape": df.shape,
                        "columns": df.columns.tolist(),
                        "dtypes": df.dtypes.to_dict(),
                        "missing_values": df.isnull().sum().to_dict()
                    }
                }
            
            # Sonuçları kontrol et
            if not error_queue.empty():
                # Hata durumunda yedek analiz
                return {
                    "success": True,
                    "response": self._generate_fallback_analysis(df, query),
                    "dataframe_info": {
                        "shape": df.shape,
                        "columns": df.columns.tolist(),
                        "dtypes": df.dtypes.to_dict(),
                        "missing_values": df.isnull().sum().to_dict()
                    }
                }
            
            if not result_queue.empty():
                response = result_queue.get()
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
            else:
                # Sonuç yok - yedek analiz
                return {
                    "success": True,
                    "response": self._generate_fallback_analysis(df, query),
                    "dataframe_info": {
                        "shape": df.shape,
                        "columns": df.columns.tolist(),
                        "dtypes": df.dtypes.to_dict(),
                        "missing_values": df.isnull().sum().to_dict()
                    }
                }
                
        except Exception as e:
            return {"success": False, "error": f"Veri analizi hatası: {str(e)}"}

    def _generate_fallback_analysis(self, df: pd.DataFrame, query: str) -> str:
        """PandasAI başarısız olduğunda yedek analiz"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            analysis_parts = []
            analysis_parts.append(f"📊 **Veri Seti Özeti:**")
            analysis_parts.append(f"• Satır sayısı: {df.shape[0]:,}")
            analysis_parts.append(f"• Sütun sayısı: {df.shape[1]}")
            analysis_parts.append(f"• Sayısal sütunlar: {len(numeric_cols)}")
            analysis_parts.append(f"• Kategorik sütunlar: {len(categorical_cols)}")
            
            if numeric_cols:
                analysis_parts.append(f"\n📈 **Sayısal Sütunlar:** {', '.join(numeric_cols)}")
                for col in numeric_cols[:3]:  # İlk 3 sayısal sütun
                    stats = df[col].describe()
                    analysis_parts.append(f"• {col}: Ortalama={stats['mean']:.2f}, Min={stats['min']:.2f}, Max={stats['max']:.2f}")
            
            if categorical_cols:
                analysis_parts.append(f"\n📝 **Kategorik Sütunlar:** {', '.join(categorical_cols)}")
                for col in categorical_cols[:2]:  # İlk 2 kategorik sütun
                    unique_count = df[col].nunique()
                    top_value = df[col].value_counts().index[0] if unique_count > 0 else "N/A"
                    analysis_parts.append(f"• {col}: {unique_count} benzersiz değer, en yaygın: {top_value}")
            
            return "\n".join(analysis_parts)
        except Exception:
            return f"Veri analizi tamamlanamadı. Veri seti: {df.shape[0]} satır, {df.shape[1]} sütun."

    def _generate_simple_summary(self, df: pd.DataFrame) -> str:
        """Basit özet oluştur"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return f"Otomatik özet: {df.shape[0]} satır, {df.shape[1]} sütun, sayısal_sütunlar={numeric_cols}"

    def _generate_basic_stats(self, df: pd.DataFrame, query: str) -> str:
        """Temel istatistikler oluştur"""
        try:
            if "ortalama" in query.lower() or "average" in query.lower():
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    means = df[numeric_cols].mean()
                    return f"Ortalama değerler:\n" + "\n".join([f"• {col}: {val:.2f}" for col, val in means.items()])
            
            if "top" in query.lower() or "en yüksek" in query.lower():
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    max_vals = df[numeric_cols].max()
                    return f"En yüksek değerler:\n" + "\n".join([f"• {col}: {val:.2f}" for col, val in max_vals.items()])
            
            return self._generate_simple_summary(df)
        except Exception:
            return self._generate_simple_summary(df)

    def create_visualization(self, df: pd.DataFrame, chart_type: str, x_col: str, y_col: str, title: str = "", color_col: str = None) -> go.Figure:
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
            fig.update_layout(template="plotly_white", title_x=0.5, height=500)
            return fig
        except Exception as e:
            st.error(f"Görselleştirme hatası: {e}")
            return None

    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        try:
            summary = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
                "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            if summary["numeric_columns"]:
                summary["numeric_stats"] = df[summary["numeric_columns"]].describe().to_dict()
            if summary["categorical_columns"]:
                summary["categorical_stats"] = {col: df[col].nunique() for col in summary["categorical_columns"]}
            return summary
        except Exception as e:
            return {"error": f"Özet oluşturma hatası: {str(e)}"}

    def suggest_visualizations(self, df: pd.DataFrame) -> List[Dict]:
        suggestions = []
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if len(numeric_cols) >= 2:
                suggestions.append({"type": "scatter", "title": f"{numeric_cols[0]} vs {numeric_cols[1]} İlişkisi", "x_col": numeric_cols[0], "y_col": numeric_cols[1], "description": "İki sayısal değişken arasındaki ilişkiyi gösterir"})
                suggestions.append({"type": "line", "title": f"{numeric_cols[0]} Trendi", "x_col": df.index.name if df.index.name else "Index", "y_col": numeric_cols[0], "description": "Zaman serisi analizi için uygun"})
            if categorical_cols and numeric_cols:
                suggestions.append({"type": "bar", "title": f"{categorical_cols[0]} Kategorilerine Göre {numeric_cols[0]}", "x_col": categorical_cols[0], "y_col": numeric_cols[0], "description": "Kategorik değişkenlere göre sayısal değerlerin karşılaştırması"})
            if numeric_cols:
                suggestions.append({"type": "histogram", "title": f"{numeric_cols[0]} Dağılımı", "x_col": numeric_cols[0], "description": "Sayısal değişkenin dağılımını gösterir"})
            if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                suggestions.append({"type": "box", "title": f"{categorical_cols[0]} Kategorilerine Göre {numeric_cols[0]} Dağılımı", "x_col": categorical_cols[0], "y_col": numeric_cols[0], "description": "Kategorilere göre sayısal değişkenin dağılımını gösterir"})
        except Exception as e:
            st.error(f"Öneri oluşturma hatası: {e}")
        return suggestions


