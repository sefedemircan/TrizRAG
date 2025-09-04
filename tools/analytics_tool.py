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

    def analyze_data_with_pandasai(self, df: pd.DataFrame, query: str) -> Dict:
        try:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                return {"success": False, "error": "OpenRouter API key bulunamadı. Lütfen .env dosyasında OPENROUTER_API_KEY'i ayarlayın."}
            os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
            llm = LiteLLM(model="openrouter/mistralai/mistral-small-3.1-24b-instruct:free")
            # Sandboxing bazı ortamlarda SQL icbarı hatasına yol açabiliyor; devre dışı bırakıyoruz
            pai.config.set({
                "llm": llm,
                "use_sandbox": False,
            })
            pai_df = pai.DataFrame(df)
            with st.spinner("🤖 AI analyzing your data..."):
                try:
                    response = pai_df.chat(query)
                except Exception as e:
                    # Bazı PandasAI sürümlerinde SQL yönlendirme hatası oluşabiliyor; basit yedek yanıt üretimi
                    msg = str(e)
                    if "execute_sql_query" in msg:
                        # Basit özet ve temel istatistikler döndür
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        summary = {
                            "rows": df.shape[0],
                            "cols": df.shape[1],
                            "numeric_cols": numeric_cols,
                        }
                        response = f"Otomatik özet: satır={summary['rows']}, sütun={summary['cols']}, sayısal_sütunlar={summary['numeric_cols']}"
                    else:
                        raise
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
            return {"success": False, "error": f"Veri analizi hatası: {str(e)}"}

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


