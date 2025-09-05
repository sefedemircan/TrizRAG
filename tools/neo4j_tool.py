import os
import json
from datetime import datetime
from typing import Dict

import pandas as pd
import requests
import streamlit as st
from neo4j import GraphDatabase
from urllib.parse import urlparse
import socket


class Neo4jTool:
    def __init__(self):
        self.neo4j_driver = None
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_username = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    def initialize_neo4j(self) -> bool:
        try:
            if self.neo4j_driver is not None:
                return True
            if not (self.neo4j_uri and self.neo4j_username and self.neo4j_password):
                st.error("Neo4j bilgileri eksik. Lütfen .env ayarlarını kontrol edin.")
                return False
            self.neo4j_driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_username, self.neo4j_password))
            # Yalnızca verify_connectivity ile doğrulama
            self.neo4j_driver.verify_connectivity()
            return True
        except Exception as e:
            st.error(f"Neo4j bağlantı hatası: {e}")
            diag = self.diagnose_neo4j_connectivity()
            if diag:
                st.info(diag)
            self.neo4j_driver = None
            return False

    def diagnose_neo4j_connectivity(self) -> str:
        try:
            # Yalnızca verify_connectivity ile teşhis
            temp_driver = None
            driver = self.neo4j_driver
            if driver is None and self.neo4j_uri and self.neo4j_username and self.neo4j_password:
                temp_driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_username, self.neo4j_password))
                driver = temp_driver
            try:
                if driver is None:
                    return "Neo4j: ❌ sürücü oluşturulamadı; ortam değişkenlerini kontrol edin"
                driver.verify_connectivity()
                return "Neo4j: ✅ verify_connectivity başarılı"
            except Exception as ve:
                return f"Neo4j: ❌ verify_connectivity hata ({ve})"
            finally:
                if temp_driver is not None:
                    try:
                        temp_driver.close()
                    except Exception:
                        pass
        except Exception:
            return ""

    def run_cypher(self, cypher: str, params: Dict = None):
        if self.neo4j_driver is None:
            return {"success": False, "error": "Neo4j bağlantısı yok"}
        try:
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                result = session.run(cypher, params or {})
                records = [r.data() for r in result]
            return {"success": True, "records": records}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_basic_schema(self):
        try:
            labels_q = "CALL db.labels()"
            rels_q = "CALL db.relationshipTypes()"
            labels = self.run_cypher(labels_q)
            rels = self.run_cypher(rels_q)
            if not labels.get("success") or not rels.get("success"):
                return {"labels": [], "relationships": []}
            return {
                "labels": [row.get("label") or list(row.values())[0] for row in labels["records"]],
                "relationships": [row.get("relationshipType") or list(row.values())[0] for row in rels["records"]]
            }
        except Exception:
            return {"labels": [], "relationships": []}

    def call_openrouter_llm(self, prompt: str, model: str) -> str:
        if not self.openrouter_api_key:
            return ""
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://streamlit.io",
            "X-Title": "Neo4j Cypher Generator"
        }
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You translate Turkish questions to Cypher. Return JSON with keys: cypher, notes. No markdown fences."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 400,
            "temperature": 0.2
        }
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=30)
        if resp.status_code != 200:
            return ""
        return resp.json()["choices"][0]["message"]["content"]

    def llm_generate_cypher(self, user_query: str, schema_hint: str = "", model: str = "microsoft/wizardlm-2-8x22b") -> Dict:
        prompt = (
            f"Soru: {user_query}\n\n"
            f"Şema ipuçları: {schema_hint}\n\n"
            "Lütfen aşağıdaki formatta yanıt ver:\n"
            "{\"cypher\": \"MATCH ... RETURN ...\", \"notes\": \"kısa açıklama\"}"
        )
        try:
            content = self.call_openrouter_llm(prompt, model)
            if not content:
                return {"success": False, "error": "LLM yanıtı alınamadı"}
            try:
                parsed = json.loads(content)
            except Exception:
                start = content.find("{")
                end = content.rfind("}")
                parsed = json.loads(content[start:end+1]) if start != -1 and end != -1 else {"cypher": content, "notes": "freeform"}
            return {"success": True, "cypher": parsed.get("cypher", ""), "notes": parsed.get("notes", "")}
        except Exception as e:
            return {"success": False, "error": str(e)}


