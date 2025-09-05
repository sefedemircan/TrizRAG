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
            
            # URI formatını kontrol et ve düzelt
            uri = self.neo4j_uri.strip()
            
            # Çifte prefix kontrolü - eğer zaten neo4j+s:// varsa, ek prefix ekleme
            if uri.startswith('neo4j+s://') and 'neo4j+s://' in uri[9:]:
                # Çifte prefix varsa düzelt
                uri = uri.replace('neo4j+s://neo4j+s://', 'neo4j+s://')
                st.warning(f"⚠️ Çifte prefix düzeltildi: {uri}")
            elif not uri.startswith(('neo4j://', 'neo4j+s://', 'bolt://', 'bolt+s://')):
                st.warning("⚠️ Neo4j URI formatı düzeltiliyor...")
                if 'databases.neo4j.io' in uri:
                    uri = f"neo4j+s://{uri}"
                else:
                    uri = f"neo4j://{uri}"
            
            # Neo4j+s URI'si için özel ayarlar (dokümantasyona göre)
            if uri.startswith('neo4j+s://'):
                # neo4j+s URI'si için encryption ayarları kullanma
                self.neo4j_driver = GraphDatabase.driver(
                    uri, 
                    auth=(self.neo4j_username, self.neo4j_password),
                    max_connection_lifetime=30 * 60,  # 30 dakika
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=2 * 60  # 2 dakika
                )
            else:
                # Diğer URI'ler için encryption ayarları
                self.neo4j_driver = GraphDatabase.driver(
                    uri, 
                    auth=(self.neo4j_username, self.neo4j_password),
                    max_connection_lifetime=30 * 60,  # 30 dakika
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=2 * 60,  # 2 dakika
                    encrypted=True,
                    trust="TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"
                )
            
            # Bağlantıyı test et
            self.neo4j_driver.verify_connectivity()
            st.success(f"✅ Neo4j bağlantısı başarılı: {uri}")
            return True
                
        except Exception as conn_error:
            st.error(f"Neo4j bağlantı hatası: {conn_error}")
            
            # Alternatif URI formatlarını dene
            base_uri = self.neo4j_uri.strip()
            if base_uri.startswith('neo4j+s://'):
                base_uri = base_uri.replace('neo4j+s://', '')
            
            alternative_uris = [
                f"neo4j+s://{base_uri}",
                f"bolt+s://{base_uri}",
                f"neo4j://{base_uri}",
                f"bolt://{base_uri}"
            ]
            
            for alt_uri in alternative_uris:
                if alt_uri != uri:
                    try:
                        st.info(f"🔄 Alternatif URI deneniyor: {alt_uri}")
                        
                        # URI tipine göre ayarları belirle
                        if alt_uri.startswith(('neo4j+s://', 'bolt+s://')):
                            # SSL URI'leri için encryption ayarları kullanma
                            temp_driver = GraphDatabase.driver(
                                alt_uri, 
                                auth=(self.neo4j_username, self.neo4j_password)
                            )
                        else:
                            # Diğer URI'ler için encryption ayarları
                            temp_driver = GraphDatabase.driver(
                                alt_uri, 
                                auth=(self.neo4j_username, self.neo4j_password),
                                encrypted=True,
                                trust="TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"
                            )
                        
                        temp_driver.verify_connectivity()
                        self.neo4j_driver = temp_driver
                        st.success(f"✅ Alternatif URI ile bağlantı başarılı: {alt_uri}")
                        return True
                    except Exception as e:
                        st.warning(f"⚠️ {alt_uri} başarısız: {str(e)[:50]}...")
                        continue
            
            # Tüm alternatifler başarısız
            diag = self.diagnose_neo4j_connectivity()
            if diag:
                st.info(diag)
            self.neo4j_driver = None
            return False
                
        except Exception as e:
            st.error(f"Neo4j bağlantı hatası: {e}")
            diag = self.diagnose_neo4j_connectivity()
            if diag:
                st.info(diag)
            self.neo4j_driver = None
            return False

    def diagnose_neo4j_connectivity(self) -> str:
        try:
            if not (self.neo4j_uri and self.neo4j_username and self.neo4j_password):
                return "Neo4j: ❌ Ortam değişkenleri eksik (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)"
            
            # URI formatını kontrol et
            uri = self.neo4j_uri
            if not uri.startswith(('neo4j://', 'neo4j+s://', 'bolt://', 'bolt+s://')):
                return f"Neo4j: ❌ URI formatı hatalı: {uri}\nDoğru format: neo4j+s://hostname:port"
            
            # DNS çözümleme testi
            import socket
            try:
                from urllib.parse import urlparse
                parsed = urlparse(uri)
                host = parsed.hostname
                port = parsed.port or 7687
                
                # DNS çözümleme
                socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
                dns_status = f"DNS: ✅ {host} çözümlendi"
            except Exception as e:
                dns_status = f"DNS: ❌ {host} çözümlenemedi ({e})"
            
            # Bağlantı testi
            try:
                # URI tipine göre ayarları belirle
                if uri.startswith(('neo4j+s://', 'bolt+s://')):
                    # SSL URI'leri için encryption ayarları kullanma
                    temp_driver = GraphDatabase.driver(
                        uri, 
                        auth=(self.neo4j_username, self.neo4j_password)
                    )
                else:
                    # Diğer URI'ler için encryption ayarları
                    temp_driver = GraphDatabase.driver(
                        uri, 
                        auth=(self.neo4j_username, self.neo4j_password),
                        encrypted=True,
                        trust="TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"
                    )
                
                temp_driver.verify_connectivity()
                conn_status = "Neo4j: ✅ Bağlantı başarılı"
                temp_driver.close()
            except Exception as e:
                conn_status = f"Neo4j: ❌ Bağlantı hatası ({e})"
            
            return f"{dns_status}\n{conn_status}\n\n💡 Öneriler:\n- .env dosyasında NEO4J_URI formatını kontrol edin\n- Neo4j Cloud hesabınızın aktif olduğundan emin olun\n- Şifrenizi doğru girdiğinizden emin olun"
            
        except Exception as e:
            return f"Teşhis hatası: {e}"

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


