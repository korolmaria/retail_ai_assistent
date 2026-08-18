# src/rag_engine.py

import nest_asyncio
import pickle
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import logging
import time
import re
import math
import numpy as np
from collections import Counter
import concurrent.futures

# ================================================================
# НАСТРОЙКА ЛОГГЕРА
# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

nest_asyncio.apply()

import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

# ================================================================
# ИМПОРТЫ LLAMAINDEX
# ================================================================
from llama_index.core import (
    StorageContext, 
    Settings,
    Document,
    VectorStoreIndex,
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore, TextNode

# ================================================================
# ИМПОРТ НАШЕГО ПАРСЕРА И КОНФИГА
# ================================================================
from src.document_parser import DocumentParser
from src.config import Config

# Настройка LLM
try:
    from src.client import llamaindex_llm
    Settings.llm = llamaindex_llm
    logger.info("✅ LLM загружена из client.py")
except ImportError:
    from llama_index.llms.openai_like import OpenAILike
    Settings.llm = OpenAILike(
        api_key=Config.LM_STUDIO_API_KEY,
        api_base=Config.LM_STUDIO_URL,
        is_chat_model=True,
        context_window=8192,
        temperature=Config.TEMPERATURE_ANALYTICAL,
        max_tokens=Config.MAX_TOKENS,
        timeout=120,
    )
    logger.info("✅ LLM создана из конфига")

# ================================================================
# ПРОВЕРКА ДОСТУПНОСТИ ГРАФА
# ================================================================
GRAPH_AVAILABLE = False
try:
    from llama_index.graph_stores.neo4j import Neo4jGraphStore
    GRAPH_AVAILABLE = True
    logger.info("✅ Графовый индекс доступен (Neo4j)")
except ImportError as e:
    logger.warning(f"⚠️ Графовый индекс НЕ ДОСТУПЕН: {e}")
    GRAPH_AVAILABLE = False

# ================================================================
# НАСТРОЙКИ
# ================================================================

# Загрузка модели эмбеддингов
try:
    logger.info(f"🔄 Загрузка модели эмбеддингов: {Config.EMBEDDING_MODEL}")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=str(Config.EMBEDDING_MODEL),
        device=Config.EMBEDDING_DEVICE,
        trust_remote_code=True,
        cache_folder=str(Config.MODELS_DIR / "embeddings"),
    )
    logger.info("✅ Модель эмбеддингов загружена")
except Exception as e:
    logger.error(f"❌ Ошибка загрузки модели: {e}")
    logger.warning("⚠️ Используем fallback модель")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=Config.EMBEDDING_DEVICE,
        trust_remote_code=True,
    )

Settings.chunk_size = Config.CHUNK_SIZE
Settings.chunk_overlap = Config.CHUNK_OVERLAP

# Создаем сплиттер для чанков
node_parser = SentenceSplitter(
    chunk_size=Config.CHUNK_SIZE,
    chunk_overlap=Config.CHUNK_OVERLAP,
    paragraph_separator="\n\n",
    secondary_chunking_regex="[^,.;]+[,.;]?",
)

# Директории
CACHE_DIR = Config.CACHE_DIR
CHUNKS_DIR = Config.CHUNKS_DIR
DOCSTORE_DIR = Config.DOCSTORE_DIR
STORAGE_CONTEXT_DIR = Config.STORAGE_CONTEXT_DIR

# Файлы для хранения чанков
CHUNKS_METADATA_FILE = CHUNKS_DIR / "chunks_metadata.json"
CHUNKS_INDEX_FILE = CHUNKS_DIR / "chunks_index.pkl"

SEARCH_K = Config.SEARCH_K
logger.info(f"🔍 SEARCH_K = {SEARCH_K}")


# ================================================================
# КЛАСС ChunkManager
# ================================================================

class ChunkManager:
    """Управление чанками документов"""
    
    def __init__(self, chunks_dir: Path = None):
        self.chunks_dir = chunks_dir or CHUNKS_DIR
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunks_metadata_file = self.chunks_dir / "chunks_metadata.json"
        self.chunks_index_file = self.chunks_dir / "chunks_index.pkl"
        
        self._chunks_cache: Dict[str, Dict[str, Any]] = {}
        self._doc_hash: Optional[str] = None
        
        self._load_chunks()
    
    def _load_chunks(self):
        if self.chunks_metadata_file.exists():
            try:
                with open(self.chunks_metadata_file, 'r', encoding='utf-8') as f:
                    self._chunks_cache = json.load(f)
                logger.info(f"📦 Загружено {len(self._chunks_cache)} чанков из кэша")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка загрузки метаданных чанков: {e}")
                self._chunks_cache = {}
        
        if self.chunks_index_file.exists():
            try:
                with open(self.chunks_index_file, 'rb') as f:
                    index_data = pickle.load(f)
                    self._doc_hash = index_data.get('doc_hash')
                    logger.info(f"📦 Загружен хеш документов: {self._doc_hash[:12] if self._doc_hash else 'None'}")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка загрузки индекса чанков: {e}")
                self._doc_hash = None
    
    def _save_chunks(self):
        try:
            with open(self.chunks_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self._chunks_cache, f, ensure_ascii=False, indent=2)
            
            with open(self.chunks_index_file, 'wb') as f:
                pickle.dump({
                    'doc_hash': self._doc_hash,
                    'chunk_count': len(self._chunks_cache),
                    'timestamp': datetime.now().isoformat()
                }, f)
            
            logger.info(f"💾 Сохранено {len(self._chunks_cache)} чанков в кэш")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения чанков: {e}")
    
    def get_doc_hash(self) -> Optional[str]:
        return self._doc_hash
    
    def set_doc_hash(self, doc_hash: str):
        self._doc_hash = doc_hash
        self._save_chunks()
    
    def has_chunks(self) -> bool:
        return len(self._chunks_cache) > 0
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        return list(self._chunks_cache.values())
    
    def get_chunks_by_document(self, doc_name: str) -> List[Dict[str, Any]]:
        return [
            chunk for chunk in self._chunks_cache.values()
            if chunk.get('metadata', {}).get('source') == doc_name
        ]
    
    def get_chunks_by_page(self, page_num: int) -> List[Dict[str, Any]]:
        return [
            chunk for chunk in self._chunks_cache.values()
            if chunk.get('metadata', {}).get('page') == page_num
        ]
    
    def get_available_pages(self) -> List[int]:
        pages = set()
        for chunk in self._chunks_cache.values():
            page = chunk.get('metadata', {}).get('page')
            if page is not None:
                pages.add(page)
        return sorted(list(pages))
    
    def save_chunks_from_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        logger.info(f"📝 Создание чанков из {len(documents)} документов...")
        
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            doc_metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            source = doc_metadata.get('source', f'doc_{doc_idx}')
            page = doc_metadata.get('page', 1)
            doc_type = doc_metadata.get('type', 'unknown')
            
            chunks = node_parser.split_text(doc.text)
            
            logger.info(f"  📄 {source} (стр. {page}): {len(chunks)} чанков")
            
            for chunk_idx, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue
                
                chunk_id = f"{source}_p{page}_c{chunk_idx}"
                
                chunk_data = {
                    'id': chunk_id,
                    'text': chunk_text,
                    'metadata': {
                        'source': source,
                        'page': page,
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'document_type': doc_type,
                        'has_ocr': doc_metadata.get('has_ocr', False),
                        'has_tables': doc_metadata.get('has_tables', False),
                    },
                    'created_at': datetime.now().isoformat(),
                    'text_length': len(chunk_text),
                }
                
                self._chunks_cache[chunk_id] = chunk_data
                all_chunks.append(chunk_data)
        
        self._save_chunks()
        logger.info(f"✅ Создано {len(all_chunks)} чанков")
        
        return all_chunks
    
    def get_chunks_for_indexing(self) -> List[Document]:
        documents = []
        
        for chunk_id, chunk_data in self._chunks_cache.items():
            doc = Document(
                text=chunk_data['text'],
                metadata={
                    'chunk_id': chunk_id,
                    'source': chunk_data['metadata']['source'],
                    'page': chunk_data['metadata']['page'],
                    'chunk_index': chunk_data['metadata']['chunk_index'],
                    'total_chunks': chunk_data['metadata']['total_chunks'],
                    'document_type': chunk_data['metadata']['document_type'],
                }
            )
            documents.append(doc)
        
        logger.info(f"📚 Подготовлено {len(documents)} чанков для индексации")
        return documents
    
    def clear(self):
        self._chunks_cache = {}
        self._doc_hash = None
        self._save_chunks()
        logger.info("🧹 Кэш чанков очищен")


# ================================================================
# КЛАСС BM25Retriever
# ================================================================

class BM25Retriever:
    """BM25 ретривер для keyword поиска"""
    
    def __init__(self, nodes, similarity_top_k: int = 5, b: float = 0.75, k1: float = 1.2):
        self.nodes = nodes
        self.similarity_top_k = similarity_top_k
        self.b = b
        self.k1 = k1
        
        self._build_index()
    
    def _build_index(self):
        """Построение BM25 индекса"""
        self.node_texts = [node.text.lower() for node in self.nodes]
        self.node_ids = [node.node_id for node in self.nodes]
        
        # Документы
        self.doc_lengths = [len(text.split()) for text in self.node_texts]
        self.avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Строим инвертированный индекс
        self.inverted_index = {}
        self.doc_freq = Counter()
        
        stop_words = {'это', 'для', 'без', 'на', 'в', 'с', 'по', 'к', 'у', 'о', 'и', 'а', 'но', 'или', 'так', 'же',
                     'что', 'как', 'все', 'его', 'ее', 'их', 'был', 'была', 'было', 'были', 'из', 'от', 'до', 'за',
                     'при', 'про', 'через', 'над', 'под', 'об', 'от', 'перед', 'между', 'среди', 'вокруг', 'около'}
        
        for doc_idx, text in enumerate(self.node_texts):
            words = re.findall(r'[а-яА-Яa-zA-Z0-9]{3,}', text)
            words = [w for w in words if w.lower() not in stop_words]
            
            word_counts = Counter(words)
            for word, count in word_counts.items():
                if word not in self.inverted_index:
                    self.inverted_index[word] = []
                self.inverted_index[word].append((doc_idx, count))
                self.doc_freq[word] += 1
        
        self.total_docs = len(self.nodes)
        self.doc_id_to_node = {i: node for i, node in enumerate(self.nodes)}
        
        logger.info(f"✅ BM25 индекс построен: {len(self.inverted_index)} уникальных слов")
    
    def retrieve(self, query: str) -> List[NodeWithScore]:
        """Поиск по BM25"""
        if not query:
            return []
        
        query_words = re.findall(r'[а-яА-Яa-zA-Z0-9]{3,}', query.lower())
        
        stop_words = {'это', 'для', 'без', 'на', 'в', 'с', 'по', 'к', 'у', 'о', 'и', 'а', 'но', 'или', 'так', 'же',
                     'что', 'как', 'все', 'его', 'ее', 'их', 'был', 'была', 'было', 'были', 'из', 'от', 'до', 'за',
                     'при', 'про', 'через', 'над', 'под', 'об', 'от', 'перед', 'между', 'среди', 'вокруг', 'около'}
        query_words = [w for w in query_words if w.lower() not in stop_words]
        
        if not query_words:
            return []
        
        # Вычисляем BM25 score для каждого документа
        scores = np.zeros(self.total_docs)
        
        for word in query_words:
            if word not in self.inverted_index:
                continue
            
            # IDF
            idf = math.log((self.total_docs - self.doc_freq[word] + 0.5) / (self.doc_freq[word] + 0.5) + 1)
            
            for doc_idx, term_freq in self.inverted_index[word]:
                # TF с нормализацией
                doc_len = self.doc_lengths[doc_idx]
                norm_doc_len = doc_len / self.avg_doc_len
                
                # BM25 score
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * norm_doc_len)
                score = idf * numerator / denominator
                
                scores[doc_idx] += score
        
        # Сортируем по score
        top_indices = np.argsort(scores)[::-1][:self.similarity_top_k]
        
        results = []
        for doc_idx in top_indices:
            if scores[doc_idx] > 0:
                node = self.doc_id_to_node[doc_idx]
                results.append(NodeWithScore(node=node, score=float(scores[doc_idx])))
        
        return results


# ================================================================
# КЛАСС GraphRetriever
# ================================================================

class GraphRetriever:
    """Графовый ретривер для Neo4j"""
    
    def __init__(self, graph_store, similarity_top_k: int = 5):
        self.graph_store = graph_store
        self.similarity_top_k = similarity_top_k
        self._driver = None
        logger.info("✅ GraphRetriever инициализирован")
    
    def _get_driver(self):
        """Получение драйвера Neo4j"""
        if self._driver:
            return self._driver
        
        if hasattr(self.graph_store, 'driver'):
            self._driver = self.graph_store.driver
            return self._driver
        
        try:
            from neo4j import GraphDatabase
            from src.config import Config
            self._driver = GraphDatabase.driver(
                Config.NEO4J_URI,
                auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
            )
            return self._driver
        except Exception as e:
            logger.error(f"❌ Ошибка создания драйвера: {e}")
            return None
    
    def retrieve(self, query: str) -> List[NodeWithScore]:
        results = []
        
        if not query or not isinstance(query, str):
            return results
        
        driver = self._get_driver()
        if not driver:
            return results
        
        try:
            entities = re.findall(r'"([^"]+)"|«([^»]+)»|([А-Я][а-я]+(?:[-\s][А-Я][а-я]+)*)', query)
            entities = [e for group in entities for e in group if e]
            
            if not entities:
                entities = re.findall(r'\b[А-Яа-яA-Za-z]{4,}\b', query)
            
            if not entities:
                return results
            
            logger.info(f"🔍 Граф: найдены сущности: {entities[:3]}")
            
            with driver.session() as session:
                for entity in entities[:3]:
                    if not entity or len(str(entity)) < 2:
                        continue
                        
                    cypher = f"""
                    MATCH (n)
                    WHERE toLower(n.text) CONTAINS toLower('{entity}')
                       OR toLower(n.name) CONTAINS toLower('{entity}')
                    OPTIONAL MATCH (n)-[r]-(related)
                    RETURN n, collect(DISTINCT related) as related_nodes, 
                           collect(DISTINCT type(r)) as relations
                    LIMIT 5
                    """
                    
                    try:
                        result = session.run(cypher)
                        for record in result:
                            node = record['n']
                            related_nodes = record['related_nodes']
                            relations = record['relations']
                            
                            if node:
                                node_text = node.get('text', node.get('name', str(node)))
                                if related_nodes:
                                    node_text += f"\n\nСвязанные сущности: {', '.join([str(r) for r in related_nodes[:5]])}"
                                if relations:
                                    node_text += f"\n\nТипы связей: {', '.join(relations)}"
                                
                                text_node = TextNode(
                                    text=node_text,
                                    metadata={
                                        'source': 'graph',
                                        'entity': entity,
                                        'relations': relations,
                                        'related_count': len(related_nodes) if related_nodes else 0
                                    }
                                )
                                
                                results.append(NodeWithScore(node=text_node, score=0.85))
                                
                    except Exception as e:
                        logger.warning(f"⚠️ Ошибка графового запроса для {entity}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка графового ретривера: {e}")
        
        results.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)
        return results[:self.similarity_top_k]


# ================================================================
# КЛАСС ImprovedHybridRetriever
# ================================================================

class ImprovedHybridRetriever:
    """
    Улучшенный гибридный ретривер с параллельным запуском
    """
    
    def __init__(self, vector_retriever, keyword_retriever=None, bm25_retriever=None, 
                 graph_retriever=None, top_k=5):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = None  # Отключаем TF-IDF
        self.bm25_retriever = bm25_retriever
        self.graph_retriever = graph_retriever
        self.top_k = top_k
        
        # Веса для разных методов
        self.weights = {
            'vector': 1.0,
            'bm25': 0.9,
            'graph': 0.7
        }
        
        self.score_threshold = 0.1
        self.use_rrf = True
        self.rrf_k = 60
        self.timeout_seconds = 30
        
        logger.info(f"✅ ImprovedHybridRetriever создан (параллельный режим)")
    
    def retrieve(self, query: str) -> List[NodeWithScore]:
        """Гибридный поиск с параллельным запуском"""
        method_results = {}
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            
            futures['vector'] = executor.submit(self._get_vector_results, query)
            
            if self.bm25_retriever:
                futures['bm25'] = executor.submit(self.bm25_retriever.retrieve, query)
            
            if self.graph_retriever:
                futures['graph'] = executor.submit(self.graph_retriever.retrieve, query)
            
            for name, future in futures.items():
                try:
                    results = future.result(timeout=self.timeout_seconds)
                    if results:
                        method_results[name] = results
                        all_results.extend(results)
                        logger.info(f"📊 {name.capitalize()}: {len(results)} результатов")
                except concurrent.futures.TimeoutError:
                    logger.warning(f"⚠️ Таймаут {name} поиска (> {self.timeout_seconds} сек)")
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка {name} поиска: {e}")
        
        if not all_results:
            logger.warning("⚠️ Нет результатов от всех методов")
            return []
        
        if self.use_rrf and len(method_results) > 1:
            combined = self._reciprocal_rank_fusion(method_results)
        else:
            combined = self._score_fusion(all_results)
        
        combined = [r for r in combined if r.score and r.score >= self.score_threshold]
        top_results = combined[:self.top_k]
        
        for node in top_results:
            if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                if isinstance(node.node.metadata, dict):
                    if 'retriever_source' not in node.node.metadata:
                        source_guess = self._guess_source(node, method_results)
                        node.node.metadata['retriever_source'] = source_guess
        
        logger.info(f"✅ Объединено {len(top_results)} результатов из {len(all_results)} (использовано {len(method_results)} методов)")
        
        return top_results
    
    def _get_vector_results(self, query: str) -> List[NodeWithScore]:
        try:
            results = self.vector_retriever.retrieve(query)
            if results:
                max_score = max(r.score for r in results if r.score is not None)
                min_score = min(r.score for r in results if r.score is not None)
                if max_score > min_score:
                    for r in results:
                        if r.score is not None:
                            r.score = (r.score - min_score) / (max_score - min_score) * self.weights['vector']
            return results
        except Exception as e:
            logger.warning(f"⚠️ Ошибка векторного поиска: {e}")
            return []
    
    def _reciprocal_rank_fusion(self, method_results: Dict[str, List[NodeWithScore]]) -> List[NodeWithScore]:
        all_nodes = {}
        node_methods = {}
        
        for method, results in method_results.items():
            for rank, result in enumerate(results, 1):
                node_id = result.node.node_id
                if node_id not in all_nodes:
                    all_nodes[node_id] = result.node
                    node_methods[node_id] = {}
                node_methods[node_id][method] = rank
        
        final_scores = {}
        for node_id, methods in node_methods.items():
            rrf_score = 0.0
            for method, rank in methods.items():
                weight = self.weights.get(method, 0.5)
                rrf_score += weight / (self.rrf_k + rank)
            final_scores[node_id] = rrf_score
        
        sorted_nodes = sorted(
            [NodeWithScore(node=all_nodes[node_id], score=score) 
             for node_id, score in final_scores.items()],
            key=lambda x: x.score if x.score is not None else 0,
            reverse=True
        )
        
        return sorted_nodes
    
    def _score_fusion(self, results: List[NodeWithScore]) -> List[NodeWithScore]:
        grouped = {}
        for result in results:
            node_id = result.node.node_id
            if node_id not in grouped:
                grouped[node_id] = {
                    'node': result.node,
                    'scores': [],
                }
            if result.score is not None:
                grouped[node_id]['scores'].append(result.score)
        
        final_results = []
        for node_id, data in grouped.items():
            if data['scores']:
                avg_score = sum(data['scores']) / len(data['scores'])
                final_results.append(NodeWithScore(node=data['node'], score=avg_score))
        
        final_results.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)
        return final_results
    
    def _guess_source(self, node: NodeWithScore, method_results: Dict[str, List[NodeWithScore]]) -> str:
        node_id = node.node.node_id
        for method, results in method_results.items():
            for r in results:
                if r.node.node_id == node_id:
                    return method
        return 'unknown'
    
    def set_weights(self, vector: float = None, bm25: float = None, graph: float = None):
        if vector is not None:
            self.weights['vector'] = vector
        if bm25 is not None:
            self.weights['bm25'] = bm25
        if graph is not None:
            self.weights['graph'] = graph
        logger.info(f"⚖️ Веса обновлены: {self.weights}")


# ================================================================
# КЛАСС HybridRAG
# ================================================================

class HybridRAG:
    """Гибридный RAG с Vector + BM25 + Graph поиском"""
    
    def __init__(self):
        self.storage_context = None
        self.persist_dir = STORAGE_CONTEXT_DIR
        self.chunk_manager = ChunkManager()
        self.is_initialized = False
        
        # QDRANT
        logger.info("🔄 Подключение к Qdrant...")
        try:
            self.qdrant_client = QdrantClient(url=Config.QDRANT_URL)
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client, 
                collection_name=Config.QDRANT_COLLECTION
            )
            logger.info("✅ Подключение к Qdrant установлено")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к Qdrant: {e}")
            raise
        
        # NEO4J
        self.graph_store = None
        self.graph_retriever = None
        self.use_graph = False
        
        if GRAPH_AVAILABLE:
            try:
                self.graph_store = Neo4jGraphStore(
                    username=Config.NEO4J_USERNAME,
                    password=Config.NEO4J_PASSWORD,
                    url=Config.NEO4J_URI,
                    database="neo4j"
                )
                self.graph_retriever = GraphRetriever(
                    graph_store=self.graph_store,
                    similarity_top_k=SEARCH_K
                )
                self.use_graph = True
                logger.info("✅ Подключение к Neo4j установлено")
            except Exception as e:
                logger.warning(f"⚠️ Neo4j не доступен: {e}")
                self.graph_store = None
                self.graph_retriever = None
                self.use_graph = False
        
        # DOCSTORE
        self.docstore = SimpleDocumentStore()
        self.docstore_file = DOCSTORE_DIR / "docstore.json"
        
        # ИНДЕКСЫ
        self.vector_index = None
        self.query_engine = None
        self.hybrid_retriever = None
        self.documents_dir = Config.DOCUMENTS_DIR
        self._nodes_cache = None
        
        # ПАРСЕР
        self.parser = DocumentParser(
            use_ocr=Config.USE_OCR,
            ocr_lang=Config.OCR_LANGUAGE,
            cache_dir=Config.PARSED_DOCS_DIR
        )
        
        # РЕ-РАНКЕР
        logger.info("🔄 Загрузка ре-ранкера...")
        try:
            self.reranker = SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                top_n=5,
                keep_retrieval_score=True
            )
            logger.info("✅ Ре-ранкер загружен")
        except Exception as e:
            logger.warning(f"⚠️ Ре-ранкер не загружен: {e}")
            self.reranker = None

    # ============================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ============================================================
    
    def _get_documents_hash(self) -> str:
        hasher = hashlib.md5()
        extensions = Config.SUPPORTED_EXTENSIONS
        for ext in extensions:
            for file_path in self.documents_dir.glob(f'*{ext}'):
                if file_path.is_file():
                    hasher.update(file_path.name.encode())
                    hasher.update(str(file_path.stat().st_mtime).encode())
                    hasher.update(str(file_path.stat().st_size).encode())
        return hasher.hexdigest()
    
    def _check_qdrant(self) -> bool:
        try:
            collections = self.qdrant_client.get_collections().collections
            exists = any(c.name == Config.QDRANT_COLLECTION for c in collections)
            if exists:
                info = self.qdrant_client.get_collection(Config.QDRANT_COLLECTION)
                logger.info(f"📊 Qdrant: {info.points_count} точек")
                return info.points_count > 0
            return False
        except Exception as e:
            logger.error(f"Ошибка Qdrant: {e}")
            return False
    
    def _load_docstore(self) -> bool:
        if not self.docstore_file.exists():
            logger.info("📁 Docstore не найден")
            return False
        
        try:
            self.docstore = SimpleDocumentStore.from_persist_path(str(self.docstore_file))
            logger.info(f"✅ Docstore загружен из {self.docstore_file}")
            self._nodes_cache = list(self.docstore.docs.values())
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки docstore: {e}")
            return False
    
    def _save_docstore(self):
        try:
            self.docstore_file.parent.mkdir(parents=True, exist_ok=True)
            self.docstore.persist(str(self.docstore_file))
            logger.info(f"💾 Docstore сохранен: {self.docstore_file}")
            self._nodes_cache = list(self.docstore.docs.values())
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения docstore: {e}")
    
    def _parse_all_documents(self) -> List[Document]:
        logger.info("📚 Обработка документов через парсер...")
        documents = []
        
        for ext in Config.SUPPORTED_EXTENSIONS:
            for file_path in self.documents_dir.glob(f'*{ext}'):
                if not file_path.is_file() or file_path.name == '.gitkeep':
                    continue
                
                logger.info(f"  📄 Парсинг: {file_path.name}")
                parsed_data = self.parser.parse_document(file_path)
                
                if not parsed_data.get('pages'):
                    logger.warning(f"  ⚠️ Нет страниц в {file_path.name}")
                    continue
                
                for page_data in parsed_data['pages']:
                    page_num = page_data.get('page_num', 1)
                    page_text = page_data.get('text', '')
                    
                    if not page_text.strip():
                        continue
                    
                    doc = Document(
                        text=page_text,
                        metadata={
                            'source': file_path.name,
                            'page': page_num,
                            'total_pages': parsed_data.get('metadata', {}).get('total_pages', 1),
                            'has_ocr': 'ocr_text' in page_data,
                            'has_tables': page_data.get('has_tables', False),
                            'has_images': page_data.get('has_images', False),
                            'page_type': page_data.get('type', 'text'),
                        }
                    )
                    documents.append(doc)
                
                logger.info(f"  ✅ {file_path.name}: {len(parsed_data['pages'])} страниц")
        
        logger.info(f"📚 Всего загружено документов: {len(documents)}")
        return documents
    
    def _create_collection(self):
        try:
            if not self.qdrant_client.collection_exists(Config.QDRANT_COLLECTION):
                logger.info(f"🛠️ Создание коллекции...")
                test_embed = Settings.embed_model.get_text_embedding("test")
                self.qdrant_client.create_collection(
                    collection_name=Config.QDRANT_COLLECTION,
                    vectors_config=qdrant_models.VectorParams(
                        size=len(test_embed),
                        distance=qdrant_models.Distance.COSINE
                    ),
                    hnsw_config=qdrant_models.HnswConfigDiff(m=16, ef_construct=100)
                )
                logger.info("✅ Коллекция создана")
        except Exception as e:
            logger.error(f"❌ Ошибка создания коллекции: {e}")
            raise
    
    def _load_from_cache(self) -> bool:
        current_hash = self._get_documents_hash()
        cached_hash = self.chunk_manager.get_doc_hash()
        
        if cached_hash != current_hash:
            logger.info(f"🔄 Хеш изменился: {cached_hash[:12] if cached_hash else 'None'} -> {current_hash[:12]}")
            return False
        
        if not self.chunk_manager.has_chunks():
            logger.info("📁 Нет чанков в кэше")
            return False
        
        if not self._check_qdrant():
            logger.info("📁 Нет данных в Qdrant")
            return False
        
        if not self._load_docstore():
            return False
        
        if self.persist_dir.exists():
            try:
                self.storage_context = StorageContext.from_defaults(
                    vector_store=self.vector_store,
                    docstore=self.docstore,
                    persist_dir=str(self.persist_dir)
                )
                logger.info(f"✅ StorageContext восстановлен из {self.persist_dir}")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка восстановления StorageContext: {e}")
                return False
        else:
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                docstore=self.docstore
            )
        
        try:
            self.vector_index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=Settings.embed_model,
                storage_context=self.storage_context
            )
            logger.info("✅ Векторный индекс восстановлен")
        except Exception as e:
            logger.error(f"❌ Ошибка восстановления индекса: {e}")
            return False
        
        self._setup_query_engine()
        self.is_initialized = True
        logger.info("✅ Система готова (из кэша)")
        return True
    
    def _save_to_cache(self):
        self._save_docstore()
        
        if self.storage_context is not None:
            try:
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                self.storage_context.persist(persist_dir=str(self.persist_dir))
                logger.info(f"✅ StorageContext сохранен в {self.persist_dir}")
            except Exception as e:
                logger.error(f"❌ Ошибка сохранения StorageContext: {e}")
        
        current_hash = self._get_documents_hash()
        self.chunk_manager.set_doc_hash(current_hash)
    
    def _get_nodes(self) -> List[Document]:
        if self._nodes_cache is None:
            self._nodes_cache = list(self.docstore.docs.values())
        return self._nodes_cache
    
    def _build_graph_index(self, documents: List[Document]):
        if not self.use_graph or not self.graph_store:
            logger.warning("⚠️ Граф не доступен, пропускаем построение")
            return
        
        logger.info("🏗️ Построение графового индекса...")
        
        try:
            from neo4j import GraphDatabase
            
            driver = GraphDatabase.driver(
                Config.NEO4J_URI,
                auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
            )
            
            with driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("🗑️ Граф очищен")
                
                count = 0
                for doc in documents:
                    try:
                        text = doc.text
                        source = doc.metadata.get('source', 'unknown')
                        page = doc.metadata.get('page', 1)
                        
                        doc_id = f"{source}_p{page}"
                        
                        check = session.run("MATCH (d:Document {id: $doc_id}) RETURN d", {'doc_id': doc_id})
                        if check.single():
                            continue
                        
                        session.run("""
                        CREATE (d:Document {
                            id: $doc_id,
                            source: $source,
                            page: $page,
                            text: $text
                        })
                        """, {
                            'doc_id': doc_id,
                            'source': source,
                            'page': page,
                            'text': text[:1000]
                        })
                        count += 1
                        
                        concepts = re.findall(r'([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*)', text)
                        concepts = [c for c in concepts if len(c) > 5 and len(c) < 50]
                        
                        for concept in set(concepts[:10]):
                            if not concept.strip():
                                continue
                            session.run("MERGE (c:Concept {name: $name})", {'name': concept.strip()})
                            session.run("""
                            MATCH (d:Document {id: $doc_id})
                            MATCH (c:Concept {name: $concept})
                            CREATE (d)-[:CONTAINS_CONCEPT]->(c)
                            """, {'doc_id': doc_id, 'concept': concept.strip()})
                        
                        if count % 5 == 0:
                            logger.info(f"   Обработано {count} документов")
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Ошибка обработки документа: {e}")
                
                logger.info(f"✅ Создано {count} узлов Document")
                
                result = session.run("MATCH (n) RETURN count(n) as total")
                total = result.single()['total']
                logger.info(f"📊 Всего узлов в графе: {total}")
                
                self._graph_driver = driver
                
        except Exception as e:
            logger.error(f"❌ Ошибка построения графа: {e}")
            self.use_graph = False
    
    def _setup_query_engine(self, filters=None):
        """Настройка гибридного query engine (Vector + BM25 + Graph)"""
        if self.vector_index is None:
            raise ValueError("Индекс не инициализирован")
        
        logger.info("🔧 Настройка ГИБРИДНОГО query engine (Vector + BM25 + Graph)")
        
        vector_retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=SEARCH_K * 2,
            filters=filters,
        )
        logger.info("✅ Векторный ретривер создан")
        
        bm25_retriever = None
        nodes = self._get_nodes()
        
        if nodes:
            try:
                bm25_retriever = BM25Retriever(
                    nodes=nodes,
                    similarity_top_k=SEARCH_K * 2
                )
                logger.info("✅ BM25 ретривер создан")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка создания BM25 ретривера: {e}")
        
        self.hybrid_retriever = ImprovedHybridRetriever(
            vector_retriever=vector_retriever,
            keyword_retriever=None,
            bm25_retriever=bm25_retriever,
            graph_retriever=self.graph_retriever if self.use_graph else None,
            top_k=SEARCH_K
        )
        logger.info("✅ ImprovedHybridRetriever создан (параллельный режим: Vector + BM25 + Graph)")
        
        template_str = """
Ты — корпоративный ассистент компании ООО «Евроторг».

## КОНТЕКСТ:
{context_str}

## ВОПРОС:
{query_str}

## ИНСТРУКЦИИ:
1. Отвечай ТОЛЬКО на основе контекста.
2. Форматируй ответ так, чтобы он был удобен для чтения.
3. НЕ добавляй источники в ответ.

## ТВОЙ ОТВЕТ:"""
        
        prompt = PromptTemplate(template_str)
        
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.hybrid_retriever,
            response_synthesizer=TreeSummarize(
                llm=Settings.llm,
                summary_template=prompt,
            ),
            node_postprocessors=[self.reranker] if self.reranker else [],
            verbose=False,
        )
        
        self.query_engine._is_default = True
        logger.info("✅ Query engine настроен (Vector + BM25 + Graph, параллельный режим)")
    
    # ============================================================
    # НОВЫЙ МЕТОД: ПАРСИНГ СТРУКТУРИРОВАННОГО ОТВЕТА
    # ============================================================
    
    def _parse_structured_response(self, answer: str, sources: List[Dict]) -> Dict[str, Any]:
        """Парсит ответ в структурированный JSON"""
        result = {
            "title": "",
            "paragraphs": [],
            "lists": [],
            "code_blocks": [],
            "sources": [],
            "search_methods": ""
        }
        
        lines = answer.split('\n')
        current_list = None
        current_code = None
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # JSON блок
            if line.startswith('```json'):
                current_code = []
                continue
            if current_code is not None:
                if line.startswith('```'):
                    result["code_blocks"].append({
                        "language": "json",
                        "code": '\n'.join(current_code)
                    })
                    current_code = None
                    continue
                current_code.append(line)
                continue
            
            # Заголовок
            if line.startswith('###'):
                result["title"] = line.replace('###', '').strip()
                continue
            
            # Нумерованный список
            if re.match(r'^\d+\.\s', line):
                if current_list != 'ol':
                    current_list = 'ol'
                    result["lists"].append({"type": "ol", "items": []})
                item = re.sub(r'^\d+\.\s', '', line)
                result["lists"][-1]["items"].append(item)
                continue
            
            # Маркированный список
            if re.match(r'^[-•*]\s', line):
                if current_list != 'ul':
                    current_list = 'ul'
                    result["lists"].append({"type": "ul", "items": []})
                item = re.sub(r'^[-•*]\s', '', line)
                result["lists"][-1]["items"].append(item)
                continue
            
            # Обычный текст
            current_list = None
            result["paragraphs"].append(line)
        
        # Добавляем источники
        for src in sources:
            source = src.get('metadata', {}).get('source', '')
            page = src.get('page', '')
            if source and page:
                result["sources"].append(f"{source}, стр. {page}")
        
        return result
        
    # ============================================================
    # ПУБЛИЧНЫЕ МЕТОДЫ
    # ============================================================
    
    def initialize_and_index(self, force_reindex: bool = False):
        logger.info("🚀 Запуск инициализации...")
        
        if not force_reindex and self._load_from_cache():
            logger.info("✅ Система готова (из кэша)")
            return
        
        logger.info("🔄 Полная переиндексация...")
        
        documents = self._parse_all_documents()
        
        if not documents:
            logger.warning("⚠️ Нет документов для индексации")
            self.is_initialized = False
            return
        
        logger.info("📝 Создание чанков...")
        chunks = self.chunk_manager.save_chunks_from_documents(documents)
        logger.info(f"✅ Создано {len(chunks)} чанков")
        
        chunk_documents = self.chunk_manager.get_chunks_for_indexing()
        logger.info(f"📚 Подготовлено {len(chunk_documents)} документов для индексации")
        
        self._create_collection()
        
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=self.docstore,
        )
        
        logger.info("🧠 Создание векторного индекса (Qdrant)...")
        self.vector_index = VectorStoreIndex.from_documents(
            chunk_documents,
            storage_context=self.storage_context,
            embed_model=Settings.embed_model,
            show_progress=True
        )
        logger.info("✅ Векторный индекс создан")
        
        if self.use_graph:
            logger.info("🏗️ Создание графового индекса...")
            self._build_graph_index(chunk_documents)
        
        self._save_to_cache()
        self._setup_query_engine()
        self.is_initialized = True
        logger.info("✅ Система готова!")
    
    def query(self, question: str, verbose: bool = False, page_filter: int = None) -> Dict[str, Any]:
        """Гибридный поиск с объединением результатов"""
        if not self.query_engine:
            return {
                "answer": "Ошибка: RAG не инициализирован.",
                "sources": [],
                "status": "error"
            }
        
        try:
            start_time = time.time()
            
            # Проверка на запрос о странице
            page_patterns = [
                r'(?:про что|что|о чем|расскажи|информация|покажи|опиши)\s+(?:на|с|со|про)\s+(\d+)\s*(?:страниц[еы]|стр\.?)',
                r'страниц[еы]\s*(\d+)\s+(?:про что|что|о чем)',
                r'(\d+)\s*страниц[еы]\s*(?:про что|что|о чем)',
                r'покажи\s+(\d+)\s*страниц[уы]',
                r'страница\s+(\d+)\s*:?',
                r'(\d+)\s*страница',
            ]
            
            requested_page = None
            is_page_query = False
            
            if page_filter is not None:
                requested_page = page_filter
                is_page_query = True
            else:
                for pattern in page_patterns:
                    match = re.search(pattern, question.lower(), re.IGNORECASE)
                    if match:
                        requested_page = int(match.group(1))
                        is_page_query = True
                        break
            
            if is_page_query and requested_page:
                page_content = self._get_page_content(requested_page)
                
                if page_content:
                    answer = self._format_page_answer(requested_page, page_content, question)
                    sources = self._get_page_sources(requested_page)
                    
                    structured = self._parse_structured_response(answer, sources)
                    
                    return {
                        "answer": answer,
                        "sources": sources,
                        "structured": structured,
                        "status": "success",
                        "elapsed": time.time() - start_time,
                        "sources_count": len(sources),
                        "requested_page": requested_page,
                        "page_mode": True,
                        "hybrid_search": False,
                    }
                else:
                    available_pages = self._get_available_pages()
                    if available_pages:
                        pages_str = ', '.join(str(p) for p in available_pages[:20])
                        return {
                            "answer": f"❌ **Страница {requested_page} не найдена** в документах.\n\n📚 **Доступные страницы:** {pages_str}",
                            "sources": [],
                            "status": "not_found",
                            "elapsed": time.time() - start_time,
                            "requested_page": requested_page,
                            "available_pages": available_pages,
                        }
                    else:
                        return {
                            "answer": "❌ В документах нет страниц. Возможно, документы еще не спарсены.",
                            "sources": [],
                            "status": "not_found",
                            "elapsed": time.time() - start_time,
                        }
            
            # Гибридный запрос
            logger.info(f"🔍 Гибридный поиск: {question[:100]}...")
            
            response = self.query_engine.query(question)
            answer = str(response)
            sources = self._extract_sources(response)
            
            source_stats = {}
            for src in sources:
                source_type = src.get('source_type', 'unknown')
                source_stats[source_type] = source_stats.get(source_type, 0) + 1
            
            if source_stats:
                logger.info(f"📊 Методы поиска: {source_stats}")
            
            answer = self._clean_answer(answer)
            
            # ============================================================
            # ПАРСИМ СТРУКТУРИРОВАННЫЙ ОТВЕТ
            # ============================================================
            
            structured = self._parse_structured_response(answer, sources)
            
            if source_stats:
                methods_info = ", ".join(source_stats.keys())
                structured["search_methods"] = methods_info
            
            # Логируем результат парсинга для отладки
            logger.info(f"📝 Структурированный ответ: title={structured.get('title')}, "
                        f"paragraphs={len(structured.get('paragraphs', []))}, "
                        f"lists={len(structured.get('lists', []))}, "
                        f"code_blocks={len(structured.get('code_blocks', []))}")
            
            return {
                "answer": answer,
                "sources": sources,
                "structured": structured,  # ← КЛЮЧЕВАЯ СТРОКА
                "status": "success",
                "elapsed": time.time() - start_time,
                "sources_count": len(sources),
                "source_stats": source_stats,
                "hybrid_search": True,
                "methods_used": list(source_stats.keys()) if source_stats else ['vector'],
                "page_mode": False,
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка запроса: {e}", exc_info=True)
            return {
                "answer": f"Ошибка: {str(e)}",
                "sources": [],
                "status": "error"
            }
    
    def _get_page_content(self, page_num: int) -> Optional[str]:
        chunks = self.chunk_manager.get_chunks_by_page(page_num)
        if not chunks:
            return None
        chunks.sort(key=lambda x: x['metadata']['chunk_index'])
        return '\n\n'.join([chunk['text'] for chunk in chunks])
    
    def _get_page_sources(self, page_num: int) -> List[Dict[str, Any]]:
        chunks = self.chunk_manager.get_chunks_by_page(page_num)
        sources = []
        for chunk in chunks:
            sources.append({
                "text": chunk['text'],
                "metadata": chunk['metadata'],
                "page": page_num,
                "source": chunk['metadata'].get('source', 'unknown'),
            })
        return sources
    
    def _format_page_answer(self, page_num: int, content: str, question: str) -> str:
        headers = []
        header_patterns = [
            r'^([А-ЯЁ][А-ЯЁ\s\d.]+[А-ЯЁ])',
            r'^(\d+\.\d+\s+[А-ЯЁ][а-яё\s\d]+)',
            r'^(\d+\s+[А-ЯЁ][а-яё\s\d]+)',
            r'^(Приложение\s+\d+[\.\s]*[А-ЯЁа-яё\s\d]+)',
        ]
        
        for pattern in header_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            headers.extend(matches)
        headers = list(dict.fromkeys(headers))
        
        answer_parts = []
        answer_parts.append(f"📄 **Страница {page_num}**\n")
        
        if headers:
            answer_parts.append("**Содержание страницы:**")
            for header in headers[:10]:
                answer_parts.append(f"  • {header.strip()}")
            answer_parts.append("")
        
        paragraphs = content.split('\n\n')
        if paragraphs:
            first_paragraph = paragraphs[0][:500]
            answer_parts.append(f"**Начало текста:**\n{first_paragraph}...")
        
        sources = self._get_page_sources(page_num)
        if sources:
            source_names = list(set([s['source'] for s in sources]))
            answer_parts.append(f"\n📚 **Источники:** {', '.join(source_names)}")
        
        return '\n'.join(answer_parts)
    
    def _extract_sources(self, response) -> List[Dict[str, Any]]:
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes[:10]:
                try:
                    node_text = None
                    node_page = None
                    node_metadata = {}
                    source_type = None
                    
                    if hasattr(node, 'text') and node.text:
                        node_text = node.text
                    elif hasattr(node, 'node') and hasattr(node.node, 'text'):
                        node_text = node.node.text
                    
                    if hasattr(node, 'metadata'):
                        node_metadata = node.metadata
                        node_page = node.metadata.get('page')
                        source_type = node.metadata.get('retriever_source')
                    
                    if not node_page and node_text:
                        page_in_text = re.search(r'\[Страница (\d+)\]', node_text)
                        if page_in_text:
                            node_page = int(page_in_text.group(1))
                    
                    if node_text:
                        sources.append({
                            "text": node_text,
                            "score": float(node.score) if hasattr(node, 'score') and node.score is not None else None,
                            "metadata": node_metadata,
                            "page": node_page,
                            "source_type": source_type,
                        })
                except Exception:
                    continue
        return sources
    
    def _clean_answer(self, answer: str) -> str:
        answer = re.sub(r'\[Страница\s+(\d+)\]', r'страница \1', answer)
        answer = re.sub(r'\[OCR Страница\s+(\d+)\]', r'страница \1', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()
        return answer
    
    def _get_available_pages(self) -> List[int]:
        return self.chunk_manager.get_available_pages()
    
    def ask(self, question: str) -> str:
        result = self.query(question)
        return result.get('answer', 'Нет ответа')
    
    def get_stats(self) -> Dict[str, Any]:
        docs_count = len(self._get_nodes()) if self._get_nodes() else 0
        
        methods = ['vector', 'bm25']
        if self.use_graph:
            methods.append('graph')
        
        return {
            'collection': Config.QDRANT_COLLECTION,
            'is_initialized': self.is_initialized,
            'chunks_count': len(self.chunk_manager._chunks_cache) if self.chunk_manager else 0,
            'pages_available': self._get_available_pages(),
            'documents_in_docstore': docs_count,
            'nodes_for_keyword': docs_count,
            'qdrant_points': self._check_qdrant(),
            'search_k': SEARCH_K,
            'use_graph': self.use_graph,
            'graph_available': GRAPH_AVAILABLE,
            'hybrid_retriever': self.hybrid_retriever is not None,
            'methods': methods
        }
    
    def set_search_weights(self, vector: float = None, bm25: float = None, graph: float = None):
        if hasattr(self.hybrid_retriever, 'set_weights'):
            self.hybrid_retriever.set_weights(vector, bm25, graph)
            logger.info("⚖️ Веса поиска обновлены")
            self._setup_query_engine()


# ================================================================
# ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР
# ================================================================

rag_agent = HybridRAG()
logger.info("✅ RAG агент создан с ГИБРИДНЫМ поиском (Vector + BM25 + Graph)")