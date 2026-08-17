# src/rag_engine.py

import nest_asyncio
import pickle
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, List 
from datetime import datetime
import logging
import time
import re
from llama_index.core.storage import StorageContext
from llama_index.core.indices.keyword_table import SimpleKeywordTableIndex

# ================================================================
# НАСТРОЙКА ЛОГГЕРА (ДО ВСЕХ ИМПОРТОВ)
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
    SimpleKeywordTableIndex,
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.retrievers import KeywordTableSimpleRetriever
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import BaseRetriever

# ================================================================
# ОПЦИОНАЛЬНО: ГРАФОВЫЙ ИНДЕКС (ЕСЛИ УСТАНОВЛЕН)
# ================================================================
try:
    from llama_index.core import KnowledgeGraphIndex
    from llama_index.graph_stores.neo4j import Neo4jGraphStore
    GRAPH_AVAILABLE = True
    logger.info("✅ Графовый индекс доступен")
except ImportError as e:
    GRAPH_AVAILABLE = False
    logger.warning(f"⚠️ Графовый индекс НЕ ДОСТУПЕН: {e}")

from src.client import llamaindex_llm
from src.config import Config
from src.document_parser import document_parser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================
# НАСТРОЙКИ
# ================================================================
Settings.llm = llamaindex_llm

# Загрузка модели эмбеддингов
try:
    logger.info(f"🔄 Загрузка модели эмбеддингов: {Config.EMBEDDING_MODEL}")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=Config.EMBEDDING_MODEL,
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
Settings.node_parser = SentenceSplitter(
    chunk_size=Config.CHUNK_SIZE,
    chunk_overlap=Config.CHUNK_OVERLAP
)

# Директории
DOCSTORE_DIR = Config.DOCSTORE_DIR
DOCSTORE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Config.CACHE_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DOCSTORE_FILE = DOCSTORE_DIR / "docstore.json"
INDEX_CACHE_FILE = CACHE_DIR / "index_cache.pkl"
METADATA_CACHE_FILE = CACHE_DIR / "metadata_cache.pkl"

SEARCH_K = Config.SEARCH_K
logger.info(f"🔍 SEARCH_K = {SEARCH_K}")


class MergedRetriever(BaseRetriever):
    """Объединяет результаты нескольких ретриверов"""
    
    def __init__(self, retrievers, similarity_top_k=5):
        self.retrievers = retrievers
        self.similarity_top_k = similarity_top_k
        super().__init__()
    
    def _retrieve(self, query_bundle):
        all_nodes = []
        seen_ids = set()
        
        for retriever in self.retrievers:
            try:
                nodes = retriever._retrieve(query_bundle)
                for node in nodes:
                    node_id = node.node.node_id
                    if node_id not in seen_ids:
                        seen_ids.add(node_id)
                        all_nodes.append(node)
            except Exception as e:
                logger.warning(f"⚠️ Ошибка ретривера: {e}")
                continue
        
        all_nodes.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)
        return all_nodes[:self.similarity_top_k]


class GraphRetriever(BaseRetriever):
    """Расширенный ретривер с использованием Neo4j графа"""
    
    def __init__(self, graph_store, graph_index, similarity_top_k=5):
        self.graph_store = graph_store
        self.graph_index = graph_index
        self.similarity_top_k = similarity_top_k
        super().__init__()
    
    def _retrieve(self, query_bundle):
        """Извлекает узлы из графа на основе запроса"""
        query_text = query_bundle.query_str
        results = []
        
        try:
            # ============================================================
            # 1. Извлекаем ключевые сущности из запроса
            # ============================================================
            # Ищем названия сущностей (слова с заглавной буквы или в кавычках)
            entities = re.findall(r'"([^"]+)"|«([^»]+)»|([А-Я][а-я]+(?:[-\s][А-Я][а-я]+)*)', query_text)
            entities = [e for group in entities for e in group if e]
            
            # Если ничего не нашли - берем все слова длиннее 3 символов
            if not entities:
                entities = re.findall(r'\b[А-Яа-яA-Za-z]{4,}\b', query_text)
            
            logger.info(f"🔍 Граф: найдены сущности: {entities[:3]}")
            
            # ============================================================
            # 2. Ищем связи между сущностями в графе
            # ============================================================
            for entity in entities[:3]:  # Берем первые 3 сущности
                # Ищем узлы, содержащие эту сущность
                cypher = f"""
                MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower('{entity}') 
                   OR toLower(n.text) CONTAINS toLower('{entity}')
                OPTIONAL MATCH (n)-[r]-(related)
                RETURN n, collect(DISTINCT related) as related_nodes, 
                       collect(DISTINCT type(r)) as relations
                LIMIT 5
                """
                
                try:
                    with self.graph_store.session as session:
                        result = session.run(cypher)
                        for record in result:
                            node = record['n']
                            related_nodes = record['related_nodes']
                            relations = record['relations']
                            
                            if node:
                                # Создаем node для совместимости с другими ретриверами
                                from llama_index.core.schema import NodeWithScore, TextNode
                                
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
                                
                                results.append(NodeWithScore(
                                    node=text_node,
                                    score=0.85  # Высокий приоритет для графовых результатов
                                ))
                                
                                logger.info(f"   ✅ Найден узел в графе: {node.get('name', node.get('text', ''))[:50]}...")
                                
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка графового запроса для {entity}: {e}")
            
            # ============================================================
            # 3. Ищем связи по типу (если запрос содержит "связи", "отношения" и т.д.)
            # ============================================================
            if any(word in query_text.lower() for word in ['связ', 'отнош', 'связан', 'граф']):
                logger.info("🔍 Граф: поиск связей по запросу...")
                cypher = f"""
                MATCH (n)-[r]-(m)
                WHERE toLower(n.name) CONTAINS toLower('{entities[0] if entities else "товар"}')
                   OR toLower(m.name) CONTAINS toLower('{entities[0] if entities else "товар"}')
                RETURN n, r, m
                LIMIT 10
                """
                
                try:
                    with self.graph_store.session as session:
                        result = session.run(cypher)
                        for record in result:
                            n = record['n']
                            r = record['r']
                            m = record['m']
                            
                            if n and m and r:
                                rel_text = f"{n.get('name', n.get('text', ''))} --[{type(r)}]--> {m.get('name', m.get('text', ''))}"
                                
                                from llama_index.core.schema import NodeWithScore, TextNode
                                text_node = TextNode(
                                    text=rel_text,
                                    metadata={
                                        'source': 'graph_relation',
                                        'relation_type': type(r),
                                        'from': n.get('name', ''),
                                        'to': m.get('name', '')
                                    }
                                )
                                
                                results.append(NodeWithScore(
                                    node=text_node,
                                    score=0.9  # Очень высокий приоритет для связей
                                ))
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка поиска связей: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Ошибка графового ретривера: {e}")
        
        # Сортируем по score и возвращаем
        results.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)
        return results[:self.similarity_top_k]


class HybridRAG:
    """Гибридный RAG с DocStore для хранения текстов и графовым поиском"""
    
    def __init__(self):
        self.storage_context = None
        self.persist_dir = CACHE_DIR / "storage_context"

        # ============================================================
        # QDRANT
        # ============================================================
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
        
        # ============================================================
        # DOCSTORE (ХРАНИЛИЩЕ ТЕКСТОВ)
        # ============================================================
        self.docstore = SimpleDocumentStore()
        
        # ============================================================
        # NEO4J (ОПЦИОНАЛЬНО)
        # ============================================================
        self.graph_store = None
        self.use_graph = False
        
        if GRAPH_AVAILABLE:
            try:
                self.graph_store = Neo4jGraphStore(
                    username=Config.NEO4J_USERNAME,
                    password=Config.NEO4J_PASSWORD,
                    url=Config.NEO4J_URI
                )
                logger.info("✅ Подключение к Neo4j установлено")
                self.use_graph = True
            except Exception as e:
                logger.warning(f"⚠️ Neo4j не доступен: {e}")
                self.graph_store = None
                self.use_graph = False
        else:
            logger.info("ℹ️ Графовый индекс отключен (зависимости не установлены)")
        
        # ============================================================
        # ИНДЕКСЫ
        # ============================================================
        self.vector_index = None
        self.keyword_index = None
        self.graph_index = None
        self.query_engine = None
        self.is_initialized = False
        self.documents_dir = Config.DOCUMENTS_DIR
        
        # ============================================================
        # РЕ-РАНКЕР
        # ============================================================
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

    def _get_documents_hash(self) -> str:
        """Вычисляет хеш документов"""
        hasher = hashlib.md5()
        extensions = getattr(Config, 'SUPPORTED_EXTENSIONS', ['.pdf', '.docx', '.doc', '.txt', '.md'])
        for ext in extensions:
            for file_path in self.documents_dir.glob(f'*{ext}'):
                if file_path.is_file():
                    hasher.update(file_path.name.encode())
                    hasher.update(str(file_path.stat().st_mtime).encode())
                    hasher.update(str(file_path.stat().st_size).encode())
        return hasher.hexdigest()

    def _check_qdrant(self) -> bool:
        """Проверяет данные в Qdrant"""
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
        """Загружает docstore из файла"""
        if not DOCSTORE_FILE.exists():
            logger.info("📁 Docstore не найден")
            return False
        
        try:
            self.docstore = SimpleDocumentStore.from_persist_path(str(DOCSTORE_FILE))
            logger.info(f"✅ Docstore загружен из {DOCSTORE_FILE}")
            
            docs = list(self.docstore.docs.values())
            logger.info(f"📄 В docstore {len(docs)} документов")
            
            if docs:
                sample = docs[0]
                text_preview = sample.text[:100] if sample.text else "ПУСТО"
                logger.info(f"   Пример: {text_preview}...")
            
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки docstore: {e}", exc_info=True)
            return False

    def _save_docstore(self):
        """Сохраняет docstore"""
        try:
            DOCSTORE_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.docstore.persist(str(DOCSTORE_FILE))
            
            file_size = DOCSTORE_FILE.stat().st_size if DOCSTORE_FILE.exists() else 0
            logger.info(f"💾 Docstore сохранен: {DOCSTORE_FILE} ({file_size} байт)")
            
            docs = list(self.docstore.docs.values())
            logger.info(f"📄 Сохранено {len(docs)} документов")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения docstore: {e}", exc_info=True)

    def _parse_all_documents(self) -> List[Document]:
        """Парсит все документы с сохранением информации о страницах"""
        logger.info("📚 Обработка документов через document_parser...")
        
        documents = []
        parsed_docs = document_parser.parse_all_documents(self.documents_dir)
        
        if not parsed_docs:
            logger.warning("⚠️ Нет документов для индексации")
            return documents
        
        for doc_data in parsed_docs:
            full_text = doc_data.get('text', '')
            
            if not full_text.strip():
                logger.warning(f"⚠️ Пропускаем пустой документ: {doc_data['metadata']['source']}")
                continue
            
            # ============================================================
            # ИЗВЛЕКАЕМ ИНФОРМАЦИЮ О СТРАНИЦАХ
            # ============================================================
            import re
            
            # Ищем все маркеры страниц: [Страница X]
            page_pattern = r'\[Страница (\d+)\]\n(.*?)(?=\[Страница \d+\]|$)'
            matches = re.findall(page_pattern, full_text, re.DOTALL)
            
            page_map = {}
            if matches:
                for page_num, page_text in matches:
                    page_map[int(page_num)] = page_text.strip()
                logger.info(f"  📄 Найдено страниц: {len(page_map)}")
            else:
                page_map[1] = full_text
            
            # ============================================================
            # СОЗДАЕМ ДОКУМЕНТЫ ПО СТРАНИЦАМ (вместо одного большого)
            # ============================================================
            for page_num, page_text in page_map.items():
                if not page_text.strip():
                    continue
                
                # Добавляем номер страницы в начало текста и в метаданные
                page_text_with_marker = f"[Страница {page_num}]\n{page_text}"
                
                document = Document(
                    text=page_text_with_marker,
                    metadata={
                        'source': doc_data['metadata'].get('source', 'unknown'),
                        'type': doc_data['metadata'].get('type', 'unknown'),
                        'page': page_num,  # ← ЯВНЫЙ НОМЕР СТРАНИЦЫ
                        'total_pages': len(page_map),
                        'has_ocr': bool(doc_data.get('ocr_text')),
                        'ocr_method': doc_data.get('ocr_method'),
                        'has_tables': bool(doc_data.get('tables')),
                    }
                )
                documents.append(document)
            
            logger.info(f"  ✅ Добавлен: {doc_data['metadata']['source']} ({len(page_map)} страниц)")
        
        logger.info(f"📚 Всего загружено документов: {len(documents)}")
        return documents

    def _create_collection(self):
        """Создает коллекцию в Qdrant"""
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
        """Загружает только векторный индекс (без keyword и graph)"""
        if not INDEX_CACHE_FILE.exists() or not METADATA_CACHE_FILE.exists():
            return False
        
        try:
            with open(METADATA_CACHE_FILE, 'rb') as f:
                cached = pickle.load(f)
            
            if cached.get('hash') != self._get_documents_hash():
                return False
            
            if not self._check_qdrant():
                return False
            
            if not self._load_docstore():
                return False
            
            # Загружаем StorageContext
            from llama_index.core.storage import StorageContext
            self.persist_dir = CACHE_DIR / "storage_context"
            
            if self.persist_dir.exists():
                self.storage_context = StorageContext.from_defaults(
                    vector_store=self.vector_store,
                    docstore=self.docstore,
                    persist_dir=str(self.persist_dir)
                )
            else:
                self.storage_context = StorageContext.from_defaults(
                    vector_store=self.vector_store,
                    docstore=self.docstore
                )
            
            # Восстанавливаем только векторный индекс
            self.vector_index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=Settings.embed_model,
                storage_context=self.storage_context
            )
            logger.info("✅ Векторный индекс восстановлен")
            
            # Keyword и Graph отключаем (из-за ошибки совместимости)
            self.keyword_index = None
            self.graph_index = None
            logger.info("ℹ️ Keyword и Graph индексы отключены (используется только векторный поиск)")
            
            self._setup_query_engine()
            self.is_initialized = True
            logger.info("✅ Система готова")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
            return False

    def _save_to_cache(self):
        """Сохраняет индексы через storage_context"""
        try:
            self._save_docstore()
            
            # Сохраняем storage_context
            if self.storage_context is not None:
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                self.storage_context.persist(persist_dir=str(self.persist_dir))
                logger.info(f"✅ StorageContext сохранен в {self.persist_dir}")
            
            metadata = {
                'hash': self._get_documents_hash(),
                'timestamp': datetime.now().isoformat(),
                'collection': Config.QDRANT_COLLECTION,
                'embedding_model': Config.EMBEDDING_MODEL,
                'use_graph': self.use_graph,
                'documents_count': len(list(self.docstore.docs.values()))
            }
            
            with open(METADATA_CACHE_FILE, 'wb') as f:
                pickle.dump(metadata, f)
            
            with open(INDEX_CACHE_FILE, 'wb') as f:
                pickle.dump({'type': 'HybridRAG', 'version': '2.0'}, f)
            
            logger.info(f"💾 Кэш сохранен")
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")

    def initialize_and_index(self, force_reindex: bool = False):
        """Инициализация и индексация"""
        logger.info("🚀 Запуск инициализации...")
        
        if not force_reindex and self._load_from_cache():
            self.is_initialized = True
            logger.info("✅ Система готова (из кэша)")
            return
        
        logger.info("🔄 Полная переиндексация...")
        documents = self._parse_all_documents()
        
        if not documents:
            logger.warning("⚠️ Нет документов для индексации")
            self.is_initialized = False
            return
        
        # Создаем коллекцию в Qdrant
        self._create_collection()
        
        # ============================================================
        # СОЗДАЕМ STORAGE_CONTEXT ДЛЯ ВСЕХ ИНДЕКСОВ
        # ============================================================
        from llama_index.core.storage import StorageContext
        
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=self.docstore,
            graph_store=self.graph_store if self.use_graph else None
        )
        
        # Создаем векторный индекс
        logger.info("🧠 Создание векторного индекса (Qdrant)...")
        self.vector_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            embed_model=Settings.embed_model,
            show_progress=True
        )
        logger.info("✅ Векторный индекс создан")
        
        # Создаем keyword индекс
        try:
            logger.info("🧠 Создание keyword индекса...")
            from llama_index.core.indices.keyword_table import SimpleKeywordTableIndex
            
            self.keyword_index = SimpleKeywordTableIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                max_keywords_per_chunk=10,
                show_progress=True
            )
            logger.info("✅ Keyword индекс создан")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка создания keyword индекса: {e}")
            self.keyword_index = None
        
        # Создаем графовый индекс
        if self.use_graph and GRAPH_AVAILABLE:
            try:
                logger.info("🧠 Создание графового индекса (Neo4j)...")
                from llama_index.core import KnowledgeGraphIndex
                
                self.graph_index = KnowledgeGraphIndex.from_documents(
                    documents,
                    storage_context=self.storage_context,
                    max_triplets_per_chunk=3,
                    show_progress=True,
                    include_embeddings=True,
                    embed_model=Settings.embed_model
                )
                logger.info("✅ Графовый индекс создан")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка создания графового индекса: {e}")
                self.graph_index = None
                self.use_graph = False
        
        # Сохраняем кэш
        self._save_to_cache()
        
        # Настраиваем query engine
        self._setup_query_engine()
        self.is_initialized = True
        logger.info("✅ Система готова!")

    def _setup_query_engine(self):
        """Настройка query engine с расширенным графовым поиском"""
        if self.vector_index is None:
            raise ValueError("Индекс не инициализирован")
        
        logger.info("🔧 Настройка ГИБРИДНОГО поиска с графом...")
        
        retrievers = []
        
        # 1. Векторный ретривер
        vector_retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=SEARCH_K,
        )
        retrievers.append(vector_retriever)
        logger.info(f"✅ Векторный ретривер (top_k={SEARCH_K})")
        
        # 2. Keyword ретривер
        if self.keyword_index is not None:
            try:
                keyword_retriever = KeywordTableSimpleRetriever(
                    index=self.keyword_index,
                    similarity_top_k=SEARCH_K,
                )
                retrievers.append(keyword_retriever)
                logger.info(f"✅ Keyword ретривер (top_k={SEARCH_K})")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка Keyword ретривера: {e}")
        
        # 3. Расширенный графовый ретривер
        if self.use_graph and self.graph_store is not None:
            try:
                graph_retriever = GraphRetriever(
                    graph_store=self.graph_store,
                    graph_index=self.graph_index,
                    similarity_top_k=SEARCH_K,
                )
                retrievers.append(graph_retriever)
                logger.info(f"✅ Расширенный графовый ретривер (top_k={SEARCH_K})")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка графового ретривера: {e}")
        
        # Объединяем ретриверы
        if len(retrievers) > 1:
            merged_retriever = MergedRetriever(
                retrievers=retrievers,
                similarity_top_k=SEARCH_K,
            )
            logger.info(f"✅ Объединенный ретривер ({len(retrievers)} методов)")
        else:
            merged_retriever = retrievers[0] if retrievers else None
            logger.info("ℹ️ Используется только один ретривер")
        
        # ============================================================
        # ОБНОВЛЕННЫЙ ПРОМПТ С ИНСТРУКЦИЕЙ ПРО СТРАНИЦЫ
        # ============================================================
        template_str = """Ты — ассистент по документации компании ООО «Евроторг».

    Ниже дан КОНТЕКСТ из документов и графа знаний. Ответь на вопрос, используя ТОЛЬКО этот контекст.

    КОНТЕКСТ (включая связи между сущностями):
    {context_str}

    ВОПРОС: {query_str}

    ИНСТРУКЦИЯ:
    1. Дай РАЗВЕРНУТЫЙ ответ (2-4 предложения).
    2. Начни с прямого ответа на вопрос.
    3. Затем добавь детали из контекста.
    4. ⚠️ ВАЖНО: Номер страницы УЖЕ указан в контексте в формате [Страница X].
    - Используй ТОЛЬКО тот номер страницы, который есть в контексте.
    - НЕ ПРИДУМЫВАЙ номера страниц, которых нет в контексте.
    - Если в контексте нет номера страницы — не пиши его.
    5. Обязательно укажи источник информации (Приложение, раздел, страницу из контекста).
    6. Если есть связи между сущностями — укажи их.
    7. НЕ ПИШИ "нет информации", если в контексте есть ответ.
    8. Ответь на русском языке, вежливо и профессионально.

    ОТВЕТ (развернуто, с указанием источника из контекста):"""
        prompt = PromptTemplate(template_str)
        
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=merged_retriever,
            response_synthesizer=TreeSummarize(
                llm=Settings.llm,
                summary_template=prompt,
            ),
            node_postprocessors=[self.reranker] if self.reranker else [],
            verbose=False,
        )
        
        logger.info("✅ ГИБРИДНЫЙ ПОИСК с графом настроен!")

    def query_graph(self, cypher_query: str) -> List[Dict]:
        """Выполняет Cypher-запрос к графу"""
        if not self.use_graph or not self.graph_store:
            logger.warning("⚠️ Граф не доступен")
            return []
        
        try:
            with self.graph_store.session as session:
                result = session.run(cypher_query)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"❌ Ошибка запроса к графу: {e}")
            return []

    def find_related_entities(self, entity_name: str, relation_type: str = None) -> List[Dict]:
        """Находит связанные сущности в графе"""
        if not self.use_graph:
            return []
        
        query = f"""
        MATCH (e {{name: '{entity_name}'}})
        OPTIONAL MATCH (e)-[r]-(related)
        WHERE {relation_type} IS NULL OR type(r) = '{relation_type}'
        RETURN e, r, related
        LIMIT 10
        """
        return self.query_graph(query)

    def query(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        """Запрос к системе с получением источников"""
        if not self.query_engine:
            return {
                "answer": "Ошибка: RAG не инициализирован.",
                "sources": [],
                "status": "error"
            }
        
        try:
            start_time = time.time()
            if verbose:
                logger.info(f"📝 Запрос: {question}")
            
            # ============================================================
            # ИЗВЛЕКАЕМ НОМЕР СТРАНИЦЫ ИЗ ЗАПРОСА
            # ============================================================
            import re
            page_match = re.search(r'(\d+)\s*страниц[еы]|страниц[еы]\s*(\d+)', question, re.IGNORECASE)
            requested_page = None
            if page_match:
                requested_page = int(page_match.group(1) or page_match.group(2))
                logger.info(f"📄 Запрошена страница: {requested_page}")
            
            response = self.query_engine.query(question)
            answer = str(response)
            
            # ============================================================
            # СБОР ИСТОЧНИКОВ
            # ============================================================
            sources = []
            sources_text = ""
            
            if hasattr(response, 'source_nodes'):
                logger.info(f"📊 Найдено source_nodes: {len(response.source_nodes)}")
                
                for node in response.source_nodes[:5]:
                    try:
                        node_text = None
                        node_page = None
                        
                        # 1. Пробуем из node.text
                        if hasattr(node, 'text') and node.text:
                            node_text = node.text
                        
                        # 2. Пробуем из node.node.text
                        if not node_text and hasattr(node, 'node'):
                            if hasattr(node.node, 'text') and node.node.text:
                                node_text = node.node.text
                        
                        # 3. Пробуем из docstore по node_id
                        if not node_text:
                            node_id = None
                            if hasattr(node, 'node') and hasattr(node.node, 'node_id'):
                                node_id = node.node.node_id
                            elif hasattr(node, 'node_id'):
                                node_id = node.node_id
                            
                            if node_id and hasattr(self.docstore, 'docs'):
                                doc = self.docstore.docs.get(node_id)
                                if doc and hasattr(doc, 'text') and doc.text:
                                    node_text = doc.text
                                    logger.debug(f"✅ Текст из docstore по ID {node_id}")
                        
                        # ============================================================
                        # ИЗВЛЕКАЕМ СТРАНИЦУ ИЗ МЕТАДАННЫХ ИЛИ ТЕКСТА
                        # ============================================================
                        if hasattr(node, 'metadata'):
                            node_page = node.metadata.get('page')
                        
                        # Если не нашли в метаданных, ищем в тексте
                        if not node_page and node_text:
                            page_in_text = re.search(r'\[Страница (\d+)\]', node_text)
                            if page_in_text:
                                node_page = int(page_in_text.group(1))
                        
                        if node_text:
                            source = {
                                "text": node_text,
                                "score": float(node.score) if hasattr(node, 'score') and node.score is not None else None,
                                "metadata": node.metadata if hasattr(node, 'metadata') else {},
                                "page": node_page,  # ← ДОБАВЛЯЕМ СТРАНИЦУ
                            }
                            
                            # Проверяем, не из графа ли источник
                            if source['metadata'].get('source') == 'graph':
                                logger.info(f"   🌐 Графовый источник: {node_text[:100]}...")
                            
                            sources.append(source)
                            sources_text += node_text + " "
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Ошибка обработки источника: {e}")
                        continue
                
                logger.info(f"📄 Получено источников: {len(sources)}")
                if sources:
                    first_text = sources[0].get('text', '')[:100]
                    first_page = sources[0].get('page')
                    logger.info(f"   Первый источник: {first_text}... (страница: {first_page})")
            
            # ============================================================
            # ФИЛЬТРАЦИЯ ПО СТРАНИЦЕ (если запрошена)
            # ============================================================
            if requested_page:
                # Фильтруем источники по странице
                filtered_sources = [s for s in sources if s.get('page') == requested_page]
                if filtered_sources:
                    sources = filtered_sources
                    logger.info(f"📄 Отфильтровано {len(sources)} источников на странице {requested_page}")
                else:
                    logger.warning(f"⚠️ Нет источников на странице {requested_page}")
                    # Обновляем ответ, чтобы пользователь знал
                    if "нет информации" not in answer.lower():
                        # Проверяем, есть ли в ответе упоминание другой страницы
                        page_in_answer = re.search(r'(\d+)\s*страниц[еы]', answer)
                        if page_in_answer:
                            wrong_page = int(page_in_answer.group(1))
                            if wrong_page != requested_page:
                                # Убираем неправильную страницу из ответа
                                answer = answer.replace(f"на странице {wrong_page}", "").replace(f"на {wrong_page} странице", "").strip()
                                # Добавляем информацию о том, что информация не найдена
                                if not answer:
                                    answer = f"Информация на странице {requested_page} не найдена. Проверьте правильность номера страницы."
            
            # ============================================================
            # ПОСТ-ОБРАБОТКА: Исправляем ответ, если LLM ошиблась
            # ============================================================
            has_forbidden = "ЗАПРЕЩЕНО" in sources_text
            has_allow = "РАЗРЕШЕНО" in sources_text
            
            if "нет информации" in answer.lower() and sources:
                logger.info("🔧 LLM сказал 'нет информации', но есть источники. Исправляем...")
                
                for source in sources:
                    text = source.get('text', '')
                    
                    if "ЗАПРЕЩЕНО" in text:
                        match = re.search(r'[^.]*ЗАПРЕЩЕНО[^.]*\.', text)
                        if match:
                            page_info = f" (страница {source.get('page')})" if source.get('page') else ""
                            answer = match.group(0).strip() + page_info
                            logger.info(f"🔧 Исправлен ответ (найдено ЗАПРЕЩЕНО): {answer}")
                            break
                    
                    if "РАЗРЕШЕНО" in text:
                        match = re.search(r'[^.]*РАЗРЕШЕНО[^.]*\.', text)
                        if match:
                            page_info = f" (страница {source.get('page')})" if source.get('page') else ""
                            answer = match.group(0).strip() + page_info
                            logger.info(f"🔧 Исправлен ответ (найдено РАЗРЕШЕНО): {answer}")
                            break
            
            if "нет информации" in answer.lower() and sources:
                for source in sources:
                    text = source.get('text', '')
                    if "Приложение 2" in text and "ЗАПРЕЩЕНО" in text:
                        match = re.search(r'Приложение 2[^.]*ЗАПРЕЩЕНО[^.]*\.', text)
                        if match:
                            page_info = f" (страница {source.get('page')})" if source.get('page') else ""
                            answer = match.group(0).strip() + page_info
                            logger.info(f"🔧 Исправлен ответ (Приложение 2): {answer}")
                            break
            
            # ============================================================
            # ДОБАВЛЯЕМ СТРАНИЦУ В ОТВЕТ, ЕСЛИ ЕЕ НЕТ
            # ============================================================
            if sources and not re.search(r'страниц[еы]', answer, re.IGNORECASE):
                # Если в ответе нет номера страницы, но есть источники
                page_from_source = sources[0].get('page')
                if page_from_source:
                    # Проверяем, не упоминается ли уже страница в ответе
                    if not re.search(rf'{page_from_source}\s*страниц[еы]', answer, re.IGNORECASE):
                        answer = f"{answer} (страница {page_from_source})"
            
            elapsed = time.time() - start_time
            logger.info(f"⏱️ Время ответа: {elapsed:.2f}с")
            
            return {
                "answer": answer,
                "sources": sources,
                "status": "success",
                "elapsed": elapsed,
                "sources_count": len(sources),
                "requested_page": requested_page,
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка запроса: {e}", exc_info=True)
            return {
                "answer": f"Ошибка: {str(e)}",
                "sources": [],
                "status": "error"
            }

    def ask(self, question: str) -> str:
        """Простой метод для получения ответа"""
        result = self.query(question)
        return result.get('answer', 'Нет ответа')

    def get_stats(self) -> Dict[str, Any]:
        """Статистика системы"""
        stats = {
            'collection': Config.QDRANT_COLLECTION,
            'docstore_file': str(DOCSTORE_FILE) if DOCSTORE_FILE.exists() else None,
            'docstore_size': DOCSTORE_FILE.stat().st_size if DOCSTORE_FILE.exists() else 0,
            'documents_in_docstore': len(list(self.docstore.docs.values())) if hasattr(self.docstore, 'docs') else 0,
            'vector_index_loaded': self.vector_index is not None,
            'keyword_index_loaded': self.keyword_index is not None,
            'graph_index_loaded': self.graph_index is not None,
            'query_engine_ready': self.query_engine is not None,
            'is_initialized': self.is_initialized,
            'use_graph': self.use_graph and GRAPH_AVAILABLE,
            'graph_available': GRAPH_AVAILABLE,
            'search_k': SEARCH_K,
            'cache_dir': str(CACHE_DIR),
            'docstore_dir': str(DOCSTORE_DIR),
        }
        
        if self._check_qdrant():
            try:
                info = self.qdrant_client.get_collection(Config.QDRANT_COLLECTION)
                stats['qdrant_points'] = info.points_count
            except Exception:
                pass
        
        return stats


# Глобальный экземпляр
rag_agent = HybridRAG()
logger.info("✅ RAG агент создан с расширенным графовым поиском")