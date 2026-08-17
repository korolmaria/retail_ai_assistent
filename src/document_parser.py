# src/document_parser.py

import fitz  # PyMuPDF
import pdfplumber
import docx
import pytesseract
from PIL import Image, ImageEnhance
import io
import re
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import tempfile
import os


from pdf2image import convert_from_path
from src.config import Config

logger = logging.getLogger(__name__)

# ============================================================================
# ПРОВЕРКА ДОСТУПНОСТИ БИБЛИОТЕК
# ============================================================================

# Проверка EasyOCR
EASYOCR_AVAILABLE = False
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    logger.info("✅ EasyOCR доступен")
except ImportError:
    logger.warning("⚠️ EasyOCR не установлен")

# Проверка docx2txt для извлечения картинок из docx
DOCX2TXT_AVAILABLE = False
try:
    import docx2txt
    DOCX2TXT_AVAILABLE = True
    logger.info("✅ docx2txt доступен")
except ImportError:
    logger.warning("⚠️ docx2txt не установлен")


class DocumentParser:
    """Парсер документов с гибридным подходом: текст + таблицы + OCR для картинок"""
    
    def __init__(self):
        self.use_ocr = getattr(Config, 'USE_OCR', True)
        self.ocr_lang = getattr(Config, 'OCR_LANGUAGE', 'rus+eng')
        self.ocr_dpi = getattr(Config, 'OCR_DPI', 600)
        self.use_easyocr = getattr(Config, 'OCR_USE_EASYOCR', True)
        
        # Инициализация OCR движков
        self._init_ocr_engines()
    
    def _init_ocr_engines(self):
        """Инициализация OCR движков"""
        # Tesseract
        self.tesseract_available = False
        try:
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info("✅ Tesseract OCR доступен")
        except Exception as e:
            logger.warning(f"⚠️ Tesseract не доступен: {e}")
        
        # EasyOCR
        self.easyocr_available = False
        self.easyocr_reader = None
        
        if EASYOCR_AVAILABLE and self.use_easyocr:
            try:
                import easyocr
                lang_list = ['en']
                if 'rus' in self.ocr_lang or 'ru' in self.ocr_lang:
                    lang_list.append('ru')
                
                self.easyocr_reader = easyocr.Reader(
                    lang_list,
                    gpu=False,
                    verbose=False
                )
                self.easyocr_available = True
                logger.info(f"✅ EasyOCR доступен (языки: {lang_list})")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка инициализации EasyOCR: {e}")
        
        if self.tesseract_available or self.easyocr_available:
            logger.info("✅ OCR система готова")
        else:
            logger.warning("⚠️ OCR не доступен!")
    
    def parse_document(self, file_path: Path) -> Dict[str, Any]:
        """Парсит документ и возвращает текст с метаданными"""
        ext = file_path.suffix.lower()
        
        if ext == '.pdf':
            return self._parse_pdf_hybrid(file_path)
        elif ext == '.docx':
            return self._parse_docx(file_path)
        elif ext == '.doc':
            return self._parse_doc(file_path)
        elif ext in ['.txt', '.md']:
            return self._parse_txt(file_path)
        else:
            logger.warning(f"Неподдерживаемый формат: {ext}")
            return {
                'text': '',
                'ocr_text': '',
                'tables': [],
                'metadata': {'source': file_path.name, 'error': 'Unsupported format'},
                'pages': 0,
                'has_images': False,
                'has_tables': False,
                'ocr_method': None,
                'extraction_details': {}
            }
    
    # =========================================================================
    # PDF ПАРСИНГ (ГИБРИДНЫЙ)
    # =========================================================================
    
    def _parse_pdf_hybrid(self, file_path: Path) -> Dict[str, Any]:
        """ГИБРИДНЫЙ парсинг PDF: текст + таблицы + OCR только для картинок"""
        logger.info(f"📄 ГИБРИДНЫЙ парсинг PDF: {file_path.name}")
        
        result = {
            'text': '',
            'ocr_text': '',
            'tables': [],
            'metadata': {
                'source': file_path.name,
                'type': 'pdf',
                'processed_at': datetime.now().isoformat()
            },
            'pages': 0,
            'has_images': False,
            'has_tables': False,
            'ocr_method': None,
            'extraction_details': {
                'text_pages': 0,
                'image_pages': 0,
                'table_pages': 0,
                'ocr_images': 0,
                'ocr_successful': 0,
                'camelot_tables': 0
            }
        }
        
        try:
            # Открываем PDF для PyMuPDF (нужен для таблиц)
            doc = fitz.open(str(file_path))
            
            # ============================================================
            # ШАГ 1: ИЗВЛЕКАЕМ ТЕКСТ И ТАБЛИЦЫ (БЕЗ OCR)
            # ============================================================
            logger.info("  📝 Извлечение текста и таблиц через pdfplumber...")
            text_by_page = {}
            tables_by_page = {}
            
            with pdfplumber.open(str(file_path)) as pdf:
                result['pages'] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Извлекаем текст (без OCR)
                    page_text = page.extract_text() or ''
                    page_text = self._clean_text(page_text)
                    
                    if page_text.strip():
                        text_by_page[page_num] = page_text
                        result['extraction_details']['text_pages'] += 1
                    
                    # Извлекаем таблицы через pdfplumber (если они есть как текст)
                    try:
                        page_tables = page.extract_tables()
                        if page_tables:
                            result['has_tables'] = True
                            result['extraction_details']['table_pages'] += 1
                            tables_by_page[page_num] = page_tables
                            logger.info(f"    📊 Страница {page_num}: найдено {len(page_tables)} таблиц (pdfplumber)")
                    except Exception as e:
                        logger.warning(f"    ⚠️ Ошибка извлечения таблиц через pdfplumber: {e}")
                    
                    # ============================================================
                    # ДОПОЛНИТЕЛЬНО: ИЗВЛЕКАЕМ ТАБЛИЦЫ ЧЕРЕЗ PyMuPDF
                    # ============================================================
                    try:
                        # Ищем таблицы через PyMuPDF
                        tabs = page.find_tables()
                        if tabs and len(tabs.tables) > 0:
                            for tab_idx, tab in enumerate(tabs.tables, 1):
                                table_data = tab.extract()
                                if table_data and len(table_data) > 1:
                                    # Проверяем, не дублируем ли таблицу из pdfplumber
                                    is_duplicate = False
                                    for existing in tables_by_page.get(page_num, []):
                                        if len(existing) == len(table_data) and len(existing[0]) == len(table_data[0]):
                                            is_duplicate = True
                                            break
                                    
                                    if not is_duplicate:
                                        result['has_tables'] = True
                                        tables_by_page.setdefault(page_num, []).append(table_data)
                                        logger.info(f"    📊 Страница {page_num}: найдена таблица через PyMuPDF")
                    except Exception as e:
                        logger.warning(f"    ⚠️ Ошибка извлечения таблиц через PyMuPDF: {e}")
            
            # ============================================================
            # ШАГ 1.5: ИЗВЛЕКАЕМ ТАБЛИЦЫ ЧЕРЕЗ CAMELOT (ДОПОЛНИТЕЛЬНО)
            # ============================================================
            try:
                logger.info("  📊 Извлечение таблиц через Camelot...")
                camelot_tables = self._extract_tables_with_camelot(file_path)
                if camelot_tables:
                    for table_data in camelot_tables:
                        # Проверяем, нет ли уже такой таблицы
                        is_duplicate = False
                        page = table_data['page']
                        
                        # Проверяем существующие таблицы на этой странице
                        for existing in tables_by_page.get(page, []):
                            # Простая проверка на дубликат
                            if len(existing) >= 2 and len(table_data['raw']) >= 2:
                                if len(existing) == len(table_data['raw']) and len(existing[0]) == len(table_data['raw'].columns):
                                    is_duplicate = True
                                    break
                        
                        if not is_duplicate:
                            # Конвертируем DataFrame в список для единообразия
                            table_list = table_data['raw'].values.tolist()
                            tables_by_page.setdefault(page, []).append(table_list)
                            result['has_tables'] = True
                            result['extraction_details']['camelot_tables'] += 1
                            logger.info(f"    ✅ Добавлена таблица через Camelot (страница {page})")
            except ImportError:
                logger.info("  ℹ️ Camelot не установлен, пропускаем")
            except Exception as e:
                logger.warning(f"  ⚠️ Ошибка Camelot: {e}")
            
            # ============================================================
            # ШАГ 2: СОХРАНЯЕМ ТЕКСТ И ТАБЛИЦЫ
            # ============================================================
            # Сохраняем текст
            if text_by_page:
                result['text'] = '\n\n'.join([
                    f"[Страница {page_num}]\n{text}"
                    for page_num, text in text_by_page.items()
                ])
            
            # Сохраняем таблицы
            if tables_by_page:
                table_texts = []
                for page_num, tables in tables_by_page.items():
                    for table_idx, table in enumerate(tables, 1):
                        if table:
                            table_str = self._format_table(table)
                            table_texts.append(
                                f"[Таблица {table_idx} на странице {page_num}]\n{table_str}"
                            )
                if table_texts:
                    result['tables'] = table_texts
                    # Добавляем таблицы в текст
                    if result['text']:
                        result['text'] += '\n\n' + '\n\n'.join(table_texts)
                    else:
                        result['text'] = '\n\n'.join(table_texts)
            
            # ============================================================
            # ШАГ 3: ИЗВЛЕКАЕМ КАРТИНКИ (ТОЛЬКО ДЛЯ OCR)
            # ============================================================
            logger.info("  🖼️ Извлечение картинок через PyMuPDF...")
            
            images_data = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                
                if image_list:
                    result['has_images'] = True
                    result['extraction_details']['image_pages'] += 1
                    
                    logger.info(f"    📸 Страница {page_num + 1}: {len(image_list)} изображений")
                    
                    for img_index, img in enumerate(image_list, 1):
                        try:
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)
                            
                            if pix.n - pix.alpha < 4:
                                img_data = pix.tobytes("png")
                                pil_image = Image.open(io.BytesIO(img_data))
                                images_data.append({
                                    'page': page_num + 1,
                                    'index': img_index,
                                    'image': pil_image,
                                    'xref': xref
                                })
                            else:
                                pix = fitz.Pixmap(fitz.csRGB, pix)
                                img_data = pix.tobytes("png")
                                pil_image = Image.open(io.BytesIO(img_data))
                                images_data.append({
                                    'page': page_num + 1,
                                    'index': img_index,
                                    'image': pil_image,
                                    'xref': xref
                                })
                            
                            pix = None
                            
                        except Exception as e:
                            logger.warning(f"    ⚠️ Ошибка извлечения изображения {img_index}: {e}")
            
            doc.close()
            logger.info(f"  ✅ Извлечено {len(images_data)} изображений")
            
            # ============================================================
            # ШАГ 4: OCR ТОЛЬКО ДЛЯ КАРТИНОК (ЕСЛИ ЕСТЬ)
            # ============================================================
            if images_data and self.use_ocr:
                logger.info(f"  🔍 Запуск OCR для {len(images_data)} изображений...")
                
                ocr_results = []
                
                for img_info in images_data:
                    try:
                        processed_image = self._preprocess_image(img_info['image'])
                        ocr_text = self._perform_hybrid_ocr(processed_image)
                        
                        # Фильтруем мусор
                        if ocr_text.strip() and self._is_valid_text(ocr_text):
                            ocr_results.append({
                                'page': img_info['page'],
                                'text': ocr_text
                            })
                            result['extraction_details']['ocr_successful'] += 1
                            logger.info(f"    ✅ OCR страница {img_info['page']}: {len(ocr_text)} символов")
                        else:
                            logger.warning(f"    ⚠️ OCR страница {img_info['page']}: текст не распознан или мусор")
                        
                        result['extraction_details']['ocr_images'] += 1
                        
                    except Exception as e:
                        logger.warning(f"    ❌ Ошибка OCR: {e}")
                
                if ocr_results:
                    ocr_text_combined = '\n\n'.join([
                        f"[OCR Страница {r['page']}]\n{r['text']}"
                        for r in ocr_results
                    ])
                    result['ocr_text'] = ocr_text_combined
                    result['ocr_method'] = 'hybrid'
                    logger.info(f"  ✅ OCR завершен: {result['extraction_details']['ocr_successful']}/{result['extraction_details']['ocr_images']} успешно")
            
            # ============================================================
            # ШАГ 5: ОБЪЕДИНЯЕМ ТЕКСТ + OCR
            # ============================================================
            # Добавляем OCR только если есть
            if result.get('ocr_text'):
                if result['text']:
                    result['text'] = result['text'] + '\n\n' + result['ocr_text']
                else:
                    result['text'] = result['ocr_text']
                logger.info(f"  📊 Добавлено {len(result['ocr_text'])} символов OCR текста")
            
            # ============================================================
            # ШАГ 6: ИЗВЛЕКАЕМ ЗАГОЛОВКИ ИЗ ТЕКСТА
            # ============================================================
            headers = []
            header_patterns = [
                r'^(?:#+\s*)?([А-ЯЁ][А-ЯЁ\s\d.]+[А-ЯЁ])$',  # ВСЕ ЗАГЛАВНЫЕ
                r'^(?:#+\s*)?(\d+\.\d+\s+[А-ЯЁ][а-яё\s\d]+)$',  # 1.1 Название
                r'^(?:#+\s*)?(\d+\s+[А-ЯЁ][а-яё\s\d]+)$',  # 1 Название
                r'^(?:#+\s*)?([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)$',  # Слово Слово Слово
                r'^(?:#+\s*)?(Приложение\s+\d+[\.\s]*[А-ЯЁа-яё\s\d]+)$',  # Приложение 1 Название
            ]
            
            for page_num, page_text in text_by_page.items():
                lines = page_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line or len(line) < 5 or len(line) > 200:
                        continue
                    
                    # Проверяем по паттернам
                    for pattern in header_patterns:
                        match = re.match(pattern, line, re.IGNORECASE)
                        if match:
                            header_text = match.group(1).strip()
                            # Проверяем, что это не просто цифры
                            if not re.match(r'^\d+$', header_text):
                                # Проверяем, что это не конец предложения
                                if not re.search(r'[.!?]$', header_text):
                                    headers.append({
                                        'page': page_num,
                                        'header': header_text,
                                        'level': self._detect_header_level(header_text, pattern)
                                    })
                                    break
            
            if headers:
                result['metadata']['headers'] = headers
                result['metadata']['header_count'] = len(headers)
                logger.info(f"  📑 Извлечено заголовков: {len(headers)}")
            
            # ============================================================
            # ШАГ 7: СОХРАНЯЕМ ИНФОРМАЦИЮ О СТРАНИЦАХ В МЕТАДАННЫЕ
            # ============================================================
            if text_by_page:
                result['metadata']['page_texts'] = text_by_page
                result['metadata']['page_count'] = len(text_by_page)
                result['metadata']['pages'] = list(text_by_page.keys())
                logger.info(f"  📄 Сохранена информация о {len(text_by_page)} страницах в метаданные")
            
            logger.info(f"✅ ГИБРИДНЫЙ парсинг завершен: {len(result['text'])} символов всего")
            logger.info(f"   📊 Таблиц: {len(result['tables'])}, 🖼️ Изображений: {len(images_data)}, Camelot: {result['extraction_details']['camelot_tables']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка парсинга: {e}", exc_info=True)
            result['metadata']['error'] = str(e)
            return result

    def _detect_header_level(self, header: str, pattern: str) -> int:
        """Определяет уровень заголовка на основе паттерна"""
        if 'Приложение' in header:
            return 1
        if re.match(r'^\d+\.\d+\s', header):
            return 3
        if re.match(r'^\d+\s', header):
            return 2
        if header.isupper() and len(header) > 5:
            return 1
        return 2

    def _format_table(self, table: List[List[str]]) -> str:
        """Форматирует таблицу для текстового представления"""
        if not table:
            return ""
        
        # Очищаем данные
        cleaned = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            cleaned.append(cleaned_row)
        
        # Находим максимальную ширину каждой колонки
        if not cleaned:
            return ""
        
        col_widths = []
        for col_idx in range(len(cleaned[0])):
            max_width = 0
            for row in cleaned:
                if col_idx < len(row):
                    max_width = max(max_width, len(row[col_idx]))
            col_widths.append(max_width + 2)
        
        # Формируем строки
        lines = []
        for row in cleaned:
            row_str = "| "
            for col_idx, cell in enumerate(row):
                if col_idx < len(col_widths):
                    row_str += cell.ljust(col_widths[col_idx]) + " | "
            lines.append(row_str.strip())
        
        # Добавляем разделитель
        if lines:
            separator = "|" + "|".join(["-" * (w + 1) for w in col_widths]) + "|"
            lines.insert(1, separator)
        
        return "\n".join(lines)

    def _ocr_full_pdf(self, file_path: Path) -> str:
        """OCR всей PDF страницы"""
        try:
            images = convert_from_path(str(file_path), dpi=self.ocr_dpi)
            all_text = []
            
            for i, img in enumerate(images, 1):
                logger.info(f"  OCR страница {i}/{len(images)}")
                processed_img = self._preprocess_image(img)
                text = self._perform_hybrid_ocr(processed_img)
                if text.strip():
                    all_text.append(f"[OCR Страница {i}]\n{text}")
            
            return '\n\n'.join(all_text)
        except Exception as e:
            logger.error(f"❌ Ошибка OCR PDF: {e}")
            return ""
    
    # =========================================================================
    # WORD ПАРСИНГ
    # =========================================================================
    
    def _parse_docx(self, file_path: Path) -> Dict[str, Any]:
        """Парсинг DOCX файла с таблицами и картинками"""
        logger.info(f"📄 Парсинг DOCX: {file_path.name}")
        
        result = {
            'text': '',
            'ocr_text': '',
            'tables': [],
            'metadata': {
                'source': file_path.name,
                'type': 'docx',
                'processed_at': datetime.now().isoformat()
            },
            'pages': 0,
            'has_images': False,
            'has_tables': False,
            'ocr_method': None,
            'extraction_details': {
                'paragraphs': 0,
                'tables': 0,
                'images_extracted': 0
            }
        }
        
        try:
            doc = docx.Document(str(file_path))
            
            # Извлекаем параграфы
            paragraphs = []
            headers = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)
                    # Проверяем, является ли параграф заголовком
                    if self._is_header(text):
                        headers.append({
                            'header': text,
                            'level': self._detect_header_level_from_text(text)
                        })
            
            result['extraction_details']['paragraphs'] = len(paragraphs)
            
            if headers:
                result['metadata']['headers'] = headers
                result['metadata']['header_count'] = len(headers)
                logger.info(f"  📑 Извлечено заголовков из DOCX: {len(headers)}")
            
            # Извлекаем таблицы
            tables_text = []
            for table_idx, table in enumerate(doc.tables, 1):
                if table.rows:
                    table_data = []
                    for row in table.rows:
                        row_data = []
                        for cell in row.cells:
                            if cell.text.strip():
                                row_data.append(cell.text.strip())
                        if row_data:
                            table_data.append(row_data)
                    
                    if table_data:
                        result['has_tables'] = True
                        table_str = self._format_table(table_data)
                        tables_text.append(f"[Таблица {table_idx}]\n{table_str}")
            
            result['extraction_details']['tables'] = len(doc.tables)
            
            # Извлечение картинок из DOCX
            images_text = []
            if self.use_ocr and DOCX2TXT_AVAILABLE:
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        docx2txt.process(str(file_path), temp_dir)
                        
                        for file_name in os.listdir(temp_dir):
                            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                                img_path = os.path.join(temp_dir, file_name)
                                try:
                                    with Image.open(img_path) as img:
                                        processed_img = self._preprocess_image(img)
                                        ocr_text = self._perform_hybrid_ocr(processed_img)
                                        
                                        if ocr_text.strip():
                                            images_text.append(f"[Изображение: {file_name}]\n{ocr_text}")
                                            result['extraction_details']['images_extracted'] += 1
                                            result['has_images'] = True
                                except Exception as e:
                                    logger.warning(f"  ⚠️ Ошибка обработки изображения {file_name}: {e}")
                except Exception as e:
                    logger.warning(f"  Ошибка извлечения картинок из docx: {e}")
            
            # Собираем всё вместе
            all_text_parts = []
            
            if paragraphs:
                all_text_parts.append('\n'.join(paragraphs))
            
            if tables_text:
                all_text_parts.append('\n\n'.join(tables_text))
                result['tables'] = tables_text
            
            if images_text:
                ocr_combined = '\n\n'.join(images_text)
                result['ocr_text'] = ocr_combined
                all_text_parts.append(ocr_combined)
                result['ocr_method'] = 'hybrid'
            
            result['text'] = '\n\n'.join(all_text_parts)
            
            logger.info(f"✅ DOCX спарсен: {len(result['text'])} символов, {len(result['tables'])} таблиц, {result['extraction_details']['images_extracted']} картинок")
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка парсинга DOCX: {e}")
            result['metadata']['error'] = str(e)
            return result
    
    def _is_header(self, text: str) -> bool:
        """Проверяет, является ли текст заголовком"""
        if not text or len(text) < 3:
            return False
        
        # Проверяем паттерны заголовков
        patterns = [
            r'^[А-ЯЁ][А-ЯЁ\s\d.]+[А-ЯЁ]$',  # ВСЕ ЗАГЛАВНЫЕ
            r'^\d+\.\d+\s+[А-ЯЁ][а-яё]',  # 1.1 Название
            r'^\d+\s+[А-ЯЁ][а-яё]',  # 1 Название
            r'^Приложение\s+\d+',  # Приложение 1
        ]
        
        for pattern in patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _detect_header_level_from_text(self, text: str) -> int:
        """Определяет уровень заголовка"""
        if 'Приложение' in text:
            return 1
        if re.match(r'^\d+\.\d+\s', text):
            return 3
        if re.match(r'^\d+\s', text):
            return 2
        if text.isupper() and len(text) > 5:
            return 1
        return 2
    
    def _parse_doc(self, file_path: Path) -> Dict[str, Any]:
        """Парсинг старого DOC файла через бинарный экстрактор"""
        logger.info(f"📄 Парсинг DOC: {file_path.name}")
        
        result = {
            'text': '',
            'ocr_text': '',
            'tables': [],
            'metadata': {
                'source': file_path.name,
                'type': 'doc_legacy',
                'processed_at': datetime.now().isoformat()
            },
            'pages': 0,
            'has_images': False,
            'has_tables': False,
            'ocr_method': None,
            'extraction_details': {
                'processor': None,
                'success': False
            }
        }
        
        # Способ 1: Бинарный экстрактор
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            text_pattern = re.compile(rb'[\x20-\x7E\x0410-\x044F]{20,}')
            matches = text_pattern.findall(content)
            
            if matches:
                text_parts = []
                for match in matches:
                    for encoding in ['utf-8', 'cp1251', 'koi8-r']:
                        try:
                            decoded = match.decode(encoding)
                            text_parts.append(decoded)
                            break
                        except UnicodeDecodeError:
                            continue
                
                if text_parts:
                    full_text = '\n'.join(text_parts)
                    result['text'] = self._clean_text(full_text)
                    result['extraction_details']['processor'] = 'binary_extract'
                    result['extraction_details']['success'] = True
                    
                    # Извлекаем заголовки из текста
                    headers = []
                    lines = full_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if self._is_header(line):
                            headers.append({
                                'header': line,
                                'level': self._detect_header_level_from_text(line)
                            })
                    
                    if headers:
                        result['metadata']['headers'] = headers
                        result['metadata']['header_count'] = len(headers)
                        logger.info(f"  📑 Извлечено заголовков из DOC: {len(headers)}")
                    
                    logger.info(f"✅ DOC спарсен через бинарный экстрактор: {len(result['text'])} символов")
                    return result
        except Exception as e:
            logger.warning(f"  Ошибка бинарного экстрактора: {e}")
        
        logger.error(f"❌ Не удалось распарсить DOC файл: {file_path.name}")
        result['metadata']['error'] = 'Не удалось распарсить DOC файл'
        return result
    
    # =========================================================================
    # TXT ПАРСИНГ
    # =========================================================================
    
    def _parse_txt(self, file_path: Path) -> Dict[str, Any]:
        """Парсинг TXT файла"""
        logger.info(f"📄 Парсинг TXT: {file_path.name}")
        
        result = {
            'text': '',
            'ocr_text': '',
            'tables': [],
            'metadata': {
                'source': file_path.name,
                'type': 'txt',
                'processed_at': datetime.now().isoformat()
            },
            'pages': 0,
            'has_images': False,
            'has_tables': False,
            'ocr_method': None,
            'extraction_details': {}
        }
        
        try:
            encodings = ['utf-8', 'cp1251', 'koi8-r', 'latin-1', 'windows-1251']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                        result['text'] = self._clean_text(text)
                        result['metadata']['encoding'] = encoding
                        result['extraction_details']['size'] = len(result['text'])
                        break
                except UnicodeDecodeError:
                    continue
            
            if not result['text']:
                with open(file_path, 'rb') as f:
                    content = f.read()
                    for encoding in encodings:
                        try:
                            text = content.decode(encoding, errors='ignore')
                            if text.strip():
                                result['text'] = self._clean_text(text)
                                result['metadata']['encoding'] = f'{encoding}(with errors)'
                                break
                        except:
                            continue
            
            # Извлекаем заголовки из TXT
            if result['text']:
                headers = []
                lines = result['text'].split('\n')
                for line in lines:
                    line = line.strip()
                    if self._is_header(line):
                        headers.append({
                            'header': line,
                            'level': self._detect_header_level_from_text(line)
                        })
                
                if headers:
                    result['metadata']['headers'] = headers
                    result['metadata']['header_count'] = len(headers)
                    logger.info(f"  📑 Извлечено заголовков из TXT: {len(headers)}")
            
            logger.info(f"✅ TXT спарсен: {len(result['text'])} символов")
        except Exception as e:
            logger.error(f"❌ Ошибка парсинга TXT: {e}")
            result['metadata']['error'] = str(e)
        
        return result
    
    # =========================================================================
    # OCR МЕТОДЫ (ГИБРИДНЫЙ OCR)
    # =========================================================================
    
    def _perform_hybrid_ocr(self, image: Image.Image) -> str:
        """Улучшенный OCR: Tesseract + EasyOCR с фильтрацией мусора"""
        results = {}
        
        # Tesseract
        if self.tesseract_available:
            try:
                text_tesseract = self._ocr_tesseract(image)
                # Фильтруем мусор
                if self._is_valid_text(text_tesseract):
                    results['tesseract'] = text_tesseract
            except:
                pass
        
        # EasyOCR
        if self.easyocr_available:
            try:
                text_easyocr = self._ocr_easyocr(image)
                if self._is_valid_text(text_easyocr):
                    results['easyocr'] = text_easyocr
            except:
                pass
        
        # Выбираем лучший
        best_text = self._select_best_ocr_result(results)
        
        # Очищаем от мусора
        return self._clean_ocr_text(best_text)

    def _is_valid_text(self, text: str) -> bool:
        """Проверяет, что текст не содержит мусора"""
        if not text or len(text) < 10:
            return False
        
        # Считаем долю читаемых символов
        readable = sum(1 for c in text if c.isalnum() or c in ' .,!?-')
        ratio = readable / len(text) if text else 0
        
        # Если меньше 50% читаемых символов — это мусор
        return ratio > 0.5

    def _clean_ocr_text(self, text: str) -> str:
        """Очищает текст от мусора"""
        if not text:
            return ""
        
        # Удаляем строки с мусорными символами
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Если в строке мало читаемых символов — пропускаем
            readable = sum(1 for c in line if c.isalnum() or c in ' .,!?-')
            if len(line) > 0 and readable / len(line) > 0.4:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    
    def _select_best_ocr_result(self, results: Dict[str, str]) -> str:
        """Выбирает лучший результат OCR"""
        tesseract_text = results.get('tesseract', '')
        easyocr_text = results.get('easyocr', '')
        
        if not tesseract_text and not easyocr_text:
            return ""
        
        if not tesseract_text:
            return easyocr_text
        if not easyocr_text:
            return tesseract_text
        
        # Проверяем таблицу
        tesseract_is_table = self._is_likely_table(tesseract_text)
        easyocr_is_table = self._is_likely_table(easyocr_text)
        
        if tesseract_is_table and not easyocr_is_table:
            return tesseract_text
        if easyocr_is_table and not tesseract_is_table:
            return easyocr_text
        
        # По длине
        if len(tesseract_text) > len(easyocr_text) * 1.3:
            return tesseract_text
        elif len(easyocr_text) > len(tesseract_text) * 1.3:
            return easyocr_text
        
        return tesseract_text
    
    def _is_likely_table(self, text: str) -> bool:
        """Проверяет, похож ли текст на таблицу"""
        if not text or len(text) < 20:
            return False
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if len(lines) < 3:
            return False
        
        has_separators = any('|' in line or '\t' in line or '  ' in line for line in lines)
        has_numbers = any(line[0].isdigit() for line in lines if line)
        is_aligned = len(set(len(line) for line in lines[:3])) <= 2
        
        return has_separators or (is_aligned and has_numbers)
    
    def _ocr_tesseract(self, image: Image.Image) -> str:
        """OCR через Tesseract"""
        try:
            custom_config = f'--oem 3 --psm 6 -l {self.ocr_lang}'
            text = pytesseract.image_to_string(image, config=custom_config)
            return self._clean_text(text)
        except Exception as e:
            return ""
    
    def _ocr_easyocr(self, image: Image.Image) -> str:
        """OCR через EasyOCR"""
        try:
            image_np = np.array(image)
            results = self.easyocr_reader.readtext(image_np, detail=0)
            text = ' '.join(results)
            return self._clean_text(text)
        except Exception as e:
            return ""

    # =========================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # =========================================================================
    
    def _clean_text(self, text: str) -> str:
        """Очистка текста от мусора"""
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-"\']', ' ', text)
        return text.strip()
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Предобработка изображения для улучшения OCR"""
        try:
            if image.mode != 'L':
                image = image.convert('L')
            
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            image = image.point(lambda x: 0 if x < 128 else 255, '1')
            image = image.convert('L')
            return image
        except Exception as e:
            return image
    
    def parse_all_documents(self, directory: Path) -> List[Dict[str, Any]]:
        """Парсит все документы в директории"""
        documents = []
        supported_extensions = getattr(Config, 'SUPPORTED_EXTENSIONS', ['.pdf', '.docx', '.doc', '.txt', '.md'])
        
        for ext in supported_extensions:
            for file_path in directory.glob(f'*{ext}'):
                if file_path.is_file():
                    logger.info(f"📂 Обработка: {file_path.name}")
                    doc_data = self.parse_document(file_path)
                    
                    has_text = bool(doc_data.get('text', '').strip())
                    has_ocr = bool(doc_data.get('ocr_text', '').strip())
                    has_tables = bool(doc_data.get('tables', []))
                    
                    if has_text or has_ocr or has_tables:
                        documents.append(doc_data)
                        logger.info(f"  ✅ Добавлен: {len(doc_data['text'])} символов текста, {len(doc_data.get('tables', []))} таблиц")
                        if has_ocr:
                            logger.info(f"     + OCR: {len(doc_data['ocr_text'])} символов")
                        if doc_data.get('metadata', {}).get('headers'):
                            logger.info(f"     + Заголовков: {len(doc_data['metadata']['headers'])}")
                    else:
                        logger.warning(f"  ⚠️ Пустой документ: {file_path.name}")
        
        return documents

    def _extract_tables_with_camelot(self, file_path: Path) -> List[Dict[str, Any]]:
        """Извлечение таблиц с помощью Camelot (если установлен)"""
        tables = []
        try:
            import camelot
            logger.info("  🐫 Используем Camelot для извлечения таблиц...")
            
            # Пробуем разные стратегии
            for flavor in ['lattice', 'stream']:
                try:
                    extracted = camelot.read_pdf(
                        str(file_path), 
                        pages='all', 
                        flavor=flavor,
                        strip_text='\n'
                    )
                    
                    if extracted:
                        for i, table in enumerate(extracted, 1):
                            df = table.df
                            # Очищаем от пустых строк
                            df = df.dropna(how='all')
                            df = df.dropna(axis=1, how='all')
                            
                            if not df.empty:
                                table_text = df.to_string(index=False, header=False)
                                tables.append({
                                    'page': table.page,
                                    'text': table_text,
                                    'html': df.to_html(index=False),
                                    'raw': df
                                })
                                logger.info(f"  📊 Camelot ({flavor}): найдена таблица {i} на странице {table.page}, {len(df)}x{len(df.columns)}")
                        
                        # Если нашли таблицы, выходим
                        if tables:
                            break
                            
                except Exception as e:
                    logger.warning(f"  ⚠️ Camelot {flavor} ошибка: {e}")
                    continue
                    
        except ImportError:
            logger.info("  ℹ️ Camelot не установлен, используем pdfplumber")
            return []
        except Exception as e:
            logger.warning(f"  ⚠️ Ошибка Camelot: {e}")
            return []
        
        return tables

    def _extract_tables_with_pdfplumber_advanced(self, file_path: Path) -> List[Dict[str, Any]]:
        """Улучшенное извлечение таблиц через pdfplumber"""
        tables = []
        try:
            with pdfplumber.open(str(file_path)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Извлекаем таблицы с улучшенными настройками
                    page_tables = page.extract_tables({
                        'vertical_strategy': 'lines',
                        'horizontal_strategy': 'lines',
                        'snap_tolerance': 3,
                        'join_tolerance': 3,
                        'edge_min_length': 3,
                        'min_words_vertical': 1,
                        'min_words_horizontal': 1,
                        'intersection_tolerance': 3,
                        'text_tolerance': 3,
                    })
                    
                    if page_tables:
                        for table_idx, table in enumerate(page_tables, 1):
                            if table and len(table) > 1:
                                # Очищаем таблицу
                                cleaned_table = []
                                for row in table:
                                    cleaned_row = [str(cell).strip() if cell else '' for cell in row]
                                    # Пропускаем пустые строки
                                    if any(cell for cell in cleaned_row):
                                        cleaned_table.append(cleaned_row)
                                
                                if cleaned_table and len(cleaned_table) > 1:
                                    # Форматируем как текст
                                    table_text = self._format_table_advanced(cleaned_table)
                                    tables.append({
                                        'page': page_num,
                                        'text': table_text,
                                        'raw': cleaned_table
                                    })
                                    logger.info(f"  📊 pdfplumber: таблица {table_idx} на странице {page_num}, {len(cleaned_table)} строк")
        except Exception as e:
            logger.warning(f"  ⚠️ Ошибка pdfplumber: {e}")
        
        return tables

    def _format_table_advanced(self, table: List[List[str]]) -> str:
        """Улучшенное форматирование таблицы для текстового представления"""
        if not table:
            return ""
        
        # Находим максимальную ширину каждой колонки
        col_count = max(len(row) for row in table)
        col_widths = []
        
        for col_idx in range(col_count):
            max_width = 0
            for row in table:
                if col_idx < len(row):
                    max_width = max(max_width, len(str(row[col_idx])))
            col_widths.append(max_width + 2)
        
        # Формируем строки
        lines = []
        for row_idx, row in enumerate(table):
            # Выравниваем строку
            padded_row = row + [''] * (col_count - len(row))
            
            # Формируем строку таблицы
            row_str = "| "
            for col_idx, cell in enumerate(padded_row):
                row_str += str(cell).ljust(col_widths[col_idx]) + " | "
            lines.append(row_str.strip())
            
            # Добавляем разделитель после заголовка
            if row_idx == 0 and len(table) > 1:
                separator = "+" + "+".join(['-' * (w + 1) for w in col_widths]) + "+"
                lines.append(separator)
        
        return "\n".join(lines)


# Глобальный экземпляр парсера
document_parser = DocumentParser()