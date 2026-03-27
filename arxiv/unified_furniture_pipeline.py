from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pdf2image import convert_from_path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from arxiv.furniture_postprocess import clean_furniture_catalog


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EXTRACTION_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"
DEFAULT_ROUTER_MODEL = "x-ai/grok-4.1-fast"
DEFAULT_DPI = 170

TARGET_EXPORT_COLUMNS = [
    "sku",
    "attribute_set_code",
    "product_type",
    "name",
    "price",
    "categories",
    "visibility",
    "guarantee",
    "description",
    "short_description",
    "weight",
    "product_online",
    "tax_class_name",
    "url_key",
    "meta_title",
    "is_in_stock",
    "astrio_discount_manufacturer",
    "astrio_margin",
    "collections",
    "brand",
    "countries",
    "astrio_height",
    "astrio_width",
    "astrio_length",
    "astrio_volume_package",
    "expected_delivery",
    "color",
    "auto_converter_currency",
]

ATTRIBUTE_MEANINGS = {
    "sku": "Уникальный артикул товара.",
    "attribute_set_code": "Набор атрибутов или товарная группа в каталоге.",
    "product_type": "Тип товара для CMS/магазина, обычно simple.",
    "name": "Название товара для каталога.",
    "price": "Цена товара в прайсе.",
    "categories": "Категории магазина, в которые должен попасть товар.",
    "visibility": "Видимость товара на витрине магазина.",
    "guarantee": "Гарантийный срок товара.",
    "description": "Полное описание товара.",
    "short_description": "Короткое описание или тип товара.",
    "weight": "Вес товара.",
    "product_online": "Флаг публикации товара в магазине.",
    "tax_class_name": "Налоговый класс товара.",
    "url_key": "Часть URL для страницы товара.",
    "meta_title": "SEO-заголовок страницы товара.",
    "is_in_stock": "Флаг наличия товара.",
    "astrio_discount_manufacturer": "Скидка производителя в процентах или условных единицах.",
    "astrio_margin": "Наценка на товар.",
    "collections": "Коллекция или серия товара.",
    "brand": "Бренд товара. Для одного прайса бренд должен быть одинаковым.",
    "countries": "Страна производства или брендовая страна.",
    "astrio_height": "Высота товара в сантиметрах.",
    "astrio_width": "Ширина товара в сантиметрах.",
    "astrio_length": "Длина или глубина товара в сантиметрах.",
    "astrio_volume_package": "Объем упаковки в кубических метрах.",
    "expected_delivery": "Ожидаемый срок поставки.",
    "color": "Основной цвет товара. Допустимы только: white, black, green, brown, beige, yellow, gray, orange, pink, blue, red, purple. Сначала берется с картинки, затем из описания.",
    "auto_converter_currency": "Флаг автоматической валютной конвертации.",
}

ALLOWED_COLORS = [
    "white",
    "black",
    "green",
    "brown",
    "beige",
    "yellow",
    "gray",
    "orange",
    "pink",
    "blue",
    "red",
    "purple",
]

COUNTRY_BY_BRAND = {
    "bt": "Turkey",
    "bralco": "Italy",
    "diemme": "Italy",
    "frezza": "Italy",
    "gaber": "Italy",
    "johanson": "Sweden",
    "kastel": "Italy",
    "lapalma": "Italy",
    "mara": "Italy",
    "martex": "Italy",
    "pedrali": "Italy",
    "poliform": "Italy",
    "quinti": "Italy",
    "quadrifoglio": "Italy",
    "sedus": "Germany",
    "tonon": "Italy",
    "vondom": "Spain",
}

COLOR_ALIASES = {
    "white": "white",
    "bianco": "white",
    "ivory": "beige",
    "cream": "beige",
    "sand": "beige",
    "beige": "beige",
    "black": "black",
    "nero": "black",
    "green": "green",
    "verde": "green",
    "brown": "brown",
    "walnut": "brown",
    "noce": "brown",
    "oak": "brown",
    "wood": "brown",
    "natural": "beige",
    "yellow": "yellow",
    "gold": "yellow",
    "gray": "gray",
    "grey": "gray",
    "grigio": "gray",
    "anthracite": "gray",
    "antracite": "gray",
    "silver": "gray",
    "aluminium": "gray",
    "aluminum": "gray",
    "chrome": "gray",
    "metal": "gray",
    "orange": "orange",
    "pink": "pink",
    "rose": "pink",
    "blue": "blue",
    "blu": "blue",
    "red": "red",
    "rosso": "red",
    "burgundy": "red",
    "purple": "purple",
    "violet": "purple",
    "lilac": "purple",
}


CATEGORY_PROMPTS = {
    "Chairs": """
Extract data for CHAIRS/SEATING from the image.
If dimensions are missing for a specific variant, copy them from the base model on the SAME page or from a clearly matching model family on the SAME spread.
If multiple configurations exist, keep the general/base SKU and the price of the maximum configuration.

Fields per item:
- name: string
- sku: string
- price: number
- color: choose exactly one dominant color from this list only: white, black, green, brown, beige, yellow, gray, orange, pink, blue, red, purple; take it from the image first, otherwise infer from the description, otherwise leave empty
- astrio_length: number
- astrio_width: number
- astrio_height: number
- description: string
- brand: string
- short_description: string
- dimension_source: one of parsed, same_page_base_model, same_family, inferred_from_image, missing
- dimension_confidence: number from 0 to 1
""",
    "Tables": """
Extract data for TABLES/DESKS from the image.
If dimensions are missing, copy them from the base model on the SAME page when clearly applicable.

Fields per item:
- name: string
- sku: string
- price: number
- color: choose exactly one dominant color from this list only: white, black, green, brown, beige, yellow, gray, orange, pink, blue, red, purple; take it from the image first, otherwise infer from the description, otherwise leave empty
- astrio_length: number
- astrio_width: number
- astrio_height: number
- description: string
- brand: string
- short_description: string
- dimension_source: one of parsed, same_page_base_model, same_family, inferred_from_image, missing
- dimension_confidence: number from 0 to 1
""",
    "Storage": """
Extract data for STORAGE from the image.
If dimensions are missing, infer them only from the same family on the SAME page. Do not invent values.

Fields per item:
- name: string
- sku: string
- price: number
- color: choose exactly one dominant color from this list only: white, black, green, brown, beige, yellow, gray, orange, pink, blue, red, purple; take it from the image first, otherwise infer from the description, otherwise leave empty
- astrio_length: number
- astrio_width: number
- astrio_height: number
- description: string
- brand: string
- short_description: string
- dimension_source: one of parsed, same_page_base_model, same_family, inferred_from_image, missing
- dimension_confidence: number from 0 to 1
""",
    "Sofas": """
Extract data for SOFAS / LOUNGE / POUFS from the image.
If dimensions are missing for a module or variant, copy them from the base model on the SAME page when appropriate.

Fields per item:
- name: string
- sku: string
- price: number
- color: choose exactly one dominant color from this list only: white, black, green, brown, beige, yellow, gray, orange, pink, blue, red, purple; take it from the image first, otherwise infer from the description, otherwise leave empty
- astrio_length: number
- astrio_width: number
- astrio_height: number
- description: string
- brand: string
- short_description: string
- dimension_source: one of parsed, same_page_base_model, same_family, inferred_from_image, missing
- dimension_confidence: number from 0 to 1
""",
    "Generic": """
Extract standard furniture product data from the image.
If dimensions are missing, do not invent them. Use same-page base model or same family only when the match is explicit.

Fields per item:
- name: string
- sku: string
- price: number
- color: choose exactly one dominant color from this list only: white, black, green, brown, beige, yellow, gray, orange, pink, blue, red, purple; take it from the image first, otherwise infer from the description, otherwise leave empty
- astrio_length: number
- astrio_width: number
- astrio_height: number
- description: string
- brand: string
- short_description: string
- dimension_source: one of parsed, same_page_base_model, same_family, inferred_from_image, missing
- dimension_confidence: number from 0 to 1
""",
}


ROUTER_PROMPT = """
Look at this page from a furniture price list.
Choose one option:
1. Chairs
2. Tables
3. Storage
4. Sofas
5. Other
6. Skip

Return ONLY one word from this list:
Chairs, Tables, Storage, Sofas, Other, Skip
"""


EXTRACTION_PROMPT_TEMPLATE = """
You are a data extraction specialist for furniture price lists.
Analyze this page image containing {category}.

{category_prompt}

Critical rules:
1. Extract products only, not materials pages, general conditions, accessories-only notes, fabric legends, or table of contents.
2. Return a JSON array only. No markdown.
3. Do not invent SKU.
4. If price is a range or multiple configurations, keep the maximum product price.
5. If dimension confidence is below 0.65, leave the value empty and use dimension_source="missing".
6. If the page does not contain actual products, return [].
7. Brand must be the manufacturer brand of the whole price list, not the designer name.
8. Color must come from the image first. If the image does not make the color clear, infer it from the description text. If still unclear, leave it empty.
9. The color value must be exactly one lowercase value from this set only: white, black, green, brown, beige, yellow, gray, orange, pink, blue, red, purple.
"""


@dataclass
class VLMConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    extraction_model: str = DEFAULT_EXTRACTION_MODEL
    router_model: str = DEFAULT_ROUTER_MODEL
    timeout_sec: int = 180
    temperature: float = 0.0


@dataclass
class RoutedPage:
    page_num: int
    category: str


def env_first(keys: Iterable[str]) -> str:
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def config_from_env() -> VLMConfig | None:
    api_key = env_first(["VLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"])
    if not api_key:
        return None
    base_url = env_first(["VLM_BASE_URL", "OPENROUTER_BASE_URL", "OPENAI_BASE_URL"]) or DEFAULT_BASE_URL
    extraction_model = env_first(["VLM_MODEL", "OPENROUTER_MODEL"]) or DEFAULT_EXTRACTION_MODEL
    router_model = env_first(["VLM_ROUTER_MODEL", "OPENROUTER_ROUTER_MODEL"]) or DEFAULT_ROUTER_MODEL
    return VLMConfig(
        api_key=api_key,
        base_url=base_url,
        extraction_model=extraction_model,
        router_model=router_model,
    )


def backend_status() -> dict[str, bool | str]:
    cfg = config_from_env()
    return {
        "pdf2image": True,
        "vlm_api_key_present": bool(cfg and cfg.api_key),
        "vlm_base_url": cfg.base_url if cfg else "",
        "vlm_extraction_model": cfg.extraction_model if cfg else "",
        "vlm_router_model": cfg.router_model if cfg else "",
    }


def safe_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    text = str(value).replace("\u00a0", " ").replace(" ", "").replace(",", ".")
    text = re.sub(r"[^0-9.]+", "", text)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace("\u00a0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def slugify(value: object) -> str:
    text = normalize_text(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def pdf_brand_from_name(pdf_name: str) -> str:
    stem = Path(pdf_name).stem
    stem = re.sub(r"\s*\(\d+\)$", "", stem).strip()
    first_chunk = re.split(r"_+", stem, maxsplit=1)[0].strip()
    if not first_chunk:
        first_chunk = re.split(r"\s+", stem, maxsplit=1)[0].strip()
    first_chunk = re.sub(r"\s+", " ", first_chunk)
    return first_chunk or "Unknown"


def normalize_brand_key(value: object) -> str:
    text = normalize_text(value).lower()
    text = text.replace("&", "")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def infer_color_from_text(*parts: object) -> str:
    haystack = " ".join(normalize_text(part).lower() for part in parts if normalize_text(part))
    best_match: tuple[int, str] | None = None
    for token, normalized in COLOR_ALIASES.items():
        pattern = rf"(?<![a-z]){re.escape(token)}(?![a-z])"
        match = re.search(pattern, haystack)
        if not match:
            continue
        if normalized not in ALLOWED_COLORS:
            continue
        if best_match is None or match.start() < best_match[0]:
            best_match = (match.start(), normalized)
    if best_match:
        return best_match[1]
    return ""


def normalize_color_choice(primary_value: object, *fallback_parts: object) -> tuple[str, str]:
    primary = normalize_text(primary_value)
    primary_color = infer_color_from_text(primary)
    if primary_color:
        return primary_color, "image_or_model"
    fallback_color = infer_color_from_text(*fallback_parts)
    if fallback_color:
        return fallback_color, "description_fallback"
    return "", "missing"


def normalize_dimension(value: object) -> float | None:
    number = safe_float(value)
    if number is None or number <= 0:
        return None
    if 250 < number <= 3000:
        number = number / 10.0
    return round(number, 2) if 10 <= number <= 500 else None


def encode_image_to_data_url(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


class OpenAICompatVLMClient:
    def __init__(self, config: VLMConfig):
        self.config = config

    def _post_chat_completions(self, *, model: str, messages: list[dict[str, Any]], temperature: float = 0.0) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_sec) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"VLM HTTP {exc.code}: {body[:800]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"VLM network error: {exc}") from exc

        try:
            return result["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Unexpected VLM response: {result}") from exc

    def classify_page(self, image, page_num: int) -> str:
        data_url = encode_image_to_data_url(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ROUTER_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        content = self._post_chat_completions(
            model=self.config.router_model,
            messages=messages,
            temperature=0.0,
        ).strip()
        normalized = content.replace(".", "").strip()
        if "Chair" in normalized:
            return "Chairs"
        if "Table" in normalized:
            return "Tables"
        if "Storage" in normalized:
            return "Storage"
        if "Sofa" in normalized:
            return "Sofas"
        if "Skip" in normalized:
            return "Skip"
        if "Other" in normalized:
            return "Generic"
        return "Generic"

    def extract_page_items(self, image, category: str) -> list[dict[str, Any]]:
        data_url = encode_image_to_data_url(image)
        category_prompt = CATEGORY_PROMPTS.get(category, CATEGORY_PROMPTS["Generic"])
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(category=category, category_prompt=category_prompt)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        content = self._post_chat_completions(
            model=self.config.extraction_model,
            messages=messages,
            temperature=self.config.temperature,
        )
        return parse_vlm_json(content)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = text.replace("```json", "```")
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    return text


def parse_vlm_json(text: str) -> list[dict[str, Any]]:
    text = strip_code_fences(text)
    if not text:
        return []
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    elif text.startswith("{") and text.endswith("}"):
        text = f"[{text}]"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    return [item for item in data if isinstance(item, dict)]


def guess_brand(pdf_name: str) -> str:
    return pdf_brand_from_name(pdf_name)


def normalize_vlm_item(item: dict[str, Any], *, pdf_name: str, page_num: int, category: str, model_name: str) -> dict[str, Any] | None:
    sku = normalize_text(item.get("sku") or item.get("article"))
    name = normalize_text(item.get("name") or item.get("title"))
    price = safe_float(item.get("price"))
    if price is None:
        prices = [safe_float(item.get("max_price")), safe_float(item.get("price_max")), safe_float(item.get("min_price"))]
        prices = [value for value in prices if value is not None]
        price = max(prices) if prices else 0
    if not sku and not name:
        return None
    description = normalize_text(item.get("description"))
    short_description = normalize_text(item.get("short_description") or item.get("type") or category)
    color, color_source = normalize_color_choice(item.get("color"), description, short_description, name)
    brand = guess_brand(pdf_name)

    normalized = {
        "pdf_name": pdf_name,
        "page": page_num,
        "sku": sku,
        "name": name or sku,
        "price": price or 0,
        "brand": brand,
        "color": color,
        "description": description,
        "short_description": short_description,
        "collections": normalize_text(item.get("collections")) or (name.split()[0] if name else ""),
        "astrio_height": normalize_dimension(item.get("astrio_height") or item.get("height")),
        "astrio_width": normalize_dimension(item.get("astrio_width") or item.get("width")),
        "astrio_length": normalize_dimension(item.get("astrio_length") or item.get("length") or item.get("depth")),
        "dimension_source": normalize_text(item.get("dimension_source")),
        "dimension_confidence": safe_float(item.get("dimension_confidence")),
        "color_source": color_source,
        "parser_backend": f"vlm:{model_name}",
        "parser_priority": 100,
    }
    return normalized


def extraction_model_name(client: Any) -> str:
    config = getattr(client, "config", None)
    return getattr(config, "extraction_model", "unknown-model")


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    frame = pd.DataFrame(records)
    frame["_key"] = (
        frame.get("page", 0).astype(str)
        + "|"
        + frame.get("sku", "").fillna("").astype(str).str.upper().str.replace(" ", "", regex=False)
        + "|"
        + frame.get("name", "").fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True)
    )
    frame = frame.sort_values(by=["parser_priority", "price"], ascending=[False, False], na_position="last")
    frame = frame.drop_duplicates(subset=["_key"], keep="first")
    return frame.drop(columns=["_key"]).to_dict("records")


def attribute_dictionary_df() -> pd.DataFrame:
    return pd.DataFrame(
        [{"attribute": key, "meaning": value} for key, value in ATTRIBUTE_MEANINGS.items()]
    )


def unify_brand_per_pdf(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "pdf_name" not in df.columns:
        return df.copy()
    out = df.copy()
    out["brand"] = out["pdf_name"].map(pdf_brand_from_name)
    return out


def infer_country_for_brand(brand: str) -> str:
    return COUNTRY_BY_BRAND.get(normalize_brand_key(brand), "")


def category_path_for_row(row: pd.Series) -> str:
    attribute_set = normalize_text(row.get("attribute_set_code"))
    brand = normalize_text(row.get("brand"))
    pieces = ["Default Category", "Default Category/All product"]
    if brand:
        pieces.append(f"Default Category/Brands/{brand}")
    if attribute_set:
        pieces.append(f"Default Category/{attribute_set}")
    return ",".join(dict.fromkeys(pieces))


def attribute_set_for_row(row: pd.Series) -> str:
    text = normalize_text(row.get("short_description") or row.get("name")).lower()
    if any(token in text for token in ["chair", "stool", "armchair", "seat", "bench"]):
        return "Chairs and stools"
    if any(token in text for token in ["table", "desk"]):
        return "Tables"
    if any(token in text for token in ["sofa", "pouf", "lounge"]):
        return "Sofas"
    if any(token in text for token in ["storage", "cabinet", "shelf", "locker"]):
        return "Storage"
    return "Furniture"


def compute_package_volume(row: pd.Series) -> float:
    dims = [safe_float(row.get("astrio_height")), safe_float(row.get("astrio_width")), safe_float(row.get("astrio_length"))]
    if any(value is None or value <= 0 for value in dims):
        return 0.0
    height, width, length = dims
    return round((height * width * length) / 1_000_000, 3)


def export_catalog_product(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=TARGET_EXPORT_COLUMNS)

    out = unify_brand_per_pdf(df)
    out = out.copy()
    out["attribute_set_code"] = out.apply(attribute_set_for_row, axis=1)
    out["product_type"] = "simple"
    out["categories"] = out.apply(category_path_for_row, axis=1)
    out["visibility"] = "Catalog, Search"
    out["guarantee"] = "2 years"
    out["description"] = out["description"].fillna("").astype(str)
    out["short_description"] = out["short_description"].fillna("").astype(str)
    out["weight"] = ""
    out["product_online"] = 1
    out["tax_class_name"] = "Taxable Goods"
    out["url_key"] = out.apply(
        lambda row: slugify("-".join(part for part in [row.get("brand", ""), row.get("name", ""), row.get("sku", "")] if normalize_text(part))),
        axis=1,
    )
    out["meta_title"] = out["name"].fillna("").astype(str)
    out["is_in_stock"] = 1
    out["astrio_discount_manufacturer"] = 0
    out["astrio_margin"] = 0
    out["countries"] = out["brand"].apply(infer_country_for_brand)
    out["astrio_volume_package"] = out.apply(compute_package_volume, axis=1)
    out["expected_delivery"] = "6-8 weeks"
    out["auto_converter_currency"] = 1
    color_resolution = out.apply(
        lambda row: normalize_color_choice(
            row.get("color"),
            row.get("description"),
            row.get("short_description"),
            row.get("name"),
        ),
        axis=1,
    )
    out["color"] = color_resolution.apply(lambda value: value[0])
    out["color_source"] = color_resolution.apply(lambda value: value[1])
    for col in TARGET_EXPORT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out["price"] = out["price"].apply(lambda value: round(value, 2) if safe_float(value) is not None else "")
    out["name"] = out["name"].fillna("").astype(str)
    out["sku"] = out["sku"].fillna("").astype(str)
    out = out[TARGET_EXPORT_COLUMNS].copy()
    return out.sort_values(by=["brand", "collections", "sku", "name"], na_position="last").reset_index(drop=True)


def parse_explicit_pages(value: str | None) -> list[int] | None:
    if not value:
        return None
    pages = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return sorted(set(page for page in pages if page > 0))


def select_page_numbers(
    total_pages: int,
    *,
    strategy: str = "uniform",
    sample_pages: int = 8,
    start_page: int = 1,
    explicit_pages: list[int] | None = None,
) -> list[int]:
    if total_pages <= 0:
        return []
    if explicit_pages:
        return [page for page in explicit_pages if 1 <= page <= total_pages]

    start_page = max(1, min(start_page, total_pages))
    candidates = list(range(start_page, total_pages + 1))
    if strategy == "all":
        return candidates
    if strategy == "from_page":
        return candidates[:sample_pages]
    if strategy == "tail":
        return candidates[-sample_pages:]
    if strategy == "uniform":
        if len(candidates) <= sample_pages:
            return candidates
        picks = set()
        last_index = len(candidates) - 1
        for i in range(sample_pages):
            idx = round(i * last_index / max(sample_pages - 1, 1))
            picks.add(candidates[idx])
        return sorted(picks)
    raise ValueError(f"Unknown page strategy: {strategy}")


def render_pages(pdf_path: str | Path, page_numbers: list[int], dpi: int = DEFAULT_DPI) -> list[tuple[int, Any]]:
    rendered: list[tuple[int, Any]] = []
    for page_num in page_numbers:
        images = convert_from_path(str(pdf_path), dpi=dpi, first_page=page_num, last_page=page_num)
        if images:
            rendered.append((page_num, images[0]))
    return rendered


def route_pdf_pages(
    pdf_path: str | Path,
    *,
    client: Any,
    page_strategy: str = "uniform",
    sample_pages: int = 8,
    start_page: int = 10,
    explicit_pages: list[int] | None = None,
    dpi: int = 120,
) -> list[RoutedPage]:
    pdf_path = Path(pdf_path)
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
    except Exception as exc:
        raise RuntimeError(f"Unable to open PDF {pdf_path}: {exc}") from exc

    selected_pages = select_page_numbers(
        total_pages,
        strategy=page_strategy,
        sample_pages=sample_pages,
        start_page=start_page,
        explicit_pages=explicit_pages,
    )
    routed: list[RoutedPage] = []
    for page_num, image in render_pages(pdf_path, selected_pages, dpi=dpi):
        category = client.classify_page(image, page_num)
        routed.append(RoutedPage(page_num=page_num, category=category))
    return routed


def extract_pdf_records(
    pdf_path: str | Path,
    *,
    client: Any,
    page_strategy: str = "uniform",
    sample_pages: int = 8,
    start_page: int = 1,
    explicit_pages: list[int] | None = None,
    dpi: int = DEFAULT_DPI,
) -> pd.DataFrame:
    pdf_path = Path(pdf_path)
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
    except Exception as exc:
        raise RuntimeError(f"Unable to open PDF {pdf_path}: {exc}") from exc

    selected_pages = select_page_numbers(
        total_pages,
        strategy=page_strategy,
        sample_pages=sample_pages,
        start_page=start_page,
        explicit_pages=explicit_pages,
    )
    rows: list[dict[str, Any]] = []
    for page_num, image in render_pages(pdf_path, selected_pages, dpi=dpi):
        category = client.classify_page(image, page_num)
        if category == "Skip":
            continue
        items = client.extract_page_items(image, category)
        for item in items:
            normalized = normalize_vlm_item(
                item,
                pdf_name=pdf_path.name,
                page_num=page_num,
                category=category,
                model_name=extraction_model_name(client),
            )
            if normalized:
                rows.append(normalized)
    return pd.DataFrame(dedupe_records(rows))


def extract_pdf_records_two_phase(
    pdf_path: str | Path,
    *,
    client: Any,
    routing_strategy: str = "all",
    routing_sample_pages: int = 32,
    routing_start_page: int = 1,
    routing_explicit_pages: list[int] | None = None,
    routing_dpi: int = 120,
    extraction_dpi: int = DEFAULT_DPI,
) -> tuple[pd.DataFrame, list[RoutedPage]]:
    pdf_path = Path(pdf_path)
    routed_pages = route_pdf_pages(
        pdf_path,
        client=client,
        page_strategy=routing_strategy,
        sample_pages=routing_sample_pages,
        start_page=routing_start_page,
        explicit_pages=routing_explicit_pages,
        dpi=routing_dpi,
    )
    extraction_targets = [page.page_num for page in routed_pages if page.category != "Skip"]
    category_by_page = {page.page_num: ("Generic" if page.category == "Other" else page.category) for page in routed_pages}
    rows: list[dict[str, Any]] = []
    for page_num, image in render_pages(pdf_path, extraction_targets, dpi=extraction_dpi):
        category = category_by_page.get(page_num, "Generic")
        items = client.extract_page_items(image, category)
        for item in items:
            normalized = normalize_vlm_item(
                item,
                pdf_name=pdf_path.name,
                page_num=page_num,
                category=category,
                model_name=extraction_model_name(client),
            )
            if normalized:
                rows.append(normalized)
    return pd.DataFrame(dedupe_records(rows)), routed_pages


def summarise_cleaned(df: pd.DataFrame) -> dict[str, int]:
    flags = df.get("quality_flag", pd.Series(dtype=str)).value_counts().to_dict()
    prices = pd.to_numeric(df["price"], errors="coerce") if "price" in df.columns else pd.Series(dtype="float64")
    return {
        "clean_rows": len(df),
        "price_nonzero": int(prices.fillna(0).gt(0).sum()),
        "needs_review": int(flags.get("needs_review", 0)),
        "imputed": int(flags.get("imputed", 0)),
    }


def process_single_pdf(
    pdf_path: str | Path,
    *,
    client: Any,
    output_dir: str | Path | None = None,
    page_strategy: str = "uniform",
    sample_pages: int = 8,
    start_page: int = 10,
    explicit_pages: list[int] | None = None,
    dpi: int = DEFAULT_DPI,
    two_phase: bool = False,
    routing_strategy: str = "all",
    routing_sample_pages: int = 32,
    routing_start_page: int = 1,
    routing_explicit_pages: list[int] | None = None,
    routing_dpi: int = 120,
) -> dict[str, Any]:
    pdf_path = Path(pdf_path)
    started = time.time()
    routed_pages: list[RoutedPage] = []
    if two_phase:
        raw_df, routed_pages = extract_pdf_records_two_phase(
            pdf_path,
            client=client,
            routing_strategy=routing_strategy,
            routing_sample_pages=routing_sample_pages,
            routing_start_page=routing_start_page,
            routing_explicit_pages=routing_explicit_pages or explicit_pages,
            routing_dpi=routing_dpi,
            extraction_dpi=dpi,
        )
    else:
        raw_df = extract_pdf_records(
            pdf_path,
            client=client,
            page_strategy=page_strategy,
            sample_pages=sample_pages,
            start_page=start_page,
            explicit_pages=explicit_pages,
            dpi=dpi,
        )
    raw_df = unify_brand_per_pdf(raw_df) if not raw_df.empty else raw_df.copy()
    clean_df = clean_furniture_catalog(raw_df) if not raw_df.empty else raw_df.copy()
    export_df = export_catalog_product(clean_df) if not clean_df.empty else pd.DataFrame(columns=TARGET_EXPORT_COLUMNS)
    result = {
        "pdf_name": pdf_path.name,
        "mode": "two_phase" if two_phase else "one_phase",
        "page_strategy": page_strategy,
        "sample_pages": sample_pages,
        "start_page": start_page,
        "raw_rows": len(raw_df),
        "elapsed_sec": round(time.time() - started, 2),
    }
    if two_phase:
        result["routing_strategy"] = routing_strategy
        result["routing_sample_pages"] = routing_sample_pages
        result["routing_start_page"] = routing_start_page
        result["routed_pages"] = len(routed_pages)
        result["product_like_pages"] = int(sum(page.category != "Skip" for page in routed_pages))
    result.update(summarise_cleaned(clean_df))

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = pdf_path.stem
        raw_path = out_dir / f"{stem}__raw.csv"
        clean_path = out_dir / f"{stem}__clean.csv"
        export_path = out_dir / f"{stem}__catalog_product.csv"
        routed_path = out_dir / f"{stem}__routed_pages.csv"
        raw_df.to_csv(raw_path, index=False)
        clean_df.to_csv(clean_path, index=False)
        export_df.to_csv(export_path, index=False)
        result["raw_csv"] = str(raw_path)
        result["clean_csv"] = str(clean_path)
        result["catalog_product_csv"] = str(export_path)
        if two_phase:
            pd.DataFrame([{"page": page.page_num, "category": page.category} for page in routed_pages]).to_csv(routed_path, index=False)
            result["routed_csv"] = str(routed_path)
    return result


def batch_process_folder(
    folder_path: str | Path,
    *,
    client: Any,
    output_dir: str | Path,
    page_strategy: str = "uniform",
    sample_pages: int = 8,
    start_page: int = 10,
    dpi: int = DEFAULT_DPI,
    two_phase: bool = False,
    routing_strategy: str = "all",
    routing_sample_pages: int = 32,
    routing_start_page: int = 1,
    explicit_pages: list[int] | None = None,
    routing_explicit_pages: list[int] | None = None,
    routing_dpi: int = 120,
) -> pd.DataFrame:
    folder = Path(folder_path)
    summaries: list[dict[str, Any]] = []
    for pdf_path in sorted(folder.glob("*.pdf")):
        try:
            summaries.append(
                process_single_pdf(
                    pdf_path,
                    client=client,
                    output_dir=output_dir,
                    page_strategy=page_strategy,
                    sample_pages=sample_pages,
                    start_page=start_page,
                    explicit_pages=explicit_pages,
                    dpi=dpi,
                    two_phase=two_phase,
                    routing_strategy=routing_strategy,
                    routing_sample_pages=routing_sample_pages,
                    routing_start_page=routing_start_page,
                    routing_explicit_pages=routing_explicit_pages,
                    routing_dpi=routing_dpi,
                )
            )
        except Exception as exc:
            summaries.append(
                {
                    "pdf_name": pdf_path.name,
                    "mode": "two_phase" if two_phase else "one_phase",
                    "page_strategy": page_strategy,
                    "sample_pages": sample_pages,
                    "start_page": start_page,
                    "raw_rows": 0,
                    "clean_rows": 0,
                    "price_nonzero": 0,
                    "needs_review": 0,
                    "imputed": 0,
                    "elapsed_sec": 0,
                    "error": repr(exc),
                }
            )
    summary_df = pd.DataFrame(summaries).sort_values("pdf_name").reset_index(drop=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(Path(output_dir) / "batch_summary.csv", index=False)
    return summary_df


def build_client_from_args(args: argparse.Namespace) -> OpenAICompatVLMClient:
    env_cfg = config_from_env()
    api_key = env_first([args.api_key_env]) if getattr(args, "api_key_env", None) else ""
    api_key = api_key or (env_cfg.api_key if env_cfg else "")
    if not api_key:
        raise RuntimeError(
            "No API key configured. Set VLM_API_KEY / OPENROUTER_API_KEY or pass --api-key-env with an env var name."
        )
    cfg = VLMConfig(
        api_key=api_key,
        base_url=args.base_url or (env_cfg.base_url if env_cfg else DEFAULT_BASE_URL),
        extraction_model=args.model or (env_cfg.extraction_model if env_cfg else DEFAULT_EXTRACTION_MODEL),
        router_model=args.router_model or (env_cfg.router_model if env_cfg else DEFAULT_ROUTER_MODEL),
        timeout_sec=args.timeout_sec,
    )
    return OpenAICompatVLMClient(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLM-first furniture PDF pipeline.")
    parser.add_argument("--folder", help="Folder with PDF files")
    parser.add_argument("--pdf", help="Single PDF file")
    parser.add_argument("-o", "--output-dir", default="pipeline_runs", help="Directory for CSV outputs")
    parser.add_argument("--page-strategy", default="uniform", choices=["uniform", "from_page", "tail", "all"])
    parser.add_argument("--sample-pages", type=int, default=8)
    parser.add_argument("--start-page", type=int, default=10)
    parser.add_argument("--explicit-pages", default="", help='Example: "10,11,12,20-24"')
    parser.add_argument("--two-phase", action="store_true")
    parser.add_argument("--routing-strategy", default="all", choices=["uniform", "from_page", "tail", "all"])
    parser.add_argument("--routing-sample-pages", type=int, default=32)
    parser.add_argument("--routing-start-page", type=int, default=1)
    parser.add_argument("--routing-explicit-pages", default="", help='Example: "1-20,40-60"')
    parser.add_argument("--routing-dpi", type=int, default=120)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--router-model", default="")
    parser.add_argument("--api-key-env", default="VLM_API_KEY")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--show-status", action="store_true")
    args = parser.parse_args()

    if args.show_status:
        print(pd.DataFrame([backend_status()]).to_string(index=False))
        return

    explicit_pages = parse_explicit_pages(args.explicit_pages)
    routing_explicit_pages = parse_explicit_pages(args.routing_explicit_pages)
    client = build_client_from_args(args)

    if args.pdf:
        summary = process_single_pdf(
            args.pdf,
            client=client,
            output_dir=args.output_dir,
            page_strategy=args.page_strategy,
            sample_pages=args.sample_pages,
            start_page=args.start_page,
            explicit_pages=explicit_pages,
            dpi=args.dpi,
            two_phase=args.two_phase,
            routing_strategy=args.routing_strategy,
            routing_sample_pages=args.routing_sample_pages,
            routing_start_page=args.routing_start_page,
            routing_explicit_pages=routing_explicit_pages,
            routing_dpi=args.routing_dpi,
        )
        print(pd.Series(summary).to_string())
        return

    if not args.folder:
        raise SystemExit("Provide either --pdf or --folder")

    summary_df = batch_process_folder(
        args.folder,
        client=client,
        output_dir=args.output_dir,
        page_strategy=args.page_strategy,
        sample_pages=args.sample_pages,
        start_page=args.start_page,
        explicit_pages=explicit_pages,
        dpi=args.dpi,
        two_phase=args.two_phase,
        routing_strategy=args.routing_strategy,
        routing_sample_pages=args.routing_sample_pages,
        routing_start_page=args.routing_start_page,
        routing_explicit_pages=routing_explicit_pages,
        routing_dpi=args.routing_dpi,
    )
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
