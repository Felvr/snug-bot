from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telegram_bot.common import ROOT_DIR, bool_from_env, int_from_env, load_env_file, sanitize_stem


SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xlsx"}


class ProcessingError(RuntimeError):
    pass


ProgressCallback = Callable[[str, str], None]


def _notify(progress_callback: ProgressCallback | None, stage: str, message: str) -> None:
    if progress_callback is not None:
        progress_callback(stage, message)


def _load_pipeline_module():
    try:
        import arxiv.unified_furniture_pipeline as pipeline
    except ModuleNotFoundError as exc:
        if exc.name == "pdf2image":
            raise ProcessingError(
                "Не найден пакет pdf2image. Установите зависимости пайплайна и poppler перед запуском бота."
            ) from exc
        raise
    return pipeline


def _job_dir_for(input_path: Path, output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = sanitize_stem(input_path.stem)
    job_dir = output_root / f"{timestamp}_{base_name}"
    counter = 1
    while job_dir.exists():
        counter += 1
        job_dir = output_root / f"{timestamp}_{base_name}_{counter}"
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_dir


def _copy_to_job_input(source_path: Path, input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    target_path = input_dir / source_path.name
    target_path.write_bytes(source_path.read_bytes())
    return target_path


def _pipeline_kwargs(pipeline_module) -> dict[str, Any]:
    return {
        "page_strategy": os.getenv("SNUG_PIPELINE_PAGE_STRATEGY", "all"),
        "sample_pages": int_from_env("SNUG_PIPELINE_SAMPLE_PAGES", 8),
        "start_page": int_from_env("SNUG_PIPELINE_START_PAGE", 10),
        "explicit_pages": pipeline_module.parse_explicit_pages(os.getenv("SNUG_PIPELINE_EXPLICIT_PAGES", "")),
        "dpi": int_from_env("SNUG_PIPELINE_DPI", pipeline_module.DEFAULT_DPI),
        "two_phase": bool_from_env("SNUG_PIPELINE_TWO_PHASE", True),
        "routing_strategy": os.getenv("SNUG_PIPELINE_ROUTING_STRATEGY", "all"),
        "routing_sample_pages": int_from_env("SNUG_PIPELINE_ROUTING_SAMPLE_PAGES", 32),
        "routing_start_page": int_from_env("SNUG_PIPELINE_ROUTING_START_PAGE", 1),
        "routing_explicit_pages": pipeline_module.parse_explicit_pages(os.getenv("SNUG_PIPELINE_ROUTING_EXPLICIT_PAGES", "")),
        "routing_dpi": int_from_env("SNUG_PIPELINE_ROUTING_DPI", 120),
    }


def _process_pdf(
    input_path: Path,
    output_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    _notify(progress_callback, "pdf_prepare", "Готовлю PDF-пайплайн и проверяю зависимости.")
    pipeline = _load_pipeline_module()
    cfg = pipeline.config_from_env()
    if not cfg or not cfg.api_key:
        raise ProcessingError(
            "Не найден VLM API key. Укажите VLM_API_KEY, OPENROUTER_API_KEY или OPENAI_API_KEY."
        )
    _notify(
        progress_callback,
        "pdf_vlm_start",
        "Запускаю VLM-разбор PDF. Это самый долгий этап, особенно на больших прайсах.",
    )
    client = pipeline.OpenAICompatVLMClient(cfg)
    result = pipeline.process_single_pdf(
        input_path,
        client=client,
        output_dir=output_dir,
        **_pipeline_kwargs(pipeline),
    )
    pd.DataFrame([result]).to_csv(output_dir / "batch_summary.csv", index=False)
    _notify(
        progress_callback,
        "pdf_vlm_done",
        f"PDF обработан. Найдено строк после очистки: {result.get('clean_rows', 0)}.",
    )
    return result


def _load_tabular_file(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    return pd.read_excel(input_path)


def _process_table(
    input_path: Path,
    output_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    _notify(progress_callback, "table_prepare", "Открываю таблицу и готовлю постобработку.")
    pipeline = _load_pipeline_module()
    from arxiv.furniture_postprocess import clean_furniture_catalog

    _notify(progress_callback, "table_read", "Читаю входную таблицу.")
    raw_df = _load_tabular_file(input_path)
    _notify(progress_callback, "table_clean", "Очищаю и нормализую данные.")
    clean_df = clean_furniture_catalog(raw_df)
    _notify(progress_callback, "table_export", "Собираю итоговый catalog_product.csv.")
    export_df = pipeline.export_catalog_product(clean_df)

    raw_path = output_dir / f"{input_path.stem}__raw.csv"
    clean_path = output_dir / f"{input_path.stem}__clean.csv"
    export_path = output_dir / f"{input_path.stem}__catalog_product.csv"

    raw_df.to_csv(raw_path, index=False)
    clean_df.to_csv(clean_path, index=False)
    export_df.to_csv(export_path, index=False)

    result = {
        "pdf_name": input_path.name,
        "mode": "table_cleanup",
        "raw_rows": len(raw_df),
        "clean_rows": len(clean_df),
        "price_nonzero": int(pd.to_numeric(clean_df.get("price"), errors="coerce").fillna(0).gt(0).sum()),
        "needs_review": int(clean_df.get("quality_flag", pd.Series(dtype="object")).eq("needs_review").sum()),
        "imputed": int(clean_df.get("quality_flag", pd.Series(dtype="object")).eq("imputed").sum()),
        "elapsed_sec": 0,
        "raw_csv": str(raw_path),
        "clean_csv": str(clean_path),
        "catalog_product_csv": str(export_path),
    }
    pd.DataFrame([result]).to_csv(output_dir / "batch_summary.csv", index=False)
    _notify(
        progress_callback,
        "table_done",
        f"Таблица обработана. Строк после очистки: {result.get('clean_rows', 0)}.",
    )
    return result


def _zip_output_dir(output_dir: Path, archive_path: Path) -> Path:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(output_dir.glob("*")):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.name)
    return archive_path


def process_input_file(
    input_path: str | Path,
    output_root: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    load_env_file()
    source_path = Path(input_path).expanduser().resolve()
    if not source_path.exists():
        raise ProcessingError(f"Файл не найден: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ProcessingError(f"Неподдерживаемый формат {suffix or '<без расширения>'}. Поддерживаются: {supported}")

    root = Path(output_root or os.getenv("SNUG_BOT_DATA_DIR", ROOT_DIR / "telegram_bot_data")).expanduser()
    jobs_root = root / "jobs"
    _notify(progress_callback, "job_prepare", "Создаю рабочую папку для обработки.")
    job_dir = _job_dir_for(source_path, jobs_root)
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    _notify(progress_callback, "job_copy_input", "Сохраняю входной файл в рабочую папку.")
    job_input_path = _copy_to_job_input(source_path, input_dir)
    _notify(progress_callback, "job_detect_type", f"Определил формат файла: {suffix}.")
    if suffix == ".pdf":
        result = _process_pdf(job_input_path, output_dir, progress_callback=progress_callback)
    else:
        result = _process_table(job_input_path, output_dir, progress_callback=progress_callback)

    _notify(progress_callback, "job_archive", "Упаковываю вспомогательные файлы в архив.")
    archive_path = _zip_output_dir(output_dir, job_dir / "artifacts.zip")
    catalog_path = Path(result["catalog_product_csv"]).resolve()
    clean_path = Path(result["clean_csv"]).resolve()
    _notify(progress_callback, "job_done", f"Обработка завершена. Готов файл {catalog_path.name}.")

    return {
        "job_dir": str(job_dir.resolve()),
        "input_file": str(job_input_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "catalog_product_csv": str(catalog_path),
        "clean_csv": str(clean_path),
        "artifacts_zip": str(archive_path.resolve()),
        "summary": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Process one file through the SNUG pipeline.")
    parser.add_argument("input_file", help="Input PDF/CSV/XLSX file")
    parser.add_argument(
        "-o",
        "--output-root",
        default="",
        help="Root folder for jobs. Defaults to SNUG_BOT_DATA_DIR or ./telegram_bot_data",
    )
    args = parser.parse_args()

    try:
        result = process_input_file(args.input_file, output_root=args.output_root or None)
    except ProcessingError as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
