from __future__ import annotations

import json
import mimetypes
import os
import queue
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telegram_bot.common import ROOT_DIR, int_from_env, load_env_file, resolve_data_dir, sanitize_stem
from telegram_bot.process_file import ProcessingError, SUPPORTED_EXTENSIONS, process_input_file


NETWORK_ERROR_TYPES = (
    urllib.error.URLError,
    TimeoutError,
    ConnectionError,
    OSError,
    EOFError,
    socket.timeout,
)


class TelegramBotAPI:
    def __init__(self, token: str):
        self.token = token
        self.api_base = f"https://api.telegram.org/bot{token}"
        self.file_base = f"https://api.telegram.org/file/bot{token}"

    def _request_json(self, method: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.api_base}/{method}",
            data=data,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        if not parsed.get("ok"):
            raise RuntimeError(f"Telegram API error in {method}: {parsed}")
        return parsed["result"]

    def _request_multipart(
        self,
        method: str,
        *,
        fields: dict[str, Any],
        file_field: str,
        file_path: Path,
        timeout: int = 120,
    ) -> dict[str, Any]:
        boundary = f"----snug{uuid.uuid4().hex}"
        chunks: list[bytes] = []

        for key, value in fields.items():
            chunks.append(f"--{boundary}\r\n".encode("utf-8"))
            chunks.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
            chunks.append(str(value).encode("utf-8"))
            chunks.append(b"\r\n")

        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'.encode("utf-8")
        )
        chunks.append(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
        chunks.append(file_path.read_bytes())
        chunks.append(b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode("utf-8"))

        request = urllib.request.Request(
            f"{self.api_base}/{method}",
            data=b"".join(chunks),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        if not parsed.get("ok"):
            raise RuntimeError(f"Telegram API error in {method}: {parsed}")
        return parsed["result"]

    def get_updates(self, offset: int | None = None, timeout: int = 30) -> list[dict[str, Any]]:
        payload = {
            "timeout": timeout,
            "allowed_updates": ["message"],
        }
        if offset is not None:
            payload["offset"] = offset
        return self._request_json("getUpdates", payload=payload, timeout=timeout + 10)

    def send_message(self, chat_id: int, text: str) -> dict[str, Any]:
        return self._request_json("sendMessage", {"chat_id": chat_id, "text": text})

    def send_document(self, chat_id: int, file_path: Path, caption: str = "") -> dict[str, Any]:
        fields: dict[str, Any] = {"chat_id": chat_id}
        if caption:
            fields["caption"] = caption
        return self._request_multipart("sendDocument", fields=fields, file_field="document", file_path=file_path)

    def get_file_path(self, file_id: str) -> str:
        result = self._request_json("getFile", {"file_id": file_id})
        file_path = result.get("file_path", "")
        if not file_path:
            raise RuntimeError(f"Telegram did not return file_path for file_id={file_id}")
        return file_path

    def download_file(self, file_path: str, target_path: Path) -> Path:
        import shutil

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{self.file_base}/{file_path}", timeout=120) as response:
            with target_path.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
        return target_path


def _format_network_error(exc: BaseException) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    if "nodename nor servname provided" in lowered or "name or service not known" in lowered:
        return f"DNS error while reaching Telegram API: {message}"
    if "connection abort" in lowered or "software caused connection abort" in lowered:
        return f"Connection to Telegram API was aborted: {message}"
    if "timed out" in lowered:
        return f"Telegram API timeout: {message}"
    return f"Telegram network error: {message}"


def _safe_notify(api: TelegramBotAPI, chat_id: int, text: str) -> None:
    try:
        api.send_message(chat_id, text)
    except NETWORK_ERROR_TYPES as exc:
        print(_format_network_error(exc))
    except Exception:
        traceback.print_exc()


def _safe_send_document(api: TelegramBotAPI, chat_id: int, file_path: Path, caption: str = "") -> None:
    try:
        api.send_document(chat_id, file_path, caption=caption)
    except NETWORK_ERROR_TYPES as exc:
        print(_format_network_error(exc))
    except Exception:
        traceback.print_exc()


def _progress_message(stage: str, message: str) -> str:
    stage_titles = {
        "job_prepare": "Создаю задачу.",
        "job_copy_input": "Сохраняю файл.",
        "job_detect_type": "Определяю тип входного файла.",
        "pdf_prepare": "Проверяю PDF-зависимости и VLM-конфиг.",
        "pdf_vlm_start": "Запускаю основной разбор PDF.",
        "pdf_vlm_done": "PDF-разбор завершён.",
        "table_prepare": "Готовлю обработку таблицы.",
        "table_read": "Читаю таблицу.",
        "table_clean": "Очищаю данные.",
        "table_export": "Собираю итоговый экспорт.",
        "table_done": "Табличная обработка завершена.",
        "job_archive": "Собираю архив артефактов.",
        "job_done": "Результат готов.",
    }
    title = stage_titles.get(stage)
    if title:
        return f"{title}\n{message}"
    return message


def _support_message() -> str:
    formats = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    return (
        "Пришлите PDF, CSV или XLSX с прайсом.\n"
        "Бот запустит текущий SNUG pipeline и вернёт готовый catalog_product.csv для выгрузки на сайт."
        f"\nПоддерживаемые форматы: {formats}\n"
        "Команды: /status, /stop"
    )


class JobCancelledError(RuntimeError):
    pass


@dataclass
class JobState:
    id: str
    chat_id: int
    file_name: str
    suffix: str
    file_id: str
    status: str = "queued"
    stage: str = "queued"
    stage_text: str = "Ожидает запуска."
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str = ""
    cancel_requested: bool = False


class JobManager:
    def __init__(self, api: TelegramBotAPI, base_dir: Path, max_workers: int = 1, history_size: int = 8):
        self.api = api
        self.base_dir = base_dir
        self.max_workers = max(1, max_workers)
        self.history_size = max(1, history_size)
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._jobs: dict[str, JobState] = {}
        self._history_by_chat: dict[int, deque[str]] = {}
        self._workers: list[threading.Thread] = []

        for idx in range(self.max_workers):
            worker = threading.Thread(target=self._worker, name=f"snug-job-worker-{idx + 1}", daemon=True)
            worker.start()
            self._workers.append(worker)

    def submit_document(self, chat_id: int, document: dict[str, Any]) -> JobState:
        file_name = document.get("file_name") or "upload.bin"
        suffix = Path(file_name).suffix.lower()
        job = JobState(
            id=uuid.uuid4().hex[:8],
            chat_id=chat_id,
            file_name=file_name,
            suffix=suffix,
            file_id=str(document.get("file_id", "")),
            created_at=time.monotonic(),
        )
        with self._lock:
            self._jobs[job.id] = job
            history = self._history_by_chat.setdefault(chat_id, deque(maxlen=self.history_size))
            history.appendleft(job.id)
        self._queue.put(job.id)
        return job

    def get_status_text(self, chat_id: int) -> str:
        with self._lock:
            active = [j for j in self._jobs.values() if j.chat_id == chat_id and j.status == "running"]
            queued = [j for j in self._jobs.values() if j.chat_id == chat_id and j.status == "queued"]
            history_ids = list(self._history_by_chat.get(chat_id, []))
            recent_jobs = [self._jobs[job_id] for job_id in history_ids if job_id in self._jobs]

        lines: list[str] = []
        lines.append(f"Очередь задач: {len(queued)}")
        lines.append(f"В работе сейчас: {len(active)}")

        if active:
            current = active[0]
            elapsed = int(time.monotonic() - (current.started_at or current.created_at))
            lines.append(
                f"Текущая задача #{current.id}: {current.file_name} | этап: {current.stage_text} | прошло: {elapsed} сек"
            )
        elif queued:
            next_job = queued[0]
            lines.append(f"Следующая задача #{next_job.id}: {next_job.file_name}")
        else:
            lines.append("Активных задач нет.")

        finished = [j for j in recent_jobs if j.status in {"done", "failed"}]
        if finished:
            last = finished[0]
            if last.status == "done":
                summary = (last.result or {}).get("summary", {})
                lines.append(
                    "Последняя завершенная: "
                    f"#{last.id} {last.file_name} | rows={summary.get('clean_rows', '?')} | "
                    f"price_nonzero={summary.get('price_nonzero', '?')}"
                )
            else:
                lines.append(f"Последняя задача завершилась с ошибкой: #{last.id} {last.file_name}")

        return "\n".join(lines)

    def request_stop(self, chat_id: int) -> str:
        with self._lock:
            running_for_chat = [j for j in self._jobs.values() if j.chat_id == chat_id and j.status == "running"]

            for job in running_for_chat:
                job.cancel_requested = True

        if running_for_chat:
            return "Останавливаю текущую задачу. Следующая задача начнется автоматически."
        return "Сейчас нет активных задач для остановки."

    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run_job(job_id)
            finally:
                self._queue.task_done()

    def _set_running(self, job: JobState) -> None:
        with self._lock:
            job.status = "running"
            job.stage = "job_prepare"
            job.stage_text = "Задача запущена."
            job.started_at = time.monotonic()

    def _set_progress(self, job: JobState, stage: str, message: str) -> None:
        with self._lock:
            job.stage = stage
            job.stage_text = _progress_message(stage, message).replace("\n", " ")

    def _set_done(self, job: JobState, result: dict[str, Any]) -> None:
        with self._lock:
            job.status = "done"
            job.stage = "job_done"
            job.stage_text = "Обработка завершена."
            job.result = result
            job.finished_at = time.monotonic()

    def _set_failed(self, job: JobState, error: str) -> None:
        with self._lock:
            job.status = "failed"
            job.stage = "job_failed"
            job.stage_text = "Обработка завершилась с ошибкой."
            job.error = error
            job.finished_at = time.monotonic()

    def _cancel_if_requested(self, job: JobState) -> None:
        with self._lock:
            should_cancel = bool(job.cancel_requested)
        if should_cancel:
            raise JobCancelledError("cancelled")

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return

        self._set_running(job)
        _safe_notify(self.api, job.chat_id, f"Задача #{job.id} принята в работу: {job.file_name}")

        if job.suffix not in SUPPORTED_EXTENSIONS:
            self._set_failed(job, f"Неподдерживаемый формат: {job.suffix or '<без расширения>'}")
            _safe_notify(
                self.api,
                job.chat_id,
                f"Задача #{job.id}: формат {job.suffix or '<без расширения>'} не поддерживается. {_support_message()}",
            )
            return

        incoming_dir = self.base_dir / "incoming"
        incoming_dir.mkdir(parents=True, exist_ok=True)

        safe_name = sanitize_stem(Path(job.file_name).stem)
        local_input_path = incoming_dir / f"{safe_name}_{uuid.uuid4().hex[:8]}{job.suffix}"

        try:
            self._cancel_if_requested(job)
            telegram_file_path = self.api.get_file_path(job.file_id)
            self.api.download_file(telegram_file_path, local_input_path)
            self._cancel_if_requested(job)

            def progress_callback(stage: str, message: str) -> None:
                self._cancel_if_requested(job)
                self._set_progress(job, stage, message)

            result = process_input_file(
                local_input_path,
                output_root=self.base_dir,
                progress_callback=progress_callback,
                cancel_callback=lambda: self._cancel_if_requested(job),
            )
            self._cancel_if_requested(job)
            self._set_done(job, result)

            _safe_send_document(
                self.api,
                job.chat_id,
                Path(result["catalog_product_csv"]),
                caption=_summary_caption(result),
            )

            archive_path = Path(result["artifacts_zip"])
            if archive_path.exists():
                _safe_send_document(
                    self.api,
                    job.chat_id,
                    archive_path,
                    caption="Архив с промежуточными файлами и summary.",
                )
        except JobCancelledError:
            self._set_failed(job, "cancelled")
            _safe_notify(self.api, job.chat_id, f"Задача #{job.id} остановлена. Можно отправлять следующий файл.")
        except ProcessingError as exc:
            self._set_failed(job, str(exc))
            _safe_notify(self.api, job.chat_id, f"Задача #{job.id}: не удалось обработать файл.")
        except Exception as exc:
            traceback.print_exc()
            self._set_failed(job, str(exc))
            _safe_notify(self.api, job.chat_id, f"Задача #{job.id}: во время обработки произошла ошибка.")
        finally:
            try:
                local_input_path.unlink(missing_ok=True)
            except OSError:
                pass


def _summary_caption(result: dict[str, Any]) -> str:
    summary = result["summary"]
    catalog_path = Path(result["catalog_product_csv"])
    clean_rows = summary.get("clean_rows", "")
    price_nonzero = summary.get("price_nonzero", "")
    return (
        f"Готово: {catalog_path.name}\n"
        f"Строк после очистки: {clean_rows}\n"
        f"С ценой: {price_nonzero}"
    )


def _handle_command(api: TelegramBotAPI, jobs: JobManager, chat_id: int, text: str) -> None:
    command = text.strip().split()[0].lower()
    if command in {"/start", "/help"}:
        api.send_message(chat_id, _support_message())
        return
    if command == "/status":
        api.send_message(chat_id, jobs.get_status_text(chat_id))
        return
    if command == "/stop":
        api.send_message(chat_id, jobs.request_stop(chat_id))
        return
    api.send_message(chat_id, _support_message())


def _handle_document(api: TelegramBotAPI, jobs: JobManager, chat_id: int, document: dict[str, Any]) -> None:
    file_name = document.get("file_name") or "upload.bin"
    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        api.send_message(
            chat_id,
            f"Формат {suffix or '<без расширения>'} пока не поддерживается. {_support_message()}",
        )
        return

    job = jobs.submit_document(chat_id, document)
    _safe_notify(
        api,
        chat_id,
        f"Файл {file_name} принят. Создал задачу #{job.id}. Проверяйте прогресс командой /status.",
    )


def handle_update(api: TelegramBotAPI, jobs: JobManager, update: dict[str, Any]) -> None:
    message = update.get("message") or {}
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    if not chat_id:
        return

    text = message.get("text", "")
    if text.startswith("/"):
        _handle_command(api, jobs, chat_id, text)
        return

    document = message.get("document")
    if document:
        try:
            _handle_document(api, jobs, chat_id, document)
        except ProcessingError as exc:
            _safe_notify(api, chat_id, "Не удалось обработать файл.")
        except Exception as exc:
            traceback.print_exc()
            _safe_notify(api, chat_id, "Во время обработки произошла ошибка.")
        return

    api.send_message(chat_id, _support_message())


def main() -> None:
    load_env_file()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Укажите TELEGRAM_BOT_TOKEN в окружении или в .env")

    poll_timeout = int_from_env("SNUG_BOT_POLL_TIMEOUT", 30)
    retry_delay_sec = int_from_env("SNUG_BOT_RETRY_DELAY_SEC", 5)
    max_workers = int_from_env("SNUG_BOT_MAX_WORKERS", 1)
    api = TelegramBotAPI(token)
    base_dir = resolve_data_dir()
    jobs = JobManager(api, base_dir=base_dir, max_workers=max_workers)
    offset: int | None = None

    print("SNUG Telegram bot started. Waiting for files...")
    while True:
        try:
            updates = api.get_updates(offset=offset, timeout=poll_timeout)
            for update in updates:
                offset = int(update["update_id"]) + 1
                handle_update(api, jobs, update)
        except NETWORK_ERROR_TYPES as exc:
            print(_format_network_error(exc))
            time.sleep(retry_delay_sec)
        except KeyboardInterrupt:
            print("Bot stopped.")
            return
        except Exception:
            traceback.print_exc()
            time.sleep(retry_delay_sec)


if __name__ == "__main__":
    main()
