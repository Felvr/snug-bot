from __future__ import annotations

import json
import mimetypes
import os
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
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


ACTIVE_JOBS: dict[int, dict[str, Any]] = {}
ACTIVE_JOBS_LOCK = threading.Lock()
LAST_JOBS: dict[int, dict[str, Any]] = {}
LAST_JOBS_LOCK = threading.Lock()
INSTANCE_ID = f"{socket.gethostname()}:{os.getpid()}"


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
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{self.file_base}/{file_path}", timeout=120) as response:
            target_path.write_bytes(response.read())
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


def _safe_notify_async(api: TelegramBotAPI, chat_id: int, text: str) -> None:
    threading.Thread(target=_safe_notify, args=(api, chat_id, text), daemon=True).start()


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


def _start_progress_heartbeat(
    api: TelegramBotAPI,
    chat_id: int,
    state: dict[str, Any],
    interval_sec: int,
) -> tuple[threading.Event, threading.Thread | None]:
    if interval_sec <= 0:
        return threading.Event(), None

    stop_event = threading.Event()

    def worker() -> None:
        while not stop_event.wait(interval_sec):
            elapsed_sec = int(time.monotonic() - state["started_at"])
            stage_text = state.get("last_stage_text", "обработка продолжается")
            elapsed_min = elapsed_sec // 60
            elapsed_tail = elapsed_sec % 60
            _safe_notify(
                api,
                chat_id,
                (
                    "Все еще работаю над файлом.\n"
                    f"Текущий этап: {stage_text}\n"
                    f"Прошло: {elapsed_min} мин {elapsed_tail} сек."
                ),
            )

    thread = threading.Thread(target=worker, name="snug-bot-heartbeat", daemon=True)
    thread.start()
    return stop_event, thread


def _support_message() -> str:
    formats = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    return (
        "Пришлите PDF, CSV или XLSX с прайсом.\n"
        "Бот запустит текущий SNUG pipeline и вернёт готовый catalog_product.csv для выгрузки на сайт."
        "\nКоманды: /status, /stop"
        f"\nПоддерживаемые форматы: {formats}"
    )


def _status_message() -> str:
    api_key_present = bool(
        os.getenv("VLM_API_KEY", "").strip()
        or os.getenv("OPENROUTER_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    data_dir = resolve_data_dir()
    with ACTIVE_JOBS_LOCK:
        active_job_count = sum(1 for state in ACTIVE_JOBS.values() if state.get("thread") and state["thread"].is_alive())
    return (
        f"Instance: {INSTANCE_ID}\n"
        f"TELEGRAM_BOT_TOKEN: {'ok' if bool(os.getenv('TELEGRAM_BOT_TOKEN', '').strip()) else 'missing'}\n"
        f"VLM API key: {'ok' if api_key_present else 'missing'}\n"
        f"SNUG_BOT_DATA_DIR: {data_dir}\n"
        f"SNUG_PIPELINE_TWO_PHASE: {os.getenv('SNUG_PIPELINE_TWO_PHASE', '1')}\n"
        f"SNUG_PIPELINE_PAGE_STRATEGY: {os.getenv('SNUG_PIPELINE_PAGE_STRATEGY', 'all')}\n"
        f"SNUG_PIPELINE_ROUTING_STRATEGY: {os.getenv('SNUG_PIPELINE_ROUTING_STRATEGY', 'all')}\n"
        f"Active jobs: {active_job_count}"
    )


def _set_last_job(chat_id: int, payload: dict[str, Any]) -> None:
    with LAST_JOBS_LOCK:
        LAST_JOBS[chat_id] = payload


def _get_last_job(chat_id: int) -> dict[str, Any] | None:
    with LAST_JOBS_LOCK:
        return LAST_JOBS.get(chat_id)


def _summary_caption(result: dict[str, Any]) -> str:
    summary = result["summary"]
    catalog_csv = str(result.get("catalog_product_csv", "") or "").strip()
    catalog_name = Path(catalog_csv).name if catalog_csv else "catalog_product.csv"
    clean_rows = summary.get("clean_rows", "")
    price_nonzero = summary.get("price_nonzero", "")
    mode = summary.get("mode", "")
    cancelled = int(summary.get("cancelled", result.get("cancelled", 0)))
    status = "Остановлено" if cancelled else "Готово"
    return (
        f"{status}: {catalog_name}\n"
        f"mode={mode}, rows={clean_rows}, price_nonzero={price_nonzero}"
    )


def _get_active_job(chat_id: int) -> dict[str, Any] | None:
    with ACTIVE_JOBS_LOCK:
        state = ACTIVE_JOBS.get(chat_id)
        if not state:
            return None
        thread = state.get("thread")
        if thread is not None and thread.is_alive():
            return state
        ACTIVE_JOBS.pop(chat_id, None)
    return None


def _set_active_job(chat_id: int, state: dict[str, Any]) -> None:
    with ACTIVE_JOBS_LOCK:
        ACTIVE_JOBS[chat_id] = state


def _clear_active_job(chat_id: int) -> None:
    with ACTIVE_JOBS_LOCK:
        ACTIVE_JOBS.pop(chat_id, None)


def _handle_command(api: TelegramBotAPI, chat_id: int, text: str) -> None:
    command = text.strip().split()[0].lower()
    if command in {"/start", "/help"}:
        api.send_message(chat_id, _support_message())
        return
    if command == "/status":
        active = _get_active_job(chat_id)
        if active:
            stage = active.get("progress_state", {}).get("last_stage_text", "обработка")
            job_id = active.get("job_id", "-")
            api.send_message(
                chat_id,
                f"{_status_message()}\nТекущая задача: выполняется (job={job_id}, stage={stage}).",
            )
        else:
            last = _get_last_job(chat_id)
            if not last:
                api.send_message(chat_id, _status_message())
            else:
                ended = time.strftime("%H:%M:%S", time.localtime(last.get("ended_at", time.time())))
                status = last.get("status", "unknown")
                job_id = last.get("job_id", "-")
                tail = f"Последняя задача: {status} (job={job_id}, ended={ended}, instance={last.get('instance', '-')})."
                error = str(last.get("error", "")).strip()
                if error:
                    tail = f"{tail}\nПричина: {error}"
                api.send_message(chat_id, f"{_status_message()}\n{tail}")
        return
    if command in {"/stop", "/cancel"}:
        active = _get_active_job(chat_id)
        if not active:
            api.send_message(chat_id, "Сейчас нет активной обработки для остановки.")
            return
        cancel_event = active.get("cancel_event")
        if cancel_event is None:
            api.send_message(chat_id, "Не удалось остановить задачу: внутреннее состояние недоступно.")
            return
        cancel_event.set()
        api.send_message(chat_id, "Принял /stop. Останавливаю обработку и подготовлю доступные результаты.")
        return
    api.send_message(chat_id, _support_message())


def _process_document_job(
    api: TelegramBotAPI,
    chat_id: int,
    document: dict[str, Any],
    cancel_event: threading.Event,
    progress_state: dict[str, Any],
) -> dict[str, Any]:
    file_name = document.get("file_name") or "upload.bin"
    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        api.send_message(
            chat_id,
            f"Формат {suffix or '<без расширения>'} пока не поддерживается. {_support_message()}",
        )
        return

    base_dir = resolve_data_dir()
    incoming_dir = base_dir / "incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)

    _safe_notify(api, chat_id, f"Файл {file_name} получен. Начинаю обработку.")
    _safe_notify(api, chat_id, "Скачиваю файл из Telegram в локальную рабочую папку.")

    telegram_file_path = api.get_file_path(document["file_id"])
    safe_name = sanitize_stem(Path(file_name).stem)
    local_input_path = incoming_dir / f"{safe_name}_{uuid.uuid4().hex[:8]}{suffix}"
    api.download_file(telegram_file_path, local_input_path)
    _safe_notify(api, chat_id, f"Файл скачан: {local_input_path.name}. Передаю его в pipeline.")

    heartbeat_interval_sec = int_from_env("SNUG_BOT_PROGRESS_INTERVAL_SEC", 45)
    progress_state["started_at"] = time.monotonic()
    progress_state["last_stage_text"] = "подготовка обработки"

    def progress_callback(stage: str, message: str) -> None:
        formatted = _progress_message(stage, message)
        progress_state["last_stage_text"] = formatted.replace("\n", " ")
        # Keep callback non-blocking: network hiccups to Telegram should not pause parsing.
        if stage in {"job_done", "pdf_vlm_done", "table_done"}:
            _safe_notify_async(api, chat_id, formatted)

    stop_event, heartbeat_thread = _start_progress_heartbeat(
        api,
        chat_id,
        progress_state,
        heartbeat_interval_sec,
    )

    try:
        _safe_notify(api, chat_id, "Запускаю обработку в pipeline. Это может занять несколько минут.")
        result = process_input_file(
            local_input_path,
            output_root=base_dir,
            progress_callback=progress_callback,
            stop_requested=cancel_event.is_set,
        )
    finally:
        stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1)

    cancelled = int(result.get("cancelled", result.get("summary", {}).get("cancelled", 0)))
    if cancelled:
        _safe_notify(api, chat_id, "Обработка остановлена по запросу. Отправляю всё, что успело сохраниться.")
    else:
        _safe_notify(api, chat_id, "Отправляю готовый catalog_product.csv.")

    catalog_csv = str(result.get("catalog_product_csv", "") or "").strip()
    if catalog_csv and Path(catalog_csv).exists():
        _safe_send_document(api, chat_id, Path(catalog_csv), caption=_summary_caption(result))
    else:
        _safe_notify(api, chat_id, "Итоговый catalog_product.csv на текущем этапе отсутствует.")

    archive_path = Path(result["artifacts_zip"])
    if archive_path.exists():
        _safe_notify(api, chat_id, "Отправляю архив с промежуточными файлами и summary.")
        _safe_send_document(api, chat_id, archive_path, caption="Дополнительно отправляю архив со всеми артефактами обработки.")

    return result


def _handle_document(api: TelegramBotAPI, chat_id: int, document: dict[str, Any]) -> None:
    if _get_active_job(chat_id):
        _safe_notify(api, chat_id, "У вас уже идет обработка. Дождитесь завершения или отправьте /stop.")
        return

    cancel_event = threading.Event()
    job_id = uuid.uuid4().hex[:8]
    state: dict[str, Any] = {
        "job_id": job_id,
        "cancel_event": cancel_event,
        "progress_state": {"last_stage_text": "ожидание запуска"},
    }

    def worker() -> None:
        try:
            result = _process_document_job(api, chat_id, document, cancel_event, state["progress_state"])
            cancelled = int(result.get("cancelled", result.get("summary", {}).get("cancelled", 0)))
            _set_last_job(
                chat_id,
                {
                    "job_id": job_id,
                    "status": "cancelled" if cancelled else "done",
                    "error": "",
                    "ended_at": time.time(),
                    "instance": INSTANCE_ID,
                },
            )
        except ProcessingError as exc:
            _set_last_job(
                chat_id,
                {
                    "job_id": job_id,
                    "status": "failed",
                    "error": f"ProcessingError: {exc}",
                    "ended_at": time.time(),
                    "instance": INSTANCE_ID,
                },
            )
            _safe_notify(api, chat_id, f"Не удалось обработать файл: {exc}")
        except Exception as exc:
            traceback.print_exc()
            _set_last_job(
                chat_id,
                {
                    "job_id": job_id,
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "ended_at": time.time(),
                    "instance": INSTANCE_ID,
                },
            )
            _safe_notify(api, chat_id, f"Во время обработки произошла ошибка: {exc}")
        finally:
            _clear_active_job(chat_id)

    thread = threading.Thread(target=worker, name=f"snug-bot-job-{chat_id}", daemon=True)
    state["thread"] = thread
    _set_active_job(chat_id, state)
    thread.start()


def handle_update(api: TelegramBotAPI, update: dict[str, Any]) -> None:
    message = update.get("message") or {}
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    if not chat_id:
        return

    text = message.get("text", "")
    if text.startswith("/"):
        _handle_command(api, chat_id, text)
        return

    document = message.get("document")
    if document:
        _handle_document(api, chat_id, document)
        return

    api.send_message(chat_id, _support_message())


def main() -> None:
    load_env_file()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Укажите TELEGRAM_BOT_TOKEN в окружении или в .env")

    poll_timeout = int_from_env("SNUG_BOT_POLL_TIMEOUT", 30)
    retry_delay_sec = int_from_env("SNUG_BOT_RETRY_DELAY_SEC", 5)
    api = TelegramBotAPI(token)
    offset: int | None = None

    print("SNUG Telegram bot started. Waiting for files...")
    while True:
        try:
            updates = api.get_updates(offset=offset, timeout=poll_timeout)
            for update in updates:
                offset = int(update["update_id"]) + 1
                handle_update(api, update)
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
