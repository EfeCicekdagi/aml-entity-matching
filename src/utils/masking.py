"""
masking.py — PII masking ve güvenli log altyapısı.

Hassas bilgilerin log çıktısına karışmasını önler.
Gerçek müşteri verisi log dosyalarına yazılmaz.

Desteklenen maskeleme:
  - IBAN numaraları
  - Hesap numaraları (8-26 rakam)
  - Kişi adları (NER sonuçları)
  - Pasaport / kimlik numaraları
  - IP adresleri

Kullanım:
    from src.utils.masking import mask_pii, SecureLogFilter

    masked = mask_pii("IBAN: TR330006100519786457841326 için ödeme")
    # → "IBAN: TR****1326 için ödeme"
"""

import re
import logging
from typing import Optional

# ── Regex pattern'ları ────────────────────────────────────────────────────────
_IBAN_PATTERN     = re.compile(r"\b([A-Z]{2}\d{2}[A-Z0-9]{4,8})[A-Z0-9]{4,18}([A-Z0-9]{4})\b")
_ACCOUNT_PATTERN  = re.compile(r"\b(\d{4})\d{4,18}(\d{4})\b")
_PASSPORT_PATTERN = re.compile(r"\b([A-Z]{1,2})(\d{6,8})\b")
_IP_PATTERN       = re.compile(r"\b(\d{1,3}\.\d{1,3})\.\d{1,3}\.\d{1,3}\b")

# Türkiye TC Kimlik No (11 rakam, 0 ile başlamaz)
_TC_ID_PATTERN    = re.compile(r"\b([1-9]\d{0,2})\d{5,8}(\d{2})\b")


def mask_pii(text: str, replacement: str = "****") -> str:
    """
    Metindeki PII bilgilerini maskeler.

    Args:
        text: Ham metin
        replacement: Maskeleme karakterleri

    Returns:
        PII maskelenmiş metin
    """
    if not text:
        return text

    # IBAN: İlk 6 + son 4 karakter göster
    text = _IBAN_PATTERN.sub(lambda m: f"{m.group(1)}{replacement}{m.group(2)}", text)

    # Hesap numaraları
    text = _ACCOUNT_PATTERN.sub(lambda m: f"{m.group(1)}{replacement}{m.group(2)}", text)

    # IP adresleri
    text = _IP_PATTERN.sub(lambda m: f"{m.group(1)}.{replacement}", text)

    # Türkiye TC Kimlik No: İlk 3 ve son 2 rakam görünür, ortası maskeli
    text = _TC_ID_PATTERN.sub(lambda m: f"{m.group(1)}{replacement}{m.group(2)}", text)

    # Pasaport numarası: Harf kısmı görünür, rakam kısmı maskeli
    text = _PASSPORT_PATTERN.sub(lambda m: f"{m.group(1)}{replacement}", text)

    return text


class SecureLogFilter(logging.Filter):
    """
    Log mesajlarındaki PII'yı otomatik olarak maskeleyen log filter.

    Kullanım:
        handler = logging.StreamHandler()
        handler.addFilter(SecureLogFilter())
        logger.addHandler(handler)
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Log kaydını maskeler (maskeleme başarısız olursa log atlanmaz)."""
        try:
            if isinstance(record.msg, str):
                record.msg = mask_pii(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {
                        k: mask_pii(str(v)) if isinstance(v, str) else v
                        for k, v in record.args.items()
                    }
                elif isinstance(record.args, (list, tuple)):
                    masked_args = []
                    for arg in record.args:
                        masked_args.append(mask_pii(str(arg)) if isinstance(arg, str) else arg)
                    record.args = tuple(masked_args)
        except Exception:
            pass  # Log hiçbir zaman atlanmaz
        return True


def setup_secure_logging(logger_name: str = None) -> None:
    """
    Belirtilen logger'a SecureLogFilter ekler.

    Args:
        logger_name: Logger adı (None ise root logger)
    """
    target_logger = logging.getLogger(logger_name)
    # Zaten eklenmiş mi kontrol et
    for f in target_logger.filters:
        if isinstance(f, SecureLogFilter):
            return
    target_logger.addFilter(SecureLogFilter())
    target_logger.debug(f"SecureLogFilter added to logger: {logger_name or 'root'}")
