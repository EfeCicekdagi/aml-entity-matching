"""
transliteration.py — Çok dilli transliteration katmanı.

Desteklenen dönüşümler:
  - Türkçe özel karakterler (ş→s, ğ→g, ı→i, ü→u, ö→o, ç→c)
  - Arap alfabesi → Latin
  - Kiril alfabesi → Latin
  - Aksanlı Latin karakterler → temel Latin
  - Unicode NFKC normalizasyonu
  - unidecode kütüphanesi (opsiyonel, kurulu değilse fallback kullanılır)

Kullanım:
    from src.utils.transliteration import Transliterator
    t = Transliterator()
    result = t.transliterate("Şirket Adı")
    # → "Sirket Adi"
"""

import unicodedata
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── unidecode opsiyonel import ────────────────────────────────────────────────
try:
    from unidecode import unidecode as _unidecode
    _UNIDECODE_AVAILABLE = True
    logger.debug("unidecode kütüphanesi yüklendi.")
except ImportError:
    _UNIDECODE_AVAILABLE = False
    logger.info(
        "unidecode kütüphanesi bulunamadı. "
        "Kurulum: pip install unidecode. "
        "Fallback manuel harita kullanılıyor."
    )


# ── Türkçe karakter haritası ──────────────────────────────────────────────────
TURKISH_CHAR_MAP: dict[str, str] = {
    "ş": "s", "Ş": "S",
    "ğ": "g", "Ğ": "G",
    "ı": "i", "İ": "I",
    "ü": "u", "Ü": "U",
    "ö": "o", "Ö": "O",
    "ç": "c", "Ç": "C",
}

# ── Kiril → Latin haritası (temel) ───────────────────────────────────────────
CYRILLIC_TO_LATIN: dict[str, str] = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
    "е": "e", "ё": "yo", "ж": "zh", "з": "z", "и": "i",
    "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
    "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch",
    "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "",
    "э": "e", "ю": "yu", "я": "ya",
    # Büyük harfler
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D",
    "Е": "E", "Ё": "Yo", "Ж": "Zh", "З": "Z", "И": "I",
    "Й": "Y", "К": "K", "Л": "L", "М": "M", "Н": "N",
    "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T",
    "У": "U", "Ф": "F", "Х": "Kh", "Ц": "Ts", "Ч": "Ch",
    "Ш": "Sh", "Щ": "Sch", "Ъ": "", "Ы": "Y", "Ь": "",
    "Э": "E", "Ю": "Yu", "Я": "Ya",
}

# ── Arap → Latin haritası (temel, AML entity matching için yeterli) ───────────
ARABIC_TO_LATIN: dict[str, str] = {
    "ا": "a", "أ": "a", "إ": "i", "آ": "aa",
    "ب": "b", "ت": "t", "ث": "th",
    "ج": "j", "ح": "h", "خ": "kh",
    "د": "d", "ذ": "dh",
    "ر": "r", "ز": "z",
    "س": "s", "ش": "sh",
    "ص": "s", "ض": "d",
    "ط": "t", "ظ": "z",
    "ع": "a", "غ": "gh",
    "ف": "f", "ق": "q",
    "ك": "k", "ل": "l",
    "م": "m", "ن": "n",
    "ه": "h", "و": "w",
    "ي": "y", "ى": "a",
    "ة": "a",  # ta marbuta
    "ء": "",   # hamza
    "ئ": "y", "ؤ": "w",
    # Harekeler (ignore)
    "\u064e": "", "\u064f": "", "\u0650": "", "\u0651": "", "\u0652": "",
}


class Transliterator:
    """
    Çok dilli metin transliteration servisi.

    Script tespiti yaparak uygun dönüşümü uygular.
    unidecode mevcut değilse manuel harita ile fallback yapar.
    """

    def __init__(self, use_unidecode: bool = True):
        """
        Args:
            use_unidecode: True ise unidecode kütüphanesini kullanmaya çalışır.
                           False ise her zaman manuel harita kullanır.
        """
        self._use_unidecode = use_unidecode and _UNIDECODE_AVAILABLE

    def transliterate_turkish(self, text: str) -> str:
        """
        Türkçe özel karakterleri Latin'e çevirir.

        Args:
            text: Türkçe metin

        Returns:
            Türkçe karakterleri Latin karşılığıyla değiştirilmiş metin
        """
        result = []
        for char in text:
            result.append(TURKISH_CHAR_MAP.get(char, char))
        return "".join(result)

    def transliterate_cyrillic(self, text: str) -> str:
        """
        Kiril karakterleri Latin'e çevirir.

        Args:
            text: Kiril metin

        Returns:
            Latin transliterasyonu
        """
        result = []
        for char in text:
            result.append(CYRILLIC_TO_LATIN.get(char, char))
        return "".join(result)

    def transliterate_arabic(self, text: str) -> str:
        """
        Arap karakterleri Latin'e çevirir.

        Args:
            text: Arapça metin

        Returns:
            Latin transliterasyonu
        """
        result = []
        for char in text:
            result.append(ARABIC_TO_LATIN.get(char, char))
        return "".join(result)

    def remove_accents(self, text: str) -> str:
        """
        Aksan ve diakritik işaretlerini kaldırır.
        NFD normalizasyonu sonrası Non-Spacing Mark kategorisini filtreler.

        Args:
            text: Aksanlı metin

        Returns:
            Aksansız metin
        """
        nfd = unicodedata.normalize("NFD", text)
        return "".join(c for c in nfd if unicodedata.category(c) != "Mn")

    def transliterate(self, text: str, aggressive: bool = False) -> str:
        """
        Tam transliteration pipeline'ı.

        Adımlar:
          1. Türkçe karakterler → Latin
          2. unidecode varsa genel dönüşüm, yoksa Kiril + Arap haritası
          3. Aksan kaldırma
          4. Çift boşluk temizle

        Args:
            text: Ham metin
            aggressive: True ise tüm non-ASCII karakterleri agresif kaldır

        Returns:
            Latin alfabesinde translitere edilmiş metin
        """
        if not text:
            return ""

        # 1. Türkçe karakterler
        text = self.transliterate_turkish(text)

        if self._use_unidecode:
            # unidecode tüm Unicode'u Latin'e çevirir (Arap, Kiril, Çince dahil)
            text = _unidecode(text)
        else:
            # Manuel haritalar
            text = self.transliterate_cyrillic(text)
            text = self.transliterate_arabic(text)

        # 3. Aksan kaldırma
        text = self.remove_accents(text)

        # 4. Agresif mod: tüm non-ASCII kaldır
        if aggressive:
            text = re.sub(r"[^\x00-\x7F]", "", text)

        # 5. Çift boşluk temizle
        text = " ".join(text.split())

        return text

    def get_all_forms(self, text: str) -> dict[str, str]:
        """
        Bir metin için tüm normalizasyon formlarını döndürür.
        Veritabanında arama için kullanılabilir.

        Args:
            text: Ham metin

        Returns:
            Dict with keys: original, normalized, transliterated, detected_script
        """
        from src.utils.text_utils import normalize_text, detect_script

        return {
            "original_text":       text,
            "normalized_text":     normalize_text(text),
            "transliterated_text": self.transliterate(text),
            "detected_script":     detect_script(text),
        }


# ── Singleton instance (modül seviyesi, lazy init) ────────────────────────────
_default_transliterator: Optional[Transliterator] = None


def get_transliterator() -> Transliterator:
    """Singleton Transliterator instance döndürür."""
    global _default_transliterator
    if _default_transliterator is None:
        _default_transliterator = Transliterator()
    return _default_transliterator


def transliterate(text: str) -> str:
    """
    Kolaylık fonksiyonu. Transliterator instance oluşturmadan kullanım için.

    Args:
        text: Ham metin

    Returns:
        Translitere edilmiş metin
    """
    return get_transliterator().transliterate(text)
