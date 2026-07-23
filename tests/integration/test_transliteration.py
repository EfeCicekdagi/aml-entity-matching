"""
test_transliteration.py — Transliteration testleri.
"""

import unittest
from src.utils.transliteration import Transliterator
from src.utils.text_utils import detect_script


class TestTransliterator(unittest.TestCase):

    def setUp(self):
        self.t = Transliterator(use_unidecode=False)  # Manuel harita ile test et

    # ── Türkçe ──────────────────────────────────────────────────────────────

    def test_turkish_special_chars(self):
        result = self.t.transliterate_turkish("Şirket Ağı")
        self.assertNotIn("ş", result.lower())
        self.assertNotIn("ğ", result.lower())

    def test_turkish_i_without_dot(self):
        result = self.t.transliterate_turkish("Işık")
        self.assertNotIn("ı", result)

    def test_turkish_casefold_equivalent(self):
        result = self.t.transliterate_turkish("İSTANBUL")
        self.assertNotIn("İ", result)

    # ── Kiril ───────────────────────────────────────────────────────────────

    def test_cyrillic_basic(self):
        result = self.t.transliterate_cyrillic("Москва")
        self.assertFalse(any("\u0400" <= c <= "\u04FF" for c in result))
        self.assertTrue(len(result) > 0)

    def test_cyrillic_sh(self):
        result = self.t.transliterate_cyrillic("Шолохов")
        self.assertIn("sh", result.lower())

    # ── Arapça ──────────────────────────────────────────────────────────────

    def test_arabic_basic(self):
        result = self.t.transliterate_arabic("بنك")
        # Sonuç tamamen Latin olmalı
        self.assertFalse(any("\u0600" <= c <= "\u06FF" for c in result))

    # ── Tam pipeline ─────────────────────────────────────────────────────────

    def test_full_pipeline_turkish(self):
        result = self.t.transliterate("Türkiye Büyük Millet Meclisi")
        self.assertNotIn("ü", result)
        self.assertNotIn("ğ", result)
        self.assertTrue(len(result) > 5)

    def test_empty_input_returns_empty(self):
        self.assertEqual(self.t.transliterate(""), "")

    def test_none_like_does_not_crash(self):
        result = self.t.transliterate("   ")
        self.assertEqual(result, "")

    def test_latin_unchanged(self):
        result = self.t.transliterate("Apple Inc")
        self.assertEqual(result, "Apple Inc")

    # ── Script detection ─────────────────────────────────────────────────────

    def test_detect_latin(self):
        self.assertEqual(detect_script("Apple Trading Co"), "LATIN")

    def test_detect_cyrillic(self):
        self.assertEqual(detect_script("Москва"), "CYRILLIC")

    def test_detect_arabic(self):
        self.assertEqual(detect_script("بنك"), "ARABIC")

    def test_detect_empty(self):
        self.assertEqual(detect_script(""), "UNKNOWN")

    # ── get_all_forms ─────────────────────────────────────────────────────────

    def test_get_all_forms_keys(self):
        forms = self.t.get_all_forms("Şirket Adı")
        self.assertIn("original_text", forms)
        self.assertIn("normalized_text", forms)
        self.assertIn("transliterated_text", forms)
        self.assertIn("detected_script", forms)


if __name__ == "__main__":
    unittest.main()
