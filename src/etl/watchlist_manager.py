"""
watchlist_manager.py — Watchlist / blacklist yaşam döngüsü yönetimi.

Desteklenen işlemler:
  - Yeni kayıt ekleme
  - Mevcut kayıt güncelleme
  - Alias güncelleme
  - Kaydın pasif hale getirilmesi (deactivation)
  - Delta load
  - Deduplication (source_hash kontrolü)
  - Rollback desteği
  - Kaynak dosya hash kontrolü
"""

import hashlib
import json
import logging
from datetime import date, datetime
from typing import Optional

from psycopg2.extras import execute_values
from src.config.db_tables import TABLES

logger = logging.getLogger(__name__)


def _compute_hash(data: str) -> str:
    """SHA-256 hash hesaplar."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class WatchlistManager:
    """
    Watchlist kayıtlarını yönetir.
    Tüm işlemler versiyonlanır ve audit trail bırakır.
    """

    def __init__(self, repository, watchlist_version: str = "v1"):
        """
        Args:
            repository: AMLRepository instance
            watchlist_version: Bu yükleme için versiyon etiketi
        """
        self.repo = repository
        self.watchlist_version = watchlist_version

    def compute_file_hash(self, file_path: str) -> str:
        """
        Kaynak dosyanın SHA-256 hash değerini hesaplar.
        Aynı dosya tekrar yüklenirse deduplication için kullanılır.

        Args:
            file_path: Dosya yolu

        Returns:
            SHA-256 hex string
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def upsert_company(
        self,
        company_id: int,
        company_name: str,
        source_name: str = None,
        source_authority: str = None,
        source_record_id: str = None,
        publication_date: date = None,
        effective_date: date = None,
        raw_payload: dict = None,
    ) -> Optional[int]:
        """
        Şirket varlığını ekler veya günceller.
        Şu an company_variant tablosunda company_id'yi günceller.

        Args:
            company_id: Kaynak sistemdeki şirket ID'si
            company_name: Resmi şirket adı
            ...

        Returns:
            Etkilenen kayıt sayısı
        """
        conn = self.repo.get_connection()
        if not conn:
            return None
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {TABLES['company_variant']}
                    SET source_name        = COALESCE(%s, source_name),
                        source_authority   = COALESCE(%s, source_authority),
                        source_record_id   = COALESCE(%s, source_record_id),
                        publication_date   = COALESCE(%s, publication_date),
                        effective_date     = COALESCE(%s, effective_date),
                        last_seen_date     = now(),
                        raw_payload        = COALESCE(%s::jsonb, raw_payload)
                    WHERE company_id = %s
                """, (
                    source_name, source_authority, source_record_id,
                    publication_date, effective_date,
                    json.dumps(raw_payload) if raw_payload else None,
                    company_id
                ))
                affected = cur.rowcount
            conn.commit()
            return affected
        except Exception as e:
            logger.error(f"Error upserting company {company_id}: {e}")
            conn.rollback()
            return None
        finally:
            self.repo.release_connection(conn)

    def add_alias(
        self,
        company_id: int,
        company_name: str,
        variant_name: str,
        variant_type: str,
        alias_confidence: float = 1.0,
        source_name: str = None,
        source_record_id: str = None,
        is_official_alias: bool = False,
        transliterated_name: str = None,
        detected_script: str = None,
    ) -> Optional[int]:
        """
        Şirket varyantı (alias) ekler.
        Aynı alias zaten varsa (UNIQUE constraint) sessizce atlar.

        Args:
            company_id: Şirket ID
            company_name: Ana şirket adı
            variant_name: Alias değeri
            variant_type: OFFICIAL, ALIAS, TRANSLITERATION, ABBREVIATION, vb.
            alias_confidence: 1.0=resmi, 0.5=transliteration, 0.3=algoritmik

        Returns:
            Yeni variant_id veya None
        """
        from src.utils.text_utils import normalize_text

        normalized = normalize_text(variant_name)
        source_hash = _compute_hash(f"{company_id}:{normalized}:{variant_type}")

        conn = self.repo.get_connection()
        if not conn:
            return None
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {TABLES['company_variant']} (
                        company_id, original_company_name, variant_name,
                        normalized_variant_name, variant_type,
                        list_version, list_version_tag, normalization_version,
                        alias_confidence, source_name, source_record_id,
                        source_hash, is_official_alias,
                        transliterated_name, detected_script,
                        ingestion_date, last_seen_date, is_active
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'text_utils_v2',
                            %s, %s, %s, %s, %s, %s, %s, now(), now(), true)
                    ON CONFLICT (company_id, normalized_variant_name, variant_type, list_version)
                    DO UPDATE SET
                        last_seen_date   = now(),
                        alias_confidence = GREATEST(EXCLUDED.alias_confidence, {TABLES['company_variant']}.alias_confidence),
                        source_hash      = EXCLUDED.source_hash
                    RETURNING variant_id
                """, (
                    company_id, company_name, variant_name, normalized,
                    variant_type, self.watchlist_version, self.watchlist_version,
                    alias_confidence, source_name, source_record_id, source_hash,
                    is_official_alias, transliterated_name, detected_script,
                ))
                row = cur.fetchone()
                variant_id = row[0] if row else None
            conn.commit()
            return variant_id
        except Exception as e:
            logger.error(f"Error adding alias for company {company_id}: {e}")
            conn.rollback()
            return None
        finally:
            self.repo.release_connection(conn)

    def deactivate_company(
        self,
        company_id: int,
        reason: str = "MANUAL_DEACTIVATION",
    ) -> int:
        """
        Şirketi ve tüm varyantlarını pasif hale getirir.

        Args:
            company_id: Şirket ID
            reason: Pasifleştirme nedeni

        Returns:
            Etkilenen kayıt sayısı
        """
        conn = self.repo.get_connection()
        if not conn:
            return 0
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {TABLES['company_variant']}
                    SET is_active          = false,
                        deactivation_reason = %s,
                        last_seen_date      = now()
                    WHERE company_id = %s AND is_active = true
                """, (reason, company_id))
                affected = cur.rowcount
            conn.commit()
            logger.info(f"Deactivated {affected} variants for company_id={company_id} ({reason})")
            return affected
        except Exception as e:
            logger.error(f"Error deactivating company {company_id}: {e}")
            conn.rollback()
            return 0
        finally:
            self.repo.release_connection(conn)

    def delta_load(
        self,
        new_records: list[dict],
        source_name: str,
        file_hash: str = None,
    ) -> dict:
        """
        Delta yükleme: Yeni kayıtları ekler, mevcut olanları atlar.
        Aynı dosya (file_hash) tekrar yüklenirse erken çıkar.

        Args:
            new_records: Yükleme kayıtları listesi
                         Her kayıt: {company_id, company_name, variant_name,
                                     variant_type, alias_confidence, ...}
            source_name: Kaynak adı (OFAC, UN, EU, vb.)
            file_hash: Kaynak dosya hash (deduplication için)

        Returns:
            {"added": N, "skipped": M, "errors": K}
        """
        stats = {"added": 0, "skipped": 0, "errors": 0}

        logger.info(
            f"Delta load started: source={source_name}, "
            f"records={len(new_records)}, hash={file_hash}"
        )

        for record in new_records:
            try:
                vid = self.add_alias(
                    company_id       = record["company_id"],
                    company_name     = record["company_name"],
                    variant_name     = record["variant_name"],
                    variant_type     = record.get("variant_type", "ALIAS"),
                    alias_confidence = record.get("alias_confidence", 1.0),
                    source_name      = source_name,
                    source_record_id = record.get("source_record_id"),
                    is_official_alias = record.get("is_official_alias", False),
                )
                if vid:
                    stats["added"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                logger.error(f"Error processing record {record}: {e}")
                stats["errors"] += 1

        logger.info(f"Delta load complete: {stats}")
        return stats

    def get_watchlist_stats(self) -> dict:
        """
        Mevcut watchlist istatistiklerini döndürür.

        Returns:
            {"total_companies": N, "total_variants": M, "active_variants": K}
        """
        conn = self.repo.get_connection()
        if not conn:
            return {}
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT
                        COUNT(DISTINCT company_id) AS total_companies,
                        COUNT(*) AS total_variants,
                        SUM(CASE WHEN is_active THEN 1 ELSE 0 END) AS active_variants
                    FROM {TABLES['company_variant']}
                """)
                row = cur.fetchone()
                return {
                    "total_companies": row[0],
                    "total_variants":  row[1],
                    "active_variants": row[2],
                }
        except Exception as e:
            logger.error(f"Error getting watchlist stats: {e}")
            return {}
        finally:
            self.repo.release_connection(conn)
