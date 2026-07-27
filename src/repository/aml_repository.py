"""
aml_repository.py — AML veritabanı erişim ve yönetim katmanı.

Özellikler:
  - ThreadedConnectionPool ve context manager yapısıyla güvenli bağlantı yönetimi (connection leak koruması) sağlar.
  - Toplu yazma (bulk insert) ve denetim izi (audit log) operasyonlarını transactional bütünlük içinde yürütür.
  - Alert dışa aktarma (alert_export) ve run log durum takibini veritabanı seviyesinde otomatize eder.
"""

import json
import logging
from contextlib import contextmanager
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool
from src.config.db_tables import TABLES

logger = logging.getLogger(__name__)


class AMLRepository:
    """PostgreSQL bağlantı havuzu ve veri erişim operasyonları."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        sslmode: str = "prefer",
        enable_audit_trail: bool = True,
        append_only_history: bool = True,
        name: str = None,
        **kwargs
    ):
        self.enable_audit_trail = enable_audit_trail
        self.append_only_history = append_only_history
        effective_dbname = dbname or name or kwargs.get("name")
        try:
            conn_kwargs = {
                "host": host,
                "port": port,
                "dbname": effective_dbname,
                "user": user,
                "password": password,
            }
            if sslmode:
                conn_kwargs["sslmode"] = sslmode
            self.pool = ThreadedConnectionPool(
                minconn=1, maxconn=50, **conn_kwargs
            )
            if self.append_only_history:
                self.enforce_append_only_policy()
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            self.pool = None


    def get_connection(self):
        """Havuzdan bağlantı al."""
        if self.pool:
            try:
                return self.pool.getconn()
            except Exception as e:
                logger.error(f"Failed to get connection from pool: {e}")
        return None

    def release_connection(self, conn) -> None:
        """Bağlantıyı havuza geri ver."""
        if self.pool and conn:
            self.pool.putconn(conn)

    @contextmanager
    def connection(self):
        """
        Context manager — pool bağlantısını güvenli şekilde alır ve geri verir.

        Kullanım:
            with self.repo.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(...)
                conn.commit()

        Hata durumunda rollback otomatik yapılır ve bağlantı havuza döner.
        """
        conn = self.get_connection()
        if conn is None:
            raise RuntimeError("Database connection could not be acquired from pool")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            self.release_connection(conn)

    def enforce_append_only_policy(self) -> None:
        """
        append_only_history yapılandırması aktifse, alert_status_history tablosu üzerinde
        UPDATE ve DELETE işlemlerini veritabanı (SQL trigger) düzeyinde engelleyen kuralları uygular.
        """
        if not getattr(self, "append_only_history", True):
            return
        conn = self.get_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE OR REPLACE FUNCTION aml_audit.prevent_update_delete_history()
                    RETURNS trigger AS $$
                    BEGIN
                        RAISE EXCEPTION 'Table aml_audit.alert_status_history is append-only! % operation is prohibited by security policy.', TG_OP;
                    END;
                    $$ LANGUAGE plpgsql;

                    DROP TRIGGER IF EXISTS trg_prevent_update_delete_history ON aml_audit.alert_status_history;

                    CREATE TRIGGER trg_prevent_update_delete_history
                    BEFORE UPDATE OR DELETE ON aml_audit.alert_status_history
                    FOR EACH ROW EXECUTE FUNCTION aml_audit.prevent_update_delete_history();
                """)
            conn.commit()
            logger.debug("Append-only policy trigger enforced on aml_audit.alert_status_history.")
        except Exception as e:
            logger.debug(f"Could not enforce append-only SQL trigger (table might not exist yet): {e}")
            conn.rollback()
        finally:
            self.release_connection(conn)


    def execute_script(self, script_path: str) -> None:
        """SQL script dosyasını çalıştırır."""
        conn = self.get_connection()
        if not conn:
            return
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                sql = f.read()
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            logger.info(f"Successfully executed script: {script_path}")
        except Exception as e:
            logger.error(f"Error executing script {script_path}: {e}")
            conn.rollback()
        finally:
            self.release_connection(conn)

    # ── Run Log ───────────────────────────────────────────────────────────────

    def start_run_log(
        self,
        run_id: str,
        pipeline_name: str = "AML_Pipeline",
        embedding_model: str = None,
        reranker_model: str = None,
        scoring_config_version: str = None,
        threshold_version: str = None,
        pipeline_version: str = None,
        # Yeni alanlar
        embedding_model_hash: str = None,
        reranker_model_hash: str = None,
        ner_model_name: str = None,
        ner_model_version: str = None,
        calibration_version: str = None,
        normalization_version: str = None,
        watchlist_version: str = None,
        git_commit_hash: str = None,
    ) -> None:
        """Run log kaydını başlatır."""
        if not getattr(self, "enable_audit_trail", True):
            logger.debug("Audit trail is disabled in config; skipping start_run_log.")
            return
        conn = self.get_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {TABLES['run_log']} (
                        run_id, pipeline_name, started_at, status,
                        embedding_model, reranker_model,
                        scoring_config_version, threshold_version, pipeline_version,
                        embedding_model_version, reranker_model_version,
                        embedding_model_hash, reranker_model_hash,
                        ner_model_name, ner_model_version,
                        calibration_version, normalization_version,
                        watchlist_version, git_commit_hash
                    )
                    VALUES (%s, %s, now(), 'STARTED', %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO NOTHING
                """, (
                    run_id, pipeline_name, embedding_model, reranker_model,
                    scoring_config_version, threshold_version, pipeline_version,
                    embedding_model, reranker_model,
                    embedding_model_hash, reranker_model_hash,
                    ner_model_name, ner_model_version,
                    calibration_version, normalization_version,
                    watchlist_version, git_commit_hash
                ))
            conn.commit()
        except Exception as e:
            logger.exception("Error starting run log")
            conn.rollback()
            raise
        finally:
            self.release_connection(conn)

    def finish_run_log(self, run_id: str, metrics: dict, duration_seconds: float = None, status: str = None, error_message: str = None) -> None:
        """Run log kaydını tamamlandı olarak günceller."""
        if not getattr(self, "enable_audit_trail", True):
            logger.debug("Audit trail is disabled in config; skipping finish_run_log.")
            return
        conn = self.get_connection()
        if not conn:
            return
        if status is None:
            err_cnt = metrics.get("error_count", 0)
            inp_cnt = metrics.get("input_count", 0)
            if err_cnt == 0:
                status = 'SUCCESS'
            elif 0 < err_cnt < inp_cnt:
                status = 'PARTIAL_SUCCESS'
            else:
                status = 'FAILED'
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {TABLES['run_log']}
                    SET finished_at              = now(),
                        completed_at             = now(),
                        processed_row_count      = %s,
                        candidate_count          = %s,
                        alert_count              = %s,
                        input_count              = %s,
                        match_result_count       = %s,
                        high_alert_count         = %s,
                        medium_alert_count       = %s,
                        no_match_count           = %s,
                        no_candidate_count       = %s,
                        error_count              = %s,
                        prescreen_skipped_count  = %s,
                        total_duration_s         = %s,
                        ner_duration_s           = %s,
                        embedding_duration_s     = %s,
                        retrieval_duration_s     = %s,
                        reranker_duration_s      = %s,
                        scoring_duration_s       = %s,
                        p50_latency_ms           = %s,
                        p95_latency_ms           = %s,
                        p99_latency_ms           = %s,
                        rows_per_second          = %s,
                        avg_candidate_per_row    = %s,
                        status                   = %s,
                        error_message            = %s
                    WHERE run_id = %s
                """, (
                    metrics.get("processed_row_count", 0),
                    metrics.get("candidate_count", 0),
                    metrics.get("alert_count", 0),
                    metrics.get("input_count", 0),
                    metrics.get("match_result_count", 0),
                    metrics.get("high_alert_count", 0),
                    metrics.get("medium_alert_count", 0),
                    metrics.get("no_match_count", 0),
                    metrics.get("no_candidate_count", 0),
                    metrics.get("error_count", 0),
                    metrics.get("prescreen_skipped_count", 0),
                    duration_seconds,
                    metrics.get("ner_duration_s"),
                    metrics.get("embedding_duration_s"),
                    metrics.get("retrieval_duration_s"),
                    metrics.get("reranker_duration_s"),
                    metrics.get("scoring_duration_s"),
                    metrics.get("p50_latency_ms"),
                    metrics.get("p95_latency_ms"),
                    metrics.get("p99_latency_ms"),
                    metrics.get("rows_per_second"),
                    metrics.get("avg_candidate_per_row"),
                    status,
                    error_message,
                    run_id
                ))
            conn.commit()
        except Exception as e:
            logger.exception("Error finishing run log")
            conn.rollback()
            raise
        finally:
            self.release_connection(conn)

    def update_run_metrics(self, run_id: str, precision: float, recall: float,
                           f1: float, exact_match: float) -> None:
        """Benchmark metriklerini run log'a günceller."""
        if not getattr(self, "enable_audit_trail", True):
            return
        conn = self.get_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {TABLES['run_log']}
                    SET precision_score = %s,
                        recall_score    = %s,
                        f1_score        = %s,
                        exact_match_score = %s
                    WHERE run_id = %s
                """, (precision, recall, f1, exact_match, run_id))
            conn.commit()
        except Exception as e:
            logger.error(f"Error updating run metrics: {e}")
            conn.rollback()
        finally:
            self.release_connection(conn)

    def fail_run_log(self, run_id: str, error_message: str) -> None:
        """Run log kaydını hata ile günceller."""
        if not getattr(self, "enable_audit_trail", True):
            return
        conn = self.get_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {TABLES['run_log']}
                    SET finished_at   = now(),
                        completed_at  = now(),
                        status        = 'FAILED',
                        error_message = %s
                    WHERE run_id = %s
                """, (str(error_message)[:2000], run_id))
            conn.commit()
        except Exception as e:
            logger.error(f"Error failing run log: {e}")
            conn.rollback()
        finally:
            self.release_connection(conn)

    # ── Match Result (tüm sonuçlar) ───────────────────────────────────────────

    def insert_match_results_bulk(self, results: list) -> None:
        """
        Tüm eşleştirme sonuçlarını toplu olarak match_result tablosuna yazar.
        HIGH, MEDIUM, NO_MATCH, NO_CANDIDATE_FOUND hepsi buraya yazılır.

        Her dict en az şu anahtarları içermelidir:
          run_id, eft_id, decision_status
        """
        if not results:
            return

        conn = self.get_connection()
        if not conn:
            return

        try:
            with conn.cursor() as cur:
                execute_values(cur, f"""
                    INSERT INTO {TABLES['match_result']} (
                        run_id, eft_id, candidate_company_id, variant_id,
                        extracted_entity, entity_type, extraction_method,
                        extraction_confidence, entity_extraction_status,
                        trigram_score, full_text_score, vector_score, fuzzy_score,
                        reranker_raw_score, reranker_normalized_score,
                        calibrated_probability, calibration_applied,
                        calibration_method, calibration_version,
                        final_score, pipeline_status, no_candidate_reason,
                        decision_status, candidate_count,
                        reason_codes, human_explanation,
                        retrieval_sources, candidate_rank,
                        matched_variant_name, variant_type, watchlist_company_name,
                        name_score, country_score, identifier_score, address_score,
                        date_of_birth_score, entity_type_score, auxiliary_field_reason_codes,
                        exact_compact_match, compact_explanation, compact_matched_variant, rule_score
                    )
                    VALUES %s
                    ON CONFLICT DO NOTHING
                """, [
                    (
                        r["run_id"],
                        r["eft_id"],
                        r.get("candidate_company_id"),
                        r.get("variant_id"),
                        r.get("extracted_entity"),
                        r.get("entity_type", "UNKNOWN"),
                        r.get("extraction_method"),
                        r.get("extraction_confidence"),
                        r.get("entity_extraction_status"),
                        r.get("trigram_score"),
                        r.get("full_text_score"),
                        r.get("vector_score"),
                        r.get("fuzzy_score"),
                        r.get("reranker_raw_score"),
                        r.get("reranker_normalized_score"),
                        r.get("calibrated_probability"),
                        r.get("calibration_applied", False),
                        r.get("calibration_method"),
                        r.get("calibration_version"),
                        r.get("final_score"),
                        r.get("pipeline_status"),
                        r.get("no_candidate_reason"),
                        r["decision_status"],
                        r.get("candidate_count", 0),
                        json.dumps(r.get("reason_codes", [])) if r.get("reason_codes") else None,
                        r.get("human_explanation"),
                        json.dumps(r.get("retrieval_sources", {})) if r.get("retrieval_sources") else None,
                        r.get("candidate_rank"),
                        r.get("matched_variant_name"),
                        r.get("variant_type"),
                        r.get("watchlist_company_name"),
                        r.get("name_score"),
                        r.get("country_score"),
                        r.get("identifier_score"),
                        r.get("address_score"),
                        r.get("date_of_birth_score"),
                        r.get("entity_type_score"),
                        json.dumps(r.get("auxiliary_field_reason_codes", {})) if r.get("auxiliary_field_reason_codes") else None,
                        r.get("exact_compact_match", False),
                        r.get("compact_explanation"),
                        r.get("compact_matched_variant"),
                        r.get("rule_score"),
                    )
                    for r in results
                ])
            conn.commit()
            logger.debug(f"Bulk inserted {len(results)} match results.")
        except Exception as e:
            logger.exception("Error bulk inserting match results")
            conn.rollback()
            raise
        finally:
            self.release_connection(conn)

    # ── Alert (sadece HIGH/MEDIUM) ────────────────────────────────────────────

    def insert_alert(self, run_id: str, eft_id: int, company_id: int,
                     variant_id: int, final_score: float, risk_level: str,
                     extracted_entity: str = None) -> None:
        """Tekil alert yazar. Toplu kullanım için insert_alerts_bulk tercih edilmeli."""
        self.insert_alerts_bulk([{
            "run_id": run_id, "eft_id": eft_id, "company_id": company_id,
            "variant_id": variant_id, "final_score": final_score,
            "risk_level": risk_level, "extracted_entity": extracted_entity
        }])

    def insert_alerts_bulk(self, alerts: list) -> None:
        """
        HIGH ve MEDIUM alertleri toplu olarak alert tablosuna yazar.
        LOW ve NO_MATCH kayıtlar bu metoda gönderilmemeli.

        Her dict en az şu anahtarları içermelidir:
          run_id, eft_id, company_id, variant_id, final_score, risk_level
        """
        if not alerts:
            return

        conn = self.get_connection()
        if not conn:
            return

        try:
            with conn.cursor() as cur:
                execute_values(cur, f"""
                    INSERT INTO {TABLES['alert']} (
                        run_id, eft_id, company_id, variant_id,
                        final_score, fuzzy_score, vector_score, reranker_score,
                        risk_level, alert_status,
                        extracted_entity, match_reason,
                        entity_extraction_status, matched_variant_name,
                        variant_type, watchlist_company_name,
                        decision_status, reason_codes, human_explanation,
                        reranker_raw_score, reranker_normalized_score,
                        calibrated_probability, calibration_applied,
                        calibration_method, calibration_version,
                        entity_type, extraction_method, extraction_confidence,
                        retrieval_sources, candidate_rank, candidate_count,
                        exact_compact_match, compact_explanation, compact_matched_variant, rule_score
                    )
                    VALUES %s
                    ON CONFLICT DO NOTHING
                """, [
                    (
                        a["run_id"], a["eft_id"], a["company_id"],
                        a["variant_id"], a["final_score"],
                        a.get("fuzzy_score", 0), a.get("vector_score", 0),
                        a.get("reranker_score", 0),
                        a["risk_level"], "OPEN",
                        a.get("extracted_entity"), a.get("match_reason"),
                        a.get("entity_extraction_status"),
                        a.get("matched_variant_name"),
                        a.get("variant_type"), a.get("watchlist_company_name"),
                        a.get("decision_status"),
                        json.dumps(a.get("reason_codes", [])) if a.get("reason_codes") else None,
                        a.get("human_explanation"),
                        a.get("reranker_raw_score"),
                        a.get("reranker_normalized_score"),
                        a.get("calibrated_probability"),
                        a.get("calibration_applied", False),
                        a.get("calibration_method"),
                        a.get("calibration_version"),
                        a.get("entity_type"),
                        a.get("extraction_method"),
                        a.get("extraction_confidence"),
                        json.dumps(a.get("retrieval_sources", {})) if a.get("retrieval_sources") else None,
                        a.get("candidate_rank"),
                        a.get("candidate_count", 0),
                        a.get("exact_compact_match", False),
                        a.get("compact_explanation"),
                        a.get("compact_matched_variant"),
                        a.get("rule_score"),
                    )
                    for a in alerts
                ])
            conn.commit()
            logger.debug(f"Bulk inserted {len(alerts)} alerts.")
        except Exception as e:
            logger.exception("Error bulk inserting alerts")
            conn.rollback()
            raise
        finally:
            self.release_connection(conn)

    def populate_alert_export(self, run_id: str, input_table: str = None):
        """
        Bu fonksiyon her run bitiminde alert_export tablosunu ilgili run_id icin doldurur.
        UI'daki agir JOIN operasyonlarindan kurtulmak amaciyla tum detay veriyi birlestirip buraya yazar.
        DELETE + INSERT ayni transaction icinde — idempotent (ayni run_id tekrar calisirsa duplicate uretmez).
        """
        if not input_table:
            input_table = TABLES["eft_input"]

        with self.connection() as conn:
            with conn.cursor() as cur:
                # Eger ayni run_id onceden yazildiysa temizle (idempotency)
                cur.execute(f"DELETE FROM {TABLES['alert_export']} WHERE run_id = %s", (run_id,))

                # Sadece mevcut run_id icin ilgili alert'leri ve EFT verilerini birlestirip bas
                query = f"""
                    INSERT INTO {TABLES['alert_export']} (
                        alert_id, run_id, eft_id, transaction_date, amount, sender_account_id, receiver_account_id, original_explanation, source_system, batch_id, original_company_name, final_score, fuzzy_score, vector_score, reranker_score, risk_level, alert_status, extracted_entity, match_reason, created_at, entity_extraction_status, matched_variant_name, variant_type, watchlist_company_name, reviewed_by, reviewed_at, review_result, analyst_note, false_positive_reason, status_updated_at, decision_status, reason_codes, calibrated_probability, calibration_applied, entity_type, extraction_method, candidate_count, human_explanation, retrieval_sources
                    )
                    SELECT
                        a.alert_id, a.run_id, a.eft_id, v.transaction_date, v.amount, v.sender_account_id, v.receiver_account_id, v.explanation, v.source_system, v.batch_id, c.original_company_name, a.final_score, a.fuzzy_score, a.vector_score, a.reranker_score, a.risk_level, a.alert_status, a.extracted_entity, a.match_reason, a.created_at, a.entity_extraction_status, a.matched_variant_name, a.variant_type, a.watchlist_company_name, a.reviewed_by, a.reviewed_at, a.review_result, a.analyst_note, a.false_positive_reason, a.status_updated_at, a.decision_status, a.reason_codes, a.calibrated_probability, a.calibration_applied, a.entity_type, a.extraction_method, a.candidate_count, a.human_explanation, a.retrieval_sources
                    FROM {TABLES['alert']} a
                    LEFT JOIN {input_table} v ON a.eft_id = v.eft_id
                    LEFT JOIN {TABLES['company_variant']} c ON a.variant_id = c.variant_id
                    WHERE a.run_id = %s
                """
                cur.execute(query, (run_id,))
            conn.commit()
        logger.info(f"Populated alert_export flat table for run_id: {run_id}")


    # ── Alert status güncelleme + history ─────────────────────────────────────

    def update_alert_status(
        self,
        alert_id: int,
        status: str,
        reviewed_by: str = None,
        review_result: str = None,
        analyst_note: str = None,
        false_positive_reason: str = None,
        decision_reason: str = None,
        confidence: float = None,
        false_positive_category: str = None,
        escalation_reason: str = None,
    ) -> None:
        """
        Alert durumunu günceller ve append-only history tablosuna kaydeder.
        """
        conn = self.get_connection()
        if not conn:
            return

        try:
            with conn.cursor() as cur:
                # Mevcut durumu al
                cur.execute(
                    f"SELECT alert_status, run_id FROM {TABLES['alert']} WHERE alert_id = %s",
                    (alert_id,)
                )
                row = cur.fetchone()
                previous_status = row[0] if row else None
                run_id = row[1] if row else None

                # Alert tablosunu güncelle
                cur.execute(f"""
                    UPDATE {TABLES['alert']}
                    SET alert_status          = %s,
                        reviewed_by           = %s,
                        reviewed_at           = now(),
                        status_updated_at     = now(),
                        review_result         = %s,
                        analyst_note          = %s,
                        false_positive_reason = %s,
                        decision_reason       = %s
                    WHERE alert_id = %s
                """, (status, reviewed_by, review_result,
                      analyst_note, false_positive_reason,
                      decision_reason, alert_id))

                # History tablosuna append
                if getattr(self, "enable_audit_trail", True):
                    cur.execute(f"""
                        INSERT INTO {TABLES['alert_history']} (
                            alert_id, run_id, reviewed_by, reviewed_at,
                            previous_status, new_status, analyst_status,
                            analyst_note, decision_reason,
                            confidence, false_positive_category,
                            escalation_reason, final_analyst_label
                        )
                        VALUES (%s, %s, %s, now(), %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        alert_id, run_id, reviewed_by,
                        previous_status, status, status,
                        analyst_note, decision_reason,
                        confidence, false_positive_category,
                        escalation_reason, status
                    ))

            conn.commit()
            logger.info(f"Updated alert {alert_id}: {previous_status} → {status}")
        except Exception as e:
            logger.error(f"Error updating alert status: {e}")
            conn.rollback()
        finally:
            self.release_connection(conn)

    def insert_alert_status_history(self, history_record: dict) -> None:
        """
        Tek bir history kaydı yazar (opsiyonel, direkt kullanım için).

        Args:
            history_record: History alanlarını içeren dict
        """
        if not getattr(self, "enable_audit_trail", True):
            return
        conn = self.get_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {TABLES['alert_history']} (
                        alert_id, run_id, reviewed_by,
                        previous_status, new_status, analyst_note,
                        decision_reason, final_analyst_label
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    history_record.get("alert_id"),
                    history_record.get("run_id"),
                    history_record.get("reviewed_by"),
                    history_record.get("previous_status"),
                    history_record.get("new_status"),
                    history_record.get("analyst_note"),
                    history_record.get("decision_reason"),
                    history_record.get("final_analyst_label"),
                ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error inserting alert history: {e}")
            conn.rollback()
        finally:
            self.release_connection(conn)
