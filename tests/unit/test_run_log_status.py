import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.pipeline.batch_processor import BatchProcessor


class TestRunLogStatusHandling(unittest.TestCase):
    """run_log kaydının hata, kısmi başarı ve başarı durumlarında doğru kapatıldığının testi."""

    def setUp(self):
        self.mock_repo = MagicMock()
        self.mock_inference = MagicMock()
        self.processor = BatchProcessor(
            repository=self.mock_repo,
            config={"processing": {"chunk_size": 10}},
            inference_service=self.mock_inference
        )

    def test_conn_for_read_failure_calls_finish_run_log_with_failed(self):
        """Veritabanı okuma bağlantısı alınamazsa finish_run_log FAILED statüsü ile çağrılmalı."""
        # İlk get_connection (count için) de None dönsün, okuma için olan ikinci get_connection da None dönsün
        self.mock_repo.get_connection.return_value = None

        self.processor.process_db_table_in_chunks("test_run", "batch_1")

        self.mock_repo.finish_run_log.assert_called_once()
        _, kwargs = self.mock_repo.finish_run_log.call_args
        self.assertEqual(kwargs.get("status"), "FAILED")
        self.assertIn("Could not get DB connection", kwargs.get("error_message", ""))

    @patch("pandas.read_sql")
    def test_unexpected_exception_in_loop_calls_finish_run_log_with_failed_and_reraises(self, mock_read_sql):
        """Döngü esnasında unhandled bir exception oluşursa finish_run_log FAILED ile çağrılıp hata fırlatılmalı."""
        mock_conn = MagicMock()
        self.mock_repo.get_connection.return_value = mock_conn
        
        # read_sql patlasın
        mock_read_sql.side_effect = RuntimeError("Database timeout during read")

        with self.assertRaisesRegex(RuntimeError, "Database timeout during read"):
            self.processor.process_db_table_in_chunks("test_run", "batch_1")

        self.mock_repo.finish_run_log.assert_called_once()
        _, kwargs = self.mock_repo.finish_run_log.call_args
        self.assertEqual(kwargs.get("status"), "FAILED")
        self.assertIn("Database timeout during read", kwargs.get("error_message", ""))
        self.mock_repo.release_connection.assert_called()

    @patch("pandas.read_sql")
    def test_partial_success_when_some_chunks_have_error(self, mock_read_sql):
        """Bazı chunklar hata alıp bazıları başarılı olursa (0 < error_count < input_count) işlem sonunda PARTIAL_SUCCESS yazılmalı."""
        mock_conn = MagicMock()
        self.mock_repo.get_connection.return_value = mock_conn

        df_chunk1 = pd.DataFrame({"eft_id": [1, 2], "explanation": ["exp1", "exp2"]})
        df_chunk2 = pd.DataFrame({"eft_id": [3, 4], "explanation": ["exp3", "exp4"]})
        mock_read_sql.return_value = [df_chunk1, df_chunk2]

        # 1. chunk hata versin, 2. chunk başarılı olsun
        self.mock_inference.analyze_batch.side_effect = [
            ValueError("NER model failure on chunk 1"),
            [{"no_candidate": True, "high_count": 0, "medium_count": 0, "no_match_count": 1, "match_results": [], "alerts": []}] * 2
        ]

        self.processor.process_db_table_in_chunks("test_run", "batch_1")

        self.mock_repo.finish_run_log.assert_called_once()
        _, kwargs = self.mock_repo.finish_run_log.call_args
        self.assertEqual(kwargs.get("status"), "PARTIAL_SUCCESS")
        self.mock_repo.release_connection.assert_called()

    @patch("pandas.read_sql")
    def test_failed_when_all_chunks_have_error(self, mock_read_sql):
        """Tüm chunklar hata alırsa (error_count == input_count) işlem sonunda FAILED yazılmalı."""
        mock_conn = MagicMock()
        self.mock_repo.get_connection.return_value = mock_conn

        df_chunk1 = pd.DataFrame({"eft_id": [1, 2], "explanation": ["exp1", "exp2"]})
        mock_read_sql.return_value = [df_chunk1]

        self.mock_inference.analyze_batch.side_effect = ValueError("NER model failure on chunk")

        self.processor.process_db_table_in_chunks("test_run", "batch_1")

        self.mock_repo.finish_run_log.assert_called_once()
        _, kwargs = self.mock_repo.finish_run_log.call_args
        self.assertEqual(kwargs.get("status"), "FAILED")
        self.mock_repo.release_connection.assert_called()


if __name__ == "__main__":
    unittest.main()
