"""
calibration.py — Reranker skor kalibrasyon altyapısı.

Cross-encoder modeli ham çıktıyı doğrudan olasılık olarak
değerlendirmek yerine, istatistiksel bir kalibrasyon modeli
(Platt Scaling veya Isotonic Regression) üzerinden geçirir.

Kalibrasyon modeli yoksa sistem eski normalize edilmiş skoru kullanır
ancak bu durum audit loglarına yazılır (calibration_applied=False).

Kullanım:
    calib = CalibrationWrapper(method="platt", repo=repo)
    prob = calib.calibrate(raw_score=0.87)
    # → calibrated_probability, calibration_applied, calibration_method
"""

import logging
import json
import pickle
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Kalibrasyon sonucu."""
    calibrated_probability: float
    calibration_applied: bool
    calibration_method: Optional[str]
    calibration_version: Optional[str]


class CalibrationWrapper:
    """
    Reranker skor kalibrasyonu için wrapper.

    İki mod:
      - calibration_model_path verilirse: pickle'dan yükle
      - yoksa: identity fallback (normalize edilmiş skor döndür)

    Desteklenen yöntemler:
      - PLATT_SCALING: sklearn LogisticRegression tabanlı
      - ISOTONIC_REGRESSION: sklearn IsotonicRegression tabanlı
    """

    def __init__(
        self,
        calibration_model_path: Optional[str] = None,
        calibration_version: Optional[str] = None,
        calibration_method: Optional[str] = None,
    ):
        """
        Args:
            calibration_model_path: Pickle dosyası yolu (opsiyonel)
            calibration_version: Versiyon etiketi (audit için)
            calibration_method: PLATT_SCALING veya ISOTONIC_REGRESSION
        """
        self.calibration_model_path = calibration_model_path
        self.calibration_version = calibration_version
        self.calibration_method = calibration_method
        self._model = None
        self._loaded = False

        if calibration_model_path:
            self._try_load_model(calibration_model_path)

    def _try_load_model(self, path: str) -> None:
        """
        Kalibrasyon modelini pickle dosyasından yükler.
        Hata durumunda sessizce fallback moduna geçer.
        """
        try:
            if not os.path.exists(path):
                logger.info(f"Calibration model not found at {path}. Using fallback.")
                return

            with open(path, "rb") as f:
                self._model = pickle.load(f)

            self._loaded = True
            logger.info(
                f"Calibration model loaded: {path} "
                f"(method={self.calibration_method}, version={self.calibration_version})"
            )
        except Exception as e:
            logger.warning(f"Failed to load calibration model from {path}: {e}. Using fallback.")
            self._model = None
            self._loaded = False

    @property
    def is_available(self) -> bool:
        """Kalibrasyon modeli yüklü mü?"""
        return self._loaded and self._model is not None

    def calibrate(self, raw_score: float) -> CalibrationResult:
        """
        Ham reranker skorunu kalibre eder.

        Kalibrasyon modeli yoksa raw_score'u olasılık olarak döndürür
        ve calibration_applied=False olarak işaretler.

        Args:
            raw_score: Reranker ham çıktısı

        Returns:
            CalibrationResult
        """
        if not self.is_available:
            # Fallback: Normalize edilmiş skoru kullan
            logger.debug(
                "Calibration model not available. "
                f"Using raw score as probability: {raw_score:.4f}"
            )
            return CalibrationResult(
                calibrated_probability=float(max(0.0, min(1.0, raw_score))),
                calibration_applied=False,
                calibration_method=None,
                calibration_version=None,
            )

        try:
            # sklearn modeli: predict_proba([score])[0][1]
            prob = self._model.predict_proba([[raw_score]])[0][1]
            prob = float(max(0.0, min(1.0, prob)))

            logger.debug(
                f"Calibrated: raw={raw_score:.4f} → prob={prob:.4f} "
                f"(method={self.calibration_method})"
            )

            return CalibrationResult(
                calibrated_probability=prob,
                calibration_applied=True,
                calibration_method=self.calibration_method,
                calibration_version=self.calibration_version,
            )

        except Exception as e:
            logger.warning(f"Calibration failed for score {raw_score}: {e}. Using fallback.")
            return CalibrationResult(
                calibrated_probability=float(max(0.0, min(1.0, raw_score))),
                calibration_applied=False,
                calibration_method=None,
                calibration_version=None,
            )

    def calibrate_batch(self, raw_scores: list[float]) -> list[CalibrationResult]:
        """
        Toplu kalibrasyon. Her skor için calibrate() çağrısı yapar.

        Args:
            raw_scores: Ham skor listesi

        Returns:
            CalibrationResult listesi
        """
        return [self.calibrate(s) for s in raw_scores]


def train_platt_scaling(
    raw_scores: list[float],
    labels: list[int],
    output_path: str
) -> dict:
    """
    Platt Scaling kalibrasyon modeli eğitir ve kaydeder.
    Yalnızca offline evaluation için kullanılır.
    Production config'i değiştirmez.

    Args:
        raw_scores: Reranker ham skorları
        labels: 0/1 etiketleri (0=NO_MATCH, 1=MATCH)
        output_path: Eğitilen modelin kaydedileceği yol

    Returns:
        Eğitim metrikleri dict
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import brier_score_loss
        import numpy as np

        X = np.array(raw_scores).reshape(-1, 1)
        y = np.array(labels)

        model = LogisticRegression()
        model.fit(X, y)

        probs = model.predict_proba(X)[:, 1]
        brier = brier_score_loss(y, probs)

        with open(output_path, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Platt scaling model saved to {output_path}. Brier score: {brier:.4f}")

        return {
            "method": "PLATT_SCALING",
            "brier_score": float(brier),
            "n_samples": len(raw_scores),
            "output_path": output_path,
        }

    except ImportError:
        logger.error("sklearn not available. Cannot train calibration model.")
        return {"error": "sklearn not available"}
    except Exception as e:
        logger.error(f"Platt scaling training failed: {e}")
        return {"error": str(e)}


def train_isotonic_regression(
    raw_scores: list[float],
    labels: list[int],
    output_path: str
) -> dict:
    """
    Isotonic Regression kalibrasyon modeli eğitir ve kaydeder.
    Yalnızca offline evaluation için kullanılır.

    Args:
        raw_scores: Reranker ham skorları
        labels: 0/1 etiketleri
        output_path: Eğitilen modelin kaydedileceği yol

    Returns:
        Eğitim metrikleri dict
    """
    try:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss
        import numpy as np
        import pickle

        X = np.array(raw_scores)
        y = np.array(labels, dtype=float)

        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(X, y)

        probs = model.predict(X)
        brier = brier_score_loss(y, probs)

        # Wrapper sınıfı: CalibrationWrapper predict_proba bekliyor
        class IsotonicWrapper:
            def __init__(self, iso_model):
                self._iso = iso_model

            def predict_proba(self, X_2d):
                import numpy as np
                scores = np.array(X_2d).flatten()
                probs = self._iso.predict(scores)
                return np.column_stack([1 - probs, probs])

        wrapper = IsotonicWrapper(model)

        with open(output_path, "wb") as f:
            pickle.dump(wrapper, f)

        logger.info(f"Isotonic regression model saved to {output_path}. Brier: {brier:.4f}")

        return {
            "method": "ISOTONIC_REGRESSION",
            "brier_score": float(brier),
            "n_samples": len(raw_scores),
            "output_path": output_path,
        }

    except ImportError:
        logger.error("sklearn not available.")
        return {"error": "sklearn not available"}
    except Exception as e:
        logger.error(f"Isotonic regression training failed: {e}")
        return {"error": str(e)}
