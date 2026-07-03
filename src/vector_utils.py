import numpy as np
from config import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE

try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    torch = None
    SentenceTransformer = None


def get_device() -> str:
    """
    GPU varsa cuda, yoksa cpu döndürür.
    """

    if torch is not None and torch.cuda.is_available():
        return "cuda"

    return "cpu"


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """
    SentenceTransformer modelini yükler.
    Kütüphane kurulu değilse None döndürür.
    """

    if SentenceTransformer is None:
        print("sentence-transformers kurulu değil. Vector score devre dışı.")
        return None

    device = get_device()
    print(f"Embedding modeli yükleniyor. Device: {device}")

    model = SentenceTransformer(model_name, device=device)

    return model


def encode_texts(model, texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE):
    """
    Metinleri embedding vektörlerine çevirir.
    """

    if model is None:
        return None

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.array(embeddings)


def cosine_score(vec1, vec2) -> float:
    """
    Normalize edilmiş iki vektör için cosine similarity hesaplar.
    normalize_embeddings=True kullanıldığı için dot product yeterlidir.
    """

    if vec1 is None or vec2 is None:
        return 0.0

    score = float(np.dot(vec1, vec2))

    # Güvenlik için 0-1 aralığına yaklaştırıyoruz.
    # Bazı modeller negatif similarity üretebilir.
    score = max(0.0, min(1.0, score))

    return score


def build_alias_embeddings(alias_df, model):
    """
    Alias DataFrame için embedding matrisi üretir.
    """

    if model is None:
        alias_df["embedding_index"] = None
        return None

    alias_texts = alias_df["normalized_alias"].fillna("").tolist()

    alias_embeddings = encode_texts(
        model=model,
        texts=alias_texts,
        batch_size=32
    )

    return alias_embeddings

def build_eft_embeddings(eft_df, model):
    """
    EFT açıklamaları için batch embedding matrisi üretir.
    """

    if model is None:
        return None

    descriptions = (
        eft_df["description"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    embeddings = encode_texts(
        model=model,
        texts=descriptions,
        batch_size=32
    )

    return embeddings