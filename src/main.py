import pandas as pd
import os
from config import (
    EFT_FILE_PATH,
    COMPANY_FILE_PATH,
    OUTPUT_DIR,
    RESULTS_OUTPUT_PATH,
    BEST_MATCHES_OUTPUT_PATH,
    SUSPICIOUS_EFTS_OUTPUT_PATH,
    CHUNK_SIZE,
    MIN_CANDIDATE_SCORE,
    MAX_CANDIDATES,
    FUZZY_FALLBACK_LIMIT
)

from text_utils import normalize_text
from alias_utils import generate_aliases
from candidate_filter import (
    build_alias_token_index,
    find_candidate_aliases_with_index
)
from matcher import (
    calculate_fuzzy_score,
    calculate_acronym_score,
    calculate_rule_score,
    calculate_final_score,
    assign_risk_level,
    build_reason
)
from vector_utils import (
    load_embedding_model,
    build_alias_embeddings,
    build_eft_embeddings,
    cosine_score,
    build_faiss_index
)


def load_company_data():
    """
    Şirket listesini okur.
    Şirket listesi EFT verisine göre daha küçük kabul edilir.
    """

    company_df = pd.read_csv(COMPANY_FILE_PATH)

    return company_df


def load_eft_data():
    """
    Küçük testler için EFT verisini komple okur.
    """

    eft_df = pd.read_csv(EFT_FILE_PATH)

    eft_df = eft_df.reset_index(drop=True)
    eft_df["eft_row_id"] = eft_df.index

    return eft_df


def prepare_company_aliases(company_df: pd.DataFrame) -> pd.DataFrame:
    """
    Her şirket için alias kayıtları üretir.
    """

    rows = []

    for _, row in company_df.iterrows():
        company_id = row["company_id"]
        company_name = row["company_name"]

        aliases = generate_aliases(company_name)

        for alias in aliases:
            rows.append({
                "company_id": company_id,
                "company_name": company_name,
                "alias": alias,
                "normalized_alias": normalize_text(alias)
            })

    alias_df = pd.DataFrame(rows)

    alias_df = alias_df.reset_index(drop=True)
    alias_df["alias_row_id"] = alias_df.index

    return alias_df


def run_matching(
    eft_df: pd.DataFrame,
    alias_df: pd.DataFrame,
    token_index: dict,
    eft_embeddings=None,
    alias_embeddings=None,
    faiss_index=None
) -> pd.DataFrame:
    """
    EFT açıklamaları için önce token index üzerinden şirket alias adayları bulunur.
    Sonra sadece bu adaylar üzerinde skor hesaplanır.

    Bu versiyonda EFT embeddingleri önceden batch olarak üretilir.
    Matching sırasında model tekrar çağrılmaz.
    """

    results = []

    for _, eft_row in eft_df.iterrows():
        eft_id = eft_row["eft_id"]
        eft_row_id = int(eft_row["eft_row_id"])
        description = eft_row["description"]
        normalized_description = normalize_text(description)

        description_embedding = None

        if eft_embeddings is not None:
            description_embedding = eft_embeddings[eft_row_id]

        candidate_aliases = find_candidate_aliases_with_index(
            description=description,
            alias_df=alias_df,
            token_index=token_index,
            min_candidate_score=MIN_CANDIDATE_SCORE,
            max_candidates=MAX_CANDIDATES,
            fuzzy_fallback_limit=FUZZY_FALLBACK_LIMIT
        )
        
        # Add FAISS vector candidates
        if faiss_index is not None and description_embedding is not None:
            D, I = faiss_index.search(description_embedding.reshape(1, -1), k=5) # Top 5 vector matches
            existing_candidate_indices = {c["alias_row_id"] for c in candidate_aliases}
            for score, idx in zip(D[0], I[0]):
                if idx != -1 and idx not in existing_candidate_indices and score >= 0.5:
                    alias_row = alias_df.iloc[idx].to_dict()
                    alias_row["candidate_filter_score"] = float(score)
                    alias_row["candidate_source"] = "Vector Match"
                    candidate_aliases.append(alias_row)
                    existing_candidate_indices.add(idx)

        for alias_row in candidate_aliases:
            company_id = alias_row["company_id"]
            company_name = alias_row["company_name"]
            alias = alias_row["alias"]
            alias_row_id = int(alias_row["alias_row_id"])
            candidate_filter_score = alias_row["candidate_filter_score"]
            candidate_source = alias_row.get("candidate_source", "Token Match")

            fuzzy_score = calculate_fuzzy_score(description, alias)
            acronym_score = calculate_acronym_score(description, company_name)
            rule_score = calculate_rule_score(description, company_name)

            vector_score = 0.0

            if description_embedding is not None and alias_embeddings is not None:
                alias_embedding = alias_embeddings[alias_row_id]
                vector_score = cosine_score(description_embedding, alias_embedding)

            final_score = calculate_final_score(
                fuzzy_score=fuzzy_score,
                acronym_score=acronym_score,
                rule_score=rule_score,
                vector_score=vector_score
            )

            risk_level = assign_risk_level(
                final_score=final_score,
                fuzzy_score=fuzzy_score,
                acronym_score=acronym_score,
                rule_score=rule_score,
                vector_score=vector_score,
                candidate_source=candidate_source
            )

            reason = build_reason(
                fuzzy_score=fuzzy_score,
                acronym_score=acronym_score,
                rule_score=rule_score,
                vector_score=vector_score,
                candidate_source=candidate_source
            )

            results.append({
                "eft_id": eft_id,
                "description": description,
                "normalized_description": normalized_description,
                "company_id": company_id,
                "company_name": company_name,
                "alias": alias,
                "candidate_source": candidate_source,
                "candidate_filter_score": candidate_filter_score,
                "fuzzy_score": round(fuzzy_score, 4),
                "vector_score": round(vector_score, 4),
                "acronym_score": round(acronym_score, 4),
                "rule_score": round(rule_score, 4),
                "final_score": final_score,
                "risk_level": risk_level,
                "reason": reason
            })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return result_df

    result_df = result_df.sort_values(
        by=["eft_id", "final_score"],
        ascending=[True, False]
    )

    top_results = result_df.groupby("eft_id").head(3).reset_index(drop=True)

    return top_results


def get_best_matches(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Her EFT için en yüksek final_score değerine sahip tek eşleşmeyi döndürür.
    """

    if result_df.empty:
        return result_df

    best_matches = (
        result_df
        .sort_values(
            by=["eft_id", "final_score"],
            ascending=[True, False]
        )
        .groupby("eft_id")
        .head(1)
        .reset_index(drop=True)
    )

    return best_matches


def get_suspicious_efts(best_matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    En iyi eşleşmeler içerisinden No Match olmayan EFT kayıtlarını döndürür.
    """

    if best_matches_df.empty:
        return best_matches_df

    suspicious_df = best_matches_df[
        best_matches_df["risk_level"] != "No Match"
    ].copy()

    suspicious_df = suspicious_df.sort_values(
        by="final_score",
        ascending=False
    ).reset_index(drop=True)

    return suspicious_df

def write_chunk_output(
    df: pd.DataFrame,
    output_path: str,
    write_header: bool
):
    """
    Chunk sonucunu CSV dosyasına ekler.
    İlk chunk için header yazılır, sonraki chunklarda header yazılmaz.
    """

    if df.empty:
        return

    df.to_csv(
        output_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False
    )
    
def main_chunked(chunk_size: int = 10000):
    """
    EFT verisini chunk'lar halinde işleyen ana pipeline.
    Büyük veri için önerilen çalışma şeklidir.
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    company_df = load_company_data()

    alias_df = prepare_company_aliases(company_df)

    print("Şirket alias kayıtları hazırlandı.")
    print(f"Toplam alias sayısı: {len(alias_df)}")
    print("-" * 80)

    token_index = build_alias_token_index(alias_df)

    print("Alias token index oluşturuldu.")
    print(f"Index içindeki unique token sayısı: {len(token_index)}")
    print("-" * 80)

    embedding_model = load_embedding_model()
    alias_embeddings = build_alias_embeddings(alias_df, embedding_model)
    faiss_index = build_faiss_index(alias_embeddings)

    if embedding_model is not None and alias_embeddings is not None:
        print("Alias embeddingleri oluşturuldu.")
        print(f"Alias embedding shape: {alias_embeddings.shape}")
        print("-" * 80)
    else:
        print("Vector score devre dışı. Sistem fuzzy + rule score ile çalışacak.")
        print("-" * 80)

    results_path = RESULTS_OUTPUT_PATH
    best_matches_path = BEST_MATCHES_OUTPUT_PATH
    suspicious_path = SUSPICIOUS_EFTS_OUTPUT_PATH

    write_results_header = True
    write_best_header = True
    write_suspicious_header = True

    total_processed = 0
    total_suspicious = 0

    eft_reader = pd.read_csv(
        EFT_FILE_PATH,
        chunksize=chunk_size
    )

    for chunk_no, eft_chunk_df in enumerate(eft_reader, start=1):
        print(f"Chunk {chunk_no} işleniyor... Satır sayısı: {len(eft_chunk_df)}")

        eft_chunk_df = eft_chunk_df.reset_index(drop=True)
        eft_chunk_df["eft_row_id"] = eft_chunk_df.index

        eft_embeddings = build_eft_embeddings(
            eft_df=eft_chunk_df,
            model=embedding_model
        )

        result_df = run_matching(
            eft_df=eft_chunk_df,
            alias_df=alias_df,
            token_index=token_index,
            eft_embeddings=eft_embeddings,
            alias_embeddings=alias_embeddings,
            faiss_index=faiss_index
        )

        best_matches_df = get_best_matches(result_df)
        suspicious_efts_df = get_suspicious_efts(best_matches_df)

        write_chunk_output(
            df=result_df,
            output_path=results_path,
            write_header=write_results_header
        )

        write_chunk_output(
            df=best_matches_df,
            output_path=best_matches_path,
            write_header=write_best_header
        )

        write_chunk_output(
            df=suspicious_efts_df,
            output_path=suspicious_path,
            write_header=write_suspicious_header
        )

        if not result_df.empty:
            write_results_header = False

        if not best_matches_df.empty:
            write_best_header = False

        if not suspicious_efts_df.empty:
            write_suspicious_header = False

        total_processed += len(eft_chunk_df)
        total_suspicious += len(suspicious_efts_df)

        print(f"Chunk {chunk_no} tamamlandı.")
        print(f"Toplam işlenen EFT: {total_processed}")
        print(f"Toplam şüpheli EFT: {total_suspicious}")
        print("-" * 80)

    print("Chunk processing tamamlandı.")
    print(f"Toplam işlenen EFT: {total_processed}")
    print(f"Toplam şüpheli EFT: {total_suspicious}")
    print("Sonuç dosyaları oluşturuldu:")
    print(results_path)
    print(best_matches_path)
    print(suspicious_path)

def main():
    eft_df = load_eft_data()
    company_df = load_company_data()

    alias_df = prepare_company_aliases(company_df)

    print("Şirket alias kayıtları:")
    print(alias_df)
    print("-" * 80)

    print(f"Toplam alias sayısı: {len(alias_df)}")
    print("-" * 80)

    token_index = build_alias_token_index(alias_df)

    print("Alias token index oluşturuldu.")
    print(f"Index içindeki unique token sayısı: {len(token_index)}")
    print("-" * 80)

    embedding_model = load_embedding_model()
    alias_embeddings = build_alias_embeddings(alias_df, embedding_model)
    eft_embeddings = build_eft_embeddings(eft_df, embedding_model)

    if embedding_model is not None and alias_embeddings is not None and eft_embeddings is not None:
        print("Alias ve EFT embeddingleri oluşturuldu.")
        print(f"Alias embedding shape: {alias_embeddings.shape}")
        print(f"EFT embedding shape: {eft_embeddings.shape}")
        print("-" * 80)
    else:
        print("Vector score devre dışı. Sistem fuzzy + rule score ile çalışacak.")
        print("-" * 80)

    result_df = run_matching(
        eft_df=eft_df,
        alias_df=alias_df,
        token_index=token_index,
        eft_embeddings=eft_embeddings,
        alias_embeddings=alias_embeddings
    )

    best_matches_df = get_best_matches(result_df)
    suspicious_efts_df = get_suspicious_efts(best_matches_df)

    print("Top eşleşme sonuçları:")
    print(result_df)
    print("-" * 80)

    print("Her EFT için en iyi eşleşme:")
    print(best_matches_df)
    print("-" * 80)

    print("Şüpheli EFT kayıtları:")
    print(suspicious_efts_df)
    print("-" * 80)

    result_df.to_csv("outputs/results.csv", index=False)
    best_matches_df.to_csv("outputs/best_matches.csv", index=False)
    suspicious_efts_df.to_csv("outputs/suspicious_efts.csv", index=False)

    print("Sonuç dosyaları oluşturuldu:")
    print(RESULTS_OUTPUT_PATH)
    print(BEST_MATCHES_OUTPUT_PATH)
    print(SUSPICIOUS_EFTS_OUTPUT_PATH)


if __name__ == "__main__":
    main_chunked(chunk_size=CHUNK_SIZE)