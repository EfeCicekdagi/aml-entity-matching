import pandas as pd

from text_utils import normalize_text
from alias_utils import generate_aliases
from matcher import (
    calculate_fuzzy_score,
    calculate_acronym_score,
    calculate_rule_score,
    calculate_final_score,
    assign_risk_level,
    build_reason
)


def load_data():
    eft_df = pd.read_csv("data/eft_samples.csv")
    company_df = pd.read_csv("data/company_list.csv")

    return eft_df, company_df


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

    return pd.DataFrame(rows)


def run_matching(eft_df: pd.DataFrame, alias_df: pd.DataFrame) -> pd.DataFrame:
    """
    EFT açıklamaları ile şirket alias'larını karşılaştırır.
    İlk MVP olduğu için tüm küçük veri üzerinde brute force çalışır.
    Büyük veride bunu candidate filtering ile değiştireceğiz.
    """

    results = []

    for _, eft_row in eft_df.iterrows():
        eft_id = eft_row["eft_id"]
        description = eft_row["description"]
        normalized_description = normalize_text(description)

        for _, alias_row in alias_df.iterrows():
            company_id = alias_row["company_id"]
            company_name = alias_row["company_name"]
            alias = alias_row["alias"]

            fuzzy_score = calculate_fuzzy_score(description, alias)
            acronym_score = calculate_acronym_score(description, company_name)
            rule_score = calculate_rule_score(description, company_name)

            final_score = calculate_final_score(
                fuzzy_score=fuzzy_score,
                acronym_score=acronym_score,
                rule_score=rule_score
            )

            risk_level = assign_risk_level(final_score)

            reason = build_reason(
                fuzzy_score=fuzzy_score,
                acronym_score=acronym_score,
                rule_score=rule_score
            )

            results.append({
                "eft_id": eft_id,
                "description": description,
                "normalized_description": normalized_description,
                "company_id": company_id,
                "company_name": company_name,
                "alias": alias,
                "fuzzy_score": round(fuzzy_score, 4),
                "acronym_score": round(acronym_score, 4),
                "rule_score": round(rule_score, 4),
                "final_score": final_score,
                "risk_level": risk_level,
                "reason": reason
            })

    result_df = pd.DataFrame(results)

    # Her EFT için en yüksek skorlu ilk 3 sonucu alalım
    result_df = result_df.sort_values(
        by=["eft_id", "final_score"],
        ascending=[True, False]
    )

    top_results = result_df.groupby("eft_id").head(3).reset_index(drop=True)

    return top_results


def main():
    eft_df, company_df = load_data()

    alias_df = prepare_company_aliases(company_df)

    print("Şirket alias kayıtları:")
    print(alias_df)
    print("-" * 80)

    result_df = run_matching(eft_df, alias_df)

    print("Eşleşme sonuçları:")
    print(result_df)

    result_df.to_csv("outputs/results.csv", index=False)
    print("Sonuç dosyası oluşturuldu: outputs/results.csv")


if __name__ == "__main__":
    main()