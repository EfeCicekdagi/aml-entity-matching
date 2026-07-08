-- AML Projesi SQL Sorguları
-- İmleci çalıştırmak istediğiniz sorgunun üzerine getirip F5'e basabilir
-- veya sağ tıklayıp "Run Query" (Sorguyu Çalıştır) diyebilirsiniz.
 -- 1. Şüpheli transferleri risk skorlarına göre listeleyelim

SELECT eft_id,
       description,
       company_name,
       risk_level,
       final_score,
       reason
FROM suspicious_efts
ORDER BY final_score DESC
LIMIT 50;

-- 2. Tablodaki toplam şüpheli işlem sayısını görelim

SELECT COUNT(*) as toplam_supheli_islem
FROM suspicious_efts;

-- 3. Veritabanındaki şirket isimlerine göz atalım

SELECT *
FROM company_aliases
LIMIT 10;

-- 4. Hangi şirket için kaç adet şüpheli transfer olduğunu görelim

SELECT company_name,
       COUNT(*) as supheli_islem_sayisi,
       AVG(final_score) as ortalama_risk_skoru
FROM suspicious_efts
GROUP BY company_name
ORDER BY supheli_islem_sayisi DESC;

-- 5. Risk seviyelerine (HIGH, MEDIUM, vb.) göre dağılım

SELECT risk_level,
       COUNT(*) as islem_sayisi
FROM suspicious_efts
GROUP BY risk_level
ORDER BY islem_sayisi DESC;

-- 6. Belirli bir sebeple (örneğin "Fuzzy match" veya "High Vector Similarity") eşleşenleri bulalım

SELECT eft_id,
       description,
       company_name,
       reason,
       final_score
FROM suspicious_efts
WHERE reason LIKE '%Vector%'
ORDER BY final_score DESC
LIMIT 20;


SELECT eft_id,
       acronym_score
FROM suspicious_efts
WHERE acronym_score > 0
    SELECT Count(acronym_score)
    FROM suspicious_efts WHERE acronym_score > 0
    SELECT *
    FROM aml_scoring_weight_config