-- 23_enforce_append_only_history.sql
-- aml_audit.alert_status_history tablosunda UPDATE ve DELETE işlemlerini veritabanı seviyesinde yasaklayan tetikleyici (trigger).

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

COMMENT ON FUNCTION aml_audit.prevent_update_delete_history() IS
    'Enforces append-only security policy on audit status history tables.';
