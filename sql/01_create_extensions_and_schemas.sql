-- 01_create_extensions_and_schemas.sql

-- 1. Yapisal Eklentiler (Extensions)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- 2. Yeni Veritabani Semalari (Schemas)
CREATE SCHEMA IF NOT EXISTS aml_source;
CREATE SCHEMA IF NOT EXISTS aml_stage;
CREATE SCHEMA IF NOT EXISTS aml_ml;
CREATE SCHEMA IF NOT EXISTS aml_core;
CREATE SCHEMA IF NOT EXISTS aml_config;
CREATE SCHEMA IF NOT EXISTS aml_audit;
