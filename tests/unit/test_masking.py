import pytest
from src.utils.masking import mask_pii

def test_mask_iban():
    text = "Ödeme için IBAN: TR330006100519786457841326 kullanıldı."
    masked = mask_pii(text)
    assert "TR3300061005****1326" in masked
    assert "1978645784" not in masked

def test_mask_tc_id():
    text = "Müşteri TC No: 12345678901 üzerinden sorgulandı."
    masked = mask_pii(text)
    assert "123****01" in masked
    assert "456789" not in masked

def test_mask_passport():
    text = "Yabancı uyruklu şahıs Pasaport: U1234567 ile geldi."
    masked = mask_pii(text)
    assert "U****" in masked
    assert "1234567" not in masked

def test_mask_ip():
    text = "Bağlantı adresi 192.168.1.100 olarak tespit edildi."
    masked = mask_pii(text)
    assert "192.168.****" in masked
    assert ".100" not in masked

def test_mask_empty():
    assert mask_pii("") == ""
    assert mask_pii(None) is None
