import pytest
from troubadour.release_integrity import write_manifest, verify_manifest


def test_manifest_detects_stale_cards_and_unlisted_files(tmp_path):
    card=tmp_path/'README.md';card.write_text('preview')
    cache=tmp_path/'.cache';cache.mkdir();(cache/'local').write_text('not published')
    write_manifest(tmp_path)
    assert verify_manifest(tmp_path)==1
    card.write_text('finished release')
    with pytest.raises(ValueError):verify_manifest(tmp_path)
    write_manifest(tmp_path)
    (tmp_path/'model.bin').write_bytes(b'weights')
    with pytest.raises(ValueError):verify_manifest(tmp_path)
