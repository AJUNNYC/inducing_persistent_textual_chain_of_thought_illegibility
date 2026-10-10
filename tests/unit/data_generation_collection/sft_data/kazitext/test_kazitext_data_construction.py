import pytest
from data_generation_collection.sft_data.kazitext.kazitext_data_construction import ALPHAS, PROFILE_SHA256, SETTINGS, changed_word_fraction, protected_spans, split_protected, OfficialKaziText


def test_literal_grid_and_all_switches():
    assert ALPHAS==('0.200','0.301','0.454','0.684','1.031','1.553','2.339','3.524','5.310','8.000')
    assert len(PROFILE_SHA256)==64
    assert SETTINGS['no_word_order'] is True and SETTINGS['alpha_std']==0
    assert SETTINGS['beta']==0 and SETTINGS['spelling_detailed_ratio']==.3
    assert not any(SETTINGS[k] for k in ('no_diacritics','no_spelling','no_casing','no_whitespace','no_punctuation','no_suffix_prefix','no_common_other'))


def test_protection_splits_exact_no_corruptible_sentinels():
    text=r'First x = 2 + 3 gives $\\frac{5}{2}$ and 1,234.50 percent. Then prose.'
    pieces=split_protected(text)
    assert ''.join(p['text'] for p in pieces)==text
    protected=''.join(p['text'] for p in pieces if p['protected'])
    assert 'x = 2 + 3' in protected and '1,234.50' in protected and '$' in protected
    assert any(not p['protected'] and 'prose' in p['text'] for p in pieces)
    spans=protected_spans(text)
    assert all(a<b for a,b in spans)
    assert all(spans[i][1]<spans[i+1][0] for i in range(len(spans)-1))
    boundary = split_protected('We get x = 12 + 12. The expression follows.')
    assert any(p['protected'] and p['text'] == ' x = 12 + 12. ' for p in boundary)


def test_measurement_is_logged_not_quality_filter():
    assert changed_word_fraction('one two three','one two three')==0
    assert changed_word_fraction('one two three','one too three')==pytest.approx(1/3)
    assert changed_word_fraction('one','many new replacements')==1


def test_missing_external_source_fails_explicitly(tmp_path):
    with pytest.raises(Exception):
        OfficialKaziText(tmp_path/'missing',tmp_path/'english-ewt-ud-2.5-191206.udpipe',aspell_dictionary_files=[])


def test_cached_udpipe_tampering_is_fatal(tmp_path):
    from data_generation_collection.sft_data.kazitext.kazitext_data_construction import provision_udpipe_model, UDPIPE_MODEL
    path = tmp_path / UDPIPE_MODEL
    path.write_bytes(b'fake binary must not be accepted or silently replaced')
    with pytest.raises(ValueError, match='identity mismatch'):
        provision_udpipe_model(path)
    assert path.read_bytes().startswith(b'fake binary')


def test_aspell_recursive_dictionary_identity_ignores_local_root(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from data_generation_collection.sft_data.kazitext.kazitext_data_construction import discover_aspell_dictionary
    root = tmp_path / 'runtime'
    root.mkdir()
    files = {'en.multi': 'add nested.multi\n', 'nested.multi': 'add en-common.rws\n',
             'en-common.rws': 'actual fixture bytes', 'en.dat': 'charset iso8859-1\naffix en\nsoundslike en\n',
             'en_affix.dat': 'affix', 'en_phonet.dat': 'phonetic', 'iso-8859-1.cmap': 'map',
             'iso-8859-1.cset': 'set', 'standard.kbd': 'keyboard'}
    for name, text in files.items():
        (root / name).write_text(text)
    def config():
        values = {key: str(root) for key in ('actual-dict-dir', 'dict-dir', 'data-dir', 'local-data-dir')}
        values.update({'master-path': str(root / 'en.multi'), 'actual-lang': 'en', 'keyboard': 'standard',
                       'encoding': 'UTF-8', 'sug-mode': 'normal', 'sug-typo-analysis': True,
                       'sug-repl-table': True, 'sug-split-char': [' ', '-'], 'ignore': 1,
                       'ignore-accents': False, 'ignore-case': False, 'use-other-dicts': False,
                       'normalize': True, 'norm-form': 'nfc', 'extra-dicts': [], 'wordlists': []})
        return {key: ('fixture', value, '') for key, value in values.items()}
    monkeypatch.setitem(sys.modules, 'aspell', SimpleNamespace(Speller=lambda *args: SimpleNamespace(
        ConfigKeys=config, check=lambda text: True, suggest=lambda text: ['English'])))
    first = discover_aspell_dictionary()
    assert len(first['files']) == len(files)
    moved = tmp_path / 'moved'
    root.rename(moved)
    root = moved
    second = discover_aspell_dictionary()
    assert first['identity_hash'] == second['identity_hash']
    (root / 'en-common.rws').write_bytes(b'changed')
    assert discover_aspell_dictionary()['identity_hash'] != first['identity_hash']
    (root / 'en_affix.dat').unlink()
    with pytest.raises(FileNotFoundError):
        discover_aspell_dictionary()
