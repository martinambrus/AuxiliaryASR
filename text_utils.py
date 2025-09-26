import csv
import logging
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd

from phoneme_inventory import UnifiedPhonemeMapper, merge_phoneme_configs


DEFAULT_DICT_PATH = os.path.join('word_index_dict.txt')
_DEFAULT_INVENTORY_PATH = os.path.join('Configs', 'unified_phoneme_inventory.csv')
_DEFAULT_MAPPING_PATH = os.path.join('Configs', 'language_token_mappings.yml')


class TextCleaner:
    """Convert token sequences to vocabulary indices."""

    logger = logging.getLogger(__name__)

    def __init__(self,
                 word_index_dict_path: Optional[str] = DEFAULT_DICT_PATH,
                 config: Optional[Mapping] = None):
        self.config = dict(config or {})
        self._tokenization_config = dict(self.config.get('tokenization', {}))
        self._keep_empty_tokens = bool(self._tokenization_config.get('keep_empty', False))

        # Determine which mode we are operating in.
        mode = self.config.get('mode')
        if mode is None:
            if self.config.get('enabled') is True:
                mode = 'unified'
            else:
                mode = 'legacy'
        mode = str(mode).lower()
        if mode not in {'legacy', 'unified'}:
            mode = 'legacy'
        self.mode = mode

        # Shared special tokens between both modes.
        self.special_tokens = OrderedDict(
            [("<pad>", 0), ("<sos>", 1), ("<eos>", 2), ("<unk>", 3), (" ", 4)]
        )

        if self.mode == 'unified':
            self._init_unified_mapper(word_index_dict_path)
        else:
            dict_path = self.config.get('dict_path', word_index_dict_path)
            self.word_index_dictionary = self.load_dictionary(dict_path)
            self.inverse_mapping = {index: word for word, index in self.word_index_dictionary.items()}
            self.mapper = None
            self._tokenization_config.setdefault('type', 'char')
            self._dynamic_vocabulary = bool(self.config.get('allow_dynamic_extension', False))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def __call__(self, text: str) -> list:
        if not isinstance(text, str):
            text = str(text)

        if self.mode == 'unified':
            tokens = self._tokenize(text)
            indexes = []
            for token in tokens:
                if not token and not self._keep_empty_tokens:
                    continue
                symbol = self.mapper.map_token(token)
                if symbol is None:
                    continue
                index = self.word_index_dictionary.get(symbol)
                if index is None:
                    if self.mapper.allow_dynamic_extension:
                        index = self._add_symbol(symbol)
                    else:
                        self.logger.warning("Symbol '%s' missing from vocabulary and dynamic extension disabled.", symbol)
                        continue
                indexes.append(index)
            return indexes

        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                self.logger.warning(
                    "(TextCleaner) Warning: Phoneme '%s' not found in dictionary. Text: %s",
                    char,
                    ''.join(text),
                )
        return indexes

    # ------------------------------------------------------------------
    # unified mode helpers
    # ------------------------------------------------------------------
    def _init_unified_mapper(self, default_dict_path: Optional[str]) -> None:
        inventory_path = self.config.get('inventory_path') or _DEFAULT_INVENTORY_PATH
        mapping_path = self.config.get('mapping_path') or _DEFAULT_MAPPING_PATH
        standard = self.config.get('standard', 'ipa')
        active_mappings = self.config.get('active_mappings')
        inline_mappings = self.config.get('inline_mappings')
        fallback_cfg = self.config.get('fallback')
        include_all = self.config.get('include_all_inventory_symbols', True)
        allow_dynamic = self.config.get('allow_dynamic_extension', False)
        enable_overrides = self.config.get('apply_language_mappings', True)
        case_sensitive = self.config.get('case_sensitive', True)

        self.mapper = UnifiedPhonemeMapper(
            inventory_path=inventory_path,
            standard=standard,
            mapping_path=mapping_path,
            active_mappings=active_mappings,
            inline_mappings=inline_mappings,
            fallback_config=fallback_cfg,
            include_all_symbols=include_all,
            special_tokens=self.special_tokens,
            allow_dynamic_extension=allow_dynamic,
            enable_language_overrides=enable_overrides,
            case_sensitive=case_sensitive,
        )

        # When using a unified mapper we always reference the internal dictionary
        # to keep the mapping consistent across datasets.
        self.word_index_dictionary = self.mapper.dictionary
        self.inverse_mapping = self.mapper.inverse_dictionary
        self._tokenization_config.setdefault('type', 'whitespace')
        self._dynamic_vocabulary = allow_dynamic

    def _tokenize(self, text: str) -> list:
        token_type = str(self._tokenization_config.get('type', 'whitespace')).lower()
        if token_type == 'regex':
            pattern = self._tokenization_config.get('pattern', r'\S+')
            tokens = re.findall(pattern, text)
        elif token_type == 'char' or token_type == 'character':
            tokens = list(text)
        elif token_type == 'auto':
            if any(ch.isspace() for ch in text):
                tokens = [tok for tok in text.split() if tok or self._keep_empty_tokens]
            else:
                tokens = list(text)
        else:  # whitespace splitting by default
            raw_tokens = text.split()
            if self._keep_empty_tokens:
                tokens = []
                last_index = 0
                for match in re.finditer(r'\S+', text):
                    if match.start() > last_index:
                        tokens.append('')
                    tokens.append(match.group())
                    last_index = match.end()
                if last_index < len(text):
                    tokens.append('')
            else:
                tokens = [tok for tok in raw_tokens if tok]
        return tokens

    def _add_symbol(self, symbol: str) -> int:
        if symbol not in self.word_index_dictionary:
            index = len(self.word_index_dictionary)
            self.word_index_dictionary[symbol] = index
            self.inverse_mapping[index] = symbol
            return index
        return self.word_index_dictionary[symbol]

    # ------------------------------------------------------------------
    # legacy helpers
    # ------------------------------------------------------------------
    def load_dictionary(self, path_or_dict: Optional[Mapping]) -> OrderedDict:
        """Load phoneme to index mapping from a path or return the given dict."""
        if isinstance(path_or_dict, Mapping):
            return OrderedDict(path_or_dict)

        path = path_or_dict or DEFAULT_DICT_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Dictionary file '{path}' was not found.")

        csv = pd.read_csv(path, header=None).values
        word_index_dict = OrderedDict((word, int(index)) for word, index in csv)
        return word_index_dict


def load_token_map_from_config(config: Mapping, *, return_cleaner: bool = False):
    """Return the token map specified by the global configuration."""
    dataset_cfg = dict(config.get('dataset_params', {}))
    text_cleaner_override = dataset_cfg.pop('text_cleaner', {}) if isinstance(dataset_cfg, Mapping) else {}
    phoneme_cfg = config.get('phoneme_settings', {}) or {}
    merged = merge_phoneme_configs(phoneme_cfg, text_cleaner_override)

    dict_path = merged.get('dict_path') or config.get('phoneme_maps_path')
    cleaner = TextCleaner(dict_path, config=merged)
    token_map = OrderedDict(cleaner.word_index_dictionary)
    if return_cleaner:
        return token_map, cleaner
    return token_map


def _parse_metadata_text(line: str) -> Optional[str]:
    """Return the phoneme sequence stored in a metadata line."""

    if not line:
        return None
    cleaned_line = line.rstrip('\n')
    if not cleaned_line.strip():
        return None

    if '|' not in cleaned_line:
        return cleaned_line.strip()

    parts = cleaned_line.split('|')
    if len(parts) < 2:
        return None
    if len(parts) == 2:
        return parts[1].strip()
    text = '|'.join(parts[1:-1])
    return text.strip()


def _collect_symbols_from_text(cleaner: TextCleaner, text: str) -> Iterable[str]:
    """Yield phoneme symbols appearing in ``text`` according to the cleaner."""

    if cleaner.mode == 'unified':
        tokens = cleaner._tokenize(text)  # pylint: disable=protected-access
        for token in tokens:
            if not token and not cleaner._keep_empty_tokens:
                continue
            symbol = cleaner.mapper.map_token(token) if cleaner.mapper else token
            if symbol is None:
                continue
            yield symbol
        return

    for char in text:
        if not char:
            continue
        yield char


def extend_token_map_from_metadata(
    token_map: Mapping[str, int],
    cleaner: TextCleaner,
    metadata_paths: Iterable[Optional[str]],
) -> Tuple[OrderedDict, Sequence[str]]:
    """Extend ``token_map`` by scanning dataset metadata files."""

    logger = logging.getLogger(__name__)
    working_map = OrderedDict(token_map)
    new_symbols = []
    seen_new = set()

    for metadata_path in metadata_paths:
        if not metadata_path:
            continue
        path = Path(metadata_path)
        if not path.is_file():
            logger.warning("Metadata file '%s' was not found while extending the phoneme map.", metadata_path)
            continue

        with path.open('r', encoding='utf-8') as handle:
            for raw_line in handle:
                text = _parse_metadata_text(raw_line)
                if text is None:
                    continue
                for symbol in _collect_symbols_from_text(cleaner, text):
                    if symbol not in working_map:
                        # Ensure legacy mode dictionaries are updated alongside
                        # the working copy so downstream components operate on
                        # the extended vocabulary.
                        if cleaner.mode != 'unified':
                            if symbol not in cleaner.word_index_dictionary:
                                index = len(cleaner.word_index_dictionary)
                                cleaner.word_index_dictionary[symbol] = index
                                cleaner.inverse_mapping[index] = symbol
                            index = cleaner.word_index_dictionary[symbol]
                        else:
                            index = cleaner.word_index_dictionary.get(symbol)
                            if index is None:
                                index = len(cleaner.word_index_dictionary)
                                cleaner.word_index_dictionary[symbol] = index
                                cleaner.inverse_mapping[index] = symbol
                        working_map[symbol] = index
                        if symbol not in seen_new:
                            new_symbols.append(symbol)
                            seen_new.add(symbol)

    ordered = OrderedDict(sorted(working_map.items(), key=lambda item: item[1]))
    return ordered, new_symbols


def save_token_map_to_file(token_map: Mapping[str, int], destination: str) -> None:
    """Persist ``token_map`` to ``destination`` using CSV formatting."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        for symbol, index in sorted(token_map.items(), key=lambda item: item[1]):
            writer.writerow([symbol, int(index)])


def load_token_map_file(path: str) -> OrderedDict:
    """Load a phoneme map from ``path``."""

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Token map file '{path}' was not found.")

    data = pd.read_csv(path, header=None)
    entries = []
    for row in data.itertuples(index=False):
        if len(row) < 2:
            continue
        symbol = str(row[0])
        index = int(row[1])
        entries.append((symbol, index))
    entries.sort(key=lambda item: item[1])
    return OrderedDict(entries)


def ensure_token_map_matches_reference(token_map: Mapping[str, int], reference_path: str) -> None:
    """Validate that ``token_map`` matches the stored reference mapping."""

    path = Path(reference_path)
    if not path.is_file():
        return

    reference = load_token_map_file(str(path))
    working = OrderedDict(sorted(token_map.items(), key=lambda item: item[1]))
    if reference != working:
        raise ValueError(
            "Phoneme map mismatch detected. The reference map at "
            f"'{reference_path}' does not match the phoneme map built from the current dataset. "
            "Please verify you are using the intended dataset or update/remove the reference map before training."
        )
