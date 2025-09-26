from __future__ import annotations

import logging
import math
import os
from collections import OrderedDict
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import pandas as pd
import yaml


_LOGGER = logging.getLogger(__name__)


def _is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _normalize_symbol(value) -> str:
    if value is None:
        return ""
    if _is_nan(value):
        return ""
    text = str(value)
    return text


def _deep_update(base: Mapping, override: Optional[Mapping]) -> Dict:
    result = dict(base or {})
    if not override:
        return result
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


class UnifiedPhonemeMapper:
    """Create and manage a unified multilingual phoneme inventory.

    Parameters
    ----------
    inventory_path:
        Path to a CSV file describing the canonical inventory. The file must
        contain the columns ``canonical``, ``ipa`` and ``xsampa``.  Additional
        columns are ignored.
    standard:
        Either ``"ipa"`` or ``"xsampa"`` indicating which representation to use
        when emitting phoneme symbols.
    mapping_path:
        Optional YAML file containing language specific token mappings.  The
        file should map group names to ``token -> canonical`` dictionaries.
    active_mappings:
        Sequence of mapping group names to load from ``mapping_path``.  When not
        provided, all groups present in the YAML file are used.
    inline_mappings:
        Extra mapping definitions embedded directly in the configuration.
    apply_mappings:
        Whether mappings should be applied at all.  This enables toggling the
        language specific layer without removing the configuration from disk.
    fallback_config:
        Dictionary with keys controlling how unmapped tokens are handled.
        Supported keys are ``on_missing_mapping`` (``"warn"`` / ``"error"`` /
        ``"keep"``), ``default_symbol`` and ``allow_identity``.
    include_all_symbols:
        When ``True`` (default) every canonical entry from the inventory is
        included in the resulting dictionary.  Disable this if you prefer to
        grow the vocabulary lazily based on the mappings actually used.
    special_tokens:
        Ordered mapping of special symbols that should be seeded into the
        vocabulary before phoneme symbols are added.
    allow_dynamic_extension:
        Allow new symbols to be added to the dictionary at runtime when the
        fallback behaviour is set to ``"keep"``.
    enable_language_overrides:
        Toggle the use of mapping groups entirely while still relying on the
        canonical inventory.
    case_sensitive:
        Whether token lookups should be case sensitive (default: ``True``).
    """

    DEFAULT_SPECIAL_TOKENS: OrderedDict[str, int] = OrderedDict(
        [("<pad>", 0), ("<sos>", 1), ("<eos>", 2), ("<unk>", 3), (" ", 4)]
    )

    def __init__(
        self,
        *,
        inventory_path: str,
        standard: str = "ipa",
        mapping_path: Optional[str] = None,
        active_mappings: Optional[Iterable[str]] = None,
        inline_mappings: Optional[Mapping[str, Mapping[str, str]]] = None,
        apply_mappings: bool = True,
        fallback_config: Optional[Mapping] = None,
        include_all_symbols: bool = True,
        special_tokens: Optional[MutableMapping[str, int]] = None,
        allow_dynamic_extension: bool = False,
        enable_language_overrides: bool = True,
        case_sensitive: bool = True,
    ) -> None:
        if not inventory_path:
            raise ValueError("inventory_path must be provided when using the unified phoneme mapper")

        self.standard = (standard or "ipa").lower()
        if self.standard not in {"ipa", "xsampa"}:
            raise ValueError(f"Unsupported phoneme standard '{standard}'. Choose from 'ipa' or 'xsampa'.")

        self.inventory_path = inventory_path
        self.inventory = self._load_inventory(inventory_path)
        self.case_sensitive = bool(case_sensitive)

        self.special_tokens = OrderedDict(special_tokens or self.DEFAULT_SPECIAL_TOKENS)
        self.dictionary: OrderedDict[str, int] = OrderedDict(self.special_tokens)
        self.inverse_dictionary: Dict[int, str] = {idx: token for token, idx in self.dictionary.items()}

        self.allow_dynamic_extension = bool(allow_dynamic_extension)
        self.enable_language_overrides = bool(enable_language_overrides)
        self.apply_mappings = bool(apply_mappings)

        fallback_config = dict(fallback_config or {})
        self.fallback_mode = fallback_config.get("on_missing_mapping", "warn").lower()
        if self.fallback_mode not in {"warn", "error", "keep"}:
            self.fallback_mode = "warn"
        self.default_symbol = fallback_config.get("default_symbol", "<unk>")
        self.allow_identity = bool(fallback_config.get("allow_identity", True))
        self.warned_tokens = set()

        if self.default_symbol and self.default_symbol not in self.dictionary:
            self._add_symbol(self.default_symbol)

        if include_all_symbols:
            self._register_inventory_symbols()

        self.token_to_canonical: Dict[str, str] = {}
        self.token_to_symbol: Dict[str, str] = {}

        if self.enable_language_overrides and self.apply_mappings:
            mapping_groups = self._load_mapping_groups(mapping_path)
            self._register_mappings(mapping_groups, active_mappings)
        else:
            mapping_groups = {}

        if inline_mappings:
            inline_groups = {
                name: {str(k): str(v) for k, v in mapping.items()}
                for name, mapping in inline_mappings.items()
                if isinstance(mapping, Mapping)
            }
            self._register_mappings(inline_groups, active_mappings)

        # Provide direct canonical access for tokens that already match a
        # canonical entry even when no mapping is explicitly provided.
        for canonical in self.inventory:
            token_key = self._normalise_token(canonical)
            if token_key not in self.token_to_canonical:
                self.token_to_canonical[token_key] = canonical

    # ------------------------------------------------------------------
    # loading helpers
    # ------------------------------------------------------------------
    def _load_inventory(self, path: str) -> Dict[str, Dict[str, str]]:
        data = pd.read_csv(path)
        required = {"canonical", "ipa", "xsampa"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"Unified phoneme inventory '{path}' is missing required columns: {', '.join(sorted(missing))}"
            )

        inventory: Dict[str, Dict[str, str]] = {}
        for _, row in data.iterrows():
            canonical = _normalize_symbol(row.get("canonical"))
            if not canonical:
                continue
            entry = {
                "canonical": canonical,
                "ipa": _normalize_symbol(row.get("ipa")),
                "xsampa": _normalize_symbol(row.get("xsampa")),
            }
            inventory[canonical] = entry
        return inventory

    def _load_mapping_groups(self, mapping_path: Optional[str]) -> Dict[str, Dict[str, str]]:
        if not mapping_path:
            return {}
        if not os.path.isfile(mapping_path):
            _LOGGER.warning("Phoneme mapping file '%s' not found. Skipping language overrides.", mapping_path)
            return {}
        with open(mapping_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        mapping_groups = {}
        for group_name, mapping in loaded.items():
            if not isinstance(mapping, Mapping):
                continue
            mapping_groups[group_name] = {str(token): str(canonical) for token, canonical in mapping.items()}
        return mapping_groups

    def _register_mappings(
        self,
        mapping_groups: Mapping[str, Mapping[str, str]],
        active_mappings: Optional[Iterable[str]],
    ) -> None:
        if not mapping_groups:
            return
        if active_mappings is None:
            selected_groups = mapping_groups.keys()
        else:
            selected_groups = []
            for group in active_mappings:
                if group in mapping_groups:
                    selected_groups.append(group)
                else:
                    _LOGGER.warning("Requested phoneme mapping group '%s' was not found in '%s'.", group, self.inventory_path)
        for group in selected_groups:
            mapping = mapping_groups.get(group)
            if not mapping:
                continue
            for token, canonical in mapping.items():
                normalised = self._normalise_token(token)
                self.token_to_canonical[normalised] = canonical

    # ------------------------------------------------------------------
    # dictionary management
    # ------------------------------------------------------------------
    def _register_inventory_symbols(self) -> None:
        for canonical in self.inventory:
            symbol = self._canonical_to_symbol(canonical)
            if symbol:
                self._add_symbol(symbol)

    def _canonical_to_symbol(self, canonical: str) -> str:
        entry = self.inventory.get(canonical)
        if entry is None:
            return str(canonical)
        symbol = _normalize_symbol(entry.get(self.standard))
        if not symbol:
            fallback_column = "ipa" if self.standard == "xsampa" else "xsampa"
            symbol = _normalize_symbol(entry.get(fallback_column))
        if not symbol:
            symbol = str(entry.get("canonical", canonical))
        return symbol

    def _normalise_token(self, token: str) -> str:
        if token is None:
            return ""
        token = str(token)
        if not self.case_sensitive:
            token = token.lower()
        return token

    def _add_symbol(self, symbol: str) -> int:
        if symbol not in self.dictionary:
            index = len(self.dictionary)
            self.dictionary[symbol] = index
            self.inverse_dictionary[index] = symbol
        return self.dictionary[symbol]

    # ------------------------------------------------------------------
    # token mapping API
    # ------------------------------------------------------------------
    def map_token(self, token: str, *, auto_extend: Optional[bool] = None) -> Optional[str]:
        if token in self.special_tokens:
            return token
        normalised = self._normalise_token(token)
        if not normalised:
            return None

        canonical = self.token_to_canonical.get(normalised)
        if canonical is None and self.allow_identity and normalised in self.inventory:
            canonical = normalised

        if canonical is None:
            if auto_extend is None:
                auto_extend = self.allow_dynamic_extension
            return self._handle_missing_mapping(token, auto_extend=bool(auto_extend))

        symbol = self._canonical_to_symbol(canonical)
        self._add_symbol(symbol)
        self.token_to_symbol[normalised] = symbol
        return symbol

    def _handle_missing_mapping(self, token: str, *, auto_extend: bool = False) -> Optional[str]:
        behaviour = self.fallback_mode
        if behaviour == "error":
            raise KeyError(f"Token '{token}' has no mapping in the unified phoneme inventory")

        if behaviour == "keep":
            symbol = str(token)
            index = self.dictionary.get(symbol)
            if index is None:
                if not (auto_extend or self.allow_dynamic_extension):
                    if token not in self.warned_tokens:
                        _LOGGER.warning(
                            "Encountered unmapped token '%s' but dynamic extension is disabled; falling back to default symbol '%s'.",
                            token,
                            self.default_symbol,
                        )
                        self.warned_tokens.add(token)
                    return self.default_symbol if self.default_symbol in self.dictionary else None
                self._add_symbol(symbol)
            elif auto_extend and symbol not in self.dictionary:
                self._add_symbol(symbol)
            return symbol

        if auto_extend:
            symbol = str(token)
            if symbol not in self.dictionary:
                self._add_symbol(symbol)
            normalised = self._normalise_token(token)
            self.token_to_symbol[normalised] = symbol
            if token not in self.warned_tokens and behaviour != "silent":
                _LOGGER.warning("Token '%s' was not found in the phoneme mapping. Automatically adding it to the vocabulary.", token)
                self.warned_tokens.add(token)
            return symbol

        if token not in self.warned_tokens and behaviour != "silent":
            _LOGGER.warning(
                "Token '%s' was not found in the phoneme mapping. Using default symbol '%s'.",
                token,
                self.default_symbol,
            )
            self.warned_tokens.add(token)
        if self.default_symbol in self.dictionary:
            return self.default_symbol
        return None

    # ------------------------------------------------------------------
    # configuration helpers
    # ------------------------------------------------------------------
    def export_config(self) -> Dict:
        """Return a dictionary summarising the mapper configuration."""
        return {
            "inventory_path": self.inventory_path,
            "standard": self.standard,
            "fallback_mode": self.fallback_mode,
            "default_symbol": self.default_symbol,
            "dictionary_size": len(self.dictionary),
        }


def merge_phoneme_configs(*configs: Optional[Mapping]) -> Dict:
    """Merge multiple configuration dictionaries into a single mapping."""
    merged: Dict = {}
    for cfg in configs:
        merged = _deep_update(merged, cfg or {})
    return merged
