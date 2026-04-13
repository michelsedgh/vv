from __future__ import annotations

import os
import site
import sys
from typing import Any


def _normalize_path(path: str) -> str:
    if not path:
        path = os.getcwd()
    return os.path.realpath(os.path.abspath(path))


def _user_site_roots() -> set[str]:
    roots: set[str] = set()
    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = None

    if isinstance(user_site, str):
        roots.add(_normalize_path(user_site))
    elif isinstance(user_site, (list, tuple, set)):
        roots.update(_normalize_path(path) for path in user_site)

    py_user_base = os.environ.get("PYTHONUSERBASE")
    if py_user_base:
        roots.add(_normalize_path(py_user_base))

    return {root for root in roots if root}


def _strip_user_site() -> None:
    roots = _user_site_roots()
    if not roots:
        return

    cleaned: list[str] = []
    for entry in sys.path:
        normalized = _normalize_path(entry)
        if any(normalized == root or normalized.startswith(root + os.sep) for root in roots):
            continue
        cleaned.append(entry)

    sys.path[:] = cleaned
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    try:
        site.ENABLE_USER_SITE = False
    except Exception:
        pass


def _patch_huggingface_hub() -> None:
    try:
        import huggingface_hub as hf
    except Exception:
        return

    if not hasattr(hf, "ModelFilter"):
        class ModelFilter:
            def __init__(self, **kwargs: Any) -> None:
                for key, value in kwargs.items():
                    setattr(self, key, value)

        hf.ModelFilter = ModelFilter
        try:
            import huggingface_hub.hf_api as hf_api

            hf_api.ModelFilter = ModelFilter
        except Exception:
            pass

    if not getattr(hf.HfApi.list_models, "__voice_compat_patch__", False):
        original_list_models = hf.HfApi.list_models

        def compat_list_models(self: Any, *args: Any, **kwargs: Any):
            if "use_auth_token" in kwargs and "token" not in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")

            model_filter = kwargs.get("filter")
            if model_filter is not None and not isinstance(model_filter, (str, list, tuple, set, frozenset)):
                attr_map = {
                    "author": "author",
                    "library": "library",
                    "language": "language",
                    "model_name": "model_name",
                    "search": "search",
                    "tags": "tags",
                    "task": "task",
                    "trained_dataset": "trained_dataset",
                }
                for attr_name, kw_name in attr_map.items():
                    value = getattr(model_filter, attr_name, None)
                    if value not in (None, "", [], (), set(), frozenset()):
                        kwargs.setdefault(kw_name, value)
                kwargs["filter"] = None

            return original_list_models(self, *args, **kwargs)

        compat_list_models.__voice_compat_patch__ = True
        hf.HfApi.list_models = compat_list_models

    if not getattr(hf.hf_hub_download, "__voice_compat_patch__", False):
        original_download = hf.hf_hub_download

        def compat_hf_hub_download(*args: Any, **kwargs: Any):
            if "use_auth_token" in kwargs and "token" not in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            return original_download(*args, **kwargs)

        compat_hf_hub_download.__voice_compat_patch__ = True
        hf.hf_hub_download = compat_hf_hub_download


_strip_user_site()
_patch_huggingface_hub()
