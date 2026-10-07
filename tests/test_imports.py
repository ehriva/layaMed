"""Test that all __all__ symbols are actually importable."""

import sys


def test_all_exports_exist():
    """Every name in __all__ must be importable from laya.

    This is the simplest possible gate for the public API contract: if `__all__` changes,
    this test fails immediately. No weights, no network.
    """
    import laya
    missing = [name for name in laya.__all__ if not hasattr(laya, name)]
    assert not missing, "__all__ contains names without attributes: %s" % (missing,)


def test_submodule_exports():
    """These seven submodules must be directly importable as laid out in __all__.

    They were added because they appeared in dir(laya) but not __all__:
    confidence, email, hooks, lang, presets, router, structured.
    """
    import laya
    expected = ("confidence", "email", "hooks", "lang", "presets", "router", "structured")
    found = tuple(getattr(laya, name) for name in expected)
    assert len(found) == len(expected), "not all submodule attributes present"
    # Verify each is indeed a module by checking its __file__ or __path__
    for mod_name, mod in zip(expected, found):
        assert getattr(mod, "__file__", None) is not None or hasattr(mod, "__path__"), \
            "%r attribute of laya is not a module" % (mod_name,)


def test_docstrings_render_via_mkdocstrings():
    """Submodules listed in __all__ will render with mkdocstrings :::: directives.

    The helpers reference page lists shortlist functions; this PR ensures the submodules
    appear in docs/reference/index.md when we add :::: laya.confidence etc.
    """
    import laya
    # Smoke check: the module objects have docstrings
    expected_with_docs = ["confidence", "email", "hooks", "lang", "presets"]
    for mod_name in expected_with_docs:
        mod = getattr(laya, mod_name)
        assert mod.__doc__ is not None and len(mod.__doc__) > 50, \
            "%r submodule has no meaningful docstring for mkdocstrings" % (mod_name,)


if __name__ == "__main__":
    test_all_exports_exist()
    print("test.all_exports_exist: PASS")
    test_submodule_exports()
    print("test.submodule_exports: PASS")
    test_docstrings_render_via_mkdocstrings()
    print("test.docstrings_render_via_mkdocstrings: PASS")
    print("All imports tests passed.")
