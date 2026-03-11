# AI Agent Operational Notes

## Issues & Gotchas

### VectorBT Name Collision
- **Error**: `AttributeError: module 'vectorbt' has no attribute 'Portfolio'`
- **Cause**: There is a local directory `src/vectorbt` which shadows the installed `vectorbt` library. If `src` is in `sys.path` (which it often is), `import vectorbt` will resolve to the local directory instead of the library.
- **Solution**: 
    1. **Always** run scripts using the project's virtual environment python executable: `.\.venv\Scripts\python.exe`.
    2. Be careful with `sys.path.insert(0, ...)` calls in scripts, as they can prioritize the local `src/vectorbt` over the library.
- **Context**: The `vectorbt` library is used for backtesting (e.g., `vbt.Portfolio.from_signals`), while the local `src/vectorbt` contains project-specific pipeline and UI code.

---

## Technical Environment Advice

- **Location**: Use the virtual environment located at `c:\dev\cursor\e-trading\.venv\`.
- **Scripts**: When executing scripts, explicitly use the `.venv` path to avoid package shadowing and environment mismatches.
