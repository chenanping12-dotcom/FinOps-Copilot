"""FinOps-Copilot 的展示层（Streamlit demo）。

只依赖 ``src/`` 的公开 API，不修改任何生产代码；核心逻辑是
``app/streamlit_app.py`` 中的纯函数，便于命令行 smoke test 直接调用。
"""