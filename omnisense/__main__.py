"""支持 `python -m omnisense ...`,与 console_script `omnisense` 等价。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
