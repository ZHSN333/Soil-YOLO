"""兼容入口：请使用 `python main.py val`。"""

import sys

from main import main

if __name__ == "__main__":
    sys.argv = ["main.py", "val", *sys.argv[1:]]
    main()
