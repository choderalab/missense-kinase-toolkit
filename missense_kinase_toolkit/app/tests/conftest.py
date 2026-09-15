import sys
from pathlib import Path

# app.py uses bare imports (`import constants`, `import visualizers`) because
# streamlit runs it with its own directory on sys.path[0]. replicate that here
# so pytest / AppTest can import the app module and its siblings.
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
