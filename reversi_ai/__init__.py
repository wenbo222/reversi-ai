import os
from urllib.request import urlretrieve
from pathlib import Path
from ._reversi_ai import *

def get_data_path():
    """
    Return the absolute path of the pre-packaged reversi_ai.dat file.
    """
    
    cache_dir=Path(os.path.expanduser("~"))/".reversi_ai"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dest_path=cache_dir/"reversi_ai.dat"
    if not dest_path.exists():
        url="https://github.com/wenbo222/reversi-ai/releases/download/v0.0.0/reversi_ai.dat"
        urlretrieve(url, dest_path)
        print("reversi_ai.dat transposition table downloaded without issues.")
    return str(dest_path)