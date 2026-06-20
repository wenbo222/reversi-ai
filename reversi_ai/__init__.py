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

    version="v0.1.3"
    dest_path=cache_dir/f"reversi_ai_{version}.dat"
    if not dest_path.exists():
        url=f"https://github.com/wenbo222/reversi-ai/releases/download/{version}/reversi_ai_{version}.dat"
        urlretrieve(url, dest_path)
        print(f"reversi_ai.dat transposition table ({version}) downloaded without issues.")
        
        # Clean up old data files to free up disk space
        for old_file in cache_dir.glob("reversi_ai*.dat"):
            if old_file!=dest_path:
                try:
                    old_file.unlink()
                except:
                    pass
    return str(dest_path)