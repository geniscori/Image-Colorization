import os

def create_result_folders():
    # Separem en diferents fitxers
    os.makedirs('outputs/color', exist_ok=True)
    os.makedirs('outputs/gray', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('test/gray', exist_ok=True)
    os.makedirs('test/color', exist_ok=True)