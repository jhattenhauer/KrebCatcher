# test_imports.py

packages = {
    "tensorflow": "tensorflow",
    "keras": "keras",
    "opencv-python": "cv2",
    "Pillow": "PIL",
    "matplotlib": "matplotlib"
}

for pkg, module in packages.items():
    try:
        exec(f"import {module}")
        print(f"[OK] {pkg} ({module}) imported successfully")
    except ModuleNotFoundError:
        print(f"[ERROR] {pkg} ({module}) is not installed")
    except Exception as e:
        print(f"[WARNING] {pkg} ({module}) import failed: {e}")
