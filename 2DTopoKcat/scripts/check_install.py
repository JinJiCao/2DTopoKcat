

import importlib
import sys


PACKAGES = [
    "numpy", "pandas", "scipy", "sklearn", "rdkit", "torch",
    "torch_geometric", "torch_sparse", "transformers", "safetensors", "esm",
]


def main():
    failed = []
    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
            print(f"[OK] {package}: {getattr(module, '__version__', 'unknown')}")
        except Exception as exc:
            failed.append((package, str(exc)))
            print(f"[FAIL] {package}: {exc}")
    if failed:
        raise SystemExit(1)

    import models.net_esmc_multilayer_tdl_contact_adapter              
    import train_kcat              
    import train_km              
    print("[OK] Project modules import successfully")


if __name__ == "__main__":
    main()
