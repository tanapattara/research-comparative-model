# Amendment verification environment

This is the runtime used to run the unit-test suite while preparing the amended
commit. The original E4/E5 manifests did not record a Python executable or a
complete package/OS snapshot, so this file must not be interpreted as proof of
the historical training environment.

- `where.exe python`: `C:\Users\tanap\AppData\Local\Microsoft\WindowsApps\python.exe`
  (WindowsApps alias; not used for verification)
- `where.exe py`: no result
- Verification executable:
  `C:\Users\tanap\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`
- Python: `3.12.13 (main, Aug 7 2026, 02:26:41) [MSC v.1944 64 bit (AMD64)]`
- Platform: `Windows-11-10.0.26200-SP0`
- NumPy: `2.3.5`
- pandas: `3.0.1`
- scikit-learn: `1.9.0`
- PyYAML: `6.0.3`
- SciPy: `1.18.0`
- PyTorch: `2.13.0`
- Test command: `-m unittest discover -s tests -v` with `PYTHONPATH=src`
- Test result: `Ran 41 tests in 8.213s` — `OK`

- Optuna: `4.9.0`
