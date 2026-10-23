# Tests

Run all suites with `scripts/run_all_tests.sh` (or `iree-coral test`). See
`docs/TESTING.md` for what each suite checks.

| file | hardware | purpose |
| --- | --- | --- |
| `test_mobilenet_v2_quant.py` | no | lossless quantisation: exact filters, exact activation quantisation, bit-exact per-operator outputs |
| `test_mobilenet_v2_hardware.py` | yes | end-to-end on the Edge TPU: one subgraph, 66/66 mapped, agreement with the original model |
| `paths.py` | — | test path resolution (delegates to `python/iree_coral/paths.py`, so it honours the same environment variables as `scripts/lib.sh`) |
| `test_data/` | — | fetched by `scripts/fetch_test_data.sh` |
