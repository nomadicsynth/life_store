# Medical cannabis logistics CLI

Track periodic package weigh-ins, estimate average usage, and forecast reorder dates using a file-first, dependency-free workflow.

- Storage: `therapeutics/cannabis/Package_<ID>/`
  - `product.json` — package metadata
  - `weighins/WeighIn_<YYYYMMDDThhmmssZ>.json` — individual weigh-in entries
- Env override: `THERAPEUTICS_CANNABIS_DIR` can change the base path.
- CLI exposes `main(argv)` for tests and can be executed as a script.

## Subcommands

- init
  - Required: `--id`, `--name`, `--form`, `--lead-time-days`, `--safety-stock-days`
  - Optional: `--tare-g`, `--initial-gross-g`, `--thc-percent`, `--cbd-percent`, `--force`, `--base`
- weigh
  - Required: `--id`, `--gross-g`
  - Optional: `--timestamp`, `--note`, `--base`
  - Validation: if the new gross package weight is greater than the last known gross at or before the given timestamp, the entry is rejected (exit 1).
- report
  - Required: `--id`
  - Optional: `--as-of`, `--json`, `--base`

- list
  - Lists all packages under the base directory, with current status
  - Options: `--json`, `--base`

- check
  - Checks reorder status for a single package (`--id`) or all packages if omitted
  - Exit code: 1 if any package needs reorder, 0 otherwise
  - Options: `--id`, `--json`, `--base`

## Example

```bash
# Initialize a flower package (tare 10g jar, initial gross 28g)
python cannabis_logistics.py init \
  --id jar_oct \
  --name "October Jar" \
  --form flower \
  --tare-g 10 \
  --initial-gross-g 28 \
  --lead-time-days 5 \
  --safety-stock-days 2

# Record a weigh-in at 37.2g gross (net 27.2g)
python cannabis_logistics.py weigh --id jar_oct --gross-g 37.2

# Get a JSON report
python cannabis_logistics.py report --id jar_oct --json

# List all packages
python cannabis_logistics.py list --json

# Check reorder status (all packages). Non-zero exit means action needed.
python cannabis_logistics.py check
```

## Notes

- Usage rate is computed from consecutive points (initial+weigh-ins) using the median rate for robustness.
- If usage cannot be determined (not enough data), reorder fields are `null` and `reorder_now` is `false`.
- Reorder date = depletion date − (lead_time_days + safety_stock_days). If that date is in the past, `reorder_now` is `true`.
