# Physical Model Consistency & Transferability Addendum v1.1

**Status: FROZEN POST-PROCESSING. No new fitting and no modification of the v1.0 benchmark.**

## What this addendum tests

- Mode A replaces Faiman temperature with measured module temperature while keeping every downstream parameter frozen.
- Mode B is the original frozen Faiman → De Soto → DC scalar → PVWatts chain.
- V/I decomposition evaluates the pre-scalar single-diode operating point, because the scalar is defined only on DC power and cannot be uniquely assigned to voltage or current.
- Diagnostic retention ratios are descriptive stability checks; they are never written back into the model.
- AC errors below are normalized by each array's DC nameplate rather than the common 1200 W inverter rating.

## Array-normalized frozen Mode B test metrics

| System | Observations | nMAE | nRMSE | Energy bias |
|---|---:|---:|---:|---:|
| T1-S1 | 7,094 | 3.447% | 7.342% | 2.122% |
| T1-S2 | 7,094 | 3.483% | 7.769% | 1.092% |
| T1-S3 | 7,094 | 5.405% | 8.199% | 3.449% |
| T1-S4 | 7,094 | 3.667% | 5.409% | -6.650% |
| T1-S5 | 7,094 | 3.551% | 7.103% | 5.717% |
| T1-S6 | 7,094 | 3.312% | 7.333% | 2.000% |

## Mode A versus Mode B on identical measured-temperature test rows

| System | Mode A nRMSE | Mode B nRMSE | A−B |
|---|---:|---:|---:|
| T1-S1 | 5.941% | 5.797% | +0.144 pp |
| T1-S2 | 5.685% | 5.546% | +0.139 pp |
| T1-S3 | 7.334% | 6.850% | +0.484 pp |
| T1-S4 | unavailable | unavailable | unavailable |
| T1-S5 | 5.569% | 5.364% | +0.205 pp |
| T1-S6 | 5.511% | 5.407% | +0.104 pp |

## Frozen DC-factor stability

| System | Frozen factor | Split range | Relative range | Status |
|---|---:|---:|---:|---|
| T1-S1 | 0.9128 | 0.8935–0.8991 | 0.62% | STABLE |
| T1-S2 | 0.9186 | 0.8992–0.9087 | 1.04% | STABLE |
| T1-S3 | 0.8336 | 0.7964–0.8008 | 0.53% | STABLE |
| T1-S4 | 0.5343 | 0.5109–0.5194 | 1.60% | STABLE |
| T1-S5 | 0.7947 | 0.7641–0.7774 | 1.67% | STABLE |
| T1-S6 | 0.8990 | 0.8864–0.8906 | 0.47% | STABLE |

## Transferability conclusion

The empirical DC retention scalar remains system-specific. Direct transfer to a new site or a different technology is not claimed, even when within-system stability is good.

| System | Observed-system use | Same-technology new site | Cross-technology | Flags |
|---|---|---|---|---|
| T1-S1 | SUPPORTED_OBSERVED_SYSTEM_DOMAIN | CONDITIONAL_NOT_VALIDATED | NOT_SUPPORTED | NONE |
| T1-S2 | SUPPORTED_OBSERVED_SYSTEM_DOMAIN | CONDITIONAL_NOT_VALIDATED | NOT_SUPPORTED | NONE |
| T1-S3 | SUPPORTED_OBSERVED_SYSTEM_DOMAIN | CONDITIONAL_NOT_VALIDATED | NOT_SUPPORTED | NONE |
| T1-S4 | CONDITIONAL_OBSERVED_SYSTEM_DOMAIN | CONDITIONAL_NOT_VALIDATED | NOT_SUPPORTED | NO_MEASURED_MODULE_TEMPERATURE;TRANSFERRED_FAIMAN_COEFFICIENTS;STRING_LEVEL_DC_CURRENT_REQUIRES_TOPOLOGY_MULTIPLIER;ARRAY_INVERTER_RATIO_OUTSIDE_NOMINAL_BAND |
| T1-S5 | SUPPORTED_OBSERVED_SYSTEM_DOMAIN | CONDITIONAL_NOT_VALIDATED | NOT_SUPPORTED | STRING_LEVEL_DC_CURRENT_REQUIRES_TOPOLOGY_MULTIPLIER |
| T1-S6 | SUPPORTED_OBSERVED_SYSTEM_DOMAIN | CONDITIONAL_NOT_VALIDATED | NOT_SUPPORTED | NONE |
