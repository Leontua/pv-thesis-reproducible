# Parametric PV Technology × Capacity × Climate Study v1.0

## Frozen design

- 144 deterministic scenarios: 6 technology archetypes × 4 DC capacities × 6 representative Köppen climates.
- PVGIS 5.3 TMY (`PVGIS-SARAH3`, source window recorded per location), normalized to a non-leap synthetic year.
- Fixed 30° equator-facing plane, Perez transposition, albedo 0.20.
- Fixed Faiman U0=25.0 W/m²K and U1=6.84 W·s/m³K for every scenario.
- De Soto/CEC module physics uses only the frozen datasheet-derived parameter archetypes T1-S1…T1-S6.
- A common non-fitted downstream DC loss of 14.0% and DC/AC=1.20 are applied identically.
- No ETSIDI calibrated Faiman coefficient, DC retention scalar, or other empirical scalar is used.

## Headline results

- Specific yield range: **1001.7–2079.5 kWh/kWp/year**.
- Highest cell: **T1-S4 / BWh**, 2079.5 kWh/kWp/year.
- Lowest cell: **T1-S6 / Cfb**, 1001.7 kWh/kWp/year.
- Thermal loss range versus a 25 °C counterfactual: **0.19% to 12.31%**.
- Clipping loss range: **0.00% to 0.09%**.
- Maximum capacity-driven specific-yield discrepancy: **6.821e-13 kWh/kWp/year** (expected numerical zero under fixed DC/AC and losses).

All annual energy quantities are direct hourly sums. Capacity changes absolute MWh but, by construction, not normalized yield, thermal percentage, or clipping percentage.
