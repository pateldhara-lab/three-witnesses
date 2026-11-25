# Lookups

Small tables that the loaders and the build script read. Names of countries and ports are translated only through these files, never inside a script.

## countries.csv

One row per country or area that appears as a destination, a partner or an exporter in the sources.

| Column | Meaning |
|---|---|
| `iso3` | Three-letter code used in every output table (`EU27` for the European Union) |
| `name` | English name |
| `iso2` | Two-letter code. Read with `keep_default_na=False` so that Namibia's `NA` stays text |
| `comtrade_code` | UN Comtrade partner code |
| `project_role` | `exporter` for the five exporters, `importer` for China and the EU27, blank otherwise |
| `eurostat_partner` | Partner code in the Eurostat Comext file |
| `comex_brazil_name` | Country label in Brazil's ComexStat file |
| `sars_name` | Destination name in South Africa's SARS files |

## ports.csv

One row per port whose ship records are read from IMF PortWatch.

| Column | Meaning |
|---|---|
| `portid`, `portwatch_name`, `portwatch_url` | The port's identifier, name and page on the PortWatch portal |
| `country_iso3` | Exporter the port serves |
| `witness_side` | Which record the port belongs to (`exporter port`) |
| `in_country_ship_total` | `yes` when the port's tonnes belong in the exporter's ship total; `no` for Richards Bay, a coal port |
| `feeds_table_column` | `ship_t` when the port is part of the ship total in `side_by_side.csv`; `ship_extra_ports_t` for ports kept in a separate column (Port Walcott) |
| `sars_office` | Name of the customs office in South Africa's files, used to give each customs row a port |
| `lat`, `lon` | Port position |

## lag_start.csv

For each exporter and importer pair, the lag that sailing time alone suggests, in whole months.

| Column | Meaning |
|---|---|
| `country`, `importer` | Exporter and importer (`CHN` or `EU27`) |
| `start_low_months`, `start_high_months` | Range of lags the sailing time allows |
| `sailing_time` | Sailing time in weeks |
| `note` | The route and any reason the range is uncertain |

The build step fits a lag of 0 to 3 months for each pair and reports whether the fitted lag falls inside this range (`inside_start_range` in `Processed/lag_table.csv`). The ranges for Peru, Liberia and Australia are estimates from the route and not published figures.
