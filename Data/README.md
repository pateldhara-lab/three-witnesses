# Data

Source files exactly as published or as extracted from the publisher's documents. Nothing in this folder is edited by the scripts in this repository: the loaders in `Pipeline/` read from here and write clean files to `Processed/by_source/`. Names of countries and ports are translated only through the tables in `Lookups/`.

Three records describe the same monthly flow of iron ore (heading 2601 of the Harmonized System, and its lines 260111 and 260112). Every folder below belongs to one record:

| Record | Folder | What it counts |
|---|---|---|
| Exporter | `Export/` | The exporting country's own statistics, at departure |
| Importers | `Import/` | China's and the EU27's customs records, on arrival |
| Ships | `Ship tracking/` | Cargo tonnes estimated from ship positions and draught at the exporter's ports |

## Exporter records

| Folder and file | Publisher and dataset | Link | Retrieved | Units as published | Coverage |
|---|---|---|---|---|---|
| `Export/Brazil/V_EXPORT_GENERAL_2016-01_2026-12_DT20261009.xlsx` | ComexStat, Ministry of Development, Industry, Trade and Services. General query, exports, heading 2601, by destination country | https://comexstat.mdic.gov.br/ | 2026-10-09 | Net weight in kg; value in US$ FOB | Jan 2016 to Sep 2026; one row per month and destination |
| `Export/South Africa/*.xls` (six files) | South African Revenue Service, trade statistics data download. Exports, chapter 26, tariff lines under 2601 | https://tools.sars.gov.za/tradestatsportal/data_download.aspx | 2026-10-09 | Quantity in kg; value in rand | Jan 2015 to Aug 2026; by customs office and destination. The `.xls` files are modern Excel files |
| `Export/Peru/by volume`, `Export/Peru/by value` | Central Reserve Bank of Peru (BCRP), statistical database, series PN38789BM (volume) and PN38788BM (value), traditional mining exports, iron | https://estadisticas.bcrp.gob.pe/ | 2026-10-09 | Million tonnes; million US$ | Jan 2012 to Aug 2026; Peru's total, no destination |
| `Export/Liberia/liberia_iron_ore_monthly.csv` | Central Bank of Liberia, Monthly Economic Review. Monthly table read from the issues of the review; one row per month and measure | https://www.cbl.org.lr/publications/document-type/monthly-economic-review | 2026-11-09 | Tonnes; million US$ | Jan 2019 to Aug 2026; Liberia's total, no destination. The file lists, for each month, every issue that prints it and the figure each issue prints |
| `Export/Australia/by tonnage/pilbara port/by_destination/*.pdf` | Pilbara Ports Authority, Port of Port Hedland monthly cargo statistics, by destination | https://www.pilbaraports.com.au/ports/port-of-port-hedland/ | 2026-10-09 | Tonnes | Jan 2019 to Aug 2026 |
| `Export/Australia/by tonnage/dampier port/` | Pilbara Ports Authority, Port of Dampier cargo statistics: annual files by financial year and monthly year-to-date files | https://www.pilbaraports.com.au/ports/port-of-dampier/ | 2026-10-09 | Tonnes | Financial years run July to June |

## Importer records

| Folder and file | Publisher and dataset | Link | Retrieved | Units as published | Coverage |
|---|---|---|---|---|---|
| `Import/China/china_imports_iron_ore_all_partners_long.csv` | UN Comtrade. Reporter China (code 156), imports, monthly, headings 2601, 260111 and 260112, all partners | https://comtradeplus.un.org/ | 2026-10-09 | Net weight in kg (`netWgt_kg`); value in US$ | Jan 2016 to Dec 2024. One row per month, partner and heading; the file includes the world total (partner code 0), so totals are made by adding partner rows |
| `Import/EU/eu_iron_ore_imports_long.csv` | Eurostat Comext, dataset DS-045409, EU trade since 1988 by HS2-4-6 and CN8. Reporter EU27_2020 and each member state | https://ec.europa.eu/eurostat/databrowser/view/DS-045409/default/table | 2026-10-09 | Quantity in units of 100 kg (`quantity_100kg`); value in EUR | Jan 2016 to Jul 2026; 27 member states and the EU27 aggregate; ten partners (the five exporters, other large suppliers and the world total). Licence CC BY 4.0 |

## Ship records

| Folder and file | Publisher and dataset | Link | Retrieved | Units as published | Coverage |
|---|---|---|---|---|---|
| `Ship tracking/portwatch/snapshot_2026-10-26/daily/` | IMF PortWatch, daily ports data (one file per port) | https://portwatch.imf.org/ | snapshot 2026-10-26 | Estimated cargo tonnes per day (the layer states no unit; we read it as metric tonnes) | Thirteen exporter ports, daily, 2019-01-01 to 2026-10-16 |
| `Ship tracking/portwatch/snapshot_2026-10-26/portwatch_monthly_selected_ports.csv` | The same ports added up by month (dry bulk and all cargo types, with the number of days present) | https://portwatch.imf.org/ | snapshot 2026-10-26 | Tonnes | Same thirteen ports, Jan 2019 to Oct 2026 |

PortWatch turns ship positions into port calls and a draught change into cargo tonnes. The portal did not state a licence for the data at retrieval.

## Things to know before reusing the files

- Exporter records for Peru and Liberia have no destination country, so they are checked against importers by total only.
- Brazil's destination is the declared destination, not necessarily where the ore is used.
- The 2026 values of the Peru series may be preliminary. The analysis window ends in December 2024.
- The terms of use of each source are set by its publisher. Check them before redistributing a file outside this repository.
