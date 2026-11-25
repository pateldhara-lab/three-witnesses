"""
Australia exporter: Pilbara Ports PDFs -> australia_exporter.csv
Port Hedland (by-destination PDFs) and Dampier (financial year tables)
Needs pdfplumber to run
"""
import re
from pathlib import Path
import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]  
RAW_DIR = ROOT / "Data" / "Export" / "Australia" / "by tonnage"
HEDLAND_DIR = RAW_DIR / "pilbara port" / "by_destination"
DAMPIER_DIR = RAW_DIR / "dampier port"
OUT = ROOT / "Processed" / "by_source" / "australia_exporter.csv"

RETRIEVED = "2026-10-09"
FIRST_MONTH = pd.Timestamp("2019-01-01")
NUMBER = re.compile(r"^[\d,]+\.\d\d$")
MONTHS = {"JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4, "MAY": 5,
          "JUNE": 6, "JULY": 7, "AUGUST": 8, "SEPTEMBER": 9,
          "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12}

rows = []
problems = []


# Port Hedland 
for path in sorted(HEDLAND_DIR.glob("*.pdf")):
    month = pd.Timestamp(path.stem + "-01")
    if month < FIRST_MONTH:
        continue

    # loading page is the one with the word LOAD on it
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            words = page.extract_words()
            if any(w["text"].upper() == "LOAD" for w in words):
                break

    # Group the words into lines left to right
    lines = []
    for w in sorted(words, key=lambda w: w["top"]):
        if len(lines) > 0 and w["top"] - lines[-1][0]["top"] < 3:
            lines[-1].append(w)
        else:
            lines.append([w])
    lines = [sorted(line, key=lambda w: w["x0"]) for line in lines]

    
    # Using Total row tells us where each column's right edge is
    total_line = [line for line in lines if line[0]["text"] == "Total"][0]
    total_words = [w for w in total_line if NUMBER.match(w["text"])]
    edges = [w["x1"] for w in total_words]
    column_totals = [float(w["text"].replace(",", "")) for w in total_words]

    # Iron Ore header is left of its numbers
    iron_x = [w["x0"] for line in lines for w in line
              if w["text"] == "Iron"][0]
    iron_column = [i for i in range(len(edges)) if edges[i] >= iron_x][0]
    if column_totals[iron_column] < 0.9 * max(column_totals):
        problems.append(f"{path.name}: Iron Ore is not the big column")

    # Place every number in its column and keep the iron ore ones
    column_sums = [0] * len(edges)
    foreign, domestic = 0, 0
    for line in lines[:lines.index(total_line)]:
        label = " ".join(w["text"] for w in line
                         if not NUMBER.match(w["text"]))
        for w in line:
            if not NUMBER.match(w["text"]):
                continue
            gaps = [abs(w["x1"] - edge) for edge in edges]
            column = gaps.index(min(gaps))
            value = float(w["text"].replace(",", ""))
            column_sums[column] = column_sums[column] + value
            if column == iron_column and label == "Australia":
                domestic = value
            elif column == iron_column:
                foreign = foreign + value

    # Every column has to add up to the printed Total row
    for i in range(len(edges)):
        if abs(column_sums[i] - column_totals[i]) >= 1:
            problems.append(f"{path.name}: column {i} does not add up")

    note = "foreign destinations only"
    if domestic > 0:
        note = f"domestic Australia row left out ({domestic:,.0f} t)"
    rows.append([month, "port955", round(foreign, 2), note])

# Dampier 
annual_files = []
for path in sorted((DAMPIER_DIR / "annual_fy").glob("FY*.pdf")):
    if int(path.name[2:6]) >= 2018:
        annual_files.append(path)

# monthly is YTD so we want the newest one per year  
newest = {}
for path in sorted((DAMPIER_DIR / "monthly_ytd").glob("*.pdf")):
    newest[path.name.split("_")[-1]] = path

for path in annual_files + list(newest.values()):
    with pdfplumber.open(path) as pdf:
        page = pdf.pages[0]
        text = page.extract_text()
        words = page.extract_words()
    start_year = int(re.search(r"(\d{4})-\d{4} FINANCIAL YEAR", text)[1]) 

    lines = []
    for w in sorted(words, key=lambda w: w["top"]):
        if len(lines) > 0 and w["top"] - lines[-1][0]["top"] < 3:
            lines[-1].append(w)
        else:
            lines.append([w])

    month_values, printed_total = {}, 0
    for line in lines:
        line = sorted(line, key=lambda w: w["x0"])
        names = [w["text"] for w in line]
        if names[0] == "MONTH" and names[1:3] != ["IRON", "ORE"]:
            problems.append(f"{path.name}: first column is not Iron Ore")
        # FY2020-21 prints the month number stuck to the name (7JULY)
        label = []
        for w in line[:2]:
            if w["text"].lstrip("0123456789") in list(MONTHS) + ["TOTALS:"]:
                label.append(w)
        if len(label) == 0:
            continue
        label_name = label[0]["text"].lstrip("0123456789")

        # Some digits are printed with a gap after them (1 41,228,181)
        # Touching pieces belong together column gaps are much wider
        groups, last_x1 = [], -99
        for w in line[line.index(label[0]) + 1:]:
            if w["x0"] - last_x1 < 3:
                groups[-1] = groups[-1] + w["text"]
            else:
                groups.append(w["text"])
            last_x1 = w["x1"]

        # First column is iron ore (a dash means the month hasn't happened)
        if groups[0] == "-":
            continue
        value = int(groups[0].replace(",", ""))
        if label_name == "TOTALS:":
            printed_total = value
            continue
        number = MONTHS[label_name]
        year = start_year if number >= 7 else start_year + 1
        month_values[pd.Timestamp(year=year, month=number, day=1)] = value

    if sum(month_values.values()) != printed_total:
        problems.append(f"{path.name}: months do not add up to TOTALS")
    for month in month_values:
        if month >= FIRST_MONTH:
            rows.append([month, "port276", month_values[month],
                         "iron ore out, financial year table"])

# Write clean-file format
data = pd.DataFrame(rows, columns=["month", "port", "tonnes", "note"])
data = data.sort_values(["port", "month"])
data["source"] = "australia_exporter"
data["exporter"] = "AUS"
data["counterparty"] = "ALL"
data["flag"] = ""
data["retrieved"] = RETRIEVED
data["month"] = data["month"].dt.strftime("%Y-%m-%d")
data = data[["source", "exporter", "counterparty", "port", "month",
             "tonnes", "flag", "retrieved", "note"]]
OUT.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(OUT, index=False)

print(data.groupby("port")["month"].agg(["count", "min", "max"]))

# Let's see if anything didn't add up to the printed totals
if len(problems) == 0:
    print("Every file adds up to its printed totals")
for problem in problems:
    print(problem)
