"""Presentation 2, slides 8 to 10 - the worked example of noise and signal.

These are the illustrative numbers on the slides, not project data. We start
from ten months of year-over-year changes (in Mt) for the exporter and the
ships, and from the importers' three matrix numbers as shown on slide 9.
Writes Charts/worked_example.csv.
Run from the repo folder:  python EDA/EDA_6.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHARTS = ROOT / "Charts"
CHARTS.mkdir(exist_ok=True)

# Slide 8 - one year-over-year change is year 2 minus year 1
# month 1 to 3 of the exporter: 20.1 to 20.5, 18.7 to 18.4, 21.4 to 20.0
year_one = [20.1, 18.7, 21.4]
year_two = [20.5, 18.4, 20.0]
for i in range(len(year_one)):
    print("month", i + 1, round(year_two[i] - year_one[i], 1))

# Slide 9 - all ten changes for the exporter and for the ships
exporter = [0.4, -0.3, -1.4, 1.8, -1.0, 0.3, -2.0, 1.4, 1.2, 0.9]
ships = [2.0, -0.5, -0.8, 1.9, -0.4, 1.1, -2.2, 0.6, 1.9, 0.3]
n = len(exporter)
exporter_mean = sum(exporter) / n   # = 0.13
ships_mean = sum(ships) / n         # = 0.39
exporter_off = []
ships_off = []
for i in range(n):
    exporter_off.append(exporter[i] - exporter_mean)
    ships_off.append(ships[i] - ships_mean)

# Multiply month by month, add the ten products and divide by n - 1
products = []
for i in range(n):
    products.append(exporter_off[i] * ships_off[i])
cov_exporter_ships = sum(products) / (n - 1)    # = 1.50
var_exporter = sum(x * x for x in exporter_off) / (n - 1)   # = 1.62
var_ships = sum(x * x for x in ships_off) / (n - 1)         # = 1.92
print("exporter variance", round(var_exporter, 2))
print("ships variance", round(var_ships, 2))
print("exporter and ships covariance", round(cov_exporter_ships, 2))

# The importers' ten changes are not on the slides, so their three numbers
# come straight from the matrix on slide 9
var_importers = 2.68
cov_exporter_importers = 1.50
cov_ships_importers = 1.47

# Slide 10 - signal is the covariance shared by all three records
# For one record: C(with second) x C(with third) / C(second with third)
signal_exporter = cov_exporter_ships * cov_exporter_importers \
    / cov_ships_importers                                   # = 1.53
signal_ships = cov_exporter_ships * cov_ships_importers \
    / cov_exporter_importers                                # = 1.47
signal_importers = cov_exporter_importers * cov_ships_importers \
    / cov_exporter_ships                                    # = 1.47

# Noise is the variance the other two records do not confirm
# Slide 10 uses the matrix numbers as printed (two decimals), so we round
variances = {"exporter": round(var_exporter, 2), "ships": round(var_ships, 2),
             "importers": var_importers}
signals = {"exporter": round(signal_exporter, 2),
           "ships": round(signal_ships, 2),
           "importers": round(signal_importers, 2)}
lines = ["record,variance,signal,noise,noise_to_signal"]
for record in variances:
    noise = round(variances[record] - signals[record], 2)
    ratio = (noise / signals[record]) ** 0.5
    lines.append("%s,%.2f,%.2f,%.2f,%.2f" % (record, variances[record],
                                             signals[record], noise, ratio))
    print(record, signals[record], noise, round(ratio, 2))
with open(CHARTS / "worked_example.csv", "w") as file:
    file.write("\n".join(lines) + "\n")
