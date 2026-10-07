"""Build the four analysis datasets from the raw report / XRF sources.

Outputs (utf-8-sig CSV) into  <repo>/data/ :
  flotation_bench.csv   29 bench flotation tests, grade + recovery
  flotation_plant.csv   3 plant fineness surveys, grade
  rheology.csv          apparent viscosity vs T / clay / shear rate
  wettability.csv       contact angle vs T / clay content / diesel

Cross-checks are printed so every number can be traced back.
"""
import os, io, csv, glob
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "origin_data")
OUT = os.path.join(os.path.dirname(HERE), "data")
os.makedirs(OUT, exist_ok=True)
log = io.StringIO()
W = log.write


def rd(p):
    txt = open(p, encoding="utf-8-sig").read()
    return [[c.strip() for c in l.split(",")] for l in txt.splitlines() if l.strip()]


def find(pat):
    hits = glob.glob(os.path.join(ORIG, pat))
    if not hits:
        raise FileNotFoundError(pat)
    return hits[0]


# ---------------------------------------------------------------- bench tests
wb = openpyxl.load_workbook(os.path.join(HERE, "XRF数据处理.xlsx"), data_only=True)
ws = wb["品位、回收率计算"]
rows = list(ws.iter_rows(values_only=True))

# The condition strings use the template  fineness / pH / diesel(drops) / time / frother / frother time.
# For the diesel series the dosage was written into the pH slot (e.g. "40% 18 12d 15min 3d 10min"),
# so the strings alone cannot be parsed. The dosages for that series were recovered from the
# report's own figure 3-39, where the same Mo grades appear against 50 / 100 / 150 / 200 g/t.
DIESEL_SERIES = {15: 50.0, 16: 150.0, 17: 200.0, 18: 50.0, 19: 150.0, 20: 200.0}
DPD = 50.0 / 6.0                                             # 6 drops == 50 g/t

bench = []
series = None
for r in rows[1:]:
    if r[0] is not None:
        series = str(r[0]).strip()
    if r[1] is None or not isinstance(r[1], (int, float)):
        continue
    cond = str(r[8]).strip()
    t = cond.split()
    fin = float(t[0].replace("%", ""))
    if series == "柴油用量":
        ph = 11.0
        diesel = DIESEL_SERIES[int(r[1])]
        tmin = 15.0
    else:
        ph = float(t[1])
        diesel = float(t[2].replace("d", "")) * DPD
        tmin = float(t[3].replace("min", ""))
    bench.append(dict(id=int(r[1]), series=series, fineness=fin, ph=ph,
                      diesel_gpt=round(diesel, 1), time_min=tmin,
                      yield_pct=float(r[3]),
                      mo_grade=float(r[4]), cu_grade=float(r[5]),
                      mo_rec=float(r[6]) * 100, cu_rec=float(r[7]) * 100,
                      cond=cond))

W("=== bench tests: mass-balance check (Mo) ===\n")
ws2 = wb["粗精矿（未归一）"]
raw = {int(r[1]): r for r in ws2.iter_rows(values_only=True)
       if isinstance(r[1], (int, float))}
FEED_G, FEED_MO_PCT, FEED_CU_PCT = 300.0, 0.5436, 0.0364
bad_mo = bad_cu = bad_yld = 0
for b in bench:
    rr = raw[b["id"]]
    conc_g = rr[6]
    mo_rec_calc = conc_g * b["mo_grade"] / 100.0 / (FEED_G * FEED_MO_PCT / 100.0) * 100
    cu_rec_calc = conc_g * b["cu_grade"] / 100.0 / (FEED_G * FEED_CU_PCT / 100.0) * 100
    yld_calc = conc_g / FEED_G * 100
    ok_mo = abs(mo_rec_calc - b["mo_rec"]) < 0.6
    ok_cu = abs(cu_rec_calc - b["cu_rec"]) < 0.6
    ok_y = abs(yld_calc - b["yield_pct"]) < 0.15
    bad_mo += not ok_mo; bad_cu += not ok_cu; bad_yld += not ok_y
    W(f"  #{b['id']:<3} conc={conc_g:>6}g  产率(表)={b['yield_pct']:>6}%  产率(算)={yld_calc:5.2f}% "
      f"{'  ' if ok_y else '<<'}   Mo品位={b['mo_grade']:>6}%  Mo回收(表)={b['mo_rec']:6.2f}% "
      f"Mo回收(算)={mo_rec_calc:6.2f}% {'  ' if ok_mo else '<<'}   "
      f"Cu回收(表)={b['cu_rec']:6.2f}%  Cu回收(算)={cu_rec_calc:6.2f}% {'  ' if ok_cu else '<<'}\n")
W(f"\n  Mo 品位/回收 mass balance : {len(bench)-bad_mo}/{len(bench)} consistent\n")
W(f"  Cu 品位/回收 mass balance : {len(bench)-bad_cu}/{len(bench)} consistent\n")
W(f"  产率 column             : {len(bench)-bad_yld}/{len(bench)} consistent\n")

with open(os.path.join(OUT, "flotation_bench.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(bench[0].keys()))
    w.writeheader(); w.writerows(bench)

# ---------------------------------------------------------------- plant surveys
plant = []
for line, pat in [("2区", "图3-32 -0.074 mm含量对矿山2区浮选指标的影响/Book1_Sheet1.csv"),
                  ("4区老线", None),
                  ("4区万吨", "图3-34 -0.074 mm含量对4区万吨浮选指标的影响_2/Book1_Sheet3.csv")]:
    if pat is None:
        continue
    for r in rd(find(pat))[1:]:
        plant.append(dict(line=line, fineness=float(r[0]),
                          mo_grade=float(r[1]), cu_grade=float(r[2])))
# 4区老线: values live in the 图数据 curve files
d33 = find("图3-33 -0.074 mm含量对4区老线浮选指标的影响")
c1 = rd(os.path.join(d33, "图数据_[Graph2]Layer1_曲线1.csv"))[1:]
c2 = rd(os.path.join(d33, "图数据_[Graph2]Layer1_曲线2.csv"))[1:]
for (x, mo, cu) in zip([r[1] for r in c1], [r[1] for r in c1], [r[1] for r in c2]):
    pass
mo33 = [float(r[1]) for r in c1]
cu33 = [float(r[1]) for r in c2]
for i, fin in enumerate([50, 55, 60, 65, 70, 75]):
    plant.append(dict(line="4区老线", fineness=float(fin),
                      mo_grade=mo33[i], cu_grade=cu33[i]))

with open(os.path.join(OUT, "flotation_plant.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["line", "fineness", "mo_grade", "cu_grade"])
    w.writeheader(); w.writerows(plant)
W(f"\n=== plant surveys: {len(plant)} rows ===\n")
for p in plant:
    W(f"  {p['line']:<6} {p['fineness']:>5}%  Mo={p['mo_grade']:>7}  Cu={p['cu_grade']:>7}\n")

# ---------------------------------------------------------------- rheology
# 图4-45 : four clay types, shear rate 100 s-1  (col order chlorite, muscovite, illite, kaolinite)
CLAYS = ["chlorite", "muscovite", "illite", "kaolinite"]
rheo = []
def add_shear(path, shear):
    rows = rd(path)[1:]
    Ts = [float(r[0]) for r in rows]
    for ci, clay in enumerate(CLAYS):
        for i, T in enumerate(Ts):
            rheo.append(dict(temperature=T, clay=clay, shear_rate=shear,
                             viscosity=float(rows[i][ci + 1]), source="fig4-45"))

add_shear(find("图4-45 不同黏土种类下辉钼矿粗精矿矿浆表观黏度-温度曲线（剪切速率a 100 s-1/Book1_100s-1.csv"), 100)
add_shear(find("图4-45 不同黏土种类下辉钼矿粗精矿矿浆表观黏度-温度曲线（剪切速率a 100 s-1_2/Book1_150.csv"), 150)

# 图4-47 : 100 s-1 replicates -> second measurement (Col3)
for clay, pat in [("muscovite", "图4-47 不同黏土组分矿浆表观黏度-温度图（a高岭石/Book2100_白云母.csv"),
                  ("chlorite", "图4-47 不同黏土组分矿浆表观黏度-温度图（a高岭石_2/Book2100_绿泥石.csv"),
                  ("illite", "图4-47 不同黏土组分矿浆表观黏度-温度图（a高岭石_3/Book2100_伊利石.csv"),
                  ("kaolinite", "图4-47 不同黏土组分矿浆表观黏度-温度图（a高岭石_4/Book2100_高岭石2g.csv")]:
    for r in rd(find(pat))[1:]:
        rheo.append(dict(temperature=float(r[0]), clay=clay, shear_rate=100,
                         viscosity=float(r[2]), source="fig4-47"))

# 图4-45_3 : chlorite vs clay content, five temperatures
for ci, T in enumerate([5, 10, 15, 25, 35]):
    pass
blk = rd(find("图4-45 表明：在5 -35 °C矿浆环境下，绿泥石与辉钼矿精矿混合体系的矿浆黏度最小，高岭石与/Book1_绿泥石.csv"))
loads = [float(r[0]) for r in blk[1:]]
for li, load in enumerate(loads):
    pass
for ci in range(1, 6):
    for li, load in enumerate(loads):
        rheo.append(dict(temperature=[5, 10, 15, 25, 35][ci - 1], clay=f"chlorite@{load:g}%",
                         shear_rate=100, viscosity=float(blk[li + 1][ci]), source="fig4-45c"))

with open(os.path.join(OUT, "rheology.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["temperature", "clay", "shear_rate", "viscosity", "source"])
    w.writeheader(); w.writerows(rheo)
W(f"\n=== rheology: {len(rheo)} rows ===\n")
for clay in CLAYS:
    for sh in (100, 150):
        v = [r for r in rheo if r["clay"] == clay and r["shear_rate"] == sh and r["source"] == "fig4-45"]
        if v:
            W(f"  {clay:<10} {sh:>4} s-1 : " +
              " ".join(f"{r['temperature']:g}C->{r['viscosity']:g}" for r in v) + "\n")

# ---------------------------------------------------------------- wettability
wet = []
def add_wet(system, temp_col, clay_pct, diesel, path, ycol=1, label=None):
    for r in rd(path)[1:]:
        if r[ycol] in ("", None):
            continue
        wet.append(dict(system=system, temperature=float(r[temp_col]), clay_pct=clay_pct,
                        diesel_gpt=diesel, contact_angle=float(r[ycol])))

# 图4-36 : pure Mo concentrate, temperature sweep
#   the plotted x axis is an Origin reconstruction (5..55); the underlying series is
#   a 10 degC ladder, confirmed by two independent cross-checks:
#     fig 4-39 at 0 g/t diesel  = 80.93  = the third point of 4-36   -> 4-39 runs at 30 degC
#     fig 4-37 at 25 % chlorite = 85.57  = the first point of 4-38   -> 4-37 runs at 10 degC
T36 = [10, 20, 30, 40, 50]
d36 = find("图4-36 温度对钼精矿润湿性的影响")
for i, r in enumerate(rd(os.path.join(d36, "图数据_[Graph1]Layer1_曲线1.csv"))[1:]):
    wet.append(dict(series="temperature", system="clean", temperature=T36[i], clay_pct=0,
                    diesel_gpt=0, contact_angle=float(r[1])))
# 图4-37 : chlorite content sweep, run at 10 degC (see above)
d37 = find("图4-37 绿泥石含量对精矿润湿性的影响")
for r in rd(os.path.join(d37, "Book1_绿泥石：辉钼矿.csv"))[1:]:
    wet.append(dict(series="content", system="content-sweep", temperature=10, clay_pct=float(r[0]),
                    diesel_gpt=0, contact_angle=float(r[1])))
# 图4-38 : 25 % chlorite, temperature sweep (temperatures documented in the sheet)
d38 = find("图4-38 温度对绿泥石含量占比为25 %体系下辉钼矿润湿性的影响")
for r in rd(os.path.join(d38, "Book1_Sheet2.csv"))[1:]:
    wet.append(dict(series="temperature", system="chlorite25", temperature=float(r[0]), clay_pct=25,
                    diesel_gpt=0, contact_angle=float(r[1])))
# 图4-39 : diesel sweep, clean, at 30 degC
d39 = find("图4-39 柴油用量对钼精矿润湿性的影响")
for r in rd(os.path.join(d39, "Book1_Sheet6.csv"))[1:]:
    wet.append(dict(series="diesel", system="clean", temperature=30, clay_pct=0,
                    diesel_gpt=float(r[0]), contact_angle=float(r[1])))
# 图4-40 : diesel sweep, 25 % chlorite, at 30 degC
d40 = find("图4-40 柴油用量对绿泥石含量占比为25 %体系下辉钼矿润湿性的影响")
for r in rd(os.path.join(d40, "Book1_Sheet4.csv"))[1:]:
    wet.append(dict(series="diesel", system="chlorite25", temperature=30, clay_pct=25,
                    diesel_gpt=float(r[0]), contact_angle=float(r[1])))

with open(os.path.join(OUT, "wettability.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["series", "system", "temperature", "clay_pct", "diesel_gpt", "contact_angle"])
    w.writeheader(); w.writerows(wet)
W(f"\n=== wettability: {len(wet)} rows ===\n")
for r in wet:
    W(f"  {r['series']:<12} {r['system']:<14} T={r['temperature']:>5}  clay={r['clay_pct']:>5}%  "
      f"diesel={r['diesel_gpt']:>5}  theta={r['contact_angle']}\n")

open(os.path.join(HERE, "_build_out.txt"), "w", encoding="utf-8").write(log.getvalue())
print("built ->", OUT)
