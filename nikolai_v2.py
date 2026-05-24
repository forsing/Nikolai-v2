# Nikolai formula
# predict the next point of intervals curve in 3 - 7 % of all cases
# average of all intervals is around 5.571428571428571 (39/7)   abs
#
# v2: evaluacija Nikolai formule kao samostalnog "skora" po broju 1..39
#     + back-test na poslednjih 100 izvlačenja + snimanje predikcije


"""
Šta Nikolai formula radi:
Velika Fourier-suma sa ~60+ parametara (A1-A6, B1-B28, C1-C14, D1-D14, E1-E28), R² = 0.987 na fit-u
Predict the next point of intervals curve in 3-7% of all cases
Prosek = 39/7 ≈ 5.57 (prosečan interval između 7 brojeva u opsegu 1-39)
Formula ne predviđa brojeve, već intervale između sortiranih brojeva u kombinaciji.

Output je problem:
nikolai.tail(5) daje vrednosti [-21, 15, 15, 31, 3, 5, 3] i slično — negativne brojeve i brojeve van 1-39 opsega
Ne može da se direktno koristi kao predikcija loto kombinacije
Treba ozbiljno "dekodovanje" (npr. uzeti samo pozitivne, kumulativno sabirati, normalizovati u 1-39)

Tvrdnja "3-7% pogodaka":
Slučajan baseline za predikciju 1 broja iz 39 je 7/39 ≈ 18%
Ako Nikolai daje 3-7% za jednu poziciju, to je gore od slučajnog
Verovatno je tvrdnja vezana za "tačku na curve intervals", ne za pravu kombinaciju

Veliki red flag:
Formula ima ~60 parametara, fit-ovana je verovatno na konkretne podatke. 
R² = 0.987 na fit-u je tipičan overfitting kod previše parametara — ne generalizuje

Da li može da se iskoristi?
Kao samostalna predikcija: ne. 
Output nije validna kombinacija, mora se dekodovati i još posle toga ne daje signal iznad slučajnog.
Kao DODATNI signal u ENSEMBLE-u: teoretski može — vrednost po broju 1..39 se može mapirati u "skor" i ubaciti kao još jedan model u prosek sa XGB, NN, RF. 
Ali tek nakon back-testa na poslednjih 100 kola: ako daje hits/7 ≥ 1.30 ima smisla, ako daje < 1.26 formulu treba odbaciti.
Back-test (poslednjih 100 izvlačenja, N=4620):
  NIKOLAI   hits/7 = 1.270  (18.1%)
  NIKOLAI*M hits/7 = 1.270  (18.1%)
  (slučajan baseline ≈ 1.256 hits/7 = 17.9%)
  AUC = 0.500   LRAP = 0.245

Nije loše, ali nije ni super — nešto u sredini između slučajnog i XGB/NN/RF.
Nažalost, u poslednjih 100 kola nikako nije bilo sličnih signala kao što je bilo na fit-u.
To je tipičan overfitting → generalizacija je kriva, formula ne radi što se očekuje.
Zaključak:
Nikolajeva formula nije dobra samostalna predikcija loto kombinacije.



koristi aktivni CSV (loto7_4620_k41.csv)
generiše Nikolai skorove za brojeve 1..39
mapira ih u "verovatnoće" (sigmoid ili rank-normalizacija)
back-test na poslednjih 100 kola sa hits/7, AUC, LRAP
poredi sa slučajnim baseline-om
ispiše top-7 predikciju


Ako rezultat back-testa pokaže signal — uključimo formulu u ensemble sa XGB/Keras/NN modelima.


Ako ne (sto jeste) — odbacimo formulu kao istorijski artefakt.
"""



import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from numpy import cos, sin
from sklearn.metrics import label_ranking_average_precision_score, roc_auc_score

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


# Coefficient of Multiple Determination (R^2)
# Multiple Coefficient of Determination
M = 0.987444305500000


# parametters

A1 = 7.299684015518730
A2 = 0
A3 = -16.685835427847000
A4 = 4.033620068208560
A5 = 11.787890199614100
A6 = -0.929875140722455

B1 = -2.183088127022560
B2 = 5.622345763015300
B3 = -2.937935197869250
B4 = 3.055160052310020
B5 = 4.155657481996250
B6 = 1.999149991032180
B7 = 1.424034844968820
B8 = 1.123154320679130
B9 = 0.587528422335690
B10 = 2.192576277398270
B11 = 0.646184039028009
B12 = 3.128750813623030
B13 = -2.639908199110780
B14 = -1.234452659544030
B15 = 0
B16 = 0
B17 = 1.684322379296770
B18 = 1.806815397870690
B19 = -0.010080723947845
B20 = 1.819868387007100
B21 = 3.421929858053530
B22 = 1.862697127062110
B23 = 0.540543148349822
B24 = 1.862484902239440
B25 = 1.229198270286820
B26 = 1.888114102763830
B27 = 0.542228454843728
B28 = 1.853126551719710

C1 = 9.582199721922110
C2 = 160.540667471316000
C3 = 235.597570526473000
C4 = 193.064941311939000
C5 = -69.904752286696000
C6 = -85.770268955927000
C7 = 276.721054209067000
C8 = 0
C9 = -374.987916855954000
C10 = 444.826536028089000
C11 = -256.601030945780000
C12 = -590.091497107117000
C13 = 173.815562399882000
C14 = -605.344333544192000

D1 = 100.982005590423000
D2 = -173.685727106493000
D3 = -28.816489276900800
D4 = -141.058426244729000
D5 = -172.520435212672000
D6 = -61.540771042905300
D7 = 114.003339618542000
D8 = 0
D9 = -58.866476964092400
D10 = -51.716911912693900
D11 = -59.068870808688700
D12 = -55.721714142108400
D13 = -51.593058043094400
D14 = -64.539817955903400

E1 = 233.179121249626000
E2 = -72.647812742405900
E3 = 154.397943691535000
E4 = 66.306079684944700
E5 = 130.975552547632000
E6 = 114.372274193839000
E7 = -194.072161107444000
E8 = 16.274381953945800
E9 = 25.157528943044000
E10 = 35.300757969358900
E11 = 4.034054329422520
E12 = -72.271746132602100
E13 = 144.393758949961000
E14 = 17.511079644164100
E15 = 0
E16 = 0
E17 = -108.758489911582000
E18 = -69.399029881088400
E19 = -114.061251203356000
E20 = -70.397643655260600
E21 = -206.192826567812000
E22 = -71.161467289090500
E23 = -169.344108721358000
E24 = -72.593728034594300
E25 = -205.741600763855000
E26 = -73.906381152311700
E27 = -169.781637338030000
E28 = -67.941409230581000


# =========================
# Konfiguracija
# =========================
CSV_PATH = "/data/loto7_4620_k41.csv"
OUT_TXT = Path("/nikolai_v2_predikcija.txt")
N_MIN, N_MAX = 1, 39
K = 7
BACKTEST_N = 100

T0 = time.time()
print()
print("START", datetime.today())
print()

# =========================
# 1) Učitavanje aktivnog CSV-a (bez headera, 7 kolona)
# =========================
t = pd.read_csv(CSV_PATH, header=None)
print()
print("###############################")
print()
print()
print('izvlacenja zadnjih 5')
print(t.tail(5).reset_index(drop=True))
print()
print()
print()

# Pretpostavljamo da prve 7 kolona sadrže brojeve lutrije
t = t.iloc[:, :7].astype(int)


# =========================
# 2) Nikolai formula — element-wise nad CSV-om (kao u v1)
# =========================
# suitable equation to reproduce all intervals curves for the 39 numbers
nikolai = A1 + A3 * sin(A4 +
          C1 * cos(B1 * t + E1)  +  D1 * sin(B2 * t + E2) +
          C2 * cos(B3 * t + E3)  +  D2 * sin(B4 * t + E4) +
          C3 * cos(B5 * t + E5)  +  D3 * sin(B6 * t + E6) +
          C4 * cos(B7 * t + E7)  +  D4 * sin(B8 * t + E8) +
          C5 * cos(B9 * t + E9)  +  D5 * sin(B10 * t + E10) +
          C6 * cos(B11 * t + E11) + D6 * sin(B12 * t + E12) +
          C7 * cos(B13 * t + E13) + D7 * sin(B14 * t + E14)) + A5 * cos(A6 +
          C9 * cos(B17 * t + E17)  +  D9 * sin(B18 * t + E18) +
          C10 * cos(B19 * t + E19)  +  D10 * sin(B20 * t + E20) +
          C11 * cos(B21 * t + E21)  +  D11 * sin(B22 * t + E22) +
          C12 * cos(B23 * t + E23)  +  D12 * sin(B24 * t + E24) +
          C13 * cos(B25 * t + E25)  +  D13 * sin(B26 * t + E26) +
          C14 * cos(B27 * t + E27)  +  D14 * sin(B28 * t + E28))

print('nikolai zadnjih 5')
print(np.round(nikolai.tail(5).reset_index(drop=True)).astype(int))
print()


nikolaiM = nikolai * M
print('nikolaiM zadnjih 5')
print(np.round(nikolaiM.tail(5).reset_index(drop=True)).astype(int))
print()


# =========================
# 3) Nikolai skor po broju 1..39
# =========================
def nikolai_score(i):
    return A1 + A3 * sin(A4 +
          C1 * cos(B1 * i + E1)  +  D1 * sin(B2 * i + E2) +
          C2 * cos(B3 * i + E3)  +  D2 * sin(B4 * i + E4) +
          C3 * cos(B5 * i + E5)  +  D3 * sin(B6 * i + E6) +
          C4 * cos(B7 * i + E7)  +  D4 * sin(B8 * i + E8) +
          C5 * cos(B9 * i + E9)  +  D5 * sin(B10 * i + E10) +
          C6 * cos(B11 * i + E11) + D6 * sin(B12 * i + E12) +
          C7 * cos(B13 * i + E13) + D7 * sin(B14 * i + E14)) + A5 * cos(A6 +
          C9 * cos(B17 * i + E17)  +  D9 * sin(B18 * i + E18) +
          C10 * cos(B19 * i + E19)  +  D10 * sin(B20 * i + E20) +
          C11 * cos(B21 * i + E21)  +  D11 * sin(B22 * i + E22) +
          C12 * cos(B23 * i + E23)  +  D12 * sin(B24 * i + E24) +
          C13 * cos(B25 * i + E25)  +  D13 * sin(B26 * i + E26) +
          C14 * cos(B27 * i + E27)  +  D14 * sin(B28 * i + E28))


print()
print('nikolai (i, f, fM)')
print()
scores = np.zeros(N_MAX, dtype=float)
for i in range(1, 40, 1):
    f = nikolai_score(i)
    fM = f * M
    scores[i - 1] = f
    print(np.round((i, f, fM), 0))
print()
print()


# =========================
# 4) Top-7 predikcija (deterministički, bez random)
# =========================
def topk_from_scores(scores_1d, k=K):
    s = np.asarray(scores_1d, dtype=float)
    order = np.lexsort((np.arange(N_MAX), -s))
    return np.sort(order[:k] + 1)


def describe(pick):
    return (
        f"suma={int(pick.sum())}, "
        f"neparnih={int((pick % 2 == 1).sum())}/{K}, "
        f"niskih(<=19)={int((pick <= 19).sum())}/{K}, "
        f"raspon={int(pick.max() - pick.min())}"
    )


pick_nikolai = topk_from_scores(scores)
pick_nikolaiM = topk_from_scores(scores * M)

assert len(set(pick_nikolai.tolist())) == K and pick_nikolai.min() >= N_MIN and pick_nikolai.max() <= N_MAX
assert list(pick_nikolai) == sorted(pick_nikolai.tolist())

print()
print("###############################")
print()
print("Top-7 po Nikolai skoru (deterministički):")
print(f"  NIKOLAI  -> {pick_nikolai.tolist()}  ({describe(pick_nikolai)})")
print(f"  NIKOLAI*M-> {pick_nikolaiM.tolist()}  ({describe(pick_nikolaiM)})")
print()


# =========================
# 5) Back-test: koliko prosečno pogađa fiksna NIKOLAI kombinacija
#    u poslednjih BACKTEST_N izvlačenja
# =========================
draws = np.sort(t.values, axis=1)
N = draws.shape[0]
back = draws[-BACKTEST_N:]

def avg_hits_fixed(fixed_pick, draws_arr):
    fixed_set = set(fixed_pick.tolist())
    h = 0
    for row in draws_arr:
        h += len(fixed_set & set(int(v) for v in row))
    return h / draws_arr.shape[0]


# Za AUC/LRAP nam treba multi-hot Y i score (isti score se ponavlja za svaki red)
def draws_to_multihot(rows):
    out = np.zeros((rows.shape[0], N_MAX), dtype=np.int8)
    for i, row in enumerate(rows):
        out[i, row - 1] = 1
    return out

Y_back = draws_to_multihot(back)
S_back = np.tile(scores.reshape(1, -1), (BACKTEST_N, 1))

def safe_auc(Y, S):
    try:
        return roc_auc_score(Y, S, average="macro")
    except Exception:
        return float("nan")

def safe_lrap(Y, S):
    try:
        return label_ranking_average_precision_score(Y.astype(int), S)
    except Exception:
        return float("nan")

baseline = 7 * 7 / 39
h_nikolai = avg_hits_fixed(pick_nikolai, back)
h_nikolaiM = avg_hits_fixed(pick_nikolaiM, back)
auc = safe_auc(Y_back, S_back)
lrap = safe_lrap(Y_back, S_back)

print(f"Back-test (poslednjih {BACKTEST_N} izvlačenja, N={N}):")
print(f"  NIKOLAI   hits/7 = {h_nikolai:.3f}  ({100*h_nikolai/K:.1f}%)")
print(f"  NIKOLAI*M hits/7 = {h_nikolaiM:.3f}  ({100*h_nikolaiM/K:.1f}%)")
print(f"  (slučajan baseline ≈ {baseline:.3f} hits/7 = {100*baseline/K:.1f}%)")
print(f"  AUC = {auc:.3f}   LRAP = {lrap:.3f}")
print()


# =========================
# 6) Snimanje u TXT
# =========================
with OUT_TXT.open("a", encoding="utf-8") as f:
    f.write(f"\n--- {datetime.today()} (N={N}) ---\n")
    f.write(f"NIKOLAI    -> {pick_nikolai.tolist()}  ({describe(pick_nikolai)})\n")
    f.write(f"NIKOLAI*M  -> {pick_nikolaiM.tolist()}  ({describe(pick_nikolaiM)})\n")
    f.write(f"back-test: NIKOLAI hits/7={h_nikolai:.3f}, NIKOLAI*M hits/7={h_nikolaiM:.3f}, "
            f"baseline={baseline:.3f}, AUC={auc:.3f}, LRAP={lrap:.3f}\n")
print(f"Snimljeno u: {OUT_TXT}")

elapsed = time.time() - T0
print()
print("STOP", datetime.today())
print(f"Ukupno vreme: {str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)")
print()



"""
START 2026-05-24 17:24:38.349389

###############################


izvlacenja zadnjih 5
   0   1   2   3   4   5   6
0  4   7  11  20  33  34  39
1  7  14  17  22  24  26  33
2  4   9  14  16  19  24  32
3  3   6  12  14  15  22  27
4  1   3   7  17  24  25  32



nikolai zadnjih 5
    0   1   2   3   4   5   6
0  15 -21  15  22  30  -3  16
1 -21  11  35  14   0 -20  30
2  15  11  11  15   6   0  -5
3  -3   5  10  11  -8  14  16
4  34  -3 -21  35   0   9  -5

nikolaiM zadnjih 5
    0   1   2   3   4   5   6
0  15 -20  15  22  29  -3  16
1 -20  11  34  14   0 -20  29
2  15  11  11  14   6   0  -5
3  -3   5  10  11  -8  14  16
4  33  -3 -20  34   0   9  -5


nikolai (i, f, fM)

[ 1. 34. 33.]
[ 2. -8. -8.]
[ 3. -3. -3.]
[ 4. 15. 15.]
[ 5. -4. -4.]
[6. 5. 5.]
[  7. -21. -20.]
[  8. -21. -21.]
[ 9. 11. 11.]
[10. 17. 16.]
[11. 15. 15.]
[12. 10. 10.]
[ 13. -19. -19.]
[14. 11. 11.]
[15. -8. -8.]
[16. 15. 14.]
[17. 35. 34.]
[18. 31. 31.]
[19.  6.  6.]
[20. 22. 22.]
[21. 29. 28.]
[22. 14. 14.]
[23.  9.  9.]
[24. -0. -0.]
[25.  9.  9.]
[ 26. -20. -20.]
[27. 16. 16.]
[ 28. -10. -10.]
[29.  3.  3.]
[30. 28. 28.]
[31.  5.  5.]
[32. -5. -5.]
[33. 30. 29.]
[34. -3. -3.]
[35. 10.  9.]
[36.  3.  3.]
[37. 10. 10.]
[38. 25. 25.]
[39. 16. 16.]



###############################

Top-7 po Nikolai skoru (deterministički):
  NIKOLAI  -> [1, x, 18, y, 30, z, 38]  (suma=158, neparnih=4/7, niskih(<=19)=3/7, raspon=37)
  NIKOLAI*M-> [1, x, 18, y, 30, z, 38]  (suma=158, neparnih=4/7, niskih(<=19)=3/7, raspon=37)

Back-test (poslednjih 100 izvlačenja, N=4620):
  NIKOLAI   hits/7 = 1.270  (18.1%)
  NIKOLAI*M hits/7 = 1.270  (18.1%)
  (slučajan baseline ≈ 1.256 hits/7 = 17.9%)
  AUC = 0.500   LRAP = 0.245

Snimljeno u: /nikolai_v2_predikcija.txt

STOP 2026-05-24 17:24:38.387023
Ukupno vreme: 0:00:00  (0.0 s)
"""


