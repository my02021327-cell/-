"""소화조 상태변수 다변량 통계적 공정관리(MSPC) — PCA T2 / Q 관리도.

고장 라벨이 없는 데이터에서 '정상 운전 포락면'을 학습해 이탈을 탐지한다.
2018~2022 정상 구간으로 학습하고, 2023 계측 단절 구간에서 탐지·기여도를 확인한다.
실행: `python -m src.mspc`
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.final_ensemble import prepare, TARGET

STATE = ["소화조_pH", "소화조_온도", "소화조_VS", "소화조_TS", "소화조_VFA",
         "소화조_TAlk", "소화조_CODcr"]          # MAlk 는 TAlk 와 중복이라 제외


def fit_mspc(Xtr, var_keep=0.90):
    sc = StandardScaler().fit(Xtr)
    Z = sc.transform(Xtr)
    p = PCA().fit(Z)
    a = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), var_keep) + 1)
    pca = PCA(n_components=a).fit(Z)
    T = pca.transform(Z)
    lam = T.var(axis=0, ddof=1)
    n = len(Z)
    # Hotelling T2 관리한계 (F 분포)
    t2_lim = a * (n - 1) / (n - a) * stats.f.ppf(0.99, a, n - a)
    E = Z - pca.inverse_transform(T)
    q = (E ** 2).sum(axis=1)
    q_lim = np.percentile(q, 99)
    return dict(sc=sc, pca=pca, lam=lam, a=a, t2_lim=t2_lim, q_lim=q_lim)


def score(m, X):
    Z = m["sc"].transform(X)
    T = m["pca"].transform(Z)
    t2 = (T ** 2 / m["lam"]).sum(axis=1)
    E = Z - m["pca"].inverse_transform(T)
    return t2, (E ** 2).sum(axis=1), E


def main():
    d = prepare().dropna(subset=STATE)
    tr = d[d.index.year <= 2021]
    m = fit_mspc(tr[STATE].values)
    print(f"[학습] 2018~2021 정상 {len(tr)}일, 주성분 {m['a']}개 "
          f"(설명 {m['pca'].explained_variance_ratio_.sum()*100:.1f}%)")
    print(f"[한계] T2 99% = {m['t2_lim']:.2f} | Q 99% = {m['q_lim']:.2f}\n")

    print(f"{'구간':10s} {'n':>5s} {'T2 초과율':>9s} {'Q 초과율':>9s} {'경보율':>7s}")
    for tag, sub in [("2018~2021", tr), ("2022", d[d.index.year == 2022]),
                     ("2023 상반", d[(d.index.year == 2023) & (d.index.month <= 6)]),
                     ("2023 하반", d[(d.index.year == 2023) & (d.index.month > 6)])]:
        if not len(sub): continue
        t2, q, _ = score(m, sub[STATE].values)
        a1, a2 = (t2 > m["t2_lim"]).mean(), (q > m["q_lim"]).mean()
        print(f"{tag:10s} {len(sub):5d} {a1*100:8.1f}% {a2*100:8.1f}% "
              f"{((t2>m['t2_lim'])|(q>m['q_lim'])).mean()*100:6.1f}%")

    # 기여도 : 2023 이탈을 어느 변수가 만들었나
    te = d[d.index.year == 2023]
    _, _, E = score(m, te[STATE].values)
    contrib = pd.Series((E ** 2).mean(axis=0), index=STATE).sort_values(ascending=False)
    print("\n[2023 이탈 기여도] Q 통계량 분해")
    for k, v in contrib.items():
        print(f"   {k:12s} {v/contrib.sum()*100:5.1f}%")


if __name__ == "__main__":
    main()
