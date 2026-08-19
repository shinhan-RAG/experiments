#!/usr/bin/env python3
"""paired 검정 도구 — 문항 단위 점수(연속·비율)에 맞는 검정만 둔다.

sign_test(a, b)        : 부호검정(정확 이항). a>b 승, a<b 패, 동점 제외
bootstrap_ci(a, b)     : 평균 차이의 BCa 부트스트랩 95% CI
holm(pvalues)          : Holm 보정
mde_sign(n, disagree)  : 불일치율 하에서 부호검정 80% 검출력 최소 효과(문항 수 기준, 정규 근사)
CLI: python stats.py --a ranks_A.jsonl --b ranks_B.jsonl --key R@5
"""
import argparse, json, math, random
from math import comb


def sign_test(a, b):
    w = sum(1 for x, y in zip(a, b) if x > y); l = sum(1 for x, y in zip(a, b) if x < y)
    n = w + l
    if n == 0:
        return {"win": 0, "lose": 0, "p": 1.0}
    k = min(w, l)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n)
    return {"win": w, "lose": l, "p": p}


def bootstrap_ci(a, b, B=10000, seed=0, alpha=0.05):
    d = [x - y for x, y in zip(a, b)]; n = len(d); rnd = random.Random(seed)
    mean = sum(d) / n
    boots = sorted(sum(d[rnd.randrange(n)] for _ in range(n)) / n for _ in range(B))
    # BCa
    z0 = _ppf(sum(1 for x in boots if x < mean) / B)
    jack = [(sum(d) - d[i]) / (n - 1) for i in range(n)]; jm = sum(jack) / n
    num = sum((jm - j) ** 3 for j in jack); den = 6 * (sum((jm - j) ** 2 for j in jack) ** 1.5)
    acc = num / den if den else 0.0
    def q(p):
        z = z0 + (z0 + _ppf(p)) / (1 - acc * (z0 + _ppf(p)))
        return boots[min(B - 1, max(0, int(_cdf(z) * B)))]
    return {"delta": mean, "ci95": [q(alpha / 2), q(1 - alpha / 2)]}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); m = len(ps); out = [0.0] * m; run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); out[i] = min(1.0, run)
    return out


def mde_sign(n, disagree, power=0.8, alpha=0.05):
    """불일치 문항 수 m=n*disagree 에서 승률 p 를 검출하려면 |p-.5| >= (z_a/2+z_b)/(2*sqrt(m)). 순효과(pp) = 2*(p-.5)*m/n"""
    m = max(1, n * disagree); z = 1.96 + 0.84
    dp = z / (2 * math.sqrt(m))
    return {"m_disagree": m, "min_winrate": 0.5 + dp, "net_effect_pp": round(100 * 2 * dp * m / n, 2)}


def _ppf(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    # Acklam 근사
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - 0.02425:
        q = math.sqrt(-2 * math.log(1 - p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--key", default="R@5")
    x = ap.parse_args()
    A = {json.loads(l)["qid"]: json.loads(l) for l in open(x.a)}; Bm = {json.loads(l)["qid"]: json.loads(l) for l in open(x.b)}
    q = sorted(set(A) & set(Bm)); a = [A[i][x.key] for i in q]; b = [Bm[i][x.key] for i in q]
    print(json.dumps({"n": len(q), "mean_a": sum(a)/len(a), "mean_b": sum(b)/len(b), "sign": sign_test(a, b), "bca": bootstrap_ci(a, b),
                      "mde_at_observed_disagree": mde_sign(len(q), sum(1 for u, v in zip(a, b) if u != v)/len(q))}, ensure_ascii=False))
