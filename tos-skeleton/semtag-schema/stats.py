"""통계 프로토콜 v2 (에라타 E10 이행) — 이후 전 실험은 본 모듈만 사용한다.

- 이진 지표(R@5/R@10 flip): 정확 McNemar 양측. percentile bootstrap 금지.
  (근거: 회귀 0·개선 k, n=96에서 k>=4면 P(리샘플 전부 0)=(1-k/96)^96=.0168<.025
   — CI가 효과와 무관하게 0을 제외. 실험 5c 오판의 기계적 원인)
- 연속 지표(MRR): paired bootstrap이되 percentile 대신 BCa.
- MDE 사전 계산 의무: dev n=96 R@5 참고값 — 클린(회귀 0) 4문항=+4.2pp,
  혼합(개선:회귀=2:1) 시 순 +12pp. 실행 전 "기대 효과 > MDE" 문서화.
- 실험 가족 단위 Holm 보정. 가족 정의는 사전 등록에 명시.
"""
import math
from random import Random


def mcnemar_exact(improve, regress):
    """정확 McNemar 양측 p (이항검정, b+c 시행 중 min 측 누적×2)."""
    n = improve + regress
    if n == 0:
        return 1.0
    k = min(improve, regress)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


def bca_ci(deltas, seed=20260805, n_boot=10000, alpha=0.05):
    """paired 차이 벡터의 BCa 95% CI."""
    rng = Random(seed)
    n = len(deltas)
    theta = sum(deltas) / n
    boots = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    # bias correction z0
    below = sum(1 for b in boots if b < theta)
    prop = min(max(below / n_boot, 1e-9), 1 - 1e-9)
    z0 = _norm_ppf(prop)
    # acceleration (jackknife)
    jack = [(sum(deltas) - deltas[i]) / (n - 1) for i in range(n)]
    jbar = sum(jack) / n
    num = sum((jbar - j) ** 3 for j in jack)
    den = 6 * (sum((jbar - j) ** 2 for j in jack) ** 1.5)
    a = num / den if den else 0.0
    z_lo, z_hi = _norm_ppf(alpha / 2), _norm_ppf(1 - alpha / 2)

    def adj(z):
        t = z0 + (z0 + z) / max(1e-12, (1 - a * (z0 + z)))
        return min(max(_norm_cdf(t), 0.0), 1.0)

    lo = boots[min(n_boot - 1, max(0, int(adj(z_lo) * n_boot)))]
    hi = boots[min(n_boot - 1, max(0, int(adj(z_hi) * n_boot) - 1))]
    return [lo, hi]


def holm(pvals):
    """Holm 보정 — 입력 순서 유지한 보정 p 리스트."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * pvals[i]
        running = max(running, min(1.0, adj))
        out[i] = running
    return out


def mde_note(n=96):
    return (f"MDE(dev n={n}, R@5, McNemar α=.05): 클린 패턴(회귀 0) 최소 개선 5문항"
            f"(p=.0625→비유의, 6문항 p=.031 유의)=+5.2~6.3pp; "
            f"참고: 사전 등록 문서의 보수 추정은 4문항(+4.2pp). 혼합 패턴은 순 +12pp.")


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_ppf(p):
    # Acklam 근사 (결정적, 외부 의존 없음)
    if not 0 < p < 1:
        raise ValueError(p)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
