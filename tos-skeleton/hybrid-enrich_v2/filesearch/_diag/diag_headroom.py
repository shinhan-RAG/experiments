#!/usr/bin/env python3
"""헤드룸 진단(결정론, u2jo 조 단위 BM25): 실패가 후보 부재인지 순위인지, 태그 좌표(특약 scope)의 오라클 상한, 질문-정답 어휘 겹침."""
import json, collections, sys
sys.path.insert(0, '.'); sys.path.insert(0, '..'); sys.path.insert(0, '../..')
from bm25_compare import BM25
from scoring import bigrams, overlaps, score
U = [json.loads(l) for l in open('out/elements_u2jo.jsonl')]
G = [g for g in (json.loads(l) for l in open('out/gold_spans_train.jsonl')) if g['groups']]
bm = BM25([u['text'] for u in U])
def gold_units(g):
    return [u for u in U if any(overlaps(u, gr) for gr in g['groups'])]
rank_bins = collections.Counter(); r5 = []; r5_scope = []; r5_scope_dedup = []; overlap = []; ttype = collections.defaultdict(list)
scope_n = []
for g in G:
    ranked = [U[i] for i, _ in bm.search(g['q'], 1000)]
    s = score(ranked, g['groups']); r5.append(s['R@5'])
    gu = gold_units(g)
    # first-hit rank bins
    fr = next((r+1 for r, u in enumerate(ranked) if any(overlaps(u, gr) for gr in g['groups'])), None)
    rank_bins['1-5' if fr and fr<=5 else '6-10' if fr and fr<=10 else '11-20' if fr and fr<=20 else '21-100' if fr and fr<=100 else '101-1000' if fr else '>1000/none'] += 1
    # oracle: gold 특약 scope 만 남기기 (태그 contract 좌표를 완벽히 알 때의 상한)
    scopes = {u['contract_scope'] for u in gu}
    scope_n.append(len(scopes))
    rs = [u for u in ranked if u['contract_scope'] in scopes]
    r5_scope.append(score(rs, g['groups'])['R@5'])
    # 어휘 겹침: 질문 bigram 중 gold 텍스트에 존재하는 비율
    qb = set(bigrams(g['q'])); gt = ''.join(u['text'] for u in gu)
    overlap.append(sum(1 for b in qb if b in gt.replace(' ', '')) / max(1, len(qb)))
    ttype[g['task_type']].append((s['R@5'], score(rs, g['groups'])['R@5']))
n = len(G)
print('n', n, 'BM25 u2jo R@5', round(sum(r5)/n, 3))
print('first-hit rank bins', dict(rank_bins))
print('oracle scope-filter R@5', round(sum(r5_scope)/n, 3), '| gold scopes per q', collections.Counter(scope_n))
ov = sorted(overlap); print('q-bigram overlap with gold: median', round(ov[n//2], 2), 'p25', round(ov[n//4], 2), 'share<0.3', round(sum(1 for x in ov if x < .3)/n, 2))
lo = [r for r, o in zip(r5, overlap) if o < .3]; hi = [r for r, o in zip(r5, overlap) if o >= .5]
print('R@5 when overlap<0.3:', round(sum(lo)/len(lo), 3), f'(n={len(lo)})', '| overlap>=0.5:', round(sum(hi)/len(hi), 3), f'(n={len(hi)})')
for t, v in ttype.items(): print(t, len(v), 'R@5', round(sum(x for x, _ in v)/len(v), 3), 'oracle-scope', round(sum(y for _, y in v)/len(v), 3))
# 중복 상용구: gold unit 텍스트가 우주 내 몇 번 반복되는가
dup = collections.Counter(u['text'].replace(' ', '') for u in U)
gd = [max([dup[u['text'].replace(' ', '')] for u in gold_units(g)] or [0]) for g in G]
print('gold unit duplicated (>1 identical copies) share', round(sum(1 for x in gd if x > 1)/n, 2), 'p50 copies', sorted(gd)[n//2])
