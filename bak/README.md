# bak — 보류 디렉터리

검색 실험에 쓰는 코드베이스를 **`tos-skeleton` 하나로 고정**하기로 합의(2026-08-07 미팅)하여,
사용하지 않는 디렉터리를 이곳으로 옮겼다.

| 디렉터리 | 마지막 작업 | 작업자 |
|---|---|---|
| `dr-dci/` | 2026-07-27 · AI-Hub 의료·법률 평가셋 | seyeong · atdev_noah |
| `part0-collection-document-baseline/` | 2026-08-05 · BM25F round-2, C4 라우터 | donggyu-ralph |

**삭제가 아니라 이동이다.** 이력은 그대로 보존되며 필요하면 되돌릴 수 있다.

```bash
git mv bak/dr-dci dr-dci
```

## 주의 — 관련 브랜치가 살아 있다

아래 브랜치들이 이 디렉터리들을 건드린다. **머지할 때 경로 충돌이 난다.**
해당 브랜치 작업자는 `bak/` 경로로 리베이스하거나, 머지 후 경로를 옮겨야 한다.

```
origin/feat/aihub-dataset
origin/fix/restore-aihub-datasets
origin/feat/noah  ·  origin/feat/noah_data
origin/feat/part0-5-lexical-improvements
origin/feat/part0-6-r1
origin/feature_ralph_noah_integration
```
