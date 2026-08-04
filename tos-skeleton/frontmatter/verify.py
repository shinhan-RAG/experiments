#!/usr/bin/env python3
"""게이트 1 — frontmatter 원문 대조 자동 검증.

composition: 각 rider가 문서 전문(목차 포함)에 등장하는가
coverage: source 조가 특약 단위에 실존하는가 + key_terms가 근거 조 원문에
          (공백 무시) 부분일치하는가 + name(급부명)이 원문에 등장하는가
사용: python3 verify.py <version>
출력: out/<version>.gate1.json + 콘솔 리포트
"""
import json
import re
import sys
from pathlib import Path

BASE = Path("/Users/seyoung/workspace/braincrew/experiments/tos-skeleton")


def ns(s):
    return re.sub(r"\s+", "", s)


def main(version):
    fm = json.loads((BASE / "frontmatter" / "out" / f"{version}.frontmatter.json")
                    .read_text(encoding="utf-8"))
    units = json.loads((BASE / "data" / "segmented" / f"{version}.json")
                       .read_text(encoding="utf-8"))
    fulltext = ns((BASE / "data" / "raw_text" / f"{version}.txt")
                  .read_text(encoding="utf-8"))
    unit_map = {u["unit"]: u for u in units}

    report = {"version": version, "composition": {"ok": 0, "fail": []},
              "coverage": {"riders": 0, "benefits": 0, "term_ok": 0,
                           "term_fail": 0, "fails": []}}

    for c in fm["composition"]:
        if ns(c["rider"]) in fulltext:
            report["composition"]["ok"] += 1
        else:
            report["composition"]["fail"].append(c["rider"])

    # 역방향: 페이지 헤더에 등장하는 특약명 전수가 composition에 있는가 (누락 검출)
    raw_lines = (BASE / "data" / "raw_text" / f"{version}.txt").read_text(
        encoding="utf-8").splitlines()
    coded = re.compile(r"^\d{15,}\s+(.+?)(?:\s+\d{1,4})?$")
    header_names = set()
    for line in raw_lines:
        m = coded.match(line)
        if m:
            name = re.sub(r"_(대면|TM)$", "", m.group(1).strip())
            if name != "SHINHAN LIFE" and "특약" in name:
                header_names.add(ns(name))
    comp_names = {ns(c["rider"]) for c in fm["composition"]}
    missing = sorted(h for h in header_names if h not in comp_names)
    report["composition"]["reverse_header_names"] = len(header_names)
    report["composition"]["reverse_missing"] = missing
    print(f"역방향: 페이지 헤더 특약명 {len(header_names)}종 중 "
          f"composition 누락 {len(missing)}건")
    for miss in missing[:8]:
        print(f"  MISSING {miss[:60]}")

    for cov in fm["coverage"]:
        unit = unit_map.get(cov["rider"])
        report["coverage"]["riders"] += 1
        # 근거 조 원문 (중복 조 병합)
        art_text = {}
        for a in (unit["articles"] if unit else []):
            art_text.setdefault(a["no"], []).extend(a["text"])
        unit_full = ns(" ".join(t for ts in art_text.values() for t in ts))
        for b in cov.get("benefits", []):
            report["coverage"]["benefits"] += 1
            src = ns(" ".join(art_text.get(b.get("source", ""), [])))
            fails = []
            if not src:
                fails.append(f"근거 조 {b.get('source')} 없음")
            else:
                # 급부명: 합성 명칭 허용 — 구성 토큰이 각각 근거 조에 실존하면 통과
                name = b.get("name", "")
                if name and ns(name) not in src:
                    tokens = [t for t in re.split(r"[\s()·]+", name) if len(t) >= 2]
                    missing = [t for t in tokens
                               if ns(t) not in src and ns(t) not in unit_full]
                    if missing:
                        fails.append(f"급부명 구성 토큰 미일치: {missing}")
                # key_terms: 원문 그대로(공백 무시) 필수
                for t in b.get("key_terms", []):
                    if t and ns(t) not in src:
                        fails.append(f"용어 미일치: {t}")
            if fails:
                report["coverage"]["term_fail"] += 1
                report["coverage"]["fails"].append(
                    {"rider": cov["rider"], "benefit": b.get("name"), "fails": fails})
            else:
                report["coverage"]["term_ok"] += 1

    dest = BASE / "frontmatter" / "out" / f"{version}.gate1.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    c, v = report["composition"], report["coverage"]
    print(f"composition: {c['ok']}/{c['ok']+len(c['fail'])} 목차 대조 통과")
    for f in c["fail"][:5]:
        print(f"  FAIL {f}")
    print(f"coverage: 급부 {v['benefits']}건 중 검증 통과 {v['term_ok']}, 실패 {v['term_fail']}")
    for f in v["fails"][:8]:
        print(f"  FAIL {f['rider'][:36]} / {f['benefit']}: {f['fails'][:2]}")


if __name__ == "__main__":
    main(sys.argv[1])
