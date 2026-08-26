import json, re, io

files = [
    r"C:\Users\Atdev-pc\.claude\projects\C--Users-Atdev-pc-Desktop-wiki-experiments-tos-skeleton-hybrid-enrich-v2-vector-search-noah-0823\295d28a9-34eb-4c74-9ad4-30bdd1e46ac7\tool-results\b13lnhj71.txt",
    r".\_batch.txt",
]

records = {}
for fp in files:
    with io.open(fp, encoding='utf-8') as f:
        content = f.read()
    parts = re.split(r'=== (\w+) ===\n', content)
    for i in range(1, len(parts), 2):
        cid = parts[i]
        jsonstr = parts[i+1].strip()
        try:
            obj = json.loads(jsonstr)
            records[cid] = obj
        except Exception as e:
            print("FAIL", cid, str(e)[:100])

with io.open('./_parsed.json', 'w', encoding='utf-8') as out:
    json.dump(records, out, ensure_ascii=False, indent=1)

print(len(records), "records parsed")
print(sorted(records.keys()))
