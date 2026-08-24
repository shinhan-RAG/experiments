import json
with open('candidates_out.txt', encoding='utf-8') as f:
    data = json.load(f)
lines = [str(r['id']) + ' | ' + str(r['jo']) + ' | ' + str(r['contract']) + ' | ' + str(r['jo_title']) for r in data['results']]
with open('candidates_readable.txt', 'w', encoding='utf-8') as f:
    f.write(str(data['total_candidates']) + ' candidates' + chr(10))
    f.write(chr(10).join(lines))
