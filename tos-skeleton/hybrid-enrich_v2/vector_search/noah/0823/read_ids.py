import sys, subprocess, json

ids = sys.argv[1:]
for i in ids:
    p = subprocess.run(['python', 'agent_tools.py', 'read', '--id', i], capture_output=True)
    data = p.stdout
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        text = data.decode('cp949')
    print(f"===== {i} =====")
    print(text)
    print()
