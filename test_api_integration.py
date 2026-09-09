"""Test API integration for calculator"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

fail=0
pas=0

try:
    from backend.app import app
    client = TestClient(app)
    print("=== API health ===")
    r = client.get("/api/health")
    print(f"/api/health {r.status_code} {r.json()}")

    # Test math bypass without data (should work even if chroma empty)
    print("\n=== POST /api/chat math ===")
    tests = [
        ("2.5 + 3.7", "6.2"),
        ("10 - 4.2", "5.8"),
        ("3 * 2.5", "7.5"),
        ("7 / 2", "3.5"),
        ("hai cộng ba", "5"),
        ("2,5 * 2", "5"),
        ("căn bậc hai của 16", "4"),
        ("5% * 200", "10"),
        ("0.1 + 0.2", "0.3"),
        ("1.000,5 + 0.5", "1001"),
    ]
    for q, expected in tests:
        resp = client.post("/api/chat", json={"question": q, "use_rerank": False, "show_context": False})
        try:
            j = resp.json()
        except:
            print(f"FAIL {q!r} non-json {resp.text}")
            fail+=1
            continue
        if resp.status_code != 200:
            print(f"FAIL {q!r} status {resp.status_code} {j}")
            fail+=1
            continue
        got = j.get("math_result")
        if got != expected:
            print(f"FAIL {q!r} got math_result {got!r} expected {expected!r} answer={j.get('answer')!r}")
            fail+=1
        else:
            pas+=1
            print(f"PASS {q!r} -> {j.get('math_expression')} = {got} session={j.get('session_id')}")

    # Test session follow-up
    print("\n=== Session follow-up ===")
    # create session
    s = client.post("/api/sessions", json={"title":"test math", "style":"casual"})
    sid = s.json().get("session_id")
    print(f"created session {sid}")
    r1 = client.post("/api/chat", json={"question":"5 * 2", "session_id": sid})
    j1 = r1.json()
    print(f"5*2 -> {j1.get('math_result')} expr {j1.get('math_expression')}")
    if j1.get("math_result") == "10":
        pas+=1
        print("PASS session 5*2")
    else:
        fail+=1
        print("FAIL session 5*2")

    sid_returned = j1.get("session_id")
    # follow up
    r2 = client.post("/api/chat", json={"question":"cộng thêm 5", "session_id": sid_returned})
    j2 = r2.json()
    print(f"cộng thêm 5 -> {j2.get('math_result')} expr {j2.get('math_expression')}")
    if j2.get("math_result") == "15":
        pas+=1
        print("PASS follow-up cộng thêm 5")
    else:
        fail+=1
        print(f"FAIL follow-up expected 15 got {j2.get('math_result')}")

    r3 = client.post("/api/chat", json={"question":"nó nhân 2", "session_id": sid_returned})
    j3 = r3.json()
    print(f"nó nhân 2 -> {j3.get('math_result')}")
    if j3.get("math_result") == "30":
        pas+=1
        print("PASS nó nhân 2")
    else:
        fail+=1
        print(f"FAIL nó nhân 2 expected 30 got {j3.get('math_result')}")

    # Test divide by zero
    print("\n=== Edge error ===")
    r = client.post("/api/chat", json={"question":"5 / 0"})
    j = r.json()
    print(f"5/0 -> status {r.status_code} answer {j.get('answer')[:200]!r}")
    if "không thể chia cho 0" in j.get("answer",""):
        pas+=1
        print("PASS divide by zero error")
    else:
        fail+=1
        print("FAIL divide by zero")

    # Test non-math (should still work, but may need mock? We'll check that it doesn't crash)
    # We mock retrieve to avoid needing real data: but app will try to retrieve; if no data, it returns not_ready? Actually early bypass only for math/social.
    # For non-math, if chroma empty, it returns "Chua co du lieu..."
    r = client.post("/api/chat", json={"question":"quy chế đào tạo là gì"})
    j = r.json()
    print(f"non-math quy chế -> {j.get('answer')[:200]!r}")
    # Should be either dummy or not_ready message
    if j.get("answer"):
        pas+=1
        print("PASS non-math returns something")
    else:
        fail+=1
        print("FAIL non-math")

    print(f"\nTOTAL API: {pas} PASS {fail} FAIL")
    sys.exit(1 if fail else 0)

except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
