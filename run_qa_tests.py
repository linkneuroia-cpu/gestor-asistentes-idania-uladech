"""
run_qa_tests.py
================
Arnés de pruebas de calidad para el asistente Idania (RD 1396). Manda las
30 preguntas de qa_test_questions.json contra el asistente real (vía HTTP,
igual que lo haría el navegador), agrupadas en 3 sesiones de 10 preguntas
(respeta max_actividades_sesion=10), y guarda tiempos, tokens, fuentes y
la respuesta completa de cada una en qa_test_results.json.
"""
import json
import time
import requests

BASE = "http://localhost:8100"
TOKEN = "XF7MaBrGZG8r_tEeK1r6MqKB_OXIQPKp"
COURSEID = 1396

with open("qa_test_questions.json", encoding="utf-8") as f:
    questions = json.load(f)

# 3 sesiones de 10 preguntas cada una (respeta el tope de 10 por sesión).
grupos = [questions[0:10], questions[10:20], questions[20:30]]
userids = [9001, 9002, 9003]

resultados = []

for grupo, userid in zip(grupos, userids):
    print(f"\n=== Sesión userid={userid} ({len(grupo)} preguntas) ===")
    r = requests.get(f"{BASE}/asistente/{TOKEN}", params={"courseid": COURSEID, "userid": userid}, timeout=30)
    r.raise_for_status()
    import re
    m = re.search(r"let SESION_ID = (\d+)", r.text)
    sesion_id = int(m.group(1))
    print(f"  sesion_id={sesion_id}")

    for q in grupo:
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{BASE}/asistente/{TOKEN}/mensaje",
                json={"sesion_id": sesion_id, "pregunta": q["pregunta"]},
                timeout=90,
            )
            wall_ms = (time.perf_counter() - t0) * 1000
            ok = resp.status_code == 200
            data = resp.json()
            resultado = {
                **q,
                "sesion_id": sesion_id,
                "userid": userid,
                "http_status": resp.status_code,
                "ok": ok,
                "wall_ms": round(wall_ms),
                "answer": data.get("answer") if ok else None,
                "detail": data.get("detail") if not ok else None,
                "sources": [s.get("filename") for s in (data.get("sources") or [])] if ok else [],
                "source_types": [s.get("source_type") for s in (data.get("sources") or [])] if ok else [],
                "tokens_consumidos": data.get("tokens_consumidos") if ok else None,
                "timings_ms": (data.get("config_used") or {}).get("timings_ms") if ok else None,
                "preguntas_usadas": data.get("preguntas_usadas") if ok else None,
                "max_actividades_sesion": data.get("max_actividades_sesion") if ok else None,
            }
        except Exception as e:
            resultado = {**q, "sesion_id": sesion_id, "userid": userid, "ok": False, "error": str(e)}
        resultados.append(resultado)
        estado = "OK" if resultado.get("ok") else f"FALLO({resultado.get('http_status')})"
        print(f"  [{q['id']:2}] {estado} {resultado.get('wall_ms','-')}ms — {q['pregunta'][:60]}")

    # Test extra: la pregunta #11 de esta sesión (más allá del tope de 10)
    # debe rechazarse con 429 — verifica el límite de actividades.
    t0 = time.perf_counter()
    resp = requests.post(
        f"{BASE}/asistente/{TOKEN}/mensaje",
        json={"sesion_id": sesion_id, "pregunta": "Pregunta extra para probar el límite de la sesión."},
        timeout=30,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    resultados.append({
        "id": f"limite-{userid}",
        "categoria": "Límite de actividades",
        "pregunta": "(11ª pregunta de la sesión — debe rechazarse)",
        "sesion_id": sesion_id,
        "userid": userid,
        "http_status": resp.status_code,
        "ok": resp.status_code == 429,
        "wall_ms": round(wall_ms),
        "answer": None,
        "detail": resp.json().get("detail"),
    })
    print(f"  [limite] esperado 429, obtuvo {resp.status_code}")

with open("qa_test_results.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print(f"\nTotal resultados: {len(resultados)} -> qa_test_results.json")
