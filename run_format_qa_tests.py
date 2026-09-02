"""
run_format_qa_tests.py
=======================
Arnés de pruebas enfocado en formato/fórmulas para el asistente Idania
(RD 1396), contra el mismo userid que pidió revisar el usuario (4).
Mismo patrón que run_qa_tests.py: sesiones de <=10 preguntas.
"""
import json
import re
import time
import requests

BASE = "http://localhost:8100"
TOKEN = "XF7MaBrGZG8r_tEeK1r6MqKB_OXIQPKp"
COURSEID = 1396
USERID = 4

with open("qa_format_test_questions.json", encoding="utf-8") as f:
    questions = json.load(f)

grupos = [questions[0:10], questions[10:15]]

resultados = []

for grupo in grupos:
    # nueva-conversacion (force_new=True) en vez de GET: el GET reutiliza la
    # sesión activa existente de userid=4 (que el usuario ya usó a mano para
    # probar la tabla de frecuencia), así que llegaba con preguntas ya
    # consumidas y pegaba contra el tope de 10 casi de inmediato.
    r = requests.post(
        f"{BASE}/asistente/{TOKEN}/nueva-conversacion",
        json={"courseid": COURSEID, "userid": USERID},
        timeout=30,
    )
    r.raise_for_status()
    sesion_id = r.json()["sesion_id"]
    print(f"\n=== sesion_id={sesion_id} ({len(grupo)} preguntas) ===")

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
                "userid": USERID,
                "http_status": resp.status_code,
                "ok": ok,
                "wall_ms": round(wall_ms),
                "answer": data.get("answer") if ok else None,
                "detail": data.get("detail") if not ok else None,
                "sources": [
                    {"filename": s.get("filename"), "source_type": s.get("source_type"), "section_name": s.get("section_name")}
                    for s in (data.get("sources") or [])
                ] if ok else [],
                "tokens_consumidos": data.get("tokens_consumidos") if ok else None,
                "timings_ms": (data.get("config_used") or {}).get("timings_ms") if ok else None,
            }
        except Exception as e:
            resultado = {**q, "sesion_id": sesion_id, "userid": USERID, "ok": False, "error": str(e)}
        resultados.append(resultado)
        estado = "OK" if resultado.get("ok") else f"FALLO({resultado.get('http_status')})"
        print(f"  [{q['id']:2}] {estado} {resultado.get('wall_ms', '-')}ms - {q['pregunta'][:60]}")

with open("qa_format_test_results.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print(f"\nTotal resultados: {len(resultados)} -> qa_format_test_results.json")
