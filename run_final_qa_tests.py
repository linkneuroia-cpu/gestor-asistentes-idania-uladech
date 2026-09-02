"""
run_final_qa_tests.py
======================
20 pruebas finales contra Idania (RD 1396) tras: fix de listas anidadas,
fix de \\[ \\] multilinea, tablas DOCX/PDF, pix2tex con GPU, sesiones/
carpetas y token editable. Mismo patron que run_format_qa_tests.py.
"""
import json
import requests

BASE = "http://localhost:8100"
TOKEN = "XF7MaBrGZG8r_tEeK1r6MqKB_OXIQPKp"
COURSEID = 1396
USERID = 9999

with open("qa_final_test_questions.json", encoding="utf-8") as f:
    questions = json.load(f)

grupos = [questions[0:10], questions[10:20]]
resultados = []

for grupo in grupos:
    r = requests.post(
        f"{BASE}/asistente/{TOKEN}/nueva-conversacion",
        json={"courseid": COURSEID, "userid": USERID},
        timeout=30,
    )
    r.raise_for_status()
    sesion_id = r.json()["sesion_id"]
    print(f"\n=== sesion_id={sesion_id} ({len(grupo)} preguntas) ===")

    for q in grupo:
        try:
            resp = requests.post(
                f"{BASE}/asistente/{TOKEN}/mensaje",
                json={"sesion_id": sesion_id, "pregunta": q["pregunta"]},
                timeout=120,
            )
            ok = resp.status_code == 200
            data = resp.json()
            resultado = {
                **q,
                "sesion_id": sesion_id,
                "ok": ok,
                "answer": data.get("answer") if ok else None,
                "detail": data.get("detail") if not ok else None,
                "sources": [s.get("filename") for s in (data.get("sources") or [])] if ok else [],
                "tokens_consumidos": data.get("tokens_consumidos") if ok else None,
                "timings_ms": (data.get("config_used") or {}).get("timings_ms") if ok else None,
            }
        except Exception as e:
            resultado = {**q, "sesion_id": sesion_id, "ok": False, "error": str(e)}
        resultados.append(resultado)
        estado = "OK" if resultado.get("ok") else f"FALLO"
        print(f"  [{q['id']:2}] {estado} - {q['pregunta'][:60]}")

with open("qa_final_test_results.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print(f"\nTotal resultados: {len(resultados)} -> qa_final_test_results.json")
