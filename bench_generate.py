import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

import requests


def pct(values, p: float):
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def worker(url: str, payload: dict, timeout_s: float):
    # IMPORTANT: one Session per thread -> thread-safe usage pattern
    with requests.Session() as sess:
        t0 = time.perf_counter()
        try:
            r = sess.post(url, json=payload, timeout=timeout_s)
            latency = time.perf_counter() - t0
            status = r.status_code

            if status != 200:
                return {"ok": False, "status": status, "latency": latency, "err": r.text[:200]}

            data = r.json()
            # server may return {"error":"OOM", ...}
            if isinstance(data, dict) and data.get("error"):
                return {"ok": False, "status": status, "latency": latency, "err": f"{data.get('error')}: {str(data)[:200]}"}

            return {"ok": True, "status": status, "latency": latency, "data": data}

        except Exception as e:
            latency = time.perf_counter() - t0
            return {"ok": False, "status": "EXC", "latency": latency, "err": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/generate")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--requests", type=int, default=15, help="total requests")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--ctx_len", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--prompt", default="In one sentence, what is KV cache?")
    args = ap.parse_args()

    payload = {
        "prompt": args.prompt,
        "ctx_len": args.ctx_len,
        "max_new_tokens": args.max_new_tokens,
    }

    t_start = time.perf_counter()
    results = []
    statuses = Counter()

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [
            ex.submit(worker, args.url, payload, args.timeout)
            for _ in range(args.requests)
        ]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            statuses[str(res["status"])] += 1

    total_s = time.perf_counter() - t_start
    oks = [r for r in results if r["ok"]]
    fails = [r for r in results if not r["ok"]]
    lat_ok = [r["latency"] for r in oks]

    rps = (len(results) / total_s) if total_s > 0 else 0.0

    tps_list = []
    vram_alloc = []
    vram_resv = []

    for r in oks:
        d = r.get("data") or {}
        if isinstance(d, dict):
            if "tokens_per_sec" in d:
                try:
                    tps_list.append(float(d["tokens_per_sec"]))
                except:
                    pass
            if "peak_vram_alloc_gb" in d:
                try:
                    vram_alloc.append(float(d["peak_vram_alloc_gb"]))
                except:
                    pass
            if "peak_vram_reserved_gb" in d:
                try:
                    vram_resv.append(float(d["peak_vram_reserved_gb"]))
                except:
                    pass

    print("=== BENCH: /generate (non-stream) ===")
    print(f"url              : {args.url}")
    print(f"concurrency      : {args.concurrency}")
    print(f"total_requests   : {args.requests}")
    print(f"ctx_len          : {args.ctx_len}")
    print(f"max_new_tokens   : {args.max_new_tokens}")
    print(f"timeout_s        : {args.timeout}")
    print("=== SUMMARY ===")
    print(f"success          : {len(oks)}")
    print(f"fail             : {len(fails)}")
    print(f"success_rate     : {100.0 * len(oks) / max(1, len(results)):.2f}%")
    print(f"total_time_s     : {total_s:.4f}")
    print(f"req_per_sec      : {rps:.2f}")

    if lat_ok:
        print("=== LATENCY (success only) ===")
        print(f"avg_s            : {sum(lat_ok)/len(lat_ok):.4f}")
        print(f"p50_s            : {pct(lat_ok, 50):.4f}")
        print(f"p90_s            : {pct(lat_ok, 90):.4f}")
        print(f"p95_s            : {pct(lat_ok, 95):.4f}")
        print(f"p99_s            : {pct(lat_ok, 99):.4f}")
        print(f"min_s            : {min(lat_ok):.4f}")
        print(f"max_s            : {max(lat_ok):.4f}")

    if tps_list:
        print("=== TOKENS/SEC (from server, if provided) ===")
        print(f"avg_tokens_s     : {sum(tps_list)/len(tps_list):.2f}")
        print(f"min_tokens_s     : {min(tps_list):.2f}")
        print(f"max_tokens_s     : {max(tps_list):.2f}")

    if vram_alloc:
        print(f"peak_alloc_gb    : {max(vram_alloc):.2f}")
    if vram_resv:
        print(f"peak_reserved_gb : {max(vram_resv):.2f}")

    print("=== STATUS CODES ===")
    for k, v in statuses.most_common():
        print(f"{k:>6} : {v}")

    if fails:
        print("=== SAMPLE FAILURES (up to 5) ===")
        for r in fails[:5]:
            print(f"- status={r['status']} latency_s={r['latency']:.3f} err={r.get('err','')}")

if __name__ == "__main__":
    main()
