#!/usr/bin/env python3
"""Why did the rollout trials fail? Reads a Harbor jobs dir written by leonardo_laion_agent_rollout.sbatch and reports, per outcome class:
turn budget use (n_episodes vs max), early termination, errors, verifier failures, token use, and one example per class.

  analyze_rollout.py <rollout dir>/jobs/rollout [--max-turns 30] [--examples 2]
"""
import argparse, collections, glob, json, os, re


def last_text(traj, n=500):
    steps = traj.get("steps") or []
    for s in reversed(steps):
        if s.get("source") == "agent":
            m = s.get("message") or ""
            return (m if isinstance(m, str) else json.dumps(m))[-n:]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobdir")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--examples", type=int, default=2)
    a = ap.parse_args()
    classes = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.jobdir, "rts*/result.json"))):
        td = os.path.dirname(f)
        r = json.load(open(f))
        ar = r.get("agent_result") or {}
        meta = ar.get("metadata") or {}
        eps = meta.get("n_episodes")
        reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        exc = r.get("exception_info")
        info = {"trial": os.path.basename(td), "reward": reward, "episodes": eps,
                "in_tok": ar.get("n_input_tokens"), "out_tok": ar.get("n_output_tokens"), "dir": td}
        if exc:
            info["exc"] = "%s: %s" % (exc.get("exception_type"), str(exc.get("exception_message"))[:160].replace("\n", " | "))
            classes["ERROR " + exc.get("exception_type", "?")].append(info)
        elif reward is not None and reward >= 1.0:
            classes["PASS"].append(info)
        elif eps is not None and eps >= a.max_turns:
            classes["FAIL: used all %d turns (never finished)" % a.max_turns].append(info)
        else:
            classes["FAIL: agent stopped early (declared done / gave up)"].append(info)
    total = sum(len(v) for v in classes.values())
    print("trials:", total)
    for k, v in sorted(classes.items(), key=lambda kv: -len(kv[1])):
        eps = [x["episodes"] for x in v if x["episodes"] is not None]
        outs = [x["out_tok"] for x in v if x["out_tok"]]
        print("\n== %s: %d (%.0f%%)  mean_episodes=%s  mean_out_tokens=%s" % (
            k, len(v), 100.0 * len(v) / max(total, 1),
            ("%.1f" % (sum(eps) / len(eps))) if eps else "-", ("%.0f" % (sum(outs) / len(outs))) if outs else "-"))
        for x in v[: a.examples]:
            print("  -", x["trial"][:40], "reward", x["reward"], "episodes", x["episodes"], x.get("exc", ""))
            vt = os.path.join(x["dir"], "verifier", "test-stdout.txt")
            if os.path.isfile(vt) and not k.startswith("PASS"):
                tail = [l for l in open(vt, errors="replace").read().splitlines() if l.strip()][-4:]
                for l in tail:
                    print("      verifier:", l[:180])
            tj = os.path.join(x["dir"], "agent", "trajectory.json")
            if os.path.isfile(tj) and not k.startswith("ERROR"):
                try:
                    t = last_text(json.load(open(tj)), 300).replace("\n", " ")
                    print("      last agent msg:", t)
                except Exception as e:  # noqa: BLE001
                    print("      (trajectory unreadable: %s)" % e)


if __name__ == "__main__":
    main()
