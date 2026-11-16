#!/usr/bin/env python3
"""Recursos por proceso cada 5 s: memoria y uso de GPU, RAM residente y CPU (servidor de pirate, grouper, sifter, emisor).
Uso: python monitor_procesos.py SALIDA.csv   (se detiene con SIGTERM)"""
import csv, os, re, signal, subprocess, sys, time

TICK = os.sysconf("SC_CLK_TCK")
seguir = True
signal.signal(signal.SIGTERM, lambda *a: globals().__setitem__("seguir", False))


def rol(args):
    if "run server" in args: return "pirate_server"
    if "grouper_v1" in args or "toy_grouper" in args or "grouper_testbench" in args: return "grouper"
    if "toy_sifter" in args: return "sifter"
    if "emisor_fil" in args: return "emisor"
    return None


def procesos():
    out = {}
    for p in os.listdir("/proc"):
        if not p.isdigit(): continue
        try:
            a = open(f"/proc/{p}/cmdline").read().replace("\0", " ")
            r = rol(a)
            if r:
                st = open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()
                rss = int(open(f"/proc/{p}/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
                out[int(p)] = (r, int(st[11]) + int(st[12]), rss)
        except OSError:
            pass
    return out


def gpu():
    mem, sm = {}, {}
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for l in q.splitlines():
        if "," in l:
            p, m = l.split(","); mem[int(p)] = float(m)
    q = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"], capture_output=True, text=True).stdout
    for l in q.splitlines():
        c = l.split()
        if len(c) > 4 and c[1].isdigit():
            sm[int(c[1])] = float(c[3]) if c[3] != "-" else 0.0
    return mem, sm


w = csv.writer(open(sys.argv[1], "w", newline=""))
w.writerow(["t", "rol", "pid", "gpu_mem_mib", "gpu_sm_pct", "rss_gib", "cpu_pct"])
prev, t_prev = procesos(), time.time()
while seguir:
    time.sleep(5)
    now, t = procesos(), time.time()
    mem, sm = gpu()
    for p, (r, cpu, rss) in now.items():
        dcpu = (cpu - prev[p][1]) / TICK / (t - t_prev) * 100 if p in prev else float("nan")
        w.writerow([round(t, 1), r, p, mem.get(p, 0.0), sm.get(p, 0.0), round(rss / 2**30, 3), round(dcpu, 1)])
    prev, t_prev = now, t
