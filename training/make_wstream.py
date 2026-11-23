#!/usr/bin/env python3
"""Stream dei pesi per la ZCU104 (w_stream_nmac256_<TASK>.mem) per H e N qualsiasi.

Formato di s4d_wload_axis.v (ZCU104: NMAC = 2H, una sola fase), per layer:
  [0]                    bn_sh (6 bit)                    quant_report.json, affine[m].SH
  2 * H*MODES beat       coef {b1,b0,a2,a1} 80 bit, 2 beat l<m>_s4d_coef.mem (ordine h*MODES + n)
  H beat                 {bsh, D}                         l<m>_s4d_D.mem
  H beat                 bn_w                             l<m>_bn_w.mem
  H beat                 bn_b                             l<m>_bn_b.mem
  2H beat                bout                             l<m>_s4d_bout.mem
  H righe x (2H*12/64)   W[c][h], c = 0..2H-1, LSB first  l<m>_s4d_wout.mem riordinato:
                         riga h = concatenazione delle fasi ph = 0..2H/64-1 della riga ph*H + h
                         del formato a 64 MAC (stessa permutazione degli script SC/RadioML del paper).
I 6 layer sono concatenati in ordine. Con H = 128 riproduce bit per bit lo stream del paper
(controllo: --check <mem_dir> <stream di riferimento>).
uso: python make_wstream.py <mem_dir> <file_uscita> [--check <stream_di_riferimento>]
"""
import json, sys
from pathlib import Path

NLAYER, W_W, NMAC_Z = 6, 12, 64


def rows(p):
    return [int(t, 16) for t in Path(p).read_text().split()]


def build(mem):
    mem = Path(mem)
    rep = json.loads((mem / "quant_report.json").read_text())
    sh = [a["SH"] for a in rep["affine"][:NLAYER]]
    H = len(rows(mem / "l0_s4d_D.mem"))
    coef0 = rows(mem / "l0_s4d_coef.mem")
    MODES = len(coef0) // H
    TH, NPH = 2 * H, (2 * H) // NMAC_Z
    BZ = NMAC_Z * W_W // 64                    # 12 beat per riga del formato a 64 MAC
    out = []
    for m in range(NLAYER):
        b = [sh[m] & 0x3F]
        coef = rows(mem / f"l{m}_s4d_coef.mem"); assert len(coef) == H * MODES
        for v in coef:
            b += [v & ((1 << 64) - 1), (v >> 64) & 0xFFFF]
        D = rows(mem / f"l{m}_s4d_D.mem"); assert len(D) == H
        b += [v & ((1 << 21) - 1) for v in D]
        bw = rows(mem / f"l{m}_bn_w.mem"); assert len(bw) == H
        b += [v & 0xFFFF for v in bw]
        bb = rows(mem / f"l{m}_bn_b.mem"); assert len(bb) == H
        b += [v & 0xFFFFFFFF for v in bb]
        bo = rows(mem / f"l{m}_s4d_bout.mem"); assert len(bo) == TH
        b += [v & 0xFFFFFFFF for v in bo]
        wo = rows(mem / f"l{m}_s4d_wout.mem"); assert len(wo) == NPH * H
        for h in range(H):
            for ph in range(NPH):
                r = wo[ph * H + h]
                b += [(r >> (64 * k)) & ((1 << 64) - 1) for k in range(BZ)]
        nbeat = 1 + 2 * H * MODES + 3 * H + TH + H * NPH * BZ
        assert len(b) == nbeat
        # verifica sui pesi: uscita c = ph*64 + mac, 12 bit a [c*12 +: 12] della riga h
        wbeg = 1 + 2 * H * MODES + 3 * H + TH
        for h in (0, H // 2, H - 1):
            row = 0
            for k in range(NPH * BZ):
                row |= b[wbeg + h * NPH * BZ + k] << (64 * k)
            for ph in range(NPH):
                for mac in (0, 17, 63):
                    assert (row >> ((ph * 64 + mac) * 12)) & 0xFFF == (wo[ph * H + h] >> (mac * 12)) & 0xFFF
        out += b
    return out, dict(H=H, MODES=MODES, NBEAT=nbeat, righe=len(out))


if __name__ == "__main__":
    beats, info = build(sys.argv[1])
    Path(sys.argv[2]).write_text("\n".join(f"{x:016x}" for x in beats) + "\n")
    print(f"{sys.argv[2]}: {info}")
    if len(sys.argv) > 4 and sys.argv[3] == "--check":
        same = Path(sys.argv[2]).read_text() == Path(sys.argv[4]).read_text()
        print("identico al riferimento" if same else "DIVERSO dal riferimento")
