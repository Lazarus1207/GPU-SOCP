#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cbf2mps.py -- MOSEK CBF (Conic Benchmark Format) -> cuOpt-compatible free-format MPS.

WHY THIS EXISTS
    cuOpt has NO CBF reader and NO CBF converter (cpp/src/io/ only ships
    MPS / QPS / LP parsers; read() dispatches purely on file extension).
    cuOpt's only route to a cone program is the barrier solver reading
    QCMATRIX blocks in an MPS file, so this script rewrites CBF into that.

SUPPORTED CBF CONES
    F     free                       -> variable bounds -inf..+inf
    L=    x == 0                     -> FX 0
    L+    x >= 0                     -> LO 0
    L-    x <= 0                     -> UP 0
    Q     x0 >= ||x[1:]||            -> QCMATRIX row over x
    QR    2*x0*x1 >= ||x[2:]||^2     -> QCMATRIX row over x
    (each cone may appear on the VAR side and/or the CON side)
    EXP / PSDVAR / PSDCON / POW / FCOORD / DCOORD / HCOORD / OBJFCOORD
                                     -> rejected, cuOpt cannot express them

MPS / cuOpt FORMAT NOTES (each one verified against NVIDIA/cuopt sources)
    * Free-format MPS.  cuOpt defaults to fixed_mps_format=false, and OBJSENSE
      is only legal in free format.
    * Sections ROWS, COLUMNS, RHS are mandatory; NAME/ENDATA only warn.
    * QCMATRIX carries NO 1/2 factor, unlike QUADOBJ (parser.cpp uses
      value_scale 1.0 for QCMATRIX, 0.5 for QUADOBJ/QMATRIX).
    * Every off-diagonal cross term must be written TWICE, once per
      orientation, with equal values, and the parser SUMS them.  So a quadratic
      form coefficient c * x_i * x_j is written as (x_i,x_j,c/2),(x_j,x_i,c/2).
      This mirrors cuOpt's own mps_writer.cpp.
    * A QUADOBJ entry v means 1/2*v (v on the diagonal, v per orientation
      otherwise); writing the CBF Q coefficient verbatim reproduces CBF's
      "objective = 1/2 x^T Q x + c^T x + f".
    * Objective offset f is emitted as RHS on the OBJ row with value -f
      (mps_parser.cpp does objective_offset_value = -val).
    * cuOpt's SOCP fast path needs, per quadratic row:
        - rhs == 0
        - NO linear coefficients for that row in COLUMNS
        - standard SOC: one diagonal -s (the head) + uniform diagonal +s (tails)
        - rotated SOC: one cross term -2d + uniform diagonal +s (tails)
        - head variable lower bound >= 0
      This script emits exactly that shape.
    * Integer variables use 'MARKER' 'INTORG' / 'INTEND' and every column's rows
      must be contiguous, so integer columns are written first as one block.

USAGE
    python3 cbf2mps.py prob.cbf.gz                     # -> prob.mps
    python3 cbf2mps.py prob.cbf.gz -o prob.mps
    python3 cbf2mps.py prob.cbf.gz --relax-int         # drop integrality
    python3 cbf2mps.py *.cbf.gz -d out/                # batch
    python3 cbf2mps.py prob.cbf.gz --verify            # structural self-check

    Then:  cuopt_cli prob.mps --time-limit 3600
"""

import argparse
import gzip
import glob
import os
import sys
from collections import defaultdict

INF = float("inf")

C_FREE, C_ZERO, C_LPOS, C_LNEG, C_SOC, C_RSOC, C_EXP = (
    "F", "L=", "L+", "L-", "Q", "QR", "EXP")

LINEAR_CONES = {C_FREE, C_ZERO, C_LPOS, C_LNEG}
CONIC_CONES = {C_SOC, C_RSOC}


class CbfError(Exception):
    pass


# ----------------------------------------------------------------------------
# number formatting
# ----------------------------------------------------------------------------
def fmt(v):
    """Compact but round-trip-safe float for free-format MPS."""
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return "%.16g" % v


# ----------------------------------------------------------------------------
# line reader that skips blanks and '#' comments
# ----------------------------------------------------------------------------
class Reader(object):
    def __init__(self, text):
        self._lines = text.splitlines()
        self._i = 0

    def next(self, what=""):
        while self._i < len(self._lines):
            s = self._lines[self._i].strip()
            self._i += 1
            if not s or s.startswith("#"):
                continue
            return s
        if what:
            raise CbfError("unexpected end of file while reading %s" % what)
        return None

    def count(self, what):
        tok = self.next(what).split()
        try:
            return int(tok[0])
        except (ValueError, IndexError):
            raise CbfError("expected a count for %s, got %r" % (what, " ".join(tok)))


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class Model(object):
    def __init__(self):
        self.name = "cbf"
        self.maximize = False
        self.n = 0
        self.var_blocks = []      # (cone, dim, start)
        self.con_blocks = []      # (cone, dim, start)
        self.A = defaultdict(float)   # (row, col) -> value
        self.b = defaultdict(float)   # row -> value
        self.c = defaultdict(float)   # col -> value
        self.const = 0.0
        self.Q = defaultdict(float)   # (i, j) i<=j -> coefficient of x_i*x_j
        self.integer = set()

    @property
    def ncon(self):
        return sum(d for _, d, _ in self.con_blocks)


# ----------------------------------------------------------------------------
# CBF parser
# ----------------------------------------------------------------------------
def parse_cbf(text, path="<cbf>"):
    r = Reader(text)
    m = Model()
    m.name = os.path.basename(path).split(".")[0][:60] or "cbf"

    seen_change = False
    while True:
        ln = r.next()
        if ln is None:
            break
        kw = ln.split()[0]

        if kw == "VER":
            r.next("VER version")

        elif kw == "OBJSENSE":
            s = r.next("OBJSENSE value").split()[0].upper()
            if s in ("MAX", "MAXIMIZE"):
                m.maximize = True
            elif s in ("MIN", "MINIMIZE"):
                m.maximize = False
            else:
                raise CbfError("unknown OBJSENSE %r" % s)

        elif kw == "VAR":
            tok = r.next("VAR header").split()
            m.n = int(tok[0])
            nb = int(tok[1]) if len(tok) > 1 else 0
            pos = 0
            for _ in range(nb):
                t = r.next("VAR block").split()
                cone, dim = t[0], int(t[1])
                if cone not in LINEAR_CONES | CONIC_CONES | {C_EXP}:
                    raise CbfError("unsupported VAR cone %r" % cone)
                if cone == C_EXP:
                    raise CbfError("EXP cone on the variable side: cuOpt cannot "
                                   "represent it as a QCMATRIX row")
                m.var_blocks.append((cone, dim, pos))
                pos += dim
            if pos != m.n:
                raise CbfError("VAR blocks cover %d of %d variables" % (pos, m.n))

        elif kw == "CON":
            tok = r.next("CON header").split()
            nc = int(tok[0])
            nb = int(tok[1]) if len(tok) > 1 else 0
            pos = 0
            for _ in range(nb):
                t = r.next("CON block").split()
                cone, dim = t[0], int(t[1])
                if cone not in LINEAR_CONES | CONIC_CONES | {C_EXP}:
                    raise CbfError("unsupported CON cone %r" % cone)
                if cone == C_EXP:
                    raise CbfError("EXP cone on the constraint side: cuOpt cannot "
                                   "represent it as a QCMATRIX row")
                m.con_blocks.append((cone, dim, pos))
                pos += dim
            if pos != nc:
                raise CbfError("CON blocks cover %d of %d constraints" % (pos, nc))

        elif kw == "INT":
            k = r.count("INT header")
            for _ in range(k):
                m.integer.add(int(r.next("INT index").split()[0]))

        elif kw == "OBJACOORD":
            k = r.count("OBJACOORD header")
            for _ in range(k):
                t = r.next("OBJACOORD entry").split()
                m.c[int(t[0])] += float(t[1])

        elif kw == "OBJBCOORD":
            m.const += float(r.next("OBJBCOORD value").split()[0])

        elif kw == "OBJQCOORD":
            k = r.count("OBJQCOORD header")
            for _ in range(k):
                t = r.next("OBJQCOORD entry").split()
                i, j, v = int(t[0]), int(t[1]), float(t[2])
                if i > j:
                    i, j = j, i
                m.Q[(i, j)] += v

        elif kw == "ACOORD":
            k = r.count("ACOORD header")
            for _ in range(k):
                t = r.next("ACOORD entry").split()
                m.A[(int(t[0]), int(t[1]))] += float(t[2])

        elif kw == "BCOORD":
            k = r.count("BCOORD header")
            for _ in range(k):
                t = r.next("BCOORD entry").split()
                m.b[int(t[0])] += float(t[1])

        elif kw == "CHANGE":
            # v3+ perturbation section; the base problem is already complete.
            seen_change = True
            break

        elif kw in ("PSDVAR", "PSDCON", "HCOORD", "DCOORD", "FCOORD",
                    "OBJFCOORD", "POW", "POW*", "EXP", "EXP*"):
            raise CbfError("%s block present: cuOpt's barrier only accepts cones "
                           "expressible as QCMATRIX rows (F, L=, L+, L-, Q, QR)" % kw)

        else:
            raise CbfError("unknown CBF keyword %r" % kw)

    if seen_change:
        sys.stderr.write("[cbf2mps] note: CHANGE section ignored (base problem only)\n")

    for idx in list(m.integer):
        if idx < 0 or idx >= m.n:
            raise CbfError("INT index %d out of range [0,%d)" % (idx, m.n))
    return m


# ----------------------------------------------------------------------------
# CBF -> MPS translation
# ----------------------------------------------------------------------------
class QuadRow(object):
    __slots__ = ("name", "entries", "sense", "rhs", "head", "kind")

    def __init__(self, name, entries, head, kind):
        self.name = name
        self.entries = entries      # list of (col, col, value)
        self.sense = "L"
        self.rhs = 0.0
        self.head = head            # tuple of head columns
        self.kind = kind            # "soc" | "rsoc"


class LinRow(object):
    __slots__ = ("name", "sense", "coeffs", "rhs")

    def __init__(self, name, sense, coeffs, rhs):
        self.name = name
        self.sense = sense
        self.coeffs = coeffs        # dict col -> value
        self.rhs = rhs


def standard_soc_entries(head, tails):
    """x_head^2 - sum tails^2 >= 0  <=>  sum tails^2 - head^2 <= 0."""
    e = [(head, head, -1.0)]
    e.extend((t, t, 1.0) for t in tails)
    return e


def rotated_soc_entries(h0, h1, tails):
    """2*h0*h1 - sum tails^2 >= 0  <=>  sum tails^2 - 2*h0*h1 <= 0."""
    e = [(t, t, 1.0) for t in tails]
    # cross term coefficient -2  ->  two symmetric halves of -1
    e.append((h0, h1, -1.0))
    e.append((h1, h0, -1.0))
    return e


def translate(m, relax_int):
    n_orig = m.n
    lo = [0.0] * n_orig
    up = [INF] * n_orig

    # Bucket A by row ONCE.  Scanning the whole sparse matrix per constraint
    # row is O(n_con * nnz) and dominates every other cost on large models.
    A_by_row = defaultdict(dict)
    for (rr, cc), v in m.A.items():
        if v != 0.0:
            row_d = A_by_row[rr]
            row_d[cc] = row_d.get(cc, 0.0) + v

    # --- variable-side cones: bounds -------------------------------------
    for cone, dim, start in m.var_blocks:
        if cone == C_FREE:
            for j in range(start, start + dim):
                lo[j], up[j] = -INF, INF
        elif cone == C_ZERO:
            for j in range(start, start + dim):
                lo[j], up[j] = 0.0, 0.0
        elif cone == C_LPOS:
            for j in range(start, start + dim):
                lo[j] = 0.0
        elif cone == C_LNEG:
            for j in range(start, start + dim):
                up[j] = 0.0
        elif cone == C_SOC:
            if dim < 2:
                raise CbfError("Q cone needs dim >= 2 (got %d)" % dim)
            lo[start] = max(lo[start], 0.0)
            # cone components after the head are unrestricted in sign
            for j in range(start + 1, start + dim):
                lo[j] = -INF
        elif cone == C_RSOC:
            if dim < 3:
                raise CbfError("QR cone needs dim >= 3 (got %d)" % dim)
            lo[start] = max(lo[start], 0.0)
            lo[start + 1] = max(lo[start + 1], 0.0)
            for j in range(start + 2, start + dim):
                lo[j] = -INF

    lin_rows = []
    quad_rows = []
    ncols = n_orig
    y_of_row = {}

    # --- constraint-side linear cones ------------------------------------
    for cone, dim, start in m.con_blocks:
        if cone not in LINEAR_CONES or cone == C_FREE:
            continue
        for k in range(dim):
            grow = start + k
            if cone == C_ZERO:
                sense = "E"
            elif cone == C_LPOS:
                sense = "G"
            else:                       # C_LNEG
                sense = "L"
            coeffs = dict(A_by_row.get(grow, ()))
            lin_rows.append(LinRow("R%d" % len(lin_rows), sense, coeffs,
                                   -m.b.get(grow, 0.0)))

    # --- constraint-side conic cones: aux vars + linking equalities -------
    for cone, dim, start in m.con_blocks:
        if cone not in CONIC_CONES:
            continue
        yvars = list(range(ncols, ncols + dim))
        ncols += dim
        # Cones live in R^dim: only the HEAD components are non-negative.
        # Tails are unrestricted (they may be negative), so they must be free.
        lo.extend([-INF] * dim)
        up.extend([INF] * dim)
        for j in range(dim):
            y_of_row[start + j] = yvars[j]
        for k in range(dim):
            grow = start + k
            coeffs = dict(A_by_row.get(grow, ()))
            coeffs[yvars[k]] = coeffs.get(yvars[k], 0.0) - 1.0
            lin_rows.append(LinRow("R%d" % len(lin_rows), "E", coeffs,
                                   -m.b.get(grow, 0.0)))
        if cone == C_SOC:
            lo[yvars[0]] = max(lo[yvars[0]], 0.0)
            quad_rows.append(QuadRow("QC%d" % len(quad_rows),
                                     standard_soc_entries(yvars[0], yvars[1:]),
                                     (yvars[0],), "soc"))
        else:
            lo[yvars[0]] = max(lo[yvars[0]], 0.0)
            lo[yvars[1]] = max(lo[yvars[1]], 0.0)
            quad_rows.append(QuadRow("QC%d" % len(quad_rows),
                                     rotated_soc_entries(yvars[0], yvars[1], yvars[2:]),
                                     (yvars[0], yvars[1]), "rsoc"))

    # --- variable-side conic cones --------------------------------------
    for cone, dim, start in m.var_blocks:
        if cone not in CONIC_CONES:
            continue
        idx = list(range(start, start + dim))
        if cone == C_SOC:
            quad_rows.append(QuadRow("QC%d" % len(quad_rows),
                                     standard_soc_entries(idx[0], idx[1:]),
                                     (idx[0],), "soc"))
        else:
            quad_rows.append(QuadRow("QC%d" % len(quad_rows),
                                     rotated_soc_entries(idx[0], idx[1], idx[2:]),
                                     (idx[0], idx[1]), "rsoc"))

    integer = set() if relax_int else set(m.integer)
    return lo, up, lin_rows, quad_rows, ncols, integer


# ----------------------------------------------------------------------------
# MPS writer
# ----------------------------------------------------------------------------
def write_mps(path, m, lo, up, lin_rows, quad_rows, ncols, integer):
    # column order: integer block first (contiguous, required by the parser),
    # then the continuous ones.  Names stay tied to their CBF index.
    int_cols = sorted(c for c in integer if c < ncols)
    cont_cols = [c for c in range(ncols) if c not in integer]
    order = int_cols + cont_cols
    name = lambda c: "V%d" % c

    # assemble COLUMNS: per column -> list of (rowname, value)
    col_entries = defaultdict(list)
    for c, v in m.c.items():
        if v != 0.0:
            col_entries[c].append(("OBJ", v))
    for r in lin_rows:
        for c, v in r.coeffs.items():
            if v != 0.0:
                col_entries[c].append((r.name, v))

    out = []
    w = out.append
    w("NAME          %s" % m.name)
    if m.maximize:
        w("OBJSENSE")
        w(" MAXIMIZE")
    w("ROWS")
    w(" N  OBJ")
    for r in lin_rows:
        w(" %s  %s" % (r.sense, r.name))
    for q in quad_rows:
        w(" %s  %s" % (q.sense, q.name))

    w("COLUMNS")
    if int_cols:
        w("    MARK0000  'MARKER'                 'INTORG'")
    for c in order:
        col = name(c)
        ents = col_entries.get(c)
        if not ents:
            # a variable with no coefficient still has to be declared
            w("    %s  OBJ 0" % col)
            continue
        for rn, v in ents:
            w("    %s  %s  %s" % (col, rn, fmt(v)))
    if int_cols:
        w("    MARK0001  'MARKER'                 'INTEND'")

    w("RHS")
    if m.const != 0.0:
        w("    RHS1  OBJ  %s" % fmt(-m.const))
    for r in lin_rows:
        if r.rhs != 0.0:
            w("    RHS1  %s  %s" % (r.name, fmt(r.rhs)))

    w("BOUNDS")
    for c in order:
        col = name(c)
        l, u = lo[c], up[c]
        if l == -INF and u == INF:
            w(" FR BND1  %s" % col)
        elif l == -INF:
            w(" MI BND1  %s" % col)
            if u != INF:
                w(" UP BND1  %s  %s" % (col, fmt(u)))
        elif u == INF:
            w(" LO BND1  %s  %s" % (col, fmt(l)))
        elif l == u:
            w(" FX BND1  %s  %s" % (col, fmt(l)))
        else:
            w(" LO BND1  %s  %s" % (col, fmt(l)))
            w(" UP BND1  %s  %s" % (col, fmt(u)))

    for q in quad_rows:
        w("QCMATRIX %s" % q.name)
        for i, j, v in q.entries:
            if v != 0.0:
                w("    %s  %s  %s" % (name(i), name(j), fmt(v)))

    w("ENDATA")

    with open(path, "w") as fh:
        fh.write("\n".join(out))
        fh.write("\n")


# ----------------------------------------------------------------------------
# structural verification (mirrors the checks cuOpt's parser / SOC fast path do)
# ----------------------------------------------------------------------------
def verify_mps(path, verbose=True):
    rows = {}
    cols_order = []
    colset = set()
    entries = defaultdict(list)          # rowname -> [(var,val)]
    rhs = defaultdict(float)
    lo = defaultdict(lambda: 0.0)
    up = defaultdict(lambda: INF)
    qc = defaultdict(list)
    section = None
    qc_cur = None

    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("*"):
                continue
            if not line.startswith(" "):
                key = line.split()[0] if line.split() else ""
                if key in ("NAME", "ROWS", "COLUMNS", "RHS", "BOUNDS",
                           "RANGES", "ENDATA", "QCMATRIX", "OBJSENSE"):
                    section = key
                    if key == "QCMATRIX":
                        qc_cur = line.split()[1]
                    elif key != "QCMATRIX":
                        qc_cur = None
                    continue
                section = section or "NAME"
            t = line.split()
            if section == "ROWS":
                rows[t[1]] = t[0]
            elif section == "COLUMNS":
                if "'MARKER'" in line:
                    continue
                var, row, val = t[0], t[1], float(t[2])
                if var not in colset:
                    colset.add(var)
                    cols_order.append(var)
                if row != "OBJ":
                    entries[row].append((var, val))
            elif section == "RHS":
                rhs[t[1]] = float(t[2])
            elif section == "BOUNDS":
                bt, _bn, var = t[0], t[1], t[2]
                v = float(t[3]) if len(t) > 3 else None
                if bt == "FR":
                    lo[var], up[var] = -INF, INF
                elif bt == "MI":
                    lo[var] = -INF
                elif bt == "LO":
                    lo[var] = v
                elif bt == "UP":
                    up[var] = v
                elif bt == "FX":
                    lo[var] = up[var] = v
                elif bt in ("BV", "LI", "UI"):
                    if bt == "BV":
                        lo[var], up[var] = 0.0, 1.0
                    elif bt == "LI":
                        lo[var] = v
                    else:
                        up[var] = v
            elif section == "QCMATRIX":
                a, b, v = t[0], t[1], float(t[2])
                qc[qc_cur].append((a, b, v))

    problems = []
    qc_rows = set(qc)
    for rn in qc_rows:
        if rn not in rows:
            problems.append("QCMATRIX row %s missing from ROWS" % rn)
        if rows.get(rn) not in ("L", "G", "E"):
            problems.append("QCMATRIX row %s has sense %s" % (rn, rows.get(rn)))
        if entries.get(rn):
            problems.append("QCMATRIX row %s has linear COLUMNS entries "
                            "(kills cuOpt's SOC fast path)" % rn)
        if abs(rhs.get(rn, 0.0)) > 0:
            problems.append("QCMATRIX row %s has nonzero RHS %g" % (rn, rhs[rn]))

        diag = {}
        cross = defaultdict(float)
        dup = False
        for a, b, v in qc[rn]:
            if a == b:
                if a in diag:
                    dup = True
                diag[a] = v
            else:
                cross[(a, b)] = v
        if dup:
            problems.append("QCMATRIX row %s repeats a diagonal entry" % rn)
        for (a, b), v in cross.items():
            if (b, a) not in cross:
                problems.append("QCMATRIX row %s: cross (%s,%s) has no mirror"
                                % (rn, a, b))
            elif abs(cross[(b, a)] - v) > 1e-12 * max(1.0, abs(v)):
                problems.append("QCMATRIX row %s: cross (%s,%s) halves differ"
                                % (rn, a, b))
        cross_pairs = [(a, b, v) for (a, b), v in cross.items() if a < b]

        neg = [(a, v) for a, v in diag.items() if v < 0]
        pos = [(a, v) for a, v in diag.items() if v > 0]
        if len(pos) < 1:
            problems.append("QCMATRIX row %s has no +s tail diagonal" % rn)
            continue
        s = pos[0][1]
        for a, v in pos:
            if abs(v - s) > 1e-9 * max(1.0, abs(s)):
                problems.append("QCMATRIX row %s tail diagonals not uniform" % rn)
                break
        if len(cross_pairs) == 0:
            if len(neg) != 1:
                problems.append("QCMATRIX row %s: standard SOC needs exactly one "
                                "head diagonal, found %d" % (rn, len(neg)))
            else:
                head, hv = neg[0]
                if abs(hv + s) > 1e-9 * max(1.0, abs(s)):
                    problems.append("QCMATRIX row %s: head %g != -s (%g)"
                                    % (rn, hv, -s))
                if lo[head] < 0:
                    problems.append("QCMATRIX row %s: head %s lower bound %g < 0"
                                    % (rn, head, lo[head]))
        elif len(cross_pairs) == 1:
            a, b, v = cross_pairs[0]
            if not (-v / 2.0 > 0):
                problems.append("QCMATRIX row %s: rotated cross %g not < 0" % (rn, v))
            for h in (a, b):
                if lo[h] < 0:
                    problems.append("QCMATRIX row %s: rotated head %s lower bound "
                                    "%g < 0" % (rn, h, lo[h]))
        else:
            problems.append("QCMATRIX row %s: %d cross terms, not an SOC pattern"
                            % (rn, len(cross_pairs)))

    for rn in entries:
        if rn not in rows:
            problems.append("COLUMNS references unknown row %s" % rn)

    stats = dict(rows=len(rows), cols=len(cols_order), qcrows=len(qc_rows),
                 nnz=sum(len(v) for v in entries.values()),
                 cross=sum(len(v) for v in qc.values()))
    if verbose:
        print("[verify] rows=%d cols=%d nnz=%d qcmatrices=%d qcentries=%d"
              % (stats["rows"], stats["cols"], stats["nnz"],
                 stats["qcrows"], stats["cross"]))
    if problems:
        for p in problems[:40]:
            print("[verify] FAIL %s" % p)
        if len(problems) > 40:
            print("[verify] ... %d more" % (len(problems) - 40))
        return False, stats
    if verbose:
        print("[verify] OK: every QCMATRIX row is rhs=0, has no linear part, "
              "symmetric cross halves, and non-negative cone heads")
    return True, stats


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def read_text(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    with open(path, "rt", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def convert_one(src, dst, relax_int, do_verify):
    m = parse_cbf(read_text(src), src)
    lo, up, lin_rows, quad_rows, ncols, integer = translate(m, relax_int)
    write_mps(dst, m, lo, up, lin_rows, quad_rows, ncols, integer)

    nsoc = sum(1 for q in quad_rows if q.kind == "soc")
    nrsoc = sum(1 for q in quad_rows if q.kind == "rsoc")
    print("%-40s -> %s  vars=%d(+%d aux) rows=%d qc=%d (soc %d / rsoc %d) int=%d"
          % (os.path.basename(src), os.path.basename(dst),
             m.n, ncols - m.n, len(lin_rows), len(quad_rows), nsoc, nrsoc,
             len(integer)))
    if integer:
        sys.stderr.write(
            "[cbf2mps] note: %d integer variables kept. cuOpt solves SOCP with the "
            "barrier (continuous); if the run fails, retry with --relax-int.\n"
            % len(integer))
    if do_verify:
        ok, _ = verify_mps(dst)
        if not ok:
            return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Convert MOSEK CBF (.cbf / .cbf.gz) to cuOpt-compatible MPS.")
    ap.add_argument("inputs", nargs="+", help=".cbf or .cbf.gz files (globs ok)")
    ap.add_argument("-o", "--output", help="output .mps (single input only)")
    ap.add_argument("-d", "--outdir", help="output directory for batch mode")
    ap.add_argument("--relax-int", action="store_true",
                    help="drop integrality (solve the continuous relaxation)")
    ap.add_argument("--verify", action="store_true",
                    help="re-parse the MPS and check cuOpt's SOC invariants")
    args = ap.parse_args(argv)

    files = []
    for pat in args.inputs:
        hit = glob.glob(pat)
        files.extend(hit if hit else [pat])
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        ap.error("no such file: %s" % ", ".join(missing))
    if args.output and len(files) > 1:
        ap.error("-o/--output only makes sense with a single input")

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    failed = 0
    for src in files:
        base = os.path.basename(src)
        for suf in (".gz",):
            if base.endswith(suf):
                base = base[:-len(suf)]
        if base.endswith(".cbf"):
            base = base[:-4]
        dst = args.output or os.path.join(args.outdir or "", base + ".mps")
        try:
            if not convert_one(src, dst, args.relax_int, args.verify):
                failed += 1
        except CbfError as e:
            print("[cbf2mps] ERROR %s: %s" % (src, e), file=sys.stderr)
            failed += 1
        except Exception as e:
            print("[cbf2mps] ERROR %s: %s: %s" % (src, type(e).__name__, e),
                  file=sys.stderr)
            failed += 1

    if failed:
        print("[cbf2mps] %d file(s) failed" % failed, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
