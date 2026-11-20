# Fusion Whisper (version actuelle) + Deepgram + MMS.
# 1. vote mot par mot : là où deux sources concordent (< TOL s), on a une « ancre » sûre ;
# 2. entre deux ancres, MMS recale lettre par lettre dans cette fenêtre seulement (vocalises comprises) ;
# 3. si MMS n'est pas sûr de lui sur un mot et que Whisper tombe dans la fenêtre, on garde Whisper.
# usage : python fusion.py index.html ID deepgram.json deepgram_mask.json mms.json em.pt sortie.json
import json, os, sys
import torch, torchaudio
html, ID, f_dg, f_mask, f_mms, f_em, out = sys.argv[1:8]
TOL = float(os.environ.get('TOL', '0.6'))
MINSC = float(os.environ.get('MINSC', '0.15'))
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mms_caler.py'), encoding='utf-8').read()
helpers = src[:src.index('# --- émissions CTC')].replace("html, ID, audio, out = sys.argv[1:5]\ncache = sys.argv[5] if len(sys.argv) > 5 else None\n", '')
G = {'html': html, 'ID': ID, '__name__': 'helpers'}
exec(helpers, G)
U, L, DICT, clean = G['U'], G['L'], G['DICT'], G['clean']
em = torch.load(f_em)
D, mask, M = json.load(open(f_dg)), set(json.load(open(f_mask))), json.load(open(f_mms))
def tget(R, u):
    v = R['v'][u['vi']]
    return (v['w'][u['k']][0], v['w'][u['k']][2]) if u['lang'] == 'cop' else (v['wt'][u['k']], None)

# pas des trames : durée réelle / nb de trames (même calcul que mms_caler)
import subprocess
dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', os.environ['AUDIO']], capture_output=True, text=True).stdout)
FR = dur / em.shape[0]

n = len(U)
Wt = [tget(L, u)[0] for u in U]; Wl = [tget(L, u)[1] for u in U]
Dt = [tget(D, u)[0] if i in mask else None for i, u in enumerate(U)]
Mt = [tget(M, u)[0] for u in U]
near = lambda a, b: a is not None and b is not None and abs(a - b) <= TOL

# 1. ancres
anc = {}
for i in range(n):
    if not U[i]['lat']: continue
    if near(Mt[i], Dt[i]) or near(Mt[i], Wt[i]): anc[i] = Mt[i]
    elif near(Dt[i], Wt[i]): anc[i] = (Dt[i] + Wt[i]) / 2
# garder la plus longue suite croissante d'ancres (les ancres qui reculent sont fausses)
ids = sorted(anc); best, prev = [], {}
import bisect
tails, tidx = [], []
for i in ids:
    k = bisect.bisect_left(tails, anc[i])
    prev[i] = tidx[k-1] if k else None
    if k == len(tails): tails.append(anc[i]); tidx.append(i)
    else: tails[k] = anc[i]; tidx[k] = i
x = tidx[-1] if tidx else None
while x is not None: best.append(x); x = prev[x]
best = best[::-1]
print('ancres', len(best), '/', n, file=sys.stderr)

# 2. MMS dans chaque fenêtre entre deux ancres
def align_window(ii, t0, t1):
    toks = [DICT[c] for i in ii for c in U[i]['lat']]
    f0, f1 = int(t0 / FR), max(int(t0 / FR) + 1, int(t1 / FR))
    sub = em[f0:f1]
    rep = sum(1 for a, b in zip(toks, toks[1:]) if a == b)
    if not toks or sub.shape[0] < len(toks) + rep: return None
    with torch.inference_mode():
        al, sc = torchaudio.functional.forced_align(sub.unsqueeze(0), torch.tensor([toks], dtype=torch.int32), blank=0)
    al, sc = al[0].tolist(), sc[0].exp().tolist()
    st, ss = [], []
    for t, tok in enumerate(al):
        if tok != 0 and (t == 0 or al[t-1] != tok):
            st.append((f0 + t) * FR); ss.append(sc[t])
    if len(st) != len(toks): return None
    res, p = {}, 0
    for i in ii:
        m = len(U[i]['lat']); res[i] = (st[p:p+m], sum(ss[p:p+m]) / m); p += m
    return res

bounds = [0.0] + [anc[i] for i in best] + [dur]
cuts = [0] + best + [n]
T, LT, SRC = [None] * n, [None] * n, [''] * n
for a in range(len(cuts) - 1):
    i0, i1 = cuts[a], cuts[a+1]
    ii = [i for i in range(i0, i1) if U[i]['lat']]
    if not ii: continue
    t0 = bounds[a]; t1 = bounds[a+1]
    if a > 0: t0 = max(0, t0 - 0.10)          # petite marge avant l'ancre
    # fenêtre très clairsemée (plus de SPARSE s par mot) sans aucune ancre : passage de musique, on garde Whisper
    if (t1 - t0) / len(ii) > float(os.environ.get('SPARSE', '6')) and all(t0 - 1 <= Wt[i] <= t1 + 1 for i in ii):
        for i in ii: T[i], LT[i], SRC[i] = Wt[i], Wl[i], 'W'
        continue
    r = align_window(ii, t0, t1)
    if r is None: continue
    for i, (cs, scv) in r.items():
        u = U[i]
        if scv < MINSC and t0 <= Wt[i] < t1 and i not in anc:
            T[i], LT[i], SRC[i] = Wt[i], Wl[i], 'W'; continue
        T[i], SRC[i] = cs[0], 'M'
        if u['lang'] == 'cop':
            if u['R'] is None:
                LT[i] = [cs[0] + (cs[-1] - cs[0]) * q / max(1, u['n'] - 1) for q in range(u['n'])] if u['n'] > 1 else [cs[0]]
            else:
                lt, q = [], 0
                for g, l in u['R']:
                    l = clean(l); lt.append(cs[q] if l else None); q += len(l)
                for q in range(len(lt)):
                    if lt[q] is None: lt[q] = next((y for y in lt[q:] if y is not None), None) or next((y for y in reversed(lt[:q]) if y is not None), cs[0])
                LT[i] = lt
# trous restants : Whisper
for i in range(n):
    if T[i] is None: T[i], LT[i], SRC[i] = Wt[i], Wl[i], 'W'
# ordre du temps garanti
for i in range(1, n):
    if T[i] < T[i-1]:
        T[i], LT[i], SRC[i] = T[i-1], [T[i-1]] * len(LT[i] or [1]) if U[i]['lang'] == 'cop' else None, SRC[i] + '!'

R = json.loads(json.dumps(L))
for i, u in enumerate(U):
    v = R['v'][u['vi']]
    if u['lang'] == 'cop':
        v['w'][u['k']][0] = round(T[i], 2); v['w'][u['k']][2] = [round(y, 2) for y in LT[i]]
    else:
        v['wt'][u['k']] = round(T[i], 2)
        if u['k'] == 0: v['t'] = round(T[i], 2)
for vi, v in enumerate(R['v']):
    nxt = R['v'][vi + 1] if vi + 1 < len(R['v']) else None
    v['e'] = round((nxt['w'][0][0] if 'w' in nxt else nxt['t']) if nxt else dur, 2)
json.dump(R, open(out, 'w'), ensure_ascii=False, separators=(',', ':'))

from collections import Counter
print('source des mots :', dict(Counter(SRC)), file=sys.stderr)
def st(X):
    X = sorted(X)
    return f"écart médian {X[len(X)//2]:.2f} s · ≤0,5 s : {sum(d<=.5 for d in X)*100//len(X)} % · >3 s : {sum(d>3 for d in X)*100//len(X)} %"
print('vs Whisper  :', st([abs(T[i] - Wt[i]) for i in range(n)]), file=sys.stderr)
print('vs Deepgram :', st([abs(T[i] - Dt[i]) for i in range(n) if Dt[i] is not None]), file=sys.stderr)
ok = sum(1 for i in range(n) if near(T[i], Wt[i]) or near(T[i], Dt[i]))
print(f'mots confirmés par Whisper ou Deepgram : {ok*100//n} %', file=sys.stderr)
# vocalises : mots tenus > 2,5 s
held = [(T[i+1] - T[i], i) for i in range(n - 1) if U[i]['lang'] == 'cop' and 2.5 < T[i+1] - T[i] < 40]
print('mots copte tenus > 2,5 s :', len(held), file=sys.stderr)
