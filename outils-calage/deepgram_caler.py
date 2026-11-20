# Recale les paroles d'un chant (copte + arabe) sur les mots horodatés de Deepgram.
# usage : python3 caler.py index.html ID dg_el.json dg_ar.json sortie.json
import json, os, re, sys, unicodedata
TH = float(os.environ.get("TH", "0.45"))
from rapidfuzz.process import cdist
from rapidfuzz.distance import Indel

html, ID, f_el, f_ar, out = sys.argv[1:6]
s = open(html, encoding='utf-8').read()
a = s.index('const LYRICS = /* @paroles:start */') + len('const LYRICS = /* @paroles:start */')
b = s.index('/* @paroles:end */', a)
L = json.loads(s[a:b].rstrip().rstrip(';'))[ID]

def words(f):
    d = json.load(open(f))
    return [(w['word'], w['start'], w['end']) for w in d['results']['channels'][0]['alternatives'][0]['words']]

def nfd(x): return ''.join(c for c in unicodedata.normalize('NFD', x) if not unicodedata.combining(c))
def graphs(x): return re.findall(r'\P{M}\p{M}*', x) if False else re.findall(r'[^̀-ͯ︠-︯⳯-⳱]+?[̀-ͯ︠-︯⳯-⳱]*(?=[^̀-ͯ︠-︯⳯-⳱]|$)', x)

# copte -> grec
COP = dict(zip('ⲁⲃⲅⲇⲉⲋⲍⲏⲑⲓⲕⲗⲙⲛⲝⲟⲡⲣⲥⲧⲩⲫⲭⲯⲱ', 'αβγδεςζηθικλμνξοπρστυφχψω'))
COP.update({'ϣ': 'σ', 'ϥ': 'φ', 'ϧ': 'χ', 'ϩ': 'χ', 'ϫ': 'τζ', 'ϭ': 'τσ', 'ϯ': 'τι'})
ABBR = {'πος': 'πτσοις', 'εθυ': 'εθουαβ', 'φτι': 'εφνουτι', 'πενσωρ': 'πενσωτηρ', 'πχς': 'πχριστος',
        'ιης': 'ιησους', 'πνα': 'πνευμα', 'ιλημ': 'ιερουσαλημ', 'ισλ': 'ισραηλ', 'χς': 'χριστος', 'θς': 'θεος'}
def cop2gr(x):
    x = nfd(x).lower()
    g = ''.join(COP.get(c, '') for c in x)
    return ABBR.get(g, g)
# grec -> clé phonétique (prononciation moderne, à peu près)
PH = [('ου', 'u'), ('ει', 'i'), ('οι', 'i'), ('αι', 'e'), ('ευ', 'ev'), ('αυ', 'av'), ('μπ', 'b'), ('ντ', 'd'), ('γκ', 'g'), ('τσ', 'c'), ('τζ', 'j')]
PH1 = dict(zip('αβγδεζηθικλμνξοπρσςτυφχψω', 'avgdeziTiklmnXoprsstifhPo'))
def key_gr(x):
    x = nfd(x).lower()
    for a_, b_ in PH: x = x.replace(a_, b_)
    return ''.join(PH1.get(c, c if c in 'uiecjvbdg' else '') for c in x)
def key_ar(x):
    x = re.sub('[ً-ْٰـ]', '', x)
    x = x.translate(str.maketrans('أإآىةؤئ', 'اااهويي'))
    return re.sub(r'[^ء-ي]', '', x)


CAR = dict(zip('ⲁⲃⲅⲇⲉⲍⲏⲑⲓⲕⲗⲙⲛⲟⲡⲣⲥⲧⲩⲫⲭⲱϣϥϧϩϫ', 'ابغدازيثيكلمنوبرستيفخوشفخهج'))
CAR.update({'ⲝ': 'كس', 'ⲯ': 'بس', 'ϭ': 'ج', 'ϯ': 'تي', 'ⲋ': ''})
ABBR_AR = {'بوس': 'ابشويس', 'اثي': 'اثواب', 'فتي': 'افنوتي', 'بنسور': 'بنسوتير', 'ييس': 'ييسوس', 'بخس': 'بخرستوس', 'بنا': 'ابنفما'}
def cop2ar(x):
    x = nfd(x).lower().replace('ⲟⲩ', 'ⲟ')
    g = ''.join(CAR.get(c, '') for c in x)
    return ABBR_AR.get(g, g)
def skel(x): return re.sub('[اويى]', '', x) or x

def align(A, B, gapA=-0.35, gapB=-0.15, same=None, A2=None, B2=None):
    """Needleman-Wunsch mot à mot ; A = paroles, B = mots Deepgram. Rend {i: j}."""
    n, m = len(A), len(B)
    S = cdist(A, B, scorer=Indel.normalized_similarity)
    if same:
        import numpy as np
        mask = np.array([[same(i, j) for j in range(m)] for i in range(n)])
        S = np.where(mask, S, 0.0)
        if A2 is not None: S = np.where(mask, S, cdist(A2, B2, scorer=Indel.normalized_similarity))
    S = S.tolist()
    INF = -1e18
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    P = [[0] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0
    for j in range(1, m + 1): D[0][j] = 0; P[0][j] = 2          # mots Deepgram en trop au début : gratuits
    for i in range(1, n + 1): D[i][0] = D[i-1][0] + gapA; P[i][0] = 1
    for i in range(1, n + 1):
        Di, Dp, Si, Pi = D[i], D[i-1], S[i-1], P[i]
        for j in range(1, m + 1):
            best, p = Dp[j-1] + (Si[j-1] - TH) * 2, 0
            v = Dp[j] + gapA
            if v > best: best, p = v, 1
            v = Di[j-1] + (0 if i == n else gapB)
            if v > best: best, p = v, 2
            Di[j], Pi[j] = best, p
    i, j, M = n, m, {}
    while i > 0 or j > 0:
        p = P[i][j]
        if p == 0:
            if S[i-1][j-1] >= TH: M[i-1] = j-1
            i, j = i-1, j-1
        elif p == 1: i -= 1
        else: j -= 1
    return M

el, ar = words(f_el), words(f_ar)
# unités à caler, dans l'ordre : (verset, mot, langue, texte, nb de lettres)
U = []
for vi, v in enumerate(L['v']):
    if 'w' in v:
        for k, (t, x, lt) in enumerate(v['w']): U.append([vi, k, 'cop', x, len(lt or [1])])
    else:
        for k, x in enumerate((v.get('ar') or v['fr']).split()): U.append([vi, k, 'ar' if v.get('ar') else 'fr', x, 1])

times = [None] * len(U)   # (début, fin)
# un seul calage : les mots Deepgram grecs et arabes fusionnés dans l'ordre du temps
B = sorted([(w[1], w[2], 'ar', key_ar(w[0])) for w in ar] + ([(w[1], w[2], 'el', key_gr(w[0])) for w in el] if os.environ.get('EL') else []))
A = [cop2ar(u[3]) if u[2] == 'cop' else key_ar(u[3]) for u in U]
if os.environ.get('SK'): A = [skel(a_) for a_ in A]; B = [(b_[0], b_[1], b_[2], skel(b_[3]) if b_[2] == 'ar' else b_[3]) for b_ in B]
la = ['ar'] * len(U)
AG = [key_gr(cop2gr(u[3])) if u[2] == 'cop' else '' for u in U]
M = align(A, [b[3] for b in B], same=lambda i, j: B[j][2] == 'ar', A2=AG, B2=[b[3] if b[2] == 'el' else '' for b in B])
for i, j in M.items(): times[i] = (B[j][0], B[j][1])
print('mots calés', len(M), '/', len(U), file=sys.stderr)
if os.environ.get('MASK'): json.dump(sorted(M), open(os.environ['MASK'], 'w'))

# ordre du temps : on écarte ce qui recule, puis on interpole les trous
last = -1
for i, t in enumerate(times):
    if t and t[0] < last: times[i] = None
    elif t: last = t[0]
known = [i for i, t in enumerate(times) if t]
dur = json.load(open(f_el))['metadata']['duration']
for i in range(len(U)):
    if times[i]: continue
    p = max([k for k in known if k < i], default=None); q = min([k for k in known if k > i], default=None)
    t0 = times[p][1] if p is not None else 0.0
    t1 = times[q][0] if q is not None else dur
    lo = p if p is not None else -1; hi = q if q is not None else len(U)
    f = (i - lo) / (hi - lo)
    st = t0 + (t1 - t0) * f
    times[i] = (st, st + (t1 - t0) / (hi - lo))
    times[i] = (st, min(times[i][1], st + 1.5))

matched = set(known)
R = json.loads(json.dumps(L))
for i, (vi, k, lang, x, n) in enumerate(U):
    st, en = times[i]
    v = R['v'][vi]
    if lang == 'cop':
        v['w'][k][0] = round(st, 2)
        v['w'][k][2] = [round(st + (en - st) * q / n, 2) for q in range(n)]
    else:
        if k == 0: v['t'] = round(st, 2)
        v['wt'][k] = round(st, 2)
for vi, v in enumerate(R['v']):
    nxt = R['v'][vi+1] if vi + 1 < len(R['v']) else None
    ns = (nxt['w'][0][0] if 'w' in nxt else nxt['t']) if nxt else dur
    v['e'] = round(ns, 2)

# comparaison avec les temps actuels (ceux de l'appli)
diffs = []
for i, (vi, k, lang, x, n) in enumerate(U):
    v = L['v'][vi]
    old = v['w'][k][0] if lang == 'cop' else v['wt'][k]
    diffs.append((abs(times[i][0] - old), lang, i in matched))
def stats(D):
    D = sorted(d for d, *_ in D)
    if not D: return 'aucun'
    return f"{len(D)} mots · écart médian {D[len(D)//2]:.2f} s · ≤0,5 s : {sum(d<=.5 for d in D)*100//len(D)} % · ≤1 s : {sum(d<=1 for d in D)*100//len(D)} % · >3 s : {sum(d>3 for d in D)*100//len(D)} %"
print('tous   :', stats(diffs), file=sys.stderr)
print('copte  :', stats([d for d in diffs if d[1] == 'cop']), file=sys.stderr)
print('arabe  :', stats([d for d in diffs if d[1] != 'cop']), file=sys.stderr)
print('reconnus par Deepgram :', stats([d for d in diffs if d[2]]), file=sys.stderr)
json.dump(R, open(out, 'w'), ensure_ascii=False, separators=(',', ':'))
