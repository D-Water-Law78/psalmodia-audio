# Calage forcé MMS (Meta) : paroles connues (copte + arabe) -> temps de chaque lettre.
# usage : python mms_caler.py index.html ID audio.m4a sortie.json [emission.pt]
import json, os, re, subprocess, sys, unicodedata
import numpy as np, torch, torchaudio
from torchaudio.pipelines import MMS_FA as bundle

html, ID, audio, out = sys.argv[1:5]
cache = sys.argv[5] if len(sys.argv) > 5 else None
s = open(html, encoding='utf-8').read()
a = s.index('const LYRICS = /* @paroles:start */') + len('const LYRICS = /* @paroles:start */')
b = s.index('/* @paroles:end */', a)
L = json.loads(s[a:b].rstrip().rstrip(';'))[ID]
DICT = bundle.get_dict(star=None)

def graphs(x):   # même découpage que l'appli : une lettre + ses signes
    g = []
    for c in x:
        if g and unicodedata.category(c).startswith('M'): g[-1] += c
        else: g.append(c)
    return g

# copte -> latin, prononciation copto-arabe
CL = {'ⲁ': 'a', 'ⲃ': 'v', 'ⲅ': 'g', 'ⲇ': 'd', 'ⲉ': 'e', 'ⲍ': 'z', 'ⲏ': 'i', 'ⲑ': 'th', 'ⲓ': 'i', 'ⲕ': 'k', 'ⲗ': 'l',
      'ⲙ': 'm', 'ⲛ': 'n', 'ⲝ': 'ks', 'ⲟ': 'o', 'ⲡ': 'p', 'ⲣ': 'r', 'ⲥ': 's', 'ⲧ': 't', 'ⲩ': 'i', 'ⲫ': 'f', 'ⲭ': 'kh',
      'ⲯ': 'ps', 'ⲱ': 'o', 'ϣ': 'sh', 'ϥ': 'f', 'ϧ': 'kh', 'ϩ': 'h', 'ϫ': 'j', 'ϭ': 'j', 'ϯ': 'ti', 'ⲋ': ''}
ABBR = {'ⲡⲟⲥ': 'epchois', 'ⲉⲑⲩ': 'ethowab', 'ⲫϯ': 'efnouti', 'ⲡⲉⲛⲥⲱⲣ': 'pensotir', 'ⲡⲭⲥ': 'pikhristos', 'ⲓⲏⲥ': 'iisous',
        'ⲡⲛⲁ': 'pnevma', 'ⲛⲓⲡⲛⲁ': 'nipnevmata', 'ⲭⲥ': 'khristos', 'ⲑⲥ': 'theos', 'ⲡⲟⲥ': 'epchois'}
def base(g): return unicodedata.normalize('NFD', g)[0].lower()
def cop_letters(word):
    """Rend [(lettre de l'appli, latin)] ; None si mot abrégé (prononciation à part)."""
    G = graphs(word)
    bare = ''.join(base(g) for g in G)
    for k, v in ABBR.items():
        if re.sub('^[ⲉ]', '', bare) == k or bare == k:
            return None, (('e' if bare.startswith('ⲉ') and not v.startswith('e') else '') + v)
    R = []
    for i, g in enumerate(G):
        c = base(g); lat = CL.get(c, '')
        if c == 'ⲩ' and i and base(G[i-1]) in 'ⲟ': lat = 'u'              # ⲟⲩ = ou
        if c == 'ⲩ' and i and base(G[i-1]) in 'ⲁⲉ': lat = 'v'             # ⲁⲩ / ⲉⲩ = av / ev
        if '̀' in g and lat and lat[0] not in 'aeiou': lat = 'e' + lat   # djinkim : e devant
        R.append((g, lat))
    return R, None

def ar_latin(words):
    p = subprocess.run([os.path.join(os.path.dirname(sys.executable), 'uroman')], input='\n'.join(words), capture_output=True, text=True)
    return p.stdout.split('\n')[:len(words)]

def clean(x): return ''.join(c for c in x.lower() if c in DICT and c != '-')

# --- unités dans l'ordre
U = []   # dict(vi, k, lang, latin, letters)
arw = []
for vi, v in enumerate(L['v']):
    if 'w' in v:
        for k, (t, x, lt) in enumerate(v['w']):
            R, ab = cop_letters(x)
            U.append(dict(vi=vi, k=k, lang='cop', x=x, R=R, ab=ab, n=len(lt or [1])))
    else:
        for k, x in enumerate((v.get('ar') or v['fr']).split()):
            U.append(dict(vi=vi, k=k, lang='ar' if v.get('ar') else 'fr', x=x))
            if v.get('ar'): arw.append(len(U) - 1)
lat = ar_latin([U[i]['x'] for i in arw])
for i, l in zip(arw, lat): U[i]['lat'] = clean(l)
for u in U:
    if u['lang'] == 'cop': u['lat'] = clean(u['ab']) if u['R'] is None else clean(''.join(l for _, l in u['R']))
    elif u['lang'] == 'fr': u['lat'] = clean(unicodedata.normalize('NFD', u['x']))

# --- émissions CTC, par tranches de 30 s
model = bundle.get_model(with_star=False); model.eval()
torch.set_num_threads(os.cpu_count())
if cache and os.path.exists(cache):
    em = torch.load(cache); dur = em.shape[0] * 0.02
else:
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', audio, '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'], capture_output=True).stdout
    wav = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())
    dur = len(wav) / 16000
    CH, parts = 16000 * 30, []
    with torch.inference_mode():
        for o in range(0, len(wav), CH):
            seg = wav[o:o + CH]
            if len(seg) < 400: break
            e, _ = model(seg.unsqueeze(0))
            parts.append(torch.log_softmax(e[0], -1))
            print(f'{o/16000/60:.1f} min', file=sys.stderr, flush=True)
    em = torch.cat(parts)
    if cache: torch.save(em, cache)
fr = dur / em.shape[0]   # secondes par trame
print('trames', em.shape[0], 'pas', round(fr, 4), file=sys.stderr)

# --- calage forcé de tout le texte d'un coup
idx = [i for i, u in enumerate(U) if u['lat']]
tokens = [DICT[c] for i in idx for c in U[i]['lat']]
with torch.inference_mode():
    al, sc = torchaudio.functional.forced_align(em.unsqueeze(0), torch.tensor([tokens], dtype=torch.int32), blank=0)
al, sc = al[0].tolist(), sc[0].exp().tolist()
# début de chaque jeton (trame où il apparaît la 1re fois)
starts, scores, prev, j = [], [], 0, 0
for t, tok in enumerate(al):
    if tok != 0 and (t == 0 or al[t-1] != tok or prev == 0):
        if j < len(tokens): starts.append(t * fr); scores.append(sc[t]); j += 1
    prev = tok
assert len(starts) == len(tokens), (len(starts), len(tokens))

# --- retour vers les mots et les lettres
p = 0
for i in idx:
    u = U[i]; n = len(u['lat'])
    cs = starts[p:p+n]; u['score'] = sum(scores[p:p+n]) / n; p += n
    u['t'] = cs[0]
    u['end'] = starts[p] if p < len(starts) else dur
    if u['lang'] == 'cop':
        if u['R'] is None:   # abrégé : lettres réparties sur la durée du mot
            u['lt'] = [u['t'] + (cs[-1] - u['t']) * q / max(1, u['n'] - 1) for q in range(u['n'])] if u['n'] > 1 else [u['t']]
        else:
            lt, q = [], 0
            for g, l in u['R']:
                l = clean(l)
                lt.append(cs[q] if l else None); q += len(l)
            for q in range(len(lt)):   # lettre muette : prend le temps de la suivante (ou précédente)
                if lt[q] is None: lt[q] = next((x for x in lt[q:] if x is not None), None) or next((x for x in reversed(lt[:q]) if x is not None), u['t'])
            u['lt'] = lt
# mots sans latin : temps du mot suivant
for i, u in enumerate(U):
    if 't' not in u:
        nx = next((U[k]['t'] for k in range(i + 1, len(U)) if 't' in U[k]), dur)
        u['t'] = nx; u['lt'] = [nx] * u.get('n', 1)

R = json.loads(json.dumps(L))
for u in U:
    v = R['v'][u['vi']]
    if u['lang'] == 'cop':
        v['w'][u['k']][0] = round(u['t'], 2); v['w'][u['k']][2] = [round(x, 2) for x in u['lt']]
    else:
        v['wt'][u['k']] = round(u['t'], 2)
        if u['k'] == 0: v['t'] = round(u['t'], 2)
for vi, v in enumerate(R['v']):
    nxt = R['v'][vi + 1] if vi + 1 < len(R['v']) else None
    v['e'] = round((nxt['w'][0][0] if 'w' in nxt else nxt['t']) if nxt else dur, 2)
json.dump(R, open(out, 'w'), ensure_ascii=False, separators=(',', ':'))

# --- comparaison avec la version actuelle (Whisper)
D = []
for u in U:
    v = L['v'][u['vi']]
    old = v['w'][u['k']][0] if u['lang'] == 'cop' else v['wt'][u['k']]
    D.append((abs(u['t'] - old), u['lang']))
def st(X):
    X = sorted(d for d, _ in X)
    return f"{len(X)} mots · écart médian {X[len(X)//2]:.2f} s · ≤0,5 s : {sum(d<=.5 for d in X)*100//len(X)} % · ≤1 s : {sum(d<=1 for d in X)*100//len(X)} % · >3 s : {sum(d>3 for d in X)*100//len(X)} %"
print('tous  :', st(D), file=sys.stderr)
print('copte :', st([d for d in D if d[1] == 'cop']), file=sys.stderr)
print('arabe :', st([d for d in D if d[1] != 'cop']), file=sys.stderr)
