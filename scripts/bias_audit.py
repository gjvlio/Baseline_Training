"""Deep bias/leakage audit for ACE-Net training. Read-only; investigates every
leakage vector before we commit to the full 50-epoch runs."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import numpy as np
from collections import Counter, defaultdict
from src.config import TrainConfig
from src.data import manifests
from src.train_utils import group_aware_split, stratified_split

cfg = TrainConfig()
SEP = "=" * 70


def hdr(t): print(f"\n{SEP}\n{t}\n{SEP}")


# ---------------------------------------------------------------------------
hdr("1. alpha-weight signature per class (P2 uniform vs genuine/P1 peaked?)")
g, f = manifests.build_stage2_samples()
p1 = [s for s in f if s.emotion == "crema_fake_p1"]
p2 = [s for s in f if s.emotion == "crema_fake_p2"]


def alpha_profile(samples, n=60):
    """std of alpha across frames: ~0 = uniform, high = peaked."""
    stds, maxs = [], []
    for s in samples[:n]:
        fw = manifests.load_frame_weights(s)
        a = np.array([w for _, w in fw], dtype=float)
        if a.sum() > 0:
            stds.append(a.std()); maxs.append(a.max())
    return round(float(np.mean(stds)), 4), round(float(np.mean(maxs)), 4)


print("  (alpha_std, alpha_max)  -- uniform=8 frames => std 0, max 0.125")
print("  genuine :", alpha_profile(g))
print("  P1 tamp :", alpha_profile(p1))
print("  P2 splc :", alpha_profile(p2))
print("  >> if P2 std~0 (uniform) and genuine/P1 std>0 (peaked): ALPHA LEAK")

# ---------------------------------------------------------------------------
hdr("2. duration (mel T) per class -- sequence-length tell?")


def dur(samples, n=80):
    Ts = [np.load(s.audio).shape[1] for s in samples[:n]]
    return round(float(np.mean(Ts)), 1), round(float(np.std(Ts)), 1), min(Ts), max(Ts)


print("  (meanT, stdT, minT, maxT)")
print("  genuine :", dur(g))
print("  P1 tamp :", dur(p1))
print("  P2 splc :", dur(p2))
print("  >> distinct mean T per class => duration leaks class. Fix: fixed-len crop.")

# ---------------------------------------------------------------------------
hdr("3. actor overlap: do FAKE audio-source actors overlap GENUINE actors?")
# P1 id: 1046_IWW_..._forged_X  -> actor 1046 (same speaker, tampered emotion)
# P2 id: A___B  -> visual actor A, AUDIO actor B (different speaker)
def actor(fid): return fid.split("___")[0].split("_")[0]
def audio_actor_p2(fid):
    parts = fid.split("___")
    return parts[1].split("_")[0] if len(parts) > 1 else actor(fid)

g_act = {actor(s.file_id) for s in g}
p1_act = {actor(s.file_id) for s in p1}
p2_vis = {actor(s.file_id) for s in p2}
p2_aud = {audio_actor_p2(s.file_id) for s in p2}
print(f"  genuine actors: {len(g_act)}")
print(f"  P1 actors (same-speaker tamper): {len(p1_act)}  overlap w/ genuine: {len(p1_act & g_act)}")
print(f"  P2 VISUAL actors: {len(p2_vis)}  overlap w/ genuine: {len(p2_vis & g_act)}")
print(f"  P2 AUDIO actors:  {len(p2_aud)}  overlap w/ genuine: {len(p2_aud & g_act)}")

# ---------------------------------------------------------------------------
hdr("4. group_aware_split: actor disjoint across train/val/test?")
rng = random.Random(cfg.seed)
ne = min(len(p1), len(p2)); rng.shuffle(p1); rng.shuffle(p2)
fakes = p1[:ne] + p2[:ne]; rng.shuffle(fakes)
n = min(len(g), len(fakes)); rng.shuffle(g)
chosen = g[:n] + fakes[:n]
for s in chosen: s.emotion = None
tr, va, te = group_aware_split(chosen, lambda s: s.group_key,
                               (cfg.train_ratio, cfg.val_ratio, cfg.test_ratio), cfg.seed)
trg = {s.group_key for s in tr}; vag = {s.group_key for s in va}; teg = {s.group_key for s in te}
print(f"  train/val/test sizes: {len(tr)}/{len(va)}/{len(te)}")
print(f"  train-test actor overlap: {len(trg & teg)} (must be 0)")
print(f"  train-val  actor overlap: {len(trg & vag)} (must be 0)")
print(f"  val-test   actor overlap: {len(vag & teg)} (must be 0)")

# ---------------------------------------------------------------------------
hdr("5. label balance per split partition (genuine vs fake)")
def bal(part):
    c = Counter(s.label for s in part)
    return f"genuine {c.get(0,0)}  fake {c.get(1,0)}"
print("  train:", bal(tr))
print("  val  :", bal(va))
print("  test :", bal(te))

# ---------------------------------------------------------------------------
hdr("6. per-type balance in TEST (Genuine/P1/P2 representation)")
# re-tag for inspection
def ptype(s):
    if s.label == 0: return "genuine"
    return "P1" if s.group_key in p1_act and s.file_id.endswith(tuple()) else "?"
# simpler: recover via file_id form
def ptype2(s):
    if s.label == 0: return "genuine"
    return "P2" if "___" in s.file_id else "P1"
print("  test:", Counter(ptype2(s) for s in te))

# ---------------------------------------------------------------------------
hdr("7. CRITICAL: Stage-1 train  vs  Stage-2 test  actor overlap (after fix)")
from src.data.splits import partition_by_actor
cre = manifests.build_emotion_samples("crema")
s1_tr, s1_va, s1_te = partition_by_actor(cre, lambda s: s.group_key,
                                         (cfg.train_ratio, cfg.val_ratio, cfg.test_ratio), cfg.seed)
# stage-2 actor partition (same fn/seed)
s2_tr, s2_va, s2_te = partition_by_actor(chosen, lambda s: s.group_key,
                                         (cfg.train_ratio, cfg.val_ratio, cfg.test_ratio), cfg.seed)
s1_train_ids = {s.file_id for s in s1_tr}
s1_train_actors = {s.group_key for s in s1_tr}
s2_test_gen_ids = {s.file_id for s in s2_te if s.label == 0}
s2_test_actors = {s.group_key for s in s2_te}
print(f"  Stage-1 train clips: {len(s1_train_ids)}  actors: {len(s1_train_actors)}")
print(f"  Stage-2 test clips: {len(s2_te)}  actors: {len(s2_test_actors)}")
print(f"  >> clip overlap (S2 test genuine seen in S1 train): {len(s2_test_gen_ids & s1_train_ids)}  (MUST be 0)")
print(f"  >> actor overlap (S2 test actors in S1 train): {len(s2_test_actors & s1_train_actors)} of {len(s2_test_actors)}  (MUST be 0)")
print(f"  S1 test actors == S2 test actors? {s1_train_actors.isdisjoint(s2_test_actors) and set({s.group_key for s in s1_te})==s2_test_actors}")

hdr("8. dataset-level fixes: uniform alpha + fixed mel length")
from src.data.dataset import _load_sample_tensors
def probe(samples, lbl, n=30):
    Ls=set(); astds=[]
    for s in samples[:n]:
        mel,_,_,frames,alpha = _load_sample_tensors(s, augment=False)
        Ls.add(mel.shape[1]); astds.append(float(alpha.std()))
    print(f"  {lbl}: mel widths={sorted(Ls)}  alpha_std~{np.mean(astds):.4f}")
probe(g,"genuine"); probe(p1,"P1"); probe(p2,"P2")
print("  >> all mel widths must be a single value (FIXED_MEL_LEN); alpha_std must be 0 for every class")
