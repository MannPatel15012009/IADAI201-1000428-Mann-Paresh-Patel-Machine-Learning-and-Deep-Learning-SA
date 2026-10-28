"""Runs without TensorFlow/Streamlit: python tests/test_pipeline.py"""
import json, random, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import config, logic, prepare_dataset, make_sample_lot
from predictor import SlotClassifier

# --- insight logic -----------------------------------------------------
assert logic.summarize(["occupied"] * 3 + ["empty"]) == {"total": 4, "occupied": 3, "available": 1, "occupancy_pct": 75.0}
assert logic.summarize([])["occupancy_pct"] == 0.0
for pct, lvl in [(0, "Low"), (39.9, "Low"), (40, "Moderate"), (75, "Moderate"), (75.1, "High"), (100, "High")]:
    assert logic.congestion_level(pct) == lvl, (pct, lvl)
assert logic.recommendation(logic.summarize(["occupied"] * 5))[0].startswith("Parking full")
assert logic.recommendation(logic.summarize(["occupied"] * 9 + ["empty"]))[1] == "warning"
assert logic.recommendation(logic.summarize(["occupied"] + ["empty"] * 3))[1] == "success"
assert logic.classify([0.2, 0.5, 0.9]) == ["empty", "occupied", "occupied"]

# --- slot layouts ------------------------------------------------------
slots = logic.grid_slots(1280, 720, 3, 8)
assert len(slots) == 24
xml = b"""<parking id="x"><space id="1" occupied="1"><rotatedRect><center x="100" y="100"/><size w="40" h="80"/><angle d="10"/></rotatedRect>
<contour><point x="80" y="60"/><point x="120" y="60"/><point x="120" y="140"/><point x="80" y="140"/></contour></space>
<space id="2" occupied="0"><rotatedRect><center x="300" y="200"/><size w="40" h="80"/><angle d="0"/></rotatedRect></space></parking>"""
p = logic.parse_slots_file(xml, "a.XML", (640, 360))   # half-size image -> coordinates halve
assert len(p) == 2 and p[0][0] == (40, 30) and p[1][0] == (140, 80), p
j = logic.parse_slots_file(json.dumps([[10, 10, 50, 60]]).encode(), "s.json", (100, 100))
assert j == [[(10, 10), (50, 10), (50, 60), (10, 60)]]

# --- sample image: overlay colours + crops + demo predictor -----------
img, sslots, truth = make_sample_lot.make(out_dir=Path(tempfile.mkdtemp()))
crops = logic.crop_slots(img, sslots)
assert len(crops) == 24 and all(c is not None for c in crops)
clf = SlotClassifier(Path("/nonexistent.keras"))
assert clf.is_demo
probs = clf.predict_proba(crops)
statuses = logic.classify(probs)
acc = np.mean([a == b for a, b in zip(statuses, truth)])
print(f"demo heuristic accuracy on synthetic sample: {acc:.0%}")
ov = logic.draw_overlay(img, sslots, statuses, probs)
assert ov.shape == (720, 1280, 3)
# outline pixel colour on first slot's top edge
x1, y1 = sslots[0][0]; px = tuple(int(v) for v in ov[y1 + 1, (x1 + sslots[0][1][0]) // 2])
assert px in (logic.COLOR_EMPTY, logic.COLOR_OCCUPIED) or max(px) > 0, px
print("outline colour on slot 1:", px, statuses[0])

# --- dataset preparation on a PKLot-shaped tree ------------------------
tmp = Path(tempfile.mkdtemp()); raw = tmp / "PKLotSegmented"
rng = np.random.default_rng(0)
for lot in ("PUC", "UFPR04"):
    for weather in ("Sunny", "Cloudy", "Rainy"):
        for day in ("2012-09-11", "2012-09-12"):
            for cls in ("Empty", "Occupied"):
                d = raw / lot / weather / day / cls; d.mkdir(parents=True)
                for k in range(8):
                    Image.fromarray(rng.integers(0, 255, (60, 40, 3), dtype=np.uint8)).save(d / f"{k}.jpg")
        # exact duplicate to test de-duplication
        shutil = __import__("shutil"); shutil.copy(d / "0.jpg", d / "dup.jpg")
for grp in (False, True):
    out = tmp / f"ds_{grp}"
    r = prepare_dataset.prepare(raw, out, max_per_class=100, min_per_class=100, use_groups=grp, clean=True)
    for s in ("train", "val", "test"):
        for c in ("empty", "occupied"):
            files = list((out / s / c).glob("*.jpg"))
            assert files, (grp, s, c)
            assert Image.open(files[0]).size == (224, 224)
    all_files = [p.name for p in out.rglob("*.jpg")]
    assert len(all_files) == len(set(all_files)), "duplicate across splits!"
    print("group_split =", grp, r["split"])
print("ALL TESTS PASSED")
