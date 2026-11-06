import csv, numpy as np
def rows(p): return list(csv.DictReader(open(p)))
f = lambda r, k: float(r[k]) if r[k] not in ("", "nan") else np.nan
for model in ("ego", "ego-novw"):
    for st in ("dev", "hard"):
        n_park = n_face = n_face_ext = n_ext = n_park_ext = 0; errs=[]
        for s in range(5):
            for r in rows(f"results/{model}_s{s}_{st}/rollouts.csv"):
                pk = np.isfinite(f(r, "parked_time"))
                ext = np.isfinite(f(r, "arm_extend_time"))
                n_ext += ext
                if pk:
                    n_park += 1; n_park_ext += ext
                    be = abs(f(r, "target_bearing_err_at_park_deg"))
                    errs.append(be)
                    if be < 6:
                        n_face += 1; n_face_ext += ext
        print(model, st, "parked", n_park, "parked&ext", n_park_ext, "facing<6deg", n_face, "of them ext", n_face_ext, "all ext", n_ext,
              "bearing err at park median", np.nanmedian(errs) if errs else None)
