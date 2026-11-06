#!/bin/bash
# Step 11 read-outs (pre-registered comparisons), from final/robotnik_arm_task. Stage: r | s | w | all
set -uo pipefail
PY="pixi run --manifest-path ${ROBOENV_DIR:?set ROBOENV_DIR to your RoboEnv checkout}/pixi.toml python"
O=results/step11/analysis; mkdir -p $O
runs () { local p=$1; for s in 0 1 2 3 4; do printf "%s " "${p//SEED/$s}"; done; }
csv () { runs "$1" | sed 's/ $//; s/ /,/g'; }
cmp () {  # name a-pattern b-pattern
  $PY analyze_rollouts.py --out $O ${ALLOW:-} compare --a-runs $(runs "$2") --b-runs $(runs "$3") --set dev \
      --predicted-classes parked_no_arm,circling > $O/$1.stdout 2>&1
  for ext in txt json; do mv $O/A_A_vs_B_dev.$ext $O/$1.$ext; done
  sed -n '/Seed-mean success/,/Paired difference/p' $O/$1.txt; grep -E "^  (extended_not_reached|circling|parked_no_arm) |^  arm_extended  " $O/$1.txt
}
C3=results/ego-novw-trig_sSEED_dev; J0=results/ego-novw_sSEED_dev
case "${1:-all}" in
r|all) echo "== R"; cmp R_vs_C3 $C3 results/step11/R_sSEED_dev; cmp R_vs_jointK0 $J0 results/step11/R_sSEED_dev
       $PY results/step11/step11_cond.py R=$(csv results/step11/R_sSEED_dev) ;;&
s|all) echo "== S (--allow-incomplete: is_hybrid() flags any policy-arg with base=; split_policy is stateless, 0 harness errors)"; ALLOW=--allow-incomplete; cmp SK16_vs_SK0 results/step11/S-K0_sSEED_dev results/step11/S-K16_sSEED_dev
       cmp SK16_vs_C3 $C3 results/step11/S-K16_sSEED_dev; cmp SK0_vs_jointK0 $J0 results/step11/S-K0_sSEED_dev
       $PY results/step11/step11_cond.py S-K0=$(csv results/step11/S-K0_sSEED_dev) S-K16=$(csv results/step11/S-K16_sSEED_dev) \
           joint-K0=$(csv $J0) joint-K16=$(csv $C3) expert=results/expert_v11_dev ;;&
w|all) echo "== W (windcharger runs copied to results/step11/wc)"; W=results/step11/wc
       cmp WK0_vs_wcK0 $W/wc_ego-novw_sSEED_dev $W/W-K0_sSEED_dev; cmp WK16_vs_wcC3 $W/wc_C3_sSEED_dev $W/W-K16_sSEED_dev
       cmp wcC3_vs_C3 $C3 $W/wc_C3_sSEED_dev
       $PY results/step11/step11_cond.py W-K0=$(csv $W/W-K0_sSEED_dev) W-K16=$(csv $W/W-K16_sSEED_dev) wc-C3=$(csv $W/wc_C3_sSEED_dev) ;;
esac
