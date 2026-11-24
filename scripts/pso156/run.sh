#!/bin/bash
# Issue #156 synthetic driver bench (crates/xcl/examples/pso156.rs). Synthetic shaders only.
#   scripts/pso156/run.sh DATA_DIR [sweep|conc|heap|sampled|cache|all]
# Needs: Windows SDK (mt.exe, DXC), an NVIDIA GPU; cdb.exe from the SDK debuggers for `sampled`.
# Every run uses --no-throttle: on a hybrid Intel CPU a process without a foreground window is
# moved to the E-cores by Windows (EcoQoS) and every number doubles. New seeds per run: the
# driver's disk cache (per exe path) must not have seen the programs.
set -e
DATA=${1:?data dir}; WHAT=${2:-all}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TARGET=${CARGO_TARGET_DIR:-$ROOT/target}
SDK="/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64"
mkdir -p "$DATA/bin" "$DATA/logs"
(cd "$ROOT" && cargo build --release -p xcl --example pso156)
B="$TARGET/release/examples/pso156.exe"
cp "$B" "$DATA/bin/pso156_nt.exe"; cp "$B" "$DATA/bin/pso156_seg.exe"
cp "$ROOT/scripts/pso156/"*.manifest "$DATA/"
cd "$DATA"
"$SDK/mt.exe" -nologo -manifest nt.manifest -outputresource:'bin\pso156_nt.exe;#1'
"$SDK/mt.exe" -nologo -manifest seg.manifest -outputresource:'bin\pso156_seg.exe;#1'
NT=./bin/pso156_nt.exe; SEG=./bin/pso156_seg.exe
SEED=$(( $(date +%s) % 1000000 * 100 ))
gen() { out=$1; name=$2; shift 2; $B gen --out "$out" --name "$name" --jobs 6 "$@" 2>&1 | grep -v "^xcl: DXC" | cut -f1-6; }
one() { exe=$1; f=$2; shift 2; n=$(basename "$f" .dxil); $exe run --threads 1 --cap-gb 24 --no-throttle --tag "$n" "$@" "$f" > "logs/$n.log" 2>&1; grep -h SUMMARY "logs/$n.log"; }
conc() { exe=$1; tag=$2; n=$3; dir=$4; shift 4; $exe run --threads "$n" --cap-gb 24 --no-throttle --tag "$tag" --logdir "logs/$tag" "$@" "$dir"/c_s*.dxil > "logs/$tag.log" 2>&1; grep -h "SUMMARY\|STACKS\|D3D Background" "logs/$tag.log"; }
S="$SEED..$((SEED + 1))"
if [[ $WHAT == sweep || $WHAT == all ]]; then
  # wave64 split, 4 windows, memory every round, a divergent if every 4th (XCL's NVIDIA form)
  for r in 125 250 500 750 1000 2000; do gen sweep ad4_r$r --rounds $r --windows 4 --seeds $S; done
  for r in 250 500 1000 2000; do gen sweep ad1_r$r --rounds $r --windows 1 --seeds $S; done
  for r in 500 750 1000; do gen sweep w32x4_r$r --rounds $r --windows 4 --wave64 0 --split 0 --seeds $S; done
  for r in 500 750 1000; do gen sweep nob4_r$r --rounds $r --windows 4 --branch-every 0 --seeds $S; done
  for r in 1000 2000; do gen sweep nob1_r$r --rounds $r --windows 1 --branch-every 0 --seeds $S; done
  for lvl in O0 O1 O3; do gen sweep ad4_r750_$lvl --rounds 750 --windows 4 --seeds $S --dxc-arg -$lvl; done
  gen sweep ps4_r500 --stage ps --rounds 500 --windows 4 --seeds $S
  for f in sweep/*.dxil; do case $f in *ps4*) one $NT "$f" --vs sweep/pass_vs.dxil.vs;; *) one $NT "$f";; esac; done
fi
if [[ $WHAT == conc || $WHAT == heap || $WHAT == all ]]; then
  k=0
  for n in 1 2 4 8 12 16 19; do a=$((SEED + 1000 + k * 100)); k=$((k + 1)); gen conc/r250_$n c --rounds 250 --windows 4 --seeds $a..$((a + 40)) | tail -1; conc $NT r250_nt$n $n conc/r250_$n; done
  for n in 8 19; do a=$((SEED + 2000 + k * 100)); k=$((k + 1)); gen conc/r250_seg$n c --rounds 250 --windows 4 --seeds $a..$((a + 40)) | tail -1; conc $SEG r250_seg$n $n conc/r250_seg$n; done
fi
if [[ $WHAT == sampled || $WHAT == all ]]; then
  a=$((SEED + 5000)); gen conc/s19 c --rounds 500 --windows 4 --seeds $a..$((a + 40)) | tail -1
  conc $NT r500_sampled19 19 conc/s19 --sample-ms 1000
fi
if [[ $WHAT == cache || $WHAT == all ]]; then
  a=$((SEED + 6000)); gen cache c --rounds 500 --windows 4 --seeds $a..$((a + 4)) | tail -1
  mkdir -p bin2; cp $NT bin2/other_path.exe
  $NT run --threads 1 --no-throttle --tag cold_store --library-out cache/lib.bin cache/c_s*.dxil | grep SUMMARY
  $NT run --threads 1 --no-throttle --tag same_exe_again cache/c_s*.dxil | grep SUMMARY
  ./bin2/other_path.exe run --threads 1 --no-throttle --tag other_path_library --library-in cache/lib.bin cache/c_s*.dxil | grep SUMMARY
  cp $NT bin2/other_path2.exe
  ./bin2/other_path2.exe run --threads 1 --no-throttle --tag other_path_create cache/c_s*.dxil | grep SUMMARY
fi
