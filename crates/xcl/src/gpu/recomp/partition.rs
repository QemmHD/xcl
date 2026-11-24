//! Whether a wave64 VS / PS gives the console's result when a GPU runs each 32-lane half of its
//! console wave as a wave of its own (#336). Classification only: nothing translates by it yet.
//!
//! `upper_lanes` answers what a lane ≥ 32 holds when it is read. This answers whether the
//! program's result depends on how its wave is cut: a value that depends on the cut must not
//! reach what the program leaves behind (an export, a store, a branch, EXEC).
//! - A wave-wide vote (`s_cbranch_vcc*`, `exec*`, `scc*` on a mask) is taken per half: the half
//!   with no lane for a side takes the other one, which the console wave did not run for those
//!   lanes. That is exact when the vote only skips lane-masked work: the side taken with the
//!   mask empty is the join (the branch's post-dominator), and for a VCC vote the other side
//!   starts by narrowing EXEC to VCC – or the compare is of values every lane has (each half
//!   then votes as the wave does).
//! - Lane counts, ranks and first-lane indices of a mask (`s_bcnt1`, `v_mbcnt`, `s_ff1`), a mask
//!   used as a number, and another lane's varying value (`v_readfirstlane`, `v_readlane`) are
//!   different in a half than in the wave; so is everything they reach.
//!
//! A forward data-flow pass over the program's blocks (`cfg`) carries those as taint bits per
//! register; values every lane has are followed too, since reading one from another lane is
//! exact.

use std::collections::BTreeSet;

use super::cfg::{Cfg, PostDom};
use crate::gpu::rdna2::{self, Decoded};

const LANE_COUNT: u8 = 1;
const RANK: u8 = 2;
const FIRST_LANE: u8 = 4;
const LANE_ID: u8 = 8;
const MASK_DATA: u8 = 16;
const CROSS_LANE: u8 = 32;
const TAINTED_EXEC: u8 = 64;

const VCC: usize = 106;
const EXEC: usize = 126;

/// How a wave64 VS / PS fares on 32-lane waves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// what it leaves behind does not depend on how the wave is cut
    Exact,
    /// it may (the reasons say how; the analysis is conservative: unknown counts as this)
    MayDiffer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partition {
    pub tier: Tier,
    pub reasons: BTreeSet<&'static str>,
}

/// What SCC holds, for a branch on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SccVote {
    /// not a vote: a scalar compare / result
    No,
    /// whether the EXEC just written is not empty (`s_and_saveexec_b64`, `s_and_b64 exec, …`)
    Exec,
    /// whether some other mask is not empty (the register it was written to, if any)
    Mask(Option<usize>),
}

#[derive(Clone, PartialEq, Eq)]
struct State {
    /// taint bits per SGPR (s0..s105, VCC, M0, EXEC: codes 0..127)
    s: Vec<u8>,
    scc: u8,
    scc_vote: SccVote,
    /// taint bits per VGPR
    v: Vec<u8>,
    /// the VGPR holds one value in every lane
    uniform: Vec<bool>,
    /// the SGPR holds a word of a lane mask
    mask: Vec<bool>,
    /// … from a compare of values every lane has, under the launch EXEC (each half votes as
    /// the wave does)
    uniform_mask: Vec<bool>,
    /// EXEC is what the program started with (or its whole-quad form)
    exec_launch: bool,
    /// the SGPR holds a word of that launch EXEC (saved by `s_mov_b64` / saveexec)
    launch: Vec<bool>,
}

impl State {
    fn entry() -> State {
        let mut mask = vec![false; 128];
        mask[EXEC] = true;
        mask[EXEC + 1] = true;
        State {
            s: vec![0; 128],
            scc: 0,
            scc_vote: SccVote::No,
            v: vec![0; 256],
            uniform: vec![false; 256],
            mask,
            uniform_mask: vec![false; 128],
            exec_launch: true,
            launch: vec![false; 128],
        }
    }

    fn join(&self, o: &State) -> State {
        let or = |a: &[u8], b: &[u8]| a.iter().zip(b).map(|(x, y)| x | y).collect();
        let and = |a: &[bool], b: &[bool]| a.iter().zip(b).map(|(x, y)| *x && *y).collect();
        let any = |a: &[bool], b: &[bool]| a.iter().zip(b).map(|(x, y)| *x || *y).collect();
        State {
            s: or(&self.s, &o.s),
            scc: self.scc | o.scc,
            scc_vote: if self.scc_vote == o.scc_vote { self.scc_vote } else { SccVote::Mask(None) },
            v: or(&self.v, &o.v),
            uniform: and(&self.uniform, &o.uniform),
            mask: any(&self.mask, &o.mask),
            uniform_mask: and(&self.uniform_mask, &o.uniform_mask),
            exec_launch: self.exec_launch && o.exec_launch,
            launch: and(&self.launch, &o.launch),
        }
    }

    fn exec_taint(&self) -> u8 {
        self.s[EXEC] | self.s[EXEC + 1]
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Reg {
    S(usize),
    V(usize),
    Scc,
}

/// Registers an instruction reads and writes, its implicit operands (VCC, EXEC, SCC, M0)
/// included.
fn regs(d: &Decoded) -> (Vec<Reg>, Vec<Reg>) {
    let (sr, sw) = rdna2::sgpr_uses(d);
    let (vr, vw) = rdna2::vgpr_uses(d);
    let mut r: Vec<Reg> = sr.iter().map(|&s| Reg::S(s as usize)).chain(vr.iter().map(|&v| Reg::V(v as usize))).collect();
    let mut w: Vec<Reg> = sw.iter().map(|&s| Reg::S(s as usize)).chain(vw.iter().map(|&v| Reg::V(v as usize))).collect();
    for o in d.ienc.operands {
        let explicit_sgpr = o.field.is_some() && ["SREG", "SSRC", "SDST", "OPR_SRC"].iter().any(|k| o.ty.contains(k));
        let regs: &[Reg] = match o.ty {
            "OPR_VCC" => &[Reg::S(VCC), Reg::S(VCC + 1)],
            "OPR_SDST_EXEC" | "OPR_EXEC" => &[Reg::S(EXEC), Reg::S(EXEC + 1)],
            "OPR_SSRC_SPECIAL_SCC" => &[Reg::Scc],
            "OPR_SDST_M0" => &[Reg::S(124)],
            _ => continue,
        };
        if explicit_sgpr {
            continue;
        }
        let dst = if o.output { &mut w } else { &mut r };
        for &g in regs {
            if !dst.contains(&g) {
                dst.push(g);
            }
        }
    }
    (r, w)
}

fn is_valu(d: &Decoded) -> bool {
    d.enc.name.starts_with("ENC_VOP") || d.enc.name.starts_with("VOP")
}

fn is_vcmp(d: &Decoded) -> bool {
    d.inst.name.starts_with("V_CMP")
}

/// Reads a lane mask as a mask (bit by bit, or as a whole): not as a number.
fn mask_op(n: &str) -> bool {
    let bitwise32 = ["S_AND_B32", "S_OR_B32", "S_XOR_B32", "S_ANDN2_B32", "S_ORN2_B32", "S_MOV_B32", "S_CSELECT_B32"];
    mask_result(n)
        || bitwise32.contains(&n)
        || n.starts_with("S_CMP_") && n.ends_with("_U64")
        || n.starts_with("S_CBRANCH")
        || n.starts_with("V_CNDMASK")
        || n.contains("_CO_CI_")
        || matches!(n, "S_BCNT0_I32_B64" | "S_BCNT1_I32_B64" | "S_FF1_I32_B64" | "S_FF0_I32_B64" | "S_FLBIT_I32_B64")
}

/// Makes a lane mask of lane masks (bitwise, moves, selects, saveexec): a number of one is not.
fn mask_result(n: &str) -> bool {
    let bitwise = ["S_AND_", "S_OR_", "S_XOR_", "S_ANDN1_", "S_ANDN2_", "S_ORN1_", "S_ORN2_", "S_NAND_", "S_NOR_", "S_XNOR_"];
    (n.ends_with("_B64") && (bitwise.iter().any(|p| n.starts_with(p)) || matches!(n, "S_NOT_B64" | "S_MOV_B64" | "S_CSELECT_B64" | "S_WQM_B64"))) || n.contains("SAVEEXEC")
}

/// The partition-dependent value an instruction makes of its own (not from its sources).
fn source_taint(d: &Decoded, st: &State, reads: &[Reg]) -> u8 {
    let n = d.inst.name;
    let mut t = 0;
    // explicit operands only: a vector or memory instruction's implicit EXEC is its lane mask
    let reads_mask = rdna2::sgpr_uses(d).0.iter().any(|&s| st.mask[s as usize]);
    if matches!(n, "S_BCNT0_I32_B64" | "S_BCNT1_I32_B64") || (reads_mask && matches!(n, "S_BCNT0_I32_B32" | "S_BCNT1_I32_B32")) {
        t |= LANE_COUNT;
    }
    if matches!(n, "S_FF1_I32_B64" | "S_FF0_I32_B64" | "S_FLBIT_I32_B64" | "S_FLBIT_I32_I64") || (reads_mask && matches!(n, "S_FF1_I32_B32" | "S_FLBIT_I32_B32")) {
        t |= FIRST_LANE;
    }
    if n.starts_with("V_MBCNT") {
        // a constant SRC0 (-1): the lane's index in the wave; a mask: its rank in it
        let constant = matches!(rdna2::field_value(d.enc, &d.words, "SRC0"), Some(128..=208 | 240..=248 | 255));
        t |= if constant { LANE_ID } else { RANK };
    }
    if matches!(n, "V_READFIRSTLANE_B32" | "V_READLANE_B32") && reads.iter().any(|r| matches!(r, Reg::V(v) if !st.uniform[*v])) {
        t |= CROSS_LANE;
    }
    if reads_mask && !mask_op(n) && !is_vcmp(d) {
        t |= MASK_DATA;
    }
    t
}

fn reasons_of(t: u8, out: &mut BTreeSet<&'static str>) {
    for (bit, name) in [
        (LANE_COUNT, "lane-count"),
        (RANK, "rank"),
        (FIRST_LANE, "first-lane"),
        (LANE_ID, "lane-id"),
        (MASK_DATA, "mask-data"),
        (CROSS_LANE, "cross-lane"),
        (TAINTED_EXEC, "tainted-exec"),
    ] {
        if t & bit != 0 {
            out.insert(name);
        }
    }
}

/// What the program leaves behind: exports, memory writes.
fn is_store(n: &str) -> bool {
    n == "EXP" || n.contains("STORE") || n.contains("ATOMIC") || n.starts_with("DS_WRITE") || n.starts_with("DS_ADD") || n.starts_with("S_SENDMSG")
}

/// One instruction: the state after it; what it leaves behind that depends on the cut goes to
/// `out`.
fn step(d: &Decoded, st: &mut State, out: &mut BTreeSet<&'static str>) {
    let n = d.inst.name;
    let (reads, writes) = regs(d);
    let mut t = source_taint(d, st, &reads);
    for r in &reads {
        t |= match *r {
            Reg::S(s) => st.s[s],
            Reg::V(v) => st.v[v],
            Reg::Scc => st.scc,
        };
    }
    let valu = is_valu(d) || !d.enc.name.starts_with("ENC_S");
    if valu && st.exec_taint() != 0 {
        t |= TAINTED_EXEC | st.exec_taint();
    }
    if is_store(n) && t != 0 {
        reasons_of(t, out);
    }
    if n.starts_with("S_CBRANCH") && t != 0 {
        reasons_of(t, out);
    }
    let writes_exec = writes.contains(&Reg::S(EXEC));
    if writes_exec && t != 0 {
        reasons_of(t, out);
    }
    let reads_mask_or_exec = reads.iter().any(|r| matches!(r, Reg::S(s) if st.mask[*s]));
    let all_uniform = reads.iter().all(|r| !matches!(r, Reg::V(v) if !st.uniform[*v]));
    // a mask that is all or none of the active lanes (EXEC itself, a compare of values every lane
    // has, a scalar condition selected into a mask): a half with active lanes votes on it as the
    // wave does. A half with none may take the other side, where it writes no lane – only its
    // scalars can differ, which this pass does not follow yet. Kept by these ops, not by complements.
    let vcmp_uniform = is_vcmp(d) && all_uniform;
    let keeps_uniform = matches!(n, "S_AND_B64" | "S_OR_B64" | "S_ANDN2_B64" | "S_MOV_B64" | "S_CSELECT_B64" | "S_WQM_B64") || n.contains("SAVEEXEC");
    let sgpr_reads: Vec<usize> = reads.iter().filter_map(|r| if let Reg::S(s) = r { Some(*s) } else { None }).collect();
    let is_launch = |s: usize| if s == EXEC || s == EXEC + 1 { st.exec_launch } else { st.launch[s] };
    let mask_uniform = keeps_uniform && sgpr_reads.iter().all(|&s| !st.mask[s] || st.uniform_mask[s] || s == EXEC || s == EXEC + 1 || is_launch(s));
    let from_launch = matches!(n, "S_MOV_B64" | "S_WQM_B64") && !sgpr_reads.is_empty() && sgpr_reads.iter().all(|&s| is_launch(s));
    let or_launch = n == "S_OR_B64" && sgpr_reads.iter().any(|&s| is_launch(s));
    let saved_exec = n.contains("SAVEEXEC") && st.exec_launch;
    let mut new_scc = None;
    for w in &writes {
        match *w {
            Reg::S(s) => {
                st.s[s] = t;
                st.mask[s] = s == EXEC || s == EXEC + 1 || is_vcmp(d) || (mask_result(n) && reads_mask_or_exec);
                st.uniform_mask[s] = vcmp_uniform || (st.mask[s] && mask_uniform);
                st.launch[s] = s != EXEC && s != EXEC + 1 && (from_launch || (saved_exec && Some(&Reg::S(s)) != writes.iter().find(|w| **w == Reg::S(EXEC))));
            }
            Reg::V(v) => {
                st.v[v] = t;
                // a value every lane has: a VALU result of such values (no lane-dependent one)
                // … or a load from an address every lane has
                let load = matches!(d.enc.name, "ENC_MUBUF" | "ENC_MTBUF" | "ENC_MIMG" | "ENC_MIMG_NSA1" | "ENC_MIMG_NSA2") && !n.contains("ATOMIC");
                st.uniform[v] = all_uniform && t & (RANK | LANE_ID | CROSS_LANE) == 0 && n != "V_WRITELANE_B32" && (d.enc.name.starts_with("ENC_VOP") || load);
            }
            Reg::Scc => {
                new_scc = Some(t);
            }
        }
    }
    if let Some(sc) = new_scc {
        st.scc = sc;
        st.scc_vote = if writes_exec {
            SccVote::Exec
        } else if reads_mask_or_exec && (mask_result(n) || n.starts_with("S_CMP")) {
            let to = writes.iter().find_map(|w| if let Reg::S(s) = w { Some(*s) } else { None });
            SccVote::Mask(if mask_result(n) { to } else { None })
        } else {
            SccVote::No
        };
    }
    if writes_exec {
        // EXEC back at the launch mask: its whole-quad form, a copy of it, or OR-ed with one
        st.exec_launch = from_launch || or_launch;
    }
}

/// Block `b` ends the program and leaves nothing behind: at most a null export (a pixel shader's
/// way out when every pixel is discarded) – a half whose mask is empty that goes there writes
/// nothing, as its lanes write nothing on the console.
fn leaves_nothing(insts: &[Decoded], g: &Cfg, b: usize) -> bool {
    g.succ[b].is_empty()
        && g.range(b).all(|i| {
            let d = &insts[i];
            let n = d.inst.name;
            if n == "EXP" {
                return rdna2::field_value(d.enc, &d.words, "TGT") == Some(9);
            }
            !is_store(n) && (d.enc.name.starts_with("ENC_SOP") || n == "S_ENDPGM")
        })
}

/// The first instruction of block `b` narrows EXEC to the lanes of the mask in `m`
/// (`s_and_saveexec_b64 s, m`, `s_and_b64 exec, m, x` / `x, m`, `s_mov_b64` / `s_wqm_b64 exec, m`).
fn narrows_exec_by(insts: &[Decoded], g: &Cfg, b: usize, m: usize) -> bool {
    let Some(d) = g.range(b).next().map(|i| &insts[i]) else { return false };
    let f = |name: &str| rdna2::field_value(d.enc, &d.words, name);
    match d.inst.name {
        "S_AND_SAVEEXEC_B64" => f("SSRC0") == Some(m as u64),
        // (the whole-quad form: the quads of the mask's pixels, helpers included)
        "S_MOV_B64" | "S_WQM_B64" => f("SDST") == Some(EXEC as u64) && f("SSRC0") == Some(m as u64),
        // EXEC = m & anything: no lane outside m runs this side
        "S_AND_B64" => f("SDST") == Some(EXEC as u64) && (f("SSRC0") == Some(m as u64) || f("SSRC1") == Some(m as u64)),
        _ => false,
    }
}

/// Classify a wave64 VS / PS (instructions in address order, far jumps resolved).
pub fn classify(insts: &[Decoded]) -> Partition {
    let mut reasons = BTreeSet::new();
    let Some(g) = Cfg::new(insts) else {
        reasons.insert("no-graph");
        return Partition { tier: Tier::MayDiffer, reasons };
    };
    let nb = g.start.len();
    if nb == 0 {
        return Partition { tier: Tier::Exact, reasons };
    }
    let ipdom = g.post_dominators();
    // block-entry states to a fixed point (the bits only grow, the flags only fall)
    let mut at: Vec<Option<State>> = vec![None; nb];
    at[0] = Some(State::entry());
    let mut work = BTreeSet::from([0usize]);
    let mut scratch = BTreeSet::new();
    while let Some(b) = work.pop_first() {
        let mut st = at[b].clone().expect("queued with a state");
        for i in g.range(b) {
            step(&insts[i], &mut st, &mut scratch);
        }
        for &s in &g.succ[b] {
            let new = match &at[s] {
                None => st.clone(),
                Some(old) => old.join(&st),
            };
            if at[s].as_ref() != Some(&new) {
                at[s] = Some(new);
                work.insert(s);
            }
        }
    }
    // the sinks, and the votes, from the fixed point
    for b in 0..nb {
        let Some(mut st) = at[b].clone() else { continue };
        let range = g.range(b);
        let last = range.end - 1;
        for i in range {
            if i == last {
                vote(insts, &g, &ipdom, b, &insts[i], &st, &mut reasons);
            }
            step(&insts[i], &mut st, &mut reasons);
        }
    }
    let tier = if reasons.is_empty() { Tier::Exact } else { Tier::MayDiffer };
    Partition { tier, reasons }
}

/// A conditional branch at the end of block `b` (state `st` before it): a vote that may run a
/// side the console wave did not run for the half's lanes.
fn vote(insts: &[Decoded], g: &Cfg, ipdom: &[PostDom], b: usize, d: &Decoded, st: &State, out: &mut BTreeSet<&'static str>) {
    let n = d.inst.name;
    // a conditional branch's successors: its target, then the next block (one if they are the same)
    let [t, next] = g.succ[b][..] else { return };
    // (the side taken with the mask empty, the other side, the mask's register)
    let (empty, run, mask) = match n {
        "S_CBRANCH_EXECZ" => (t, next, None),
        "S_CBRANCH_EXECNZ" => (next, t, None),
        "S_CBRANCH_VCCZ" => (t, next, Some(VCC)),
        "S_CBRANCH_VCCNZ" => (next, t, Some(VCC)),
        "S_CBRANCH_SCC0" | "S_CBRANCH_SCC1" if st.scc_vote == SccVote::Exec => {
            if n == "S_CBRANCH_SCC0" {
                (t, next, None)
            } else {
                (next, t, None)
            }
        }
        // SCC from some other mask: the register it was written to, if any, is the one to narrow
        // EXEC by (unknown: counted as a vote that may differ)
        "S_CBRANCH_SCC0" | "S_CBRANCH_SCC1" => match st.scc_vote {
            SccVote::Mask(Some(m)) if n == "S_CBRANCH_SCC0" => (t, next, Some(m)),
            SccVote::Mask(Some(m)) => (next, t, Some(m)),
            SccVote::Mask(None) => {
                out.insert("vote-branch");
                return;
            }
            _ => return,
        },
        _ => return,
    };
    // a mask every half votes on as the wave does
    let m = mask.unwrap_or(EXEC);
    let exec_uniform = mask.is_none() && st.exec_launch;
    if exec_uniform || (m + 1 < 128 && st.uniform_mask[m] && st.uniform_mask[m + 1]) {
        return;
    }
    let skips = ipdom[b] == PostDom::Block(empty) || leaves_nothing(insts, g, empty);
    let narrows = match mask {
        None => true,
        Some(m) => narrows_exec_by(insts, g, run, m),
    };
    if !(skips && narrows) {
        out.insert(if mask.is_none() { "unsafe-execz-branch" } else { "vote-branch" });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::rdna2::encode;

    const V: u64 = 256;
    type I = (&'static str, &'static str, &'static [(&'static str, u64)], Option<u32>);

    fn program(lines: &[I]) -> Vec<Decoded> {
        let mut bytes: Vec<u8> = lines.iter().flat_map(|(n, e, f, l)| encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}"))).collect();
        bytes.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        let mut out = Vec::new();
        let mut off = 0usize;
        while off < bytes.len() {
            let d = rdna2::decode(&bytes[off..], off as u64).unwrap();
            off += d.len;
            out.push(d);
        }
        rdna2::resolve_far_jumps(&mut out);
        out
    }

    fn reasons(lines: &[I]) -> Vec<&'static str> {
        let p = classify(&program(lines));
        assert_eq!(p.tier == Tier::Exact, p.reasons.is_empty());
        p.reasons.into_iter().collect()
    }

    // export v0..v3 (a pixel's colour)
    const EXP: I = ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1), ("VM", 1)], None);
    // vcc = v4 > v5 (per pixel)
    const CMP: I = ("V_CMP_GT_F32", "ENC_VOPC", &[("SRC0", V + 4), ("VSRC1", 5)], None);
    // s[10:11] = exec; exec &= vcc
    const NARROW: I = ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", VCC as u64)], None);
    const RESTORE: I = ("S_OR_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 10)], None);
    // v0 = 1.0
    const BODY: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 242)], None);

    #[test]
    fn per_lane_code_is_exact() {
        assert_eq!(reasons(&[BODY, EXP]), Vec::<&str>::new());
    }

    #[test]
    fn an_if_that_narrows_exec_and_skips_to_the_join_is_exact() {
        // 0 cmp, 1 saveexec, 2 execz → 4; 3 body; 4 restore, 5 exp
        let p: &[I] = &[CMP, NARROW, ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, RESTORE, EXP];
        assert_eq!(reasons(&p), Vec::<&str>::new());
    }

    #[test]
    fn a_vcc_vote_whose_side_runs_with_the_whole_exec_may_differ() {
        // 0 cmp, 1 vccz → 3; 2 body (every lane of the wave, whatever it voted); 3 exp
        let p: &[I] = &[CMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, EXP];
        assert_eq!(reasons(&p), vec!["vote-branch"]);
    }

    #[test]
    fn a_vcc_vote_that_narrows_exec_first_is_exact() {
        // 0 cmp, 1 vccz → 5; 2 saveexec vcc, 3 body, 4 restore; 5 exp
        let p: &[I] = &[CMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 3)], None), NARROW, BODY, RESTORE, EXP];
        assert_eq!(reasons(&p), Vec::<&str>::new());
    }

    #[test]
    fn a_vote_on_a_compare_of_values_every_lane_has_is_exact() {
        // vcc = s2 > s3 for every lane, under the launch exec: each half votes as the wave does
        const UCMP: I = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 2), ("VSRC1", 7)], None);
        const UV7: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 3)], None);
        let p: &[I] = &[UV7, UCMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, EXP];
        assert_eq!(reasons(&p), Vec::<&str>::new());
    }

    #[test]
    fn whole_quad_mode_at_the_start_keeps_the_launch_mask() {
        // s[88:89] = exec; exec = wqm(exec); a uniform compare then votes as the wave does
        const SAVE: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 88), ("SSRC0", EXEC as u64)], None);
        const WQM: I = ("S_WQM_B64", "ENC_SOP1", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64)], None);
        const UCMP: I = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 2), ("VSRC1", 7)], None);
        const UV7: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 3)], None);
        let p: &[I] = &[SAVE, WQM, UV7, UCMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, EXP];
        assert_eq!(reasons(p), Vec::<&str>::new());
    }

    #[test]
    fn an_scc_vote_on_a_mask_is_exact_when_its_side_narrows_exec_to_it() {
        // s[0:1] = vcc & exec (scc: not empty); scc0 → join; exec = s[0:1] (saved in s[2:3]); body; restore
        const AND: I = ("S_AND_B64", "ENC_SOP2", &[("SDST", 0), ("SSRC0", VCC as u64), ("SSRC1", EXEC as u64)], None);
        const SAVE: I = ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 0)], None);
        const BACK: I = ("S_OR_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 2)], None);
        let p: &[I] = &[CMP, AND, ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 3)], None), SAVE, BODY, BACK, EXP];
        assert_eq!(reasons(p), Vec::<&str>::new());
        // the same vote with the side running under the whole exec
        let q: &[I] = &[CMP, AND, ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, EXP];
        assert_eq!(reasons(q), vec!["vote-branch"]);
    }

    #[test]
    fn an_early_out_when_every_pixel_is_discarded_is_exact() {
        // 0 s[88:89] = exec, 1 cmp, 2 s[88:89] &= ~vcc (discard), 3 scc0 → 7; 4 exec = wqm(s[88:89]),
        // 5 exp (8 bytes), 6 end; 7 null export, 8 end
        const SAVE: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 88), ("SSRC0", EXEC as u64)], None);
        const KILL: I = ("S_ANDN2_B64", "ENC_SOP2", &[("SDST", 88), ("SSRC0", 88), ("SSRC1", VCC as u64)], None);
        const LIVE: I = ("S_WQM_B64", "ENC_SOP1", &[("SDST", EXEC as u64), ("SSRC0", 88)], None);
        const NULL: I = ("EXP", "ENC_EXP", &[("TGT", 9), ("EN", 0), ("VSRC0", 0), ("VSRC1", 0), ("VSRC2", 0), ("VSRC3", 0), ("DONE", 1), ("VM", 1)], None);
        let p: &[I] = &[SAVE, CMP, KILL, ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 4)], None), LIVE, EXP, ("S_ENDPGM", "ENC_SOPP", &[], None), NULL];
        assert_eq!(reasons(p), Vec::<&str>::new());
    }

    #[test]
    fn a_load_from_an_address_every_lane_has_gives_a_value_every_lane_has() {
        // v7 = buffer[s[4:7]] at offset 0 (no VGPR address); vcc = s2 > v7; vccz over code
        const LOAD: I = ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 7), ("SRSRC", 1), ("SOFFSET", 128)], None);
        const UCMP: I = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 2), ("VSRC1", 7)], None);
        let p: &[I] = &[LOAD, UCMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, EXP];
        assert_eq!(reasons(p), Vec::<&str>::new());
    }

    #[test]
    fn a_scalar_condition_selected_into_a_mask_is_a_vote_every_half_takes_as_the_wave() {
        // scc = s0 < s1; s[10:11] = scc ? exec : 0; vcc = exec & s[10:11]; vccnz → 6 over code on both sides
        const SCMP: I = ("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 1)], None);
        const SEL: I = ("S_CSELECT_B64", "ENC_SOP2", &[("SDST", 10), ("SSRC0", EXEC as u64), ("SSRC1", 128)], None);
        const VAND: I = ("S_AND_B64", "ENC_SOP2", &[("SDST", VCC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 10)], None);
        const ELSE: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 240)], None);
        let p: &[I] = &[SCMP, SEL, VAND, ("S_CBRANCH_VCCNZ", "ENC_SOPP", &[("SIMM16", 2)], None), ELSE, ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, EXP];
        assert_eq!(reasons(p), Vec::<&str>::new());
    }

    #[test]
    fn a_lane_count_that_reaches_an_export_may_differ() {
        // s0 = popcount(exec); v0 = s0; exp
        let p: &[I] = &[("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", EXEC as u64)], None), ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 0)], None), EXP];
        assert_eq!(reasons(&p), vec!["lane-count"]);
    }

    #[test]
    fn a_lane_count_that_reaches_nothing_is_exact() {
        let p: &[I] = &[("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", EXEC as u64)], None), BODY, EXP];
        assert_eq!(reasons(&p), Vec::<&str>::new());
    }

    #[test]
    fn the_lane_index_and_a_rank_may_differ() {
        let lane: &[I] = &[("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 193), ("SRC1", 128)], None), EXP];
        assert_eq!(reasons(&lane), vec!["lane-id"]);
        let rank: &[I] = &[("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 20), ("SRC1", 128)], None), EXP];
        // s20 is not known as a mask here, but mbcnt over it is still a rank
        assert_eq!(reasons(&rank), vec!["rank"]);
    }

    #[test]
    fn reading_another_lanes_value_may_differ_unless_every_lane_has_it() {
        // s0 = v4 of the first active lane; v0 = s0; exp
        let varying: &[I] = &[("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None), ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 0)], None), EXP];
        assert_eq!(reasons(&varying), vec!["cross-lane"]);
        // v4 = s5 first: every lane has it
        let uniform: &[I] = &[("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 5)], None), varying[0], varying[1], EXP];
        assert_eq!(reasons(&uniform), Vec::<&str>::new());
    }

    #[test]
    fn a_mask_used_as_a_number_may_differ() {
        // v0 = exec_lo
        let p: &[I] = &[("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", EXEC as u64)], None), EXP];
        assert_eq!(reasons(&p), vec!["mask-data"]);
    }
    // =====================================================================================
    // Review of PR #517 (not part of the PR): counter-examples. Each `review_517_*` test first
    // RUNS its program as one 64-lane wave and as two 32-lane halves with a small reference
    // interpreter (`cut`), shows that a launched lane exports something else, and then asserts
    // what a sound classifier has to say (not Exact). The second assert fails at ed04b05.
    // =====================================================================================
    mod review_517 {
        use super::*;
        use std::collections::BTreeMap;

        const ALL: u64 = u64::MAX;
        const F1: u32 = 0x3f80_0000; // 1.0
        const VCCZ: I = ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)], None);

        /// What each lane exported: (target, v0..v3) in order.
        type Lane = Vec<(u64, [u32; 4])>;
        type Out = Vec<Lane>;

        struct Wave {
            s: [u32; 128],
            scc: bool,
            v: Vec<Vec<u32>>,
            out: Out,
        }

        impl Wave {
            fn m64(&self, r: usize) -> u64 {
                self.s[r] as u64 | (self.s[r + 1] as u64) << 32
            }
            fn set64(&mut self, r: usize, x: u64) {
                self.s[r] = x as u32;
                self.s[r + 1] = (x >> 32) as u32;
            }
            fn exec(&self) -> u64 {
                self.m64(EXEC)
            }
            fn active(&self, l: usize) -> bool {
                self.exec() >> l & 1 == 1
            }
            /// a 32-bit scalar source code
            fn s32(&self, c: u64, lit: Option<u32>) -> u32 {
                match c {
                    0..=127 => self.s[c as usize],
                    128..=192 => (c - 128) as u32,
                    193..=208 => (-((c - 192) as i32)) as u32,
                    240 => 0x3f00_0000,
                    242 => F1,
                    251 => (self.m64(VCC) == 0) as u32,
                    252 => (self.exec() == 0) as u32,
                    253 => self.scc as u32,
                    255 => lit.expect("literal"),
                    c => panic!("review interpreter: source {c}"),
                }
            }
            /// a 64-bit scalar source code
            fn s64(&self, c: u64) -> u64 {
                match c {
                    0..=127 => self.m64(c as usize),
                    128..=192 => c - 128,
                    193..=208 => (-((c - 192) as i64)) as u64,
                    c => panic!("review interpreter: source {c}"),
                }
            }
            /// a vector source code in lane `l`
            fn vs(&self, c: u64, l: usize, lit: Option<u32>) -> u32 {
                if c >= 256 { self.v[(c - 256) as usize][l] } else { self.s32(c, lit) }
            }
        }

        /// Run the program in one wave of `lanes` lanes launched with `exec`; `sin` are the SGPRs,
        /// `vin(reg, lane)` what each lane's VGPRs hold at the start (inputs; stale values in
        /// lanes that were not launched).
        fn run(insts: &[Decoded], lanes: usize, exec: u64, sin: &[u32; 128], vin: &dyn Fn(usize, usize) -> u32) -> Out {
            let mut w = Wave { s: *sin, scc: false, v: (0..64).map(|r| (0..lanes).map(|l| vin(r, l)).collect()).collect(), out: vec![Vec::new(); lanes] };
            w.set64(EXEC, exec);
            w.set64(VCC, 0);
            let pcs: BTreeMap<u64, usize> = insts.iter().enumerate().map(|(i, d)| (d.pc, i)).collect();
            let mut i = 0usize;
            for _ in 0..100_000 {
                let d = &insts[i];
                let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or_else(|| panic!("{}: no field {n}", d.inst.name));
                let lit = d.literal;
                let target = || pcs[&rdna2::branch_target(d).expect("target")];
                let mut next = i + 1;
                let act: Vec<usize> = (0..lanes).filter(|&l| w.active(l)).collect();
                match d.inst.name {
                    "S_ENDPGM" => return w.out,
                    "S_BRANCH" => next = target(),
                    "S_CBRANCH_SCC0" => {
                        if !w.scc {
                            next = target()
                        }
                    }
                    "S_CBRANCH_SCC1" => {
                        if w.scc {
                            next = target()
                        }
                    }
                    "S_CBRANCH_VCCZ" => {
                        if w.m64(VCC) == 0 {
                            next = target()
                        }
                    }
                    "S_CBRANCH_VCCNZ" => {
                        if w.m64(VCC) != 0 {
                            next = target()
                        }
                    }
                    "S_CBRANCH_EXECZ" => {
                        if w.exec() == 0 {
                            next = target()
                        }
                    }
                    "S_CBRANCH_EXECNZ" => {
                        if w.exec() != 0 {
                            next = target()
                        }
                    }
                    "S_MOV_B32" => w.s[f("SDST") as usize] = w.s32(f("SSRC0"), lit),
                    "S_MOV_B64" => {
                        let x = w.s64(f("SSRC0"));
                        w.set64(f("SDST") as usize, x);
                    }
                    "S_AND_B64" | "S_OR_B64" | "S_ANDN2_B64" => {
                        let (a, b) = (w.s64(f("SSRC0")), w.s64(f("SSRC1")));
                        let x = match d.inst.name {
                            "S_AND_B64" => a & b,
                            "S_OR_B64" => a | b,
                            _ => a & !b,
                        };
                        w.set64(f("SDST") as usize, x);
                        w.scc = x != 0;
                    }
                    "S_AND_B32" | "S_OR_B32" => {
                        let (a, b) = (w.s32(f("SSRC0"), lit), w.s32(f("SSRC1"), lit));
                        let x = if d.inst.name == "S_AND_B32" { a & b } else { a | b };
                        w.s[f("SDST") as usize] = x;
                        w.scc = x != 0;
                    }
                    "S_AND_SAVEEXEC_B64" => {
                        let (old, m) = (w.exec(), w.s64(f("SSRC0")));
                        w.set64(f("SDST") as usize, old);
                        w.set64(EXEC, old & m);
                        w.scc = old & m != 0;
                    }
                    // every lane of a quad (4 lanes) that has a lane in the mask
                    "S_WQM_B64" => {
                        let m = w.s64(f("SSRC0"));
                        let x = (0..16).filter(|q| m >> (4 * q) & 0xf != 0).fold(0u64, |x, q| x | 0xf << (4 * q));
                        w.set64(f("SDST") as usize, x);
                        w.scc = x != 0;
                    }
                    "S_CMP_LG_U64" => w.scc = w.s64(f("SSRC0")) != w.s64(f("SSRC1")),
                    "S_CMP_LT_U32" => w.scc = w.s32(f("SSRC0"), lit) < w.s32(f("SSRC1"), lit),
                    "S_CSELECT_B32" => w.s[f("SDST") as usize] = if w.scc { w.s32(f("SSRC0"), lit) } else { w.s32(f("SSRC1"), lit) },
                    "S_CSELECT_B64" => {
                        let x = if w.scc { w.s64(f("SSRC0")) } else { w.s64(f("SSRC1")) };
                        w.set64(f("SDST") as usize, x);
                    }
                    "S_CMOV_B32" => {
                        if w.scc {
                            w.s[f("SDST") as usize] = w.s32(f("SSRC0"), lit);
                        }
                    }
                    "S_ADDK_I32" => {
                        let r = f("SDST") as usize;
                        w.s[r] = w.s[r].wrapping_add(f("SIMM16") as u16 as i16 as i32 as u32);
                    }
                    "S_ADD_U32" => {
                        let (x, c) = w.s32(f("SSRC0"), lit).overflowing_add(w.s32(f("SSRC1"), lit));
                        w.s[f("SDST") as usize] = x;
                        w.scc = c;
                    }
                    "S_BCNT1_I32_B64" => {
                        let x = w.s64(f("SSRC0")).count_ones();
                        w.s[f("SDST") as usize] = x;
                        w.scc = x != 0;
                    }
                    // a DPP16 move: the source lane of the same row (quad_perm, row_shl, row_shr); a
                    // source outside the row or (FI = 0) inactive reads 0 with BOUND_CTRL, else the lane
                    // is not written; ROW_MASK / BANK_MASK pick the rows (16 lanes) / banks (4) written
                    "V_MOV_B32" if d.variant == rdna2::Variant::Dpp16 => {
                        let e = rdna2::encoding("VOP1_VOP_DPP16").expect("layout");
                        let g = |n: &str| rdna2::field_value(e, &d.words, n).expect("dpp field");
                        let (ctrl, src) = (g("DPP_CTRL") as usize, g("VSRC0") as usize);
                        let old = w.v[src].clone();
                        for l in act.iter().copied() {
                            if g("ROW_MASK") >> (l >> 4) & 1 == 0 || g("BANK_MASK") >> (l >> 2 & 3) & 1 == 0 {
                                continue;
                            }
                            let from = match ctrl {
                                0..=0xff => Some(l & !3 | ctrl >> (2 * (l & 3)) & 3),
                                0x101..=0x10f => Some(l + (ctrl & 15)).filter(|&x| x >> 4 == l >> 4),
                                0x111..=0x11f => l.checked_sub(ctrl & 15).filter(|&x| x >> 4 == l >> 4),
                                c => panic!("review interpreter: dpp_ctrl {c:#x}"),
                            };
                            match from.filter(|&x| g("FI") == 1 || w.active(x)) {
                                Some(x) => w.v[f("VDST") as usize][l] = old[x],
                                None if g("BOUND_CTRL") == 1 => w.v[f("VDST") as usize][l] = 0,
                                None => {}
                            }
                        }
                    }
                    "V_MOV_B32" => {
                        for l in act.iter().copied() {
                            w.v[f("VDST") as usize][l] = w.vs(f("SRC0"), l, lit);
                        }
                    }
                    "V_ADD_NC_U32" => {
                        for l in act.iter().copied() {
                            w.v[f("VDST") as usize][l] = w.vs(f("SRC0"), l, lit).wrapping_add(w.v[f("VSRC1") as usize][l]);
                        }
                    }
                    "V_CMP_GT_F32" | "V_CMP_GT_U32" | "V_CMP_EQ_U32" => {
                        let mut m = 0u64;
                        for l in act.iter().copied() {
                            let (a, b) = (w.vs(f("SRC0"), l, lit), w.v[f("VSRC1") as usize][l]);
                            let gt = match d.inst.name {
                                "V_CMP_GT_F32" => f32::from_bits(a) > f32::from_bits(b),
                                "V_CMP_GT_U32" => a > b,
                                _ => a == b,
                            };
                            m |= (gt as u64) << l;
                        }
                        w.set64(VCC, m);
                    }
                    // vop2: vcc ? vsrc1 : src0
                    "V_CNDMASK_B32" => {
                        for l in act.iter().copied() {
                            w.v[f("VDST") as usize][l] = if w.m64(VCC) >> l & 1 == 1 { w.v[f("VSRC1") as usize][l] } else { w.vs(f("SRC0"), l, lit) };
                        }
                    }
                    // vop3 with a mask destination: the sum, and the carries of the active lanes
                    "V_ADD_CO_U32" => {
                        let mut m = 0u64;
                        for l in act.iter().copied() {
                            let (x, c) = w.vs(f("SRC0"), l, lit).overflowing_add(w.vs(f("SRC1"), l, lit));
                            w.v[f("VDST") as usize][l] = x;
                            m |= (c as u64) << l;
                        }
                        w.set64(f("SDST") as usize, m);
                    }
                    // the first active lane (lane 0 with EXEC empty)
                    "V_READFIRSTLANE_B32" => {
                        let l = (0..lanes).find(|&l| w.active(l)).unwrap_or(0);
                        w.s[f("VDST") as usize] = w.v[(f("SRC0") - 256) as usize][l];
                    }
                    // any lane, active or not; a 32-lane wave has lanes 0..31 (lane & 31)
                    "V_READLANE_B32" => {
                        let l = w.s32(f("SRC1"), lit) as usize % lanes;
                        w.s[f("VDST") as usize] = w.v[(f("SRC0") - 256) as usize][l];
                    }
                    "V_WRITELANE_B32" => {
                        let l = w.s32(f("SRC1"), lit) as usize % lanes;
                        w.v[f("VDST") as usize][l] = w.s32(f("SRC0"), lit);
                    }
                    // (a null export writes nothing)
                    "EXP" if f("TGT") == 9 => {}
                    "EXP" => {
                        for l in act.iter().copied() {
                            let x = ["VSRC0", "VSRC1", "VSRC2", "VSRC3"].map(|k| w.v[f(k) as usize][l]);
                            w.out[l].push((f("TGT"), x));
                        }
                    }
                    n => panic!("review interpreter: {n} is not modelled"),
                }
                i = next;
            }
            panic!("review interpreter: no end");
        }

        struct Case {
            /// the lanes the console wave is launched with
            launch: u64,
            /// lanes whose v4 > v5 (`CMP`)
            pass: u64,
        }

        fn sgprs() -> [u32; 128] {
            let mut s = [0u32; 128];
            for (i, x) in s.iter_mut().enumerate() {
                *x = 0x100 + i as u32;
            }
            // s2 > s3: the uniform compare of the PR's tests is true
            s[2] = 7;
            s[3] = 5;
            s
        }

        /// The console lanes (launched ones) that export something else when the wave is cut in two
        /// 32-lane waves: (lane, whole wave, half).
        fn cut(lines: &[I], c: &Case) -> Vec<(usize, Lane, Lane)> {
            let insts = program(lines);
            let pass = c.pass;
            // v4 = ±1.0 by `pass`, v5 = 0.0, v8 = 0, v9 = a per-lane count (1 + lane % 4, lane 3: 9);
            // every other register differs lane by lane
            let vin = move |r: usize, lane: usize| -> u32 {
                match r {
                    4 => {
                        if pass >> lane & 1 == 1 {
                            F1
                        } else {
                            0xbf80_0000
                        }
                    }
                    5 | 8 => 0,
                    9 => {
                        if lane == 3 {
                            9
                        } else {
                            1 + lane as u32 % 4
                        }
                    }
                    _ => 0x0100_0000 * r as u32 + 0x1000 + lane as u32,
                }
            };
            let s = sgprs();
            let whole = run(&insts, 64, c.launch, &s, &vin);
            let mut diff = Vec::new();
            for h in 0..2usize {
                let m = c.launch >> (32 * h) & 0xffff_ffff;
                if m == 0 {
                    continue;
                }
                let half = run(&insts, 32, m, &s, &|r, l| vin(r, l + 32 * h));
                for l in (0..32).filter(|&l| m >> l & 1 == 1) {
                    if whole[32 * h + l] != half[l] {
                        diff.push((32 * h + l, whole[32 * h + l].clone(), half[l].clone()));
                    }
                }
            }
            diff
        }

        /// The program differs under the cut (shown by running it), so the classifier must not
        /// call it Exact.
        fn must_differ(what: &str, lines: &[I], c: &Case) {
            let diff = cut(lines, c);
            assert!(!diff.is_empty(), "{what}: the reference run shows no difference – the counter-example is wrong");
            let (lane, whole, half) = &diff[0];
            let r = reasons(lines);
            assert!(
                !r.is_empty(),
                "{what}: classified Exact (no reasons), but {} launched lanes export something else in a 32-lane half; e.g. lane {lane}: wave64 {whole:x?}, half {half:x?}",
                diff.len()
            );
        }

        const ONE_LOW: Case = Case { launch: ALL, pass: 1 << 3 };
        const ONE_HIGH: Case = Case { launch: ALL, pass: 1 << 32 };
        // v0 = s<n> / the first lane's v6, and the export
        const V0_S20: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 20)], None);
        const V0_S0: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 0)], None);
        const FIRST_V6: I = ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 6)], None);
        const EXECZ1: I = ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)], None);

        // ---- the reference run itself: it agrees with the PR's own tests ----------------------

        #[test]
        fn review_517_reference_run_agrees_with_the_prs_tests() {
            for c in [&ONE_LOW, &ONE_HIGH, &Case { launch: 0x0000_00ff_0000_ff00, pass: 0x0000_0011_0000_1100 }, &Case { launch: ALL, pass: 0 }] {
                // exact ones
                assert!(cut(&[BODY, EXP], c).is_empty());
                assert!(cut(&[CMP, NARROW, EXECZ1, BODY, RESTORE, EXP], c).is_empty());
                assert!(cut(&[CMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 3)], None), NARROW, BODY, RESTORE, EXP], c).is_empty());
            }
            // the PR's vote that may differ does differ (one lane of the lower half votes yes)
            assert!(!cut(&[CMP, VCCZ, BODY, EXP], &ONE_LOW).is_empty());
            // and its lane count
            assert!(!cut(&[("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", EXEC as u64)], None), V0_S0, EXP], &Case { launch: 0x0000_00ff_0000_000f, pass: 0 }).is_empty());
        }

        // ---- A. `uniform` is not "every lane has it" ---------------------------------------------

        #[test]
        fn review_517_a1_a_write_under_a_narrowed_exec_is_not_uniform() {
            // idx = s5; if (v4 > v5) idx = s6; s0 = first lane's idx; export it
            let p: &[I] = &[
                ("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 5)], None),
                CMP,
                NARROW,
                EXECZ1,
                ("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 6)], None),
                RESTORE,
                FIRST_V6,
                V0_S0,
                EXP,
            ];
            must_differ("masked write", p, &ONE_HIGH);
        }

        #[test]
        fn review_517_a2_a_select_by_a_per_lane_mask_is_not_uniform() {
            // v7 = s5; v6 = (v4 > v5) ? v7 : 0; s0 = first lane's v6; export it
            const UV7: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 5)], None);
            const SEL: I = ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 128), ("VSRC1", 7)], None);
            must_differ("v_cndmask", &[UV7, CMP, SEL, FIRST_V6, V0_S0, EXP], &ONE_HIGH);
        }

        #[test]
        fn review_517_a2_the_select_compared_is_a_vote_let_through_as_uniform() {
            // v7 = s5; v10 = 0; v6 = (v4 > v5) ? v7 : 0; vcc = v6 > v10; vccz → join; v0 = 1.0 (every lane); join
            const UV7: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 5)], None);
            const SEL: I = ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 128), ("VSRC1", 7)], None);
            const ZERO10: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", 128)], None);
            const UCMP6: I = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", V + 6), ("VSRC1", 10)], None);
            must_differ("v_cndmask into a vote", &[UV7, ZERO10, CMP, SEL, UCMP6, VCCZ, BODY, EXP], &ONE_LOW);
        }

        #[test]
        fn review_517_a3_readlane_of_a_lane_the_half_did_not_launch() {
            // v6 = s5 in the launched lanes; s0 = v6 of lane 5; export. Lanes 5 and 40 are launched:
            // the upper half's lane 5 (console lane 37) is not, and holds a stale value.
            let p: &[I] = &[("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 5)], None), ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 0), ("SRC0", V + 6), ("SRC1", 128 + 5)], None), V0_S0, EXP];
            must_differ("v_readlane", p, &Case { launch: 1 << 5 | 1 << 40, pass: 0 });
        }

        // ---- B. a vote read as data, or made with 32-bit operations --------------------------------

        #[test]
        fn review_517_b1_scc_of_a_mask_read_by_a_select() {
            // scc = (vcc != 0); s20 = scc ? s6 : s5; export s20
            const ANY: I = ("S_CMP_LG_U64", "ENC_SOPC", &[("SSRC0", VCC as u64), ("SSRC1", 128)], None);
            const SEL: I = ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 20), ("SSRC0", 6), ("SSRC1", 5)], None);
            must_differ("scc into s_cselect_b32", &[CMP, ANY, SEL, V0_S20, EXP], &ONE_LOW);
        }

        #[test]
        fn review_517_b1_scc_of_a_mask_selected_into_a_uniform_mask() {
            // the PR's `a_scalar_condition_selected_into_a_mask…` with the vote as its condition:
            // scc = (vcc != 0); s[10:11] = scc ? exec : 0; vcc = exec & s[10:11]; vccnz over code on both sides
            const ANY: I = ("S_CMP_LG_U64", "ENC_SOPC", &[("SSRC0", VCC as u64), ("SSRC1", 128)], None);
            const SEL64: I = ("S_CSELECT_B64", "ENC_SOP2", &[("SDST", 10), ("SSRC0", EXEC as u64), ("SSRC1", 128)], None);
            const VAND: I = ("S_AND_B64", "ENC_SOP2", &[("SDST", VCC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 10)], None);
            const ELSE: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 240)], None);
            let p: &[I] = &[CMP, ANY, SEL64, VAND, ("S_CBRANCH_VCCNZ", "ENC_SOPP", &[("SIMM16", 2)], None), ELSE, ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, EXP];
            must_differ("scc into a selected mask", p, &ONE_LOW);
        }

        #[test]
        fn review_517_b2_src_vccz_as_an_operand() {
            // s20 = (vcc == 0); export it
            must_differ("src_vccz", &[CMP, ("S_MOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 251)], None), V0_S20, EXP], &ONE_LOW);
        }

        #[test]
        fn review_517_b3_a_mask_word_moved_with_a_32_bit_operation() {
            // s20 = exec_lo; export it: the PR's `a_mask_used_as_a_number_may_differ` with one move
            let half_masks = Case { launch: 0x0000_00ff_0000_000f, pass: 0 };
            must_differ("s_mov_b32 of a mask word", &[("S_MOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", EXEC as u64)], None), V0_S20, EXP], &half_masks);
        }

        #[test]
        fn review_517_b3_a_vote_made_with_32_bit_operations() {
            // scc = (vcc_lo | vcc_hi) != 0: a vote over unmasked code
            const ANY32: I = ("S_OR_B32", "ENC_SOP2", &[("SDST", 20), ("SSRC0", VCC as u64), ("SSRC1", VCC as u64 + 1)], None);
            must_differ("s_or_b32 vote", &[CMP, ANY32, ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, EXP], &ONE_LOW);
        }

        // ---- C. what the exempted votes assume ------------------------------------------------------

        #[test]
        fn review_517_c1_a_scalar_written_behind_an_exempted_vote() {
            // s20 = s5; if (any lane: v4 > v5) { s20 = s6 }; export s20
            let p: &[I] = &[("S_MOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 5)], None), CMP, NARROW, EXECZ1, ("S_MOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 6)], None), RESTORE, V0_S20, EXP];
            must_differ("scalar behind execz", p, &ONE_LOW);
        }

        #[test]
        fn review_517_c1_a_loop_counter_behind_the_loops_vote() {
            // s20 = 0; do { s20++; v9--; exec &= (v9 > 0) } while (exec != 0); exec = saved; export s20
            let p: &[I] = &[
                ("S_MOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 128)], None),
                ("S_MOV_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", EXEC as u64)], None),
                ("S_ADD_U32", "ENC_SOP2", &[("SDST", 20), ("SSRC0", 20), ("SSRC1", 129)], None),
                ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 193), ("VSRC1", 9)], None),
                ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", V + 9), ("VSRC1", 8)], None),
                ("S_AND_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64), ("SSRC1", VCC as u64)], None),
                ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xfffb)], None),
                ("S_MOV_B64", "ENC_SOP1", &[("SDST", EXEC as u64), ("SSRC0", 10)], None),
                V0_S20,
                EXP,
            ];
            must_differ("loop counter", p, &Case { launch: ALL, pass: 0 });
        }

        #[test]
        fn review_517_c2_the_side_widens_exec_after_narrowing_it() {
            // vccz → join; exec &= vcc; v0 = 1.0; exec = saved; v1 = 1.0 (every lane); join: export
            const BODY1: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 242)], None);
            let p: &[I] = &[CMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 4)], None), NARROW, BODY, RESTORE, BODY1, EXP];
            must_differ("widened side", p, &ONE_LOW);
        }

        #[test]
        fn review_517_c3_a_uniform_compare_under_a_narrowed_exec() {
            // exec &= (v4 > v5); vcc = (s2 > s3) in those lanes; vccz → join; exec = saved; v0 = 1.0; join
            const UV7: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 3)], None);
            const UCMP: I = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 2), ("VSRC1", 7)], None);
            let p: &[I] = &[UV7, CMP, NARROW, UCMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 2)], None), RESTORE, BODY, RESTORE, EXP];
            must_differ("uniform compare, narrowed exec", p, &ONE_LOW);
        }

        #[test]
        fn review_517_c4_scc_no_longer_says_what_exec_holds() {
            // exec &= vcc (scc: not empty); exec = saved; scc0 → join; v0 = 1.0 (every lane); join
            const BACK: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", EXEC as u64), ("SSRC0", 10)], None);
            must_differ("stale exec vote", &[CMP, NARROW, BACK, ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, EXP], &ONE_LOW);
        }

        #[test]
        fn review_517_c4_scc_no_longer_says_what_the_mask_register_holds() {
            // s[0:1] = vcc & exec (scc); s[0:1] = exec; scc0 → join; exec &= s[0:1] (all of it); v0 = 1.0
            const AND: I = ("S_AND_B64", "ENC_SOP2", &[("SDST", 0), ("SSRC0", VCC as u64), ("SSRC1", EXEC as u64)], None);
            const OVER: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", EXEC as u64)], None);
            const SAVE: I = ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 0)], None);
            const BACK2: I = ("S_OR_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 2)], None);
            must_differ("stale mask vote", &[CMP, AND, OVER, ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 3)], None), SAVE, BODY, BACK2, EXP], &ONE_LOW);
        }

        #[test]
        fn review_517_c5_the_way_out_that_writes_nothing_while_the_wave_goes_on_to_write() {
            // exec &= vcc; execz → out (null export, end); v0 = 1.0; exec = saved; export (every lane); end
            const NULL: I = ("EXP", "ENC_EXP", &[("TGT", 9), ("EN", 0), ("VSRC0", 0), ("VSRC1", 0), ("VSRC2", 0), ("VSRC3", 0), ("DONE", 1), ("VM", 1)], None);
            let p: &[I] = &[CMP, NARROW, ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 5)], None), BODY, RESTORE, EXP, ("S_ENDPGM", "ENC_SOPP", &[], None), NULL];
            must_differ("way out", p, &ONE_LOW);
        }

        // ---- D. registers the pass does not see ------------------------------------------------------

        const COUNTS: Case = Case { launch: 0x0000_00ff_0000_000f, pass: 0 };
        const COUNT_S20: I = ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", EXEC as u64)], None);

        #[test]
        fn review_517_d1_s_addk_keeps_the_old_value() {
            // s20 = popcount(exec); s20 += 1; export
            must_differ("s_addk_i32", &[COUNT_S20, ("S_ADDK_I32", "ENC_SOPK", &[("SDST", 20), ("SIMM16", 1)], None), V0_S20, EXP], &COUNTS);
        }

        #[test]
        fn review_517_d1_s_cmov_keeps_the_old_value() {
            // s20 = popcount(exec); scc = s2 < s3 (false); if (scc) s20 = s6; export
            const SCMP: I = ("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 2), ("SSRC1", 3)], None);
            must_differ("s_cmov_b32", &[COUNT_S20, SCMP, ("S_CMOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 6)], None), V0_S20, EXP], &COUNTS);
        }

        #[test]
        fn review_517_d3_exec_hi_written_alone() {
            // exec_hi = 0: the console's lanes 32..63 export nothing; a 32-lane wave of them does
            must_differ("exec_hi = 0", &[("S_MOV_B32", "ENC_SOP1", &[("SDST", EXEC as u64 + 1), ("SSRC0", 128)], None), BODY, EXP], &Case { launch: ALL, pass: 0 });
        }

        #[test]
        fn review_517_e_a_number_used_as_a_mask() {
            // exec = 1: lane 0 of the wave, lane 0 of each half
            must_differ("exec = 1", &[("S_MOV_B64", "ENC_SOP1", &[("SDST", EXEC as u64), ("SSRC0", 129)], None), BODY, EXP], &Case { launch: ALL, pass: 0 });
        }

        // ---- F. masks that no v_cmp made; H. v_writelane --------------------------------------------

        #[test]
        fn review_517_f_a_carry_mask_is_a_mask() {
            // vcc = carries of v9 + 0xfffffff8 (lane 3 only: v9 = 9 there); scc = (vcc != 0); scc0 → join; v0 = 1.0
            const CARRY: I = ("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 6), ("SDST", VCC as u64), ("SRC0", V + 9), ("SRC1", 255)], Some(0xffff_fff8));
            const ANY: I = ("S_CMP_LG_U64", "ENC_SOPC", &[("SSRC0", VCC as u64), ("SSRC1", 128)], None);
            must_differ("carry vote", &[CARRY, ANY, ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, EXP], &Case { launch: ALL, pass: 0 });
        }

        #[test]
        fn review_517_h_writelane_then_a_per_lane_read() {
            // v0[lane 5] = s5; export v0: lane 5 of the wave, lane 5 of each half (console lanes 5 and 37)
            must_differ("v_writelane", &[("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 5), ("SRC1", 128 + 5)], None), EXP], &Case { launch: ALL, pass: 0 });
        }

        // ---- DPP (run), then classifier-only ones (instructions the reference run does not model) --

        const LANE_ID_V9: I = ("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 9), ("SRC0", 193), ("SRC1", 128)], None);

        #[test]
        fn review_517_a4_dpp_of_a_uniform_value_is_not_uniform() {
            // v6 = s5; v7 = v6 of the next lane in the row, 0 if that lane is not active (row_shl:1
            // bound_ctrl); s0 = first lane's v7; export. Lanes 0, 2, 32, 33 are launched: lane 1 is
            // not, lane 33 is.
            let p: &[I] = &[
                ("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 5)], None),
                ("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 7), ("SRC0", 0xFA), ("VSRC0", 6), ("DPP_CTRL", 0x101), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF), ("BOUND_CTRL", 1)], None),
                ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 7)], None),
                V0_S0,
                EXP,
            ];
            must_differ("dpp of a uniform value", p, &Case { launch: 0b101 | 0b11 << 32, pass: 0 });
        }

        #[test]
        fn review_517_a4_a_dpp_row_mask_names_rows_of_the_wave() {
            // v0 = v6 in rows 2 and 3 (lanes 32..63) only: quad_perm:[0,1,2,3] row_mask:0xc. A 32-lane
            // wave has rows 0 and 1.
            let p: &[I] = &[("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 0), ("SRC0", 0xFA), ("VSRC0", 6), ("DPP_CTRL", 0xE4), ("ROW_MASK", 0xC), ("BANK_MASK", 0xF)], None), EXP];
            must_differ("dpp row mask", p, &Case { launch: ALL, pass: 0 });
        }

        #[test]
        fn review_517_a5_an_image_load_writes_dmask_registers_not_four() {
            // v1 = lane id; v0 = sample(v4, v5).x (DMASK 1: only v0 is written); export v0..v3
            let p: &[I] = &[
                ("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", 193), ("SRC1", 128)], None),
                ("IMAGE_SAMPLE", "ENC_MIMG", &[("VDATA", 0), ("VADDR", 4), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x1), ("DIM", 1)], None),
                EXP,
            ];
            assert_eq!(reasons(p), vec!["lane-id"]);
        }

        #[test]
        fn review_517_a6_nsa_address_registers_are_read() {
            // v9 = lane id; v[0:3] = sample(x = v4, y = v9 – the NSA dword names it); export
            let p: &[I] = &[LANE_ID_V9, ("IMAGE_SAMPLE", "ENC_MIMG", &[("VDATA", 0), ("VADDR", 4), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 1), ("NSA", 1)], Some(9)), EXP];
            let d = &program(p)[1];
            let nsa = rdna2::encoding("MIMG_NSA1").expect("layout");
            assert_eq!((d.len, rdna2::field_value(nsa, &d.words, "VADDRA")), (12, Some(9)), "the second address register is v9");
            assert_eq!(reasons(p), vec!["lane-id"]);
        }

        #[test]
        fn review_517_d2_an_smem_offset_register_is_read() {
            // s0 = popcount(exec); s20 = buffer[s[8:11]][s0]; export it
            const COUNT: I = ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", EXEC as u64)], None);
            let load: &[I] = &[COUNT, ("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 20), ("SBASE", 4), ("OFFSET", 0), ("SOFFSET", 0)], None), V0_S20, EXP];
            assert_eq!(reasons(load), vec!["lane-count"]);
        }

        #[test]
        fn review_517_d2_an_sdwa_scalar_source_is_read() {
            // s0 = popcount(exec); v0 = s0 (an SDWA move whose source is the scalar register: S0 = 1)
            const COUNT: I = ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", EXEC as u64)], None);
            let sdwa: &[I] = &[COUNT, ("V_MOV_B32", "VOP1_VOP_SDWA", &[("VDST", 0), ("SRC0", 0xF9), ("VSRC0", 0), ("S0", 1), ("SRC0_SEL", 6), ("DST_SEL", 6)], None), EXP];
            assert_eq!(reasons(sdwa), vec!["lane-count"]);
        }

        #[test]
        fn review_517_g_lds_writes_other_than_write_and_add_are_stores() {
            // lds[v4] |= lane id
            let p: &[I] = &[LANE_ID_V9, ("DS_OR_B32", "ENC_DS", &[("ADDR", 4), ("DATA0", 9)], None)];
            assert_eq!(reasons(p), vec!["lane-id"]);
        }

        // ---- precision: exact programs reported MayDiffer (these pass at ed04b05) -------------------

        #[test]
        fn review_517_p1_an_sgpr_spill_slot_is_exact_but_reported() {
            // v63[lane 5] = s6; s6 = 0; s6 = v63[lane 5]; v0 = s6; export: upper_lanes reads it from a shadow
            let p: &[I] = &[
                ("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 63), ("SRC0", 6), ("SRC1", 128 + 5)], None),
                ("S_MOV_B32", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 128)], None),
                ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 6), ("SRC0", V + 63), ("SRC1", 128 + 5)], None),
                ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 6)], None),
                EXP,
            ];
            for c in [&ONE_LOW, &Case { launch: 0x0000_00ff_0000_ff00, pass: 0 }] {
                assert!(cut(p, c).is_empty());
            }
            assert_eq!(reasons(p), vec!["cross-lane"]);
        }
        // ---- verified: the PR's Exact programs do not change under random cuts -----------------

        #[test]
        fn review_517_the_prs_exact_programs_do_not_change_under_random_cuts() {
            const UCMP: I = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 2), ("VSRC1", 7)], None);
            const UV7: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 3)], None);
            const SAVE88: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 88), ("SSRC0", EXEC as u64)], None);
            const WQM: I = ("S_WQM_B64", "ENC_SOP1", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64)], None);
            const AND: I = ("S_AND_B64", "ENC_SOP2", &[("SDST", 0), ("SSRC0", VCC as u64), ("SSRC1", EXEC as u64)], None);
            const SAVE: I = ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 0)], None);
            const BACK: I = ("S_OR_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 2)], None);
            const KILL: I = ("S_ANDN2_B64", "ENC_SOP2", &[("SDST", 88), ("SSRC0", 88), ("SSRC1", VCC as u64)], None);
            const LIVE: I = ("S_WQM_B64", "ENC_SOP1", &[("SDST", EXEC as u64), ("SSRC0", 88)], None);
            const NULL: I = ("EXP", "ENC_EXP", &[("TGT", 9), ("EN", 0), ("VSRC0", 0), ("VSRC1", 0), ("VSRC2", 0), ("VSRC3", 0), ("DONE", 1), ("VM", 1)], None);
            const SCMP: I = ("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 1)], None);
            const SEL: I = ("S_CSELECT_B64", "ENC_SOP2", &[("SDST", 10), ("SSRC0", EXEC as u64), ("SSRC1", 128)], None);
            const VAND: I = ("S_AND_B64", "ENC_SOP2", &[("SDST", VCC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 10)], None);
            const ELSE: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 240)], None);
            let exact: &[&[I]] = &[
                &[BODY, EXP],
                &[CMP, NARROW, EXECZ1, BODY, RESTORE, EXP],
                &[CMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 3)], None), NARROW, BODY, RESTORE, EXP],
                &[UV7, UCMP, VCCZ, BODY, EXP],
                &[SAVE88, WQM, UV7, UCMP, VCCZ, BODY, EXP],
                &[CMP, AND, ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 3)], None), SAVE, BODY, BACK, EXP],
                &[SAVE88, CMP, KILL, ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 4)], None), LIVE, EXP, ("S_ENDPGM", "ENC_SOPP", &[], None), NULL],
                &[SCMP, SEL, VAND, ("S_CBRANCH_VCCNZ", "ENC_SOPP", &[("SIMM16", 2)], None), ELSE, ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, EXP],
                &[("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", EXEC as u64)], None), BODY, EXP],
                &[("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 5)], None), FIRST_V6, V0_S0, EXP],
            ];
            let mut x = 0x9e37_79b9_7f4a_7c15u64;
            let mut rnd = move || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            };
            let mut runs = 0;
            for (k, p) in exact.iter().enumerate() {
                assert_eq!(reasons(p), Vec::<&str>::new(), "program {k}");
                for n in 0..500 {
                    // dense, sparse and one-sided masks
                    let launch = match n % 4 {
                        0 => ALL,
                        1 => rnd(),
                        2 => rnd() & rnd() & rnd(),
                        _ => rnd() | 0xffff_ffff,
                    };
                    let pass = match n % 5 {
                        0 => rnd(),
                        1 => rnd() & rnd() & rnd() & rnd(),
                        2 => 1 << (rnd() % 64),
                        3 => 0,
                        _ => rnd() & 0xffff_ffff,
                    };
                    let d = cut(p, &Case { launch, pass });
                    assert!(d.is_empty(), "program {k}, launch {launch:#x}, pass {pass:#x}: {:x?}", d[0]);
                    runs += 1;
                }
            }
            println!("review_517: {runs} random cuts of {} Exact programs, no difference", exact.len());
        }

        // ---- verified: a tainted branch is reported whichever register carries the condition ------

        #[test]
        fn review_517_tainted_conditions_reach_their_branches() {
            const COUNT: I = ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", EXEC as u64)], None);
            const UV7: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 3)], None);
            // vcc = (popcount > v7), a compare of values every lane has: vccz
            const CCMP: I = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 0), ("VSRC1", 7)], None);
            assert_eq!(reasons(&[COUNT, UV7, CCMP, VCCZ, BODY, EXP]), vec!["lane-count"]);
            // scc = (popcount < s1): scc1
            const SCMP: I = ("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 1)], None);
            assert_eq!(reasons(&[COUNT, SCMP, ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, EXP]), vec!["lane-count"]);
            // exec &= that vcc: execz, and everything under it
            assert_eq!(reasons(&[COUNT, UV7, CCMP, NARROW, EXECZ1, BODY, RESTORE, EXP]), vec!["lane-count", "tainted-exec"]);
            // through a loop's back edge: s1 = s0 at the top, s0 = popcount at the bottom
            const S1_S0: I = ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 0)], None);
            const V0_S1: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 1)], None);
            const UCMP: I = ("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 2), ("SSRC1", 3)], None);
            assert_eq!(reasons(&[S1_S0, V0_S1, COUNT, UCMP, ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xfffb)], None), EXP]), vec!["lane-count"]);
        }

        // ---- time: a large program, and one that needs many passes ----------------------------------

        #[test]
        fn review_517_time_of_large_programs() {
            // 8 000 ifs in a row (40 001 instructions, 160 KB of code)
            let mut ifs: Vec<I> = Vec::new();
            for _ in 0..8000 {
                ifs.extend([CMP, NARROW, EXECZ1, BODY, RESTORE]);
            }
            ifs.push(EXP);
            let insts = program(&ifs);
            let t = std::time::Instant::now();
            let p = classify(&insts);
            println!("review_517: {} instructions in a row: {:?} ({:?})", insts.len(), t.elapsed(), p.tier);
            // a loop over 1 000 ifs in which a taint moves one register per trip (s20 → s21 → … → s100;
            // the ifs save EXEC in s[10:11])
            let moves: Vec<Vec<(&'static str, u64)>> = (20..100u64).rev().map(|i| vec![("SDST", i + 1), ("SSRC0", i)]).collect();
            let moves: &'static Vec<Vec<(&'static str, u64)>> = Box::leak(Box::new(moves));
            let mut lp: Vec<I> = moves.iter().map(|f| ("S_MOV_B32", "ENC_SOP1", &f[..], None)).collect();
            for _ in 0..1000 {
                lp.extend([CMP, NARROW, EXECZ1, BODY, RESTORE]);
            }
            lp.push(("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", EXEC as u64)], None));
            lp.push(("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 102), ("SSRC1", 103)], None));
            let back: &'static [(&'static str, u64)] = Box::leak(Box::new([("SIMM16", (0x10000 - (lp.len() as u64 + 1)) & 0xffff)]));
            lp.push(("S_CBRANCH_SCC1", "ENC_SOPP", back, None));
            lp.push(("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 100)], None));
            lp.push(EXP);
            let insts = program(&lp);
            let t = std::time::Instant::now();
            let p = classify(&insts);
            println!("review_517: {} instructions, 80 trips to the fixed point: {:?} ({:?})", insts.len(), t.elapsed(), p.reasons);
            assert_eq!(p.reasons.into_iter().collect::<Vec<_>>(), vec!["lane-count"]);
        }
        fn random_cases(n: usize) -> Vec<Case> {
            let mut x = 0x2545_f491_4f6c_dd1du64;
            let mut rnd = move || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            };
            (0..n)
                .map(|k| Case {
                    launch: match k % 4 {
                        0 => ALL,
                        1 => rnd(),
                        2 => rnd() & rnd() & rnd(),
                        _ => rnd() | 0xffff_ffff,
                    },
                    pass: match k % 5 {
                        0 => rnd(),
                        1 => rnd() & rnd() & rnd() & rnd(),
                        2 => 1 << (rnd() % 64),
                        3 => 0,
                        _ => rnd() & 0xffff_ffff,
                    },
                })
                .collect()
        }

        #[test]
        fn review_517_p2_a_waterfall_loop_is_exact_but_reported() {
            // rest = exec; do { s0 = first lane's v9; exec = rest & (v9 == s0); v0 = s0; rest &= ~exec; exec = rest }
            // while (rest != 0); exec = saved; export: every lane gets its own v9, however the wave is cut
            let p: &[I] = &[
                ("S_MOV_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", EXEC as u64)], None),
                ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 9)], None),
                ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 0), ("VSRC1", 9)], None),
                ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 12), ("SSRC0", VCC as u64)], None),
                V0_S0,
                ("S_ANDN2_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", 12), ("SSRC1", EXEC as u64)], None),
                ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xfffa)], None),
                ("S_MOV_B64", "ENC_SOP1", &[("SDST", EXEC as u64), ("SSRC0", 10)], None),
                EXP,
            ];
            for c in random_cases(500) {
                let d = cut(p, &c);
                assert!(d.is_empty(), "launch {:#x}: {:x?}", c.launch, d[0]);
            }
            assert_eq!(reasons(p), vec!["cross-lane", "tainted-exec"]);
        }

        #[test]
        fn review_517_p3_an_early_out_inside_an_if_makes_the_ifs_vote_unsafe() {
            // live = exec; if (v4 > v5) { live &= ~(2 > v9); if (live == 0) → out; exec &= live; v0 = 1.0 }
            // exec = saved & live; export; end; out: null export, end
            const NULL: I = ("EXP", "ENC_EXP", &[("TGT", 9), ("EN", 0), ("VSRC0", 0), ("VSRC1", 0), ("VSRC2", 0), ("VSRC3", 0), ("DONE", 1), ("VM", 1)], None);
            let p: &[I] = &[
                ("S_MOV_B64", "ENC_SOP1", &[("SDST", 88), ("SSRC0", EXEC as u64)], None),
                CMP,
                NARROW,
                ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 5)], None),
                ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 130), ("VSRC1", 9)], None),
                ("S_ANDN2_B64", "ENC_SOP2", &[("SDST", 88), ("SSRC0", 88), ("SSRC1", VCC as u64)], None),
                ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 6)], None),
                ("S_AND_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 88)], None),
                BODY,
                ("S_AND_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", 10), ("SSRC1", 88)], None),
                EXP,
                ("S_ENDPGM", "ENC_SOPP", &[], None),
                NULL,
            ];
            for c in random_cases(500) {
                let d = cut(p, &c);
                assert!(d.is_empty(), "launch {:#x}, pass {:#x}: {:x?}", c.launch, c.pass, d[0]);
            }
            // the same if without the early out is Exact
            let plain: &[I] = &[
                ("S_MOV_B64", "ENC_SOP1", &[("SDST", 88), ("SSRC0", EXEC as u64)], None),
                CMP,
                NARROW,
                ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 4)], None),
                ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 130), ("VSRC1", 9)], None),
                ("S_ANDN2_B64", "ENC_SOP2", &[("SDST", 88), ("SSRC0", 88), ("SSRC1", VCC as u64)], None),
                ("S_AND_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 88)], None),
                BODY,
                ("S_AND_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", 10), ("SSRC1", 88)], None),
                EXP,
            ];
            assert_eq!(reasons(plain), Vec::<&str>::new());
            assert_eq!(reasons(p), vec!["unsafe-execz-branch"]);
        }
    }
}
