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
const STALE_SCALAR: u8 = 128;

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

/// What an instruction reads and writes.
struct Uses {
    /// registers whose whole value is read (`x64_masks::uses` for the SGPRs: destinations a write
    /// keeps part of, SDWA scalar sources, SMEM offsets, `src_vccz` / `src_execz`; SCC, also as
    /// `src_scc`; the VGPRs, NSA addresses included)
    reads: Vec<Reg>,
    /// the SGPRs of those read as an explicit operand (not an implicit EXEC / VCC / M0)
    explicit: Vec<usize>,
    /// a mask read lane by lane as a selector or carry (`v_cndmask`, `v_add_co_ci`, …)
    lane_mask: Vec<usize>,
    /// registers written (a write that keeps part of the old value reads it too); image loads
    /// by their DMASK
    writes: Vec<Reg>,
}

fn uses(d: &Decoded) -> Uses {
    let n = d.inst.name;
    let (whole, _) = super::x64_masks::uses(d);
    let (explicit_r, explicit_w) = rdna2::sgpr_uses(d);
    let mut implicit_in: u128 = 0;
    let mut reads: Vec<Reg> = Vec::new();
    let mut writes: Vec<Reg> = explicit_w.iter().map(|&s| Reg::S(s as usize)).collect();
    for o in d.ienc.operands {
        let regs: &[Reg] = match o.ty {
            "OPR_VCC" => &[Reg::S(VCC), Reg::S(VCC + 1)],
            "OPR_SDST_EXEC" | "OPR_EXEC" => &[Reg::S(EXEC), Reg::S(EXEC + 1)],
            "OPR_SSRC_SPECIAL_SCC" => &[Reg::Scc],
            "OPR_SDST_M0" => &[Reg::S(124)],
            _ => continue,
        };
        if o.output {
            for &g in regs {
                if !writes.contains(&g) {
                    writes.push(g);
                }
            }
        } else if o.field.is_none() {
            for &g in regs {
                match g {
                    Reg::S(s) => implicit_in |= 1 << s,
                    Reg::Scc => reads.push(Reg::Scc),
                    Reg::V(_) => {}
                }
            }
        }
    }
    let lane_mask: Vec<usize> = if matches!(n, "V_CNDMASK_B32" | "V_ADD_CO_CI_U32" | "V_SUB_CO_CI_U32" | "V_SUBREV_CO_CI_U32") {
        match rdna2::field_value(d.enc, &d.words, "SRC2") {
            Some(s) if s < 126 => vec![s as usize, s as usize + 1],
            _ => vec![VCC, VCC + 1],
        }
    } else {
        Vec::new()
    };
    for s in 0..128usize {
        if whole >> s & 1 == 1 {
            reads.push(Reg::S(s));
        }
    }
    // src_scc as an operand
    for f in ["SRC0", "SRC1", "SRC2", "SSRC0", "SSRC1"] {
        if rdna2::field_value(d.enc, &d.words, f) == Some(253) && d.enc.field(f).is_some() && !reads.contains(&Reg::Scc) {
            reads.push(Reg::Scc);
        }
    }
    let explicit: Vec<usize> = explicit_r.iter().map(|&s| s as usize).chain((0..128usize).filter(|&s| whole >> s & 1 == 1 && implicit_in >> s & 1 == 0)).collect();
    // VGPRs: an image load by its DMASK, NSA addresses
    let (vr, _) = rdna2::vgpr_uses(d);
    reads.extend(vr.iter().map(|&v| Reg::V(v as usize)));
    if d.enc.name.starts_with("ENC_MIMG") {
        let nsa = rdna2::field_value(d.enc, &d.words, "NSA").unwrap_or(0);
        if let Some(e) = (nsa > 0).then(|| rdna2::encoding(&format!("MIMG_NSA{nsa}"))).flatten() {
            for f in e.fields.iter().filter(|f| f.name.starts_with("VADDR") && f.name != "VADDR") {
                if let Some(v) = rdna2::field_value(e, &d.words, f.name) {
                    reads.push(Reg::V(v as usize));
                }
            }
        }
    }
    writes.extend(super::upper_lanes::vgpr_writes(d).iter().map(|&v| Reg::V(v as usize)));
    Uses { reads, explicit, lane_mask, writes }
}

fn is_valu(d: &Decoded) -> bool {
    d.enc.name.starts_with("ENC_VOP") || d.enc.name.starts_with("VOP")
}

fn is_vcmp(d: &Decoded) -> bool {
    d.inst.name.starts_with("V_CMP")
}

/// Reads a lane mask as a mask (bit by bit, or as a whole): not as a number.
fn mask_op(n: &str) -> bool {
    mask_result(n)
        || n.starts_with("S_CMP_") && n.ends_with("_U64")
        || n.starts_with("S_CBRANCH")
        || n.starts_with("V_CNDMASK")
        || n.contains("_CO_CI_")
        || matches!(n, "S_BCNT0_I32_B64" | "S_BCNT1_I32_B64" | "S_FF1_I32_B64" | "S_FF0_I32_B64" | "S_FLBIT_I32_B64")
}

/// Makes a lane mask (or a word of one) of lane masks (bitwise, moves, selects, saveexec): a
/// number of one is not.
fn mask_result(n: &str) -> bool {
    let bitwise = ["S_AND_", "S_OR_", "S_XOR_", "S_ANDN1_", "S_ANDN2_", "S_ORN1_", "S_ORN2_", "S_NAND_", "S_NOR_", "S_XNOR_"];
    let word = n.ends_with("_B64") || n.ends_with("_B32");
    (word && (bitwise.iter().any(|p| n.starts_with(p)) || n.starts_with("S_NOT_") || n.starts_with("S_MOV_B") || n.starts_with("S_CSELECT_B") || n.starts_with("S_CMOV_B") || n == "S_WQM_B64"))
        || n.contains("SAVEEXEC")
}

/// The partition-dependent value an instruction makes of its own (not from its sources).
fn source_taint(d: &Decoded, st: &State, u: &Uses) -> u8 {
    let n = d.inst.name;
    let reads = &u.reads;
    let mut t = 0;
    // explicit operands only: a vector or memory instruction's implicit EXEC is its lane mask
    let reads_mask = u.explicit.iter().any(|&s| st.mask[s]);
    // a vote's outcome used as a number (s_cselect, s_addc, … on SCC; not the branch on it)
    if reads.contains(&Reg::Scc) && st.scc_vote != SccVote::No && !n.starts_with("S_CBRANCH") {
        t |= MASK_DATA;
    }
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
    // another lane's value: exact only for a value every lane has, read from the first active
    // lane (a given lane may be one the half did not launch)
    if (n == "V_READFIRSTLANE_B32" && reads.iter().any(|r| matches!(r, Reg::V(v) if !st.uniform[*v]))) || n == "V_READLANE_B32" {
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
        (STALE_SCALAR, "stale-scalar"),
    ] {
        if t & bit != 0 {
            out.insert(name);
        }
    }
}

/// What the program leaves behind: exports, memory writes (LDS / GDS ones included).
fn is_store(n: &str) -> bool {
    let ds_read = n.starts_with("DS_READ") || matches!(n, "DS_SWIZZLE_B32" | "DS_PERMUTE_B32" | "DS_BPERMUTE_B32" | "DS_NOP");
    n == "EXP" || n.contains("STORE") || n.contains("ATOMIC") || (n.starts_with("DS_") && !ds_read) || n.starts_with("S_SENDMSG")
}

/// One instruction: the state after it; what it leaves behind that depends on the cut goes to
/// `out`.
fn step(d: &Decoded, st: &mut State, out: &mut BTreeSet<&'static str>) {
    let n = d.inst.name;
    let u = uses(d);
    let (reads, writes) = (&u.reads, &u.writes);
    let mut t = source_taint(d, st, &u);
    for r in reads {
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
    let writes_exec = writes.contains(&Reg::S(EXEC)) || writes.contains(&Reg::S(EXEC + 1));
    if writes_exec && t != 0 {
        reasons_of(t, out);
    }
    // one word of EXEC written on its own: a 32-lane half's lanes are all under its low word
    if writes.contains(&Reg::S(EXEC)) != writes.contains(&Reg::S(EXEC + 1)) {
        out.insert("mask-data");
    }
    // a number written to EXEC (not all lanes, not none): lane k of each half is a different
    // console lane
    if writes_exec && matches!(n, "S_MOV_B64" | "S_MOV_B32") && !matches!(rdna2::field_value(d.enc, &d.words, "SSRC0"), Some(128 | 193) | Some(0..=127)) {
        out.insert("mask-data");
    }
    let reads_mask_or_exec = reads.iter().any(|r| matches!(r, Reg::S(s) if st.mask[*s])) || u.lane_mask.iter().any(|&s| st.mask[s]);
    // a per-lane selector counts as a source: it is every lane's only if all or none of them
    let all_uniform = reads.iter().all(|r| !matches!(r, Reg::V(v) if !st.uniform[*v])) && u.lane_mask.iter().all(|&s| st.uniform_mask[s]);
    // a mask that is all or none of the active lanes (EXEC itself, a compare of values every lane
    // has, a scalar condition selected into a mask): a half with active lanes votes on it as the
    // wave does. A half with none may take the other side: it writes no lane there, and its
    // scalars are followed as stale where the sides join. Kept by these ops, not by complements.
    let vcmp_uniform = is_vcmp(d) && all_uniform;
    let keeps_uniform = matches!(n, "S_AND_B64" | "S_OR_B64" | "S_ANDN2_B64" | "S_MOV_B64" | "S_CSELECT_B64" | "S_WQM_B64") || n.contains("SAVEEXEC");
    let sgpr_reads: Vec<usize> = reads.iter().filter_map(|r| if let Reg::S(s) = r { Some(*s) } else { None }).collect();
    let is_launch = |s: usize| if s == EXEC || s == EXEC + 1 { st.exec_launch } else { st.launch[s] };
    let mask_uniform = keeps_uniform && sgpr_reads.iter().all(|&s| !st.mask[s] || st.uniform_mask[s] || s == EXEC || s == EXEC + 1 || is_launch(s));
    let from_launch = matches!(n, "S_MOV_B64" | "S_WQM_B64") && !sgpr_reads.is_empty() && sgpr_reads.iter().all(|&s| is_launch(s));
    let or_launch = n == "S_OR_B64" && sgpr_reads.iter().any(|&s| is_launch(s));
    let saved_exec = n.contains("SAVEEXEC") && st.exec_launch;
    let mut new_scc = None;
    for w in writes {
        match *w {
            Reg::S(s) => {
                st.s[s] = t;
                // (a carry-out is a mask too)
                st.mask[s] = s == EXEC || s == EXEC + 1 || is_vcmp(d) || (is_valu(d) && n.contains("_CO_")) || (mask_result(n) && reads_mask_or_exec);
                st.uniform_mask[s] = vcmp_uniform || (st.mask[s] && mask_uniform);
                st.launch[s] = s != EXEC && s != EXEC + 1 && (from_launch || (saved_exec && Some(&Reg::S(s)) != writes.iter().find(|w| **w == Reg::S(EXEC))));
            }
            Reg::V(v) => {
                st.v[v] = t;
                // a value every lane has: written under the launch EXEC (a narrowed one leaves the
                // other lanes their old value) by a plain VALU op of such values (not DPP / SDWA,
                // which read other lanes or parts), or a load from an address every lane has
                let load = matches!(d.enc.name, "ENC_MUBUF" | "ENC_MTBUF" | "ENC_MIMG") && !n.contains("ATOMIC");
                let plain = d.enc.name.starts_with("ENC_VOP") && d.variant == rdna2::Variant::Plain && !n.starts_with("V_PERMLANE");
                st.uniform[v] = st.exec_launch && all_uniform && t & (RANK | LANE_ID | CROSS_LANE) == 0 && n != "V_WRITELANE_B32" && (plain || load);
            }
            Reg::Scc => {
                new_scc = Some(t);
            }
        }
    }
    // SCC no longer says whether the mask it was about is empty once that mask is rewritten
    if new_scc.is_none() {
        let stale_vote = match st.scc_vote {
            SccVote::Exec => writes_exec,
            SccVote::Mask(Some(m)) => writes.contains(&Reg::S(m)) || writes.contains(&Reg::S(m + 1)),
            _ => false,
        };
        if stale_vote {
            st.scc_vote = SccVote::Mask(None);
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

/// What an instruction does to EXEC, for `widens_after`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ExecWrite {
    No,
    /// an AND with EXEC (`s_and_b64` / `s_andn2_b64 exec, exec, x`, `s_and_saveexec_b64`, `v_cmpx`)
    /// or the whole-quad form (quad-local: it adds no lane of a quad with none)
    Narrows,
    /// EXEC from (or with) the mask in this SGPR: `s_or_b64` / `s_xor_b64` / `s_mov_b64 exec, s`,
    /// `s_or_saveexec_b64 _, s` – no wider than that mask and EXEC
    From(usize),
    /// EXEC = this SGPR's mask AND something (`s_and_b64 exec, s, x`, `s_andn2(_wrexec)_b64 exec, s, x`):
    /// no wider than that mask
    AndWith(usize),
    /// anything else
    Widens,
}

fn exec_write(d: &Decoded) -> ExecWrite {
    let u = uses(d);
    if !u.writes.contains(&Reg::S(EXEC)) && !u.writes.contains(&Reg::S(EXEC + 1)) {
        return ExecWrite::No;
    }
    let n = d.inst.name;
    let f = |name: &str| rdna2::field_value(d.enc, &d.words, name);
    let with_exec = |a: Option<u64>| a == Some(EXEC as u64);
    if n.starts_with("V_CMPX")
        || n == "S_AND_SAVEEXEC_B64"
        || n == "S_WQM_B64"
        || (n == "S_AND_B64" && (with_exec(f("SSRC0")) || with_exec(f("SSRC1"))))
        || (matches!(n, "S_ANDN2_B64" | "S_ANDN2_WREXEC_B64") && with_exec(f("SSRC0")))
    {
        return ExecWrite::Narrows;
    }
    match n {
        // (an AND of two masks: either bounds it; the first is the one followed)
        "S_AND_B64" | "S_ANDN2_B64" | "S_ANDN2_WREXEC_B64" => {
            if let Some(s) = f("SSRC0").filter(|&s| s < 126) {
                return ExecWrite::AndWith(s as usize);
            }
        }
        _ => {}
    }
    let other = |a: Option<u64>, b: Option<u64>| match (a, b) {
        (Some(x), Some(y)) if x == EXEC as u64 && y < 126 => Some(y as usize),
        (Some(x), Some(y)) if y == EXEC as u64 && x < 126 => Some(x as usize),
        _ => None,
    };
    let src = match n {
        "S_OR_B64" | "S_XOR_B64" => other(f("SSRC0"), f("SSRC1")),
        "S_MOV_B64" | "S_OR_SAVEEXEC_B64" | "S_XOR_SAVEEXEC_B64" => f("SSRC0").filter(|&s| s < 126).map(|s| s as usize),
        _ => None,
    };
    src.map_or(ExecWrite::Widens, ExecWrite::From)
}

/// SGPRs an instruction saves EXEC into (saveexec's destination, `s_mov_b64 s, exec`).
fn saves_exec(d: &Decoded) -> Option<usize> {
    let f = |name: &str| rdna2::field_value(d.enc, &d.words, name);
    match d.inst.name {
        n if n.contains("SAVEEXEC") => f("SDST").map(|s| s as usize),
        "S_MOV_B64" if f("SSRC0") == Some(EXEC as u64) => f("SDST").map(|s| s as usize),
        _ => None,
    }
}

/// From the first instruction after `run`'s first (its narrowing) up to `stop` (exclusive; None:
/// the program's end), some block widens EXEC past what it was narrowed to: EXEC from a mask
/// not saved inside these blocks (a nested if's restore is fine), or EXEC restored on the way
/// into the join with lane work after it.
fn widens_after(insts: &[Decoded], g: &Cfg, exec_w: &[ExecWrite], run: usize, stop: Option<usize>, mask: Option<usize>) -> bool {
    let mut seen = BTreeSet::new();
    let mut todo = vec![run];
    let mut blocks = Vec::new();
    while let Some(b) = todo.pop() {
        if Some(b) == stop || !seen.insert(b) {
            continue;
        }
        blocks.push(b);
        todo.extend(g.succ[b].iter().copied());
    }
    // (not what the narrowing itself saved: that is the EXEC from before it)
    let first = g.range(run).next();
    let mut saved: BTreeSet<usize> = blocks.iter().flat_map(|&b| g.range(b)).filter(|&i| Some(i) != first).filter_map(|i| saves_exec(&insts[i])).collect();
    // the vote's own mask bounds what its side runs
    saved.extend(mask);
    for &b in &blocks {
        let skip = usize::from(b == run);
        let range: Vec<usize> = g.range(b).skip(skip).collect();
        for (k, &i) in range.iter().enumerate() {
            let wide = match exec_w[i] {
                ExecWrite::No | ExecWrite::Narrows => false,
                ExecWrite::From(s) | ExecWrite::AndWith(s) => !saved.contains(&s),
                ExecWrite::Widens => true,
            };
            // an if/else's else: `s_or_saveexec_b64 s, s` then `s_xor_b64 exec, exec, s` – the
            // lanes of s that were not running, no wider than s and the EXEC before it
            let else_flip = insts[i].inst.name == "S_OR_SAVEEXEC_B64"
                && range.get(k + 1).is_some_and(|&j| insts[j].inst.name == "S_XOR_B64" && matches!(exec_w[j], ExecWrite::From(_)));
            let wide = wide && !else_flip;
            // EXEC restored on the way into the join is fine if no lane work follows before it
            let trailing = stop.is_some() && g.succ[b].iter().all(|&s| Some(s) == stop) && range[k + 1..].iter().all(|&j| insts[j].enc.name.starts_with("ENC_SOP"));
            if wide && !trailing {
                return true;
            }
        }
    }
    false
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
    let exec_w: Vec<ExecWrite> = insts.iter().map(exec_write).collect();
    // first without stale scalars, to find the votes; then with the scalars each vote's
    // dependent blocks write marked stale where its sides join
    let none = vec![Vec::new(); nb];
    let at = fixed_point(insts, &g, &none);
    let stale = stale_at_joins(insts, &g, &ipdom, &at);
    let at = fixed_point(insts, &g, &stale);
    // the sinks, and the votes, from the fixed point
    for b in 0..nb {
        let Some(mut st) = at[b].clone() else { continue };
        mark_stale(&mut st, &stale[b]);
        let range = g.range(b);
        let last = range.end - 1;
        for i in range {
            if i == last {
                vote(insts, &g, &ipdom, &exec_w, b, &insts[i], &st, &mut reasons);
            }
            step(&insts[i], &mut st, &mut reasons);
        }
    }
    let tier = if reasons.is_empty() { Tier::Exact } else { Tier::MayDiffer };
    Partition { tier, reasons }
}

/// Marks scalar registers (and SCC) stale.
fn mark_stale(st: &mut State, regs: &[Reg]) {
    for r in regs {
        match *r {
            Reg::S(s) => st.s[s] |= STALE_SCALAR,
            Reg::Scc => st.scc |= STALE_SCALAR,
            Reg::V(_) => {}
        }
    }
}

/// Block-entry states to a fixed point (the bits only grow, the flags only fall), with the
/// registers of `stale[b]` marked stale on entering block `b`.
fn fixed_point(insts: &[Decoded], g: &Cfg, stale: &[Vec<Reg>]) -> Vec<Option<State>> {
    let nb = g.start.len();
    let mut at: Vec<Option<State>> = vec![None; nb];
    at[0] = Some(State::entry());
    let mut work = BTreeSet::from([0usize]);
    let mut scratch = BTreeSet::new();
    while let Some(b) = work.pop_first() {
        let mut st = at[b].clone().expect("queued with a state");
        mark_stale(&mut st, &stale[b]);
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
    at
}

/// Whether the conditional branch `d` (state `st` before it) is a vote a half can take
/// differently from the wave: on a mask, unless every half has lanes and votes as the wave does
/// (the launch EXEC, or a mask that is all or none of its lanes).
fn is_vote(d: &Decoded, st: &State) -> bool {
    let mask = match d.inst.name {
        "S_CBRANCH_EXECZ" | "S_CBRANCH_EXECNZ" => EXEC,
        "S_CBRANCH_VCCZ" | "S_CBRANCH_VCCNZ" => VCC,
        "S_CBRANCH_SCC0" | "S_CBRANCH_SCC1" => match st.scc_vote {
            SccVote::No => return false,
            SccVote::Exec => EXEC,
            SccVote::Mask(Some(m)) => m,
            SccVote::Mask(None) => return true,
        },
        _ => return false,
    };
    let uniform = mask == EXEC || (mask + 1 < 128 && st.uniform_mask[mask] && st.uniform_mask[mask + 1]);
    !(st.exec_launch && uniform)
}

/// For each block, the scalars (SGPRs, VCC, M0, SCC; not EXEC, which the join restores) that
/// the blocks a vote decides write, when the vote's sides join there: a half that took the
/// other side has not written them, so what it reads there is not the wave's.
fn stale_at_joins(insts: &[Decoded], g: &Cfg, ipdom: &[PostDom], at: &[Option<State>]) -> Vec<Vec<Reg>> {
    let nb = g.start.len();
    let cd = g.control_dependence(ipdom);
    // branch block -> the blocks that depend on it
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); nb];
    for (x, on) in cd.iter().enumerate() {
        for &c in on {
            dependents[c].push(x);
        }
    }
    let mut stale: Vec<Vec<Reg>> = vec![Vec::new(); nb];
    for b in 0..nb {
        let (Some(st), PostDom::Block(join)) = (&at[b], ipdom[b]) else { continue };
        let mut st = st.clone();
        let range = g.range(b);
        let last = range.end - 1;
        let mut scratch = BTreeSet::new();
        for i in range.start..last {
            step(&insts[i], &mut st, &mut scratch);
        }
        if g.succ[b].len() < 2 || !is_vote(&insts[last], &st) {
            continue;
        }
        // the blocks it decides, nested branches' blocks included
        let mut region: BTreeSet<usize> = BTreeSet::new();
        let mut todo = dependents[b].clone();
        while let Some(x) = todo.pop() {
            if region.insert(x) {
                todo.extend(dependents[x].iter().copied());
            }
        }
        for &x in &region {
            for i in g.range(x) {
                for w in uses(&insts[i]).writes {
                    let scalar = matches!(w, Reg::Scc) || matches!(w, Reg::S(s) if s != EXEC && s != EXEC + 1);
                    if scalar && !stale[join].contains(&w) {
                        stale[join].push(w);
                    }
                }
            }
        }
    }
    stale
}

/// A conditional branch at the end of block `b` (state `st` before it): a vote that may run a
/// side the console wave did not run for the half's lanes.
#[allow(clippy::too_many_arguments)]
fn vote(insts: &[Decoded], g: &Cfg, ipdom: &[PostDom], exec_w: &[ExecWrite], b: usize, d: &Decoded, st: &State, out: &mut BTreeSet<&'static str>) {
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
    let joins = ipdom[b] == PostDom::Block(empty);
    let skips = joins || leaves_nothing(insts, g, empty);
    // … and keeps it narrowed: up to the join, or for a way out that ends the program, to the end
    // (a half whose mask is empty must not miss lanes the wave turns on again)
    let narrows = match mask {
        None => true,
        Some(m) => narrows_exec_by(insts, g, run, m),
    } && !widens_after(insts, g, exec_w, run, if joins { Some(empty) } else { None }, mask);
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
    fn a_scalar_written_behind_a_vote_and_read_after_the_join_may_differ() {
        // 0 cmp, 1 saveexec, 2 execz → 4; 3 s5 = 1 (only where some lane voted yes); 4 restore,
        // 5 v0 = s5, 6 exp
        const DEF: I = ("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 129)], None);
        const USE: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 5)], None);
        let p: &[I] = &[CMP, NARROW, ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)], None), DEF, RESTORE, USE, EXP];
        assert_eq!(reasons(p), vec!["stale-scalar"]);
        // not read after the join: exact
        let q: &[I] = &[CMP, NARROW, ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)], None), DEF, RESTORE, EXP];
        assert_eq!(reasons(q), Vec::<&str>::new());
    }

    #[test]
    fn a_scalar_behind_a_vote_every_half_takes_as_the_wave_is_exact() {
        // a compare of values every lane has, under the launch exec: every half takes the same side
        const UCMP: I = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 2), ("VSRC1", 7)], None);
        const UV7: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 3)], None);
        const DEF: I = ("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 129)], None);
        const USE: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 5)], None);
        let p: &[I] = &[UV7, UCMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)], None), DEF, USE, EXP];
        assert_eq!(reasons(p), Vec::<&str>::new());
    }

    // ----- counter-examples from the review of #517 (each was Exact and exports something else
    // when the wave is cut)

    #[test]
    fn a_value_written_under_a_narrowed_exec_is_not_every_lanes() {
        // v6 = s5; if (v4 > v5) v6 = s6; s0 = first lane's v6; export it
        let p: &[I] = &[
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 5)], None),
            CMP,
            NARROW,
            ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 6)], None),
            RESTORE,
            ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 6)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 0)], None),
            EXP,
        ];
        assert!(reasons(p).contains(&"cross-lane"));
    }

    #[test]
    fn a_per_lane_select_is_not_every_lanes() {
        // v7 = s3; v6 = vcc ? v7 : 0 (vcc per pixel); s0 = first lane's v6; export it
        let p: &[I] = &[
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 3)], None),
            CMP,
            ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 128), ("VSRC1", 7)], None),
            ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 6)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 0)], None),
            EXP,
        ];
        assert!(reasons(p).contains(&"cross-lane"));
    }

    #[test]
    fn a_vote_read_as_a_number_may_differ() {
        // scc = (vcc != 0); s20 = scc ? s6 : s5; export s20
        let p: &[I] = &[
            CMP,
            ("S_CMP_LG_U64", "ENC_SOPC", &[("SSRC0", VCC as u64), ("SSRC1", 128)], None),
            ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 20), ("SSRC0", 6), ("SSRC1", 5)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 20)], None),
            EXP,
        ];
        assert!(!reasons(p).is_empty());
    }

    #[test]
    fn a_mask_word_moved_into_another_register_is_still_a_mask() {
        // s20 = exec_lo; v0 = s20; export
        let p: &[I] = &[("S_MOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", EXEC as u64)], None), ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 20)], None), EXP];
        assert_eq!(reasons(p), vec!["mask-data"]);
    }

    #[test]
    fn a_write_that_keeps_the_old_value_keeps_its_taint() {
        // s20 = popcount(exec); s20 += 1 (s_addk); v0 = s20; export
        let p: &[I] = &[
            ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", EXEC as u64)], None),
            ("S_ADDK_I32", "ENC_SOPK", &[("SDST", 20), ("SIMM16", 1)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 20)], None),
            EXP,
        ];
        assert_eq!(reasons(p), vec!["lane-count"]);
    }

    #[test]
    fn a_vote_whose_side_widens_exec_again_may_differ() {
        // 0 cmp, 1 vccz → 6; 2 exec &= vcc (saved in s[10:11]), 3 exec = s[10:11] (widened again),
        // 4 v0 = 1.0 under the whole exec, 5 s_nop; 6 exp
        const WIDEN: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", EXEC as u64), ("SSRC0", 10)], None);
        let p: &[I] = &[CMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 4)], None), NARROW, WIDEN, BODY, ("S_NOP", "ENC_SOPP", &[("SIMM16", 0)], None), EXP];
        assert!(reasons(p).contains(&"vote-branch"));
    }

    #[test]
    fn an_scc_vote_is_forgotten_when_its_mask_is_rewritten() {
        // 0 cmp, 1 s[0:1] = vcc & exec (scc), 2 s[0:1] = s[2:3] (scc is no longer about s[0:1]),
        // 3 scc0 → 6; 4 exec = s[0:1] & exec, 5 body; 6 exp
        const AND: I = ("S_AND_B64", "ENC_SOP2", &[("SDST", 0), ("SSRC0", VCC as u64), ("SSRC1", EXEC as u64)], None);
        const OTHER: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 2)], None);
        const RUN: I = ("S_AND_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", 0), ("SSRC1", EXEC as u64)], None);
        let p: &[I] = &[CMP, AND, OTHER, ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 2)], None), RUN, BODY, EXP];
        assert!(reasons(p).contains(&"vote-branch"));
    }

    #[test]
    fn the_high_word_of_exec_and_a_number_in_exec_are_seen() {
        // exec_hi = 0: in a 32-lane half its lanes are not under exec_hi – a sink of its own
        let hi: &[I] = &[("S_MOV_B32", "ENC_SOP1", &[("SDST", EXEC as u64 + 1), ("SSRC0", 128)], None), BODY, EXP];
        assert!(!reasons(hi).is_empty());
        // exec = lane 0 only: lane 0 of each half is a different console lane
        let one: &[I] = &[("S_MOV_B64", "ENC_SOP1", &[("SDST", EXEC as u64), ("SSRC0", 129)], None), BODY, EXP];
        assert!(!reasons(one).is_empty());
    }

    #[test]
    fn an_image_load_writes_only_the_registers_its_dmask_names() {
        // v3 = popcount(exec) (tainted); v0 = sample (DMASK 1: v0 only); export v0..v3 – v3 still tainted
        let p: &[I] = &[
            ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", EXEC as u64)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 20)], None),
            ("IMAGE_SAMPLE", "ENC_MIMG", &[("VDATA", 0), ("VADDR", 4), ("SRSRC", 2), ("SSAMP", 6), ("DMASK", 1), ("DIM", 1)], None),
            EXP,
        ];
        assert_eq!(reasons(p), vec!["lane-count"]);
    }

    #[test]
    fn a_ds_write_is_a_store() {
        // m0 = popcount(exec); ds_write_b32 v4, v0 with v0 = m0
        let p: &[I] = &[
            ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", EXEC as u64)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 20)], None),
            ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 4), ("DATA0", 0)], None),
        ];
        assert_eq!(reasons(p), vec!["lane-count"]);
        let q: &[I] = &[
            ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", EXEC as u64)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 20)], None),
            ("DS_MAX_U32", "ENC_DS", &[("ADDR", 4), ("DATA0", 0)], None),
        ];
        assert_eq!(reasons(q), vec!["lane-count"]);
    }

    // ----- every reason has a test

    #[test]
    fn a_first_lane_index_may_differ() {
        let p: &[I] = &[("S_FF1_I32_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", EXEC as u64)], None), ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 20)], None), EXP];
        assert_eq!(reasons(p), vec!["first-lane"]);
    }

    #[test]
    fn work_under_an_exec_that_depends_on_the_cut_may_differ() {
        // s[20:21] = 1 << popcount(exec) → exec (a number, and a lane count); v0 = 1.0 under it
        let p: &[I] = &[
            ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", EXEC as u64)], None),
            ("S_LSHL_B64", "ENC_SOP2", &[("SDST", 22), ("SSRC0", 129), ("SSRC1", 20)], None),
            ("S_AND_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 22)], None),
            BODY,
            EXP,
        ];
        assert!(reasons(p).contains(&"tainted-exec"));
    }

    #[test]
    fn an_execz_that_does_not_skip_to_the_join_may_differ() {
        // 0 cmp, 1 saveexec, 2 execz → 5 (past the join's code); 3 body, 4 s_branch → 6; 5 v0 = 0; 6 exp
        let p: &[I] = &[
            CMP,
            NARROW,
            ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 2)], None),
            BODY,
            ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 1)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 128)], None),
            EXP,
        ];
        assert!(reasons(p).contains(&"unsafe-execz-branch"));
    }

    #[test]
    fn an_indirect_jump_is_not_classified() {
        let p: &[I] = &[("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 4)], None)];
        assert_eq!(reasons(p), vec!["no-graph"]);
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
}
