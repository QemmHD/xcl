//! Review of #172 (`structure.rs`, the general structured form): a counter-example hunt.
//!
//! * [`Graph`]: a program's control flow graph as its bytes give it (the decoder and the
//!   branch targets only, nothing of `structure.rs`), a reducibility test and a walk with
//!   given branch decisions;
//! * [`cpu`]: a reference run of the instructions these programs use. The `pc` chain is no
//!   reference: with `structured: false` the translator still emits the first structured form
//!   inside its `pc` guards, and that form is wrong for some shapes (the counter-examples at
//!   the end);
//! * [`skeleton`]: the control statements of the generated HLSL (`do { } while (false)`,
//!   `for (;;)`, `if`, `break`, `continue`, `xcl_br`, the first form's skip flags) run with
//!   the same decisions: the instructions they reach must be the graph's. `structure::check`
//!   walks the tree the HLSL is rendered from; this walks the text DXC gets;
//! * [`generate`]: random reducible and irreducible programs whose loops end by
//!   construction (a step counter forces every scalar branch towards the exit), with EXEC
//!   masking, masks, lane reads, DPP, barriers and stores inside the branches;
//! * GPU: the general form (and, counted only, the default path and the `pc` chain) against
//!   the reference run, registers and memory, as a wave32 program and as a wave64 one (split
//!   where the GPU has 32 lanes).

use std::collections::{BTreeSet, HashMap};

use super::*;
use crate::gpu::rdna2;

// ---------------------------------------------------------------- the program's graph

#[derive(Clone, Debug, PartialEq)]
enum Next {
    Exit,
    Jump(usize),
    /// the conditional branch at `at` (`name`, lower case): `taken`, or `fall` (None: the end)
    Cond { at: u64, name: String, taken: usize, fall: Option<usize> },
}

struct Graph {
    /// per block: the addresses of its instructions
    blocks: Vec<Vec<u64>>,
    next: Vec<Next>,
    /// the `s_endpgm` instructions (no code of their own)
    ends: BTreeSet<u64>,
}

/// Conditional branches the emitter translates (the others are never taken, in every form).
fn translated_branch(name: &str) -> bool {
    matches!(name, "s_cbranch_scc0" | "s_cbranch_scc1" | "s_cbranch_vccz" | "s_cbranch_vccnz" | "s_cbranch_execz" | "s_cbranch_execnz")
}

impl Graph {
    fn of(code: &[u8]) -> Graph {
        let mut insts = Vec::new();
        let mut off = 0usize;
        while off + 4 <= code.len() {
            let d = rdna2::decode(&code[off..], off as u64).unwrap();
            off += d.len;
            insts.push(d);
        }
        rdna2::resolve_far_jumps(&mut insts);
        let mut leaders = BTreeSet::from([0u64]);
        for d in &insts {
            let next = d.pc + d.len as u64;
            if let Some(t) = rdna2::branch_target(d) {
                leaders.insert(t);
                leaders.insert(next);
            } else if d.inst.name.starts_with("S_ENDPGM") {
                leaders.insert(next);
            }
        }
        let starts: Vec<u64> = leaders.into_iter().filter(|l| insts.iter().any(|d| d.pc == *l)).collect();
        let index = |pc: u64| starts.binary_search(&pc).unwrap_or_else(|_| panic!("branch to {pc:#x}"));
        let (mut blocks, mut next) = (Vec::new(), Vec::new());
        for (i, &s) in starts.iter().enumerate() {
            let end = starts.get(i + 1).copied().unwrap_or(u64::MAX);
            let body: Vec<_> = insts.iter().filter(|d| d.pc >= s && d.pc < end).collect();
            let last = body.last().unwrap();
            let fall = (i + 1 < starts.len()).then_some(i + 1);
            next.push(match (rdna2::branch_target(last), last.inst.name) {
                (Some(t), n) if n.starts_with("S_CBRANCH") => Next::Cond { at: last.pc, name: n.to_ascii_lowercase(), taken: index(t), fall },
                (Some(t), _) => Next::Jump(index(t)),
                (None, n) if n.starts_with("S_ENDPGM") => Next::Exit,
                _ => fall.map_or(Next::Exit, Next::Jump),
            });
            blocks.push(body.iter().map(|d| d.pc).collect());
        }
        let ends = insts.iter().filter(|d| d.inst.name.starts_with("S_ENDPGM")).map(|d| d.pc).collect();
        Graph { blocks, next, ends }
    }

    fn successors(&self, b: usize) -> Vec<usize> {
        match &self.next[b] {
            Next::Exit => vec![],
            Next::Jump(t) => vec![*t],
            Next::Cond { taken, fall, .. } => {
                let mut v = vec![*taken];
                v.extend(fall.filter(|f| f != taken));
                v
            }
        }
    }

    fn reachable(&self) -> Vec<bool> {
        let mut seen = vec![false; self.next.len()];
        let mut work = vec![0usize];
        while let Some(b) = work.pop() {
            if !std::mem::replace(&mut seen[b], true) {
                work.extend(self.successors(b));
            }
        }
        seen
    }

    /// Reducible: without the edges to a dominator of their source, the reachable part has
    /// no cycle.
    fn reducible(&self) -> bool {
        let n = self.next.len();
        let reach = self.reachable();
        let mut preds = vec![Vec::new(); n];
        for b in (0..n).filter(|&b| reach[b]) {
            for t in self.successors(b) {
                preds[t].push(b);
            }
        }
        // dom[b][a]: a dominates b
        let mut dom = vec![vec![true; n]; n];
        dom[0] = (0..n).map(|a| a == 0).collect();
        loop {
            let mut changed = false;
            for b in (1..n).filter(|&b| reach[b]) {
                let d: Vec<bool> = (0..n).map(|a| a == b || preds[b].iter().all(|&p| dom[p][a])).collect();
                changed |= std::mem::replace(&mut dom[b], d.clone()) != d;
            }
            if !changed {
                break;
            }
        }
        // Kahn over the edges that are not back edges
        let dom = &dom;
        let forward = |u: usize| self.successors(u).into_iter().filter(move |&v| !dom[u][v]).collect::<Vec<_>>();
        let mut indeg = vec![0usize; n];
        for u in (0..n).filter(|&b| reach[b]) {
            for v in forward(u) {
                indeg[v] += 1;
            }
        }
        let mut work: Vec<usize> = (0..n).filter(|&b| reach[b] && indeg[b] == 0).collect();
        let mut left = reach.iter().filter(|&&r| r).count();
        while let Some(u) = work.pop() {
            left -= 1;
            for v in forward(u) {
                indeg[v] -= 1;
                if indeg[v] == 0 {
                    work.push(v);
                }
            }
        }
        left == 0
    }

    /// The instructions a run with these decisions executes (at most `cap` and a block).
    fn walk(&self, decide: &mut Decider, cap: usize) -> Vec<u64> {
        let mut trace = Vec::new();
        let mut b = Some(0usize);
        while let Some(x) = b {
            if trace.len() >= cap {
                break;
            }
            trace.extend(&self.blocks[x]);
            b = match &self.next[x] {
                Next::Exit => None,
                Next::Jump(t) => Some(*t),
                Next::Cond { at, name, taken, fall } => {
                    if translated_branch(name) && decide.at(*at) {
                        Some(*taken)
                    } else {
                        *fall
                    }
                }
            };
        }
        trace
    }
}

/// Branch decisions: a function of the branch and of how often it ran, so that the graph
/// and the HLSL take the same ones without sharing a sequence.
struct Decider {
    seed: u64,
    /// of 256: how often a branch is taken
    bias: u64,
    visits: HashMap<u64, u64>,
}

impl Decider {
    fn new(seed: u64, bias: u64) -> Decider {
        Decider { seed, bias, visits: HashMap::new() }
    }

    fn at(&mut self, branch: u64) -> bool {
        let k = self.visits.entry(branch).or_default();
        *k += 1;
        let mut x = self.seed ^ branch.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xD1B5_4A32_D192_ED03);
        x ^= x >> 31;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 29;
        (x & 255) < self.bias
    }
}

// ---------------------------------------------------------------- the program on the CPU

/// A reference run of the instructions the generated and hand-written programs use (ISA
/// semantics: SCC, EXEC-masked vector writes, masks over the active lanes, lane reads, DPP16).
/// The `pc` chain cannot be the reference: with `structured: false` the translator still emits
/// the first structured form for single-entry stretches (`structured_span`).
mod cpu {
    use std::collections::{BTreeSet, HashMap};

    use super::super::super::emit::ext_encoding;
    use super::super::KernelState;
    use crate::gpu::rdna2::{self, Decoded, Variant};

    pub struct Machine {
        pub s: Vec<u32>,
        pub scc: u32,
        /// `v[lane][reg]`
        pub v: Vec<Vec<u32>>,
        /// registers some instruction of the program writes, run or not (the kernel stores
        /// them; a form that runs a block it should not shows there)
        pub written_s: BTreeSet<usize>,
        pub written_v: BTreeSet<usize>,
        /// instructions run, in order
        pub trace: Vec<u64>,
        /// the kernel's flat memory (dwords)
        pub mem: Vec<u32>,
        lanes: usize,
    }

    impl Machine {
        fn field(d: &Decoded, name: &str) -> Result<u64, String> {
            rdna2::field_value(d.enc, &d.words, name).ok_or_else(|| format!("{}: no field {name}", d.inst.name))
        }

        fn scalar(&self, d: &Decoded, code: u64) -> Result<u32, String> {
            Ok(match code {
                0..=127 => self.s[code as usize],
                128..=192 => (code - 128) as u32,
                193..=208 => (-((code - 192) as i32)) as u32,
                255 => d.literal.ok_or("literal")?,
                _ => return Err(format!("{}: scalar operand {code}", d.inst.name)),
            })
        }

        fn scalar64(&self, d: &Decoded, code: u64) -> Result<u64, String> {
            Ok(match code {
                0..=126 => self.s[code as usize] as u64 | (self.s[code as usize + 1] as u64) << 32,
                _ => self.scalar(d, code)? as i32 as i64 as u64,
            })
        }

        fn set(&mut self, r: u64, v: u32) {
            self.s[r as usize] = v;
            self.written_s.insert(r as usize);
        }

        fn set64(&mut self, r: u64, v: u64) {
            self.set(r, v as u32);
            self.set(r + 1, (v >> 32) as u32);
        }

        fn exec(&self) -> u64 {
            let e = self.s[126] as u64 | (self.s[127] as u64) << 32;
            if self.lanes == 64 {
                e
            } else {
                e & 0xFFFF_FFFF
            }
        }

        fn vcc(&self) -> u64 {
            let m = self.s[106] as u64 | (self.s[107] as u64) << 32;
            if self.lanes == 64 {
                m
            } else {
                m & 0xFFFF_FFFF
            }
        }

        /// A VALU source in lane `l`: a VGPR (256 + n) or a scalar.
        fn vsrc(&self, d: &Decoded, code: u64, l: usize) -> Result<u32, String> {
            if code >= 256 {
                Ok(self.v[l][code as usize - 256])
            } else {
                self.scalar(d, code)
            }
        }

        fn vset(&mut self, r: u64, l: usize, v: u32) {
            self.v[l][r as usize] = v;
            self.written_v.insert(r as usize);
        }

        /// The source lane of a DPP16 control for lane `l` (None: outside the row).
        fn dpp16(ctrl: u64, l: usize) -> Result<Option<usize>, String> {
            let n = (ctrl & 15) as usize;
            Ok(match ctrl {
                0x000..=0x0FF => Some((l & !3) | ((ctrl as usize >> (2 * (l & 3))) & 3)),
                0x101..=0x10F => ((l & 15) + n < 16).then_some(l + n),
                0x111..=0x11F => ((l & 15) >= n).then(|| l - n),
                0x140 => Some((l & !15) | (15 - (l & 15))),
                _ => return Err(format!("DPP_CTRL {ctrl:#x}")),
            })
        }

        /// One instruction; Ok(next pc), None: the program ended.
        fn step(&mut self, d: &Decoded) -> Result<Option<u64>, String> {
            let next = d.pc + d.len as u64;
            let f = |n: &str| Self::field(d, n);
            let name = d.inst.name;
            let lanes = self.lanes;
            if name.starts_with("S_CBRANCH") || name == "S_BRANCH" {
                let t = rdna2::branch_target(d).ok_or("branch target")?;
                let taken = match name {
                    "S_BRANCH" => true,
                    "S_CBRANCH_SCC0" => self.scc == 0,
                    "S_CBRANCH_SCC1" => self.scc != 0,
                    "S_CBRANCH_VCCZ" => self.vcc() == 0,
                    "S_CBRANCH_VCCNZ" => self.vcc() != 0,
                    "S_CBRANCH_EXECZ" => self.exec() == 0,
                    "S_CBRANCH_EXECNZ" => self.exec() != 0,
                    _ => return Err(format!("{name}: not translated")),
                };
                return Ok(Some(if taken { t } else { next }));
            }
            if d.variant != Variant::Plain && !(name == "V_MOV_B32" && d.variant == Variant::Dpp16) {
                return Err(format!("{name}: SDWA / DPP variant"));
            }
            // the lanes EXEC enables, as it is before the instruction
            let active: Vec<usize> = (0..lanes).filter(|&l| self.exec() >> l & 1 == 1).collect();
            match name {
                "S_ENDPGM" => return Ok(None),
                "S_NOP" | "S_BARRIER" => {}
                "S_ADD_U32" | "S_SUB_U32" | "S_MUL_I32" | "S_AND_B32" | "S_OR_B32" | "S_XOR_B32" | "S_ANDN2_B32" | "S_LSHL_B32" | "S_LSHR_B32" | "S_CSELECT_B32" => {
                    let (a, b) = (self.scalar(d, f("SSRC0")?)?, self.scalar(d, f("SSRC1")?)?);
                    let (r, scc) = match name {
                        "S_ADD_U32" => (a.wrapping_add(b), Some(a.checked_add(b).is_none())),
                        "S_SUB_U32" => (a.wrapping_sub(b), Some(b > a)),
                        "S_MUL_I32" => (a.wrapping_mul(b), None),
                        "S_AND_B32" => (a & b, Some(a & b != 0)),
                        "S_OR_B32" => (a | b, Some(a | b != 0)),
                        "S_XOR_B32" => (a ^ b, Some(a ^ b != 0)),
                        "S_ANDN2_B32" => (a & !b, Some(a & !b != 0)),
                        "S_LSHL_B32" => (a << (b & 31), Some(a << (b & 31) != 0)),
                        "S_LSHR_B32" => (a >> (b & 31), Some(a >> (b & 31) != 0)),
                        _ => (if self.scc != 0 { a } else { b }, None),
                    };
                    self.set(f("SDST")?, r);
                    if let Some(c) = scc {
                        self.scc = c as u32;
                    }
                }
                "S_AND_B64" | "S_OR_B64" | "S_XOR_B64" | "S_ANDN2_B64" => {
                    let (a, b) = (self.scalar64(d, f("SSRC0")?)?, self.scalar64(d, f("SSRC1")?)?);
                    let r = match name {
                        "S_AND_B64" => a & b,
                        "S_OR_B64" => a | b,
                        "S_XOR_B64" => a ^ b,
                        _ => a & !b,
                    };
                    self.set64(f("SDST")?, r);
                    self.scc = (r != 0) as u32;
                }
                "S_MOV_B32" => {
                    let a = self.scalar(d, f("SSRC0")?)?;
                    self.set(f("SDST")?, a);
                }
                "S_MOV_B64" => {
                    let a = self.scalar64(d, f("SSRC0")?)?;
                    self.set64(f("SDST")?, a);
                }
                "S_MOVK_I32" => {
                    let k = f("SIMM16")? as u16 as i16 as i32 as u32;
                    self.set(f("SDST")?, k);
                }
                "S_CMP_EQ_U32" | "S_CMP_LG_U32" | "S_CMP_GT_U32" | "S_CMP_GE_U32" | "S_CMP_LT_U32" | "S_CMP_LE_U32" | "S_BITCMP0_B32" | "S_BITCMP1_B32" => {
                    let (a, b) = (self.scalar(d, f("SSRC0")?)?, self.scalar(d, f("SSRC1")?)?);
                    self.scc = match name {
                        "S_CMP_EQ_U32" => a == b,
                        "S_CMP_LG_U32" => a != b,
                        "S_CMP_GT_U32" => a > b,
                        "S_CMP_GE_U32" => a >= b,
                        "S_CMP_LT_U32" => a < b,
                        "S_CMP_LE_U32" => a <= b,
                        "S_BITCMP0_B32" => a >> (b & 31) & 1 == 0,
                        _ => a >> (b & 31) & 1 == 1,
                    } as u32;
                }
                "S_BCNT1_I32_B32" | "S_BCNT1_I32_B64" => {
                    let r = if name.ends_with("B64") { self.scalar64(d, f("SSRC0")?)?.count_ones() } else { self.scalar(d, f("SSRC0")?)?.count_ones() };
                    self.set(f("SDST")?, r);
                    self.scc = (r != 0) as u32;
                }
                "S_AND_SAVEEXEC_B32" => {
                    let (a, old) = (self.scalar(d, f("SSRC0")?)?, self.s[126]);
                    self.set(126, a & old);
                    self.scc = (a & old != 0) as u32;
                    self.set(f("SDST")?, old);
                }
                "S_AND_SAVEEXEC_B64" => {
                    let (a, old) = (self.scalar64(d, f("SSRC0")?)?, self.scalar64(d, 126)?);
                    self.set64(126, a & old);
                    self.scc = (a & old != 0) as u32;
                    self.set64(f("SDST")?, old);
                }
                "V_ADD_NC_U32" | "V_CNDMASK_B32" => {
                    let (dst, a, b) = (f("VDST")?, f("SRC0")?, f("VSRC1")?);
                    let vcc = self.vcc();
                    for l in active {
                        let (x, y) = (self.vsrc(d, a, l)?, self.v[l][b as usize]);
                        let r = if name == "V_ADD_NC_U32" {
                            x.wrapping_add(y)
                        } else if vcc >> l & 1 == 1 {
                            y
                        } else {
                            x
                        };
                        self.vset(dst, l, r);
                    }
                }
                "V_MOV_B32" if d.variant == Variant::Dpp16 => {
                    let e = ext_encoding(d).ok_or("DPP")?;
                    let x = |n: &str| rdna2::field_value(e, &d.words, n).unwrap_or(0);
                    if x("BOUND_CTRL") != 1 || x("FI") != 0 {
                        return Err("DPP without bound_ctrl".into());
                    }
                    let (dst, src) = (f("VDST")?, x("VSRC0") as usize);
                    let old: Vec<u32> = (0..lanes).map(|l| self.v[l][src]).collect();
                    for &l in &active {
                        if x("ROW_MASK") >> (l >> 4) & 1 == 0 || x("BANK_MASK") >> ((l >> 2) & 3) & 1 == 0 {
                            continue;
                        }
                        let from = Self::dpp16(x("DPP_CTRL"), l)?.filter(|k| active.contains(k));
                        self.vset(dst, l, from.map_or(0, |k| old[k]));
                    }
                }
                "V_MOV_B32" => {
                    let (dst, a) = (f("VDST")?, f("SRC0")?);
                    for l in active {
                        let x = self.vsrc(d, a, l)?;
                        self.vset(dst, l, x);
                    }
                }
                "V_CMP_GT_U32" | "V_CMP_LT_U32" | "V_CMP_EQ_U32" | "V_CMP_NE_U32" | "V_CMP_GE_U32" | "V_CMP_LE_U32" if d.enc.name == "ENC_VOPC" => {
                    let (a, b) = (f("SRC0")?, f("VSRC1")?);
                    let mut m = 0u64;
                    for l in active {
                        let (x, y) = (self.vsrc(d, a, l)?, self.v[l][b as usize]);
                        let r = match name {
                            "V_CMP_GT_U32" => x > y,
                            "V_CMP_LT_U32" => x < y,
                            "V_CMP_EQ_U32" => x == y,
                            "V_CMP_NE_U32" => x != y,
                            "V_CMP_GE_U32" => x >= y,
                            _ => x <= y,
                        };
                        m |= (r as u64) << l;
                    }
                    self.set(106, m as u32);
                    if lanes == 64 {
                        self.set(107, (m >> 32) as u32);
                    }
                }
                "GLOBAL_STORE_DWORD" => {
                    // s[SADDR:SADDR + 1] + v[ADDR] + OFFSET, the active lanes
                    let (base, addr, data) = (f("SADDR")?, f("ADDR")? as usize, f("DATA")? as usize);
                    if base >= 106 {
                        return Err("global store without a scalar base".into());
                    }
                    for l in active {
                        let at = self.s[base as usize].wrapping_add(self.v[l][addr]).wrapping_add(f("OFFSET")? as u32) as usize / 4;
                        *self.mem.get_mut(at).ok_or("store outside the memory")? = self.v[l][data];
                    }
                }
                "V_READFIRSTLANE_B32" => {
                    let l = if self.exec() == 0 { 0 } else { self.exec().trailing_zeros() as usize };
                    let x = self.vsrc(d, f("SRC0")?, l)?;
                    self.set(f("VDST")?, x);
                }
                "V_READLANE_B32" => {
                    let l = self.scalar(d, f("SRC1")?)? as usize & (lanes - 1);
                    let x = self.vsrc(d, f("SRC0")?, l)?;
                    self.set(f("VDST")?, x);
                }
                _ => return Err(format!("{name}: not in the reference")),
            }
            Ok(Some(next))
        }
    }

    /// Run `code` from `st`. Err: an instruction the reference does not know, or no end after
    /// `limit` instructions.
    pub fn run(code: &[u8], st: &KernelState, mem: &[u32], wave64: bool, limit: usize) -> Result<Machine, String> {
        let mut insts = Vec::new();
        let mut off = 0usize;
        while off + 4 <= code.len() {
            let d = rdna2::decode(&code[off..], off as u64).map_err(|e| format!("{e:?}"))?;
            off += d.len;
            insts.push(d);
        }
        let at: HashMap<u64, usize> = insts.iter().enumerate().map(|(i, d)| (d.pc, i)).collect();
        let mut m = Machine {
            s: st.sgpr.clone(),
            scc: st.scc,
            v: st.vgpr.clone(),
            written_s: BTreeSet::new(),
            written_v: BTreeSet::new(),
            trace: Vec::new(),
            mem: mem.to_vec(),
            lanes: if wave64 { 64 } else { 32 },
        };
        // every instruction once, with all lanes on, for the registers it writes
        let mut all = Machine { s: m.s.clone(), scc: 0, v: m.v.clone(), written_s: BTreeSet::new(), written_v: BTreeSet::new(), trace: Vec::new(), mem: m.mem.clone(), lanes: m.lanes };
        for d in &insts {
            (all.s[126], all.s[127]) = (u32::MAX, u32::MAX);
            let _ = all.step(d);
        }
        (m.written_s, m.written_v) = (all.written_s, all.written_v);
        let mut pc = Some(0u64);
        while let Some(x) = pc {
            // past the last instruction: the end
            let Some(&i) = at.get(&x) else { break };
            if m.trace.len() >= limit {
                return Err(format!("no end after {limit} instructions"));
            }
            m.trace.push(x);
            pc = m.step(&insts[i])?;
        }
        Ok(m)
    }
}

// ---------------------------------------------------------------- the HLSL's control flow

mod skeleton {
    use super::Decider;

    #[derive(Debug)]
    pub enum St {
        /// `// <pc>: <instruction>`
        Mark(u64),
        If(Cond, Vec<St>, Vec<St>),
        For(Vec<St>),
        Do(Vec<St>),
        Break,
        Continue,
        SetBr(u32),
        /// `xcl_skip_<target> = …` (the first structured form)
        SetFlag(String, bool),
        /// `LOOP_GUARD` at the top of a loop
        Guard,
    }

    #[derive(Debug)]
    pub enum Cond {
        /// the condition of the branch last marked (`negated`: `!(…)` of it)
        Branch { negated: bool },
        /// a branch the emitter does not translate: never taken
        Const(bool),
        BrSet,
        BrIs(u32),
        /// one of these skip flags is set
        AnyFlag(Vec<String>),
        /// none of them is
        NoFlag(Vec<String>),
    }

    struct Parser<'a> {
        lines: Vec<&'a str>,
        at: usize,
        wave64: bool,
        /// the condition the emitter gives the conditional branch last seen
        branch: Option<String>,
    }

    /// The body of `main` of a kernel-mode wave32 translation: after the register loads, up
    /// to the stores.
    fn body(hlsl: &str) -> Vec<&str> {
        let lines: Vec<&str> = hlsl.lines().collect();
        let mut start = lines.iter().position(|l| l.trim() == "uint scc = io.Load(500u);").expect("kernel prologue") + 1;
        while lines[start].contains("= io.Load(") {
            start += 1;
        }
        let end = start + lines[start..].iter().position(|l| *l == "    if (xcl_hw_lane == 0u) {").expect("kernel epilogue");
        lines[start..end].iter().map(|l| l.trim()).collect()
    }

    fn go(s: &str) -> St {
        match s {
            "break" => St::Break,
            "continue" => St::Continue,
            _ => panic!("not a jump: {s}"),
        }
    }

    impl<'a> Parser<'a> {
        fn condition(&self, text: &str) -> Cond {
            let base = self.branch.as_deref().unwrap_or_else(|| panic!("`if ({text})` without a branch before it"));
            if base == "false" && (text == "false" || text == "!(false)") {
                Cond::Const(text != "false")
            } else if text == base {
                Cond::Branch { negated: false }
            } else if text == format!("!({base})") {
                Cond::Branch { negated: true }
            } else {
                panic!("`if ({text})`: not the condition of the branch before it ({base})")
            }
        }

        /// Statements up to one of `closers`, which is returned.
        fn block(&mut self, closers: &[&str]) -> (Vec<St>, &'a str) {
            let mut out = Vec::new();
            loop {
                let Some(&l) = self.lines.get(self.at) else {
                    assert!(closers.is_empty(), "the text ends inside a block");
                    return (out, "");
                };
                self.at += 1;
                if closers.contains(&l) {
                    return (out, l);
                }
                if let Some(rest) = l.strip_prefix("// ") {
                    let (pc, text) = rest.split_once(": ").unwrap_or_else(|| panic!("comment: {l}"));
                    out.push(St::Mark(u64::from_str_radix(pc, 16).unwrap_or_else(|_| panic!("comment: {l}"))));
                    let name = text.split(' ').next().unwrap();
                    let wave64 = self.wave64;
                    let zero = |r: &str| if wave64 { format!("({r}_lo | {r}_hi) == 0u") } else { format!("{r}_lo == 0u") };
                    self.branch = match name {
                        "s_cbranch_scc0" => Some("scc == 0u".into()),
                        "s_cbranch_scc1" => Some("scc != 0u".into()),
                        "s_cbranch_vccz" => Some(zero("vcc")),
                        "s_cbranch_vccnz" => Some(format!("!({})", zero("vcc"))),
                        "s_cbranch_execz" => Some(zero("exec")),
                        "s_cbranch_execnz" => Some(format!("!({})", zero("exec"))),
                        // a branch the emitter does not translate: `false`
                        n if n.starts_with("s_cbranch") => Some("false".into()),
                        _ => None,
                    };
                } else if l == "do {" {
                    out.push(St::Do(self.block(&["} while (false);"]).0));
                } else if l == "[loop] for (;;) {" {
                    out.push(St::For(self.block(&["}"]).0));
                } else if l == "if (xcl_budget == 0u) break; xcl_budget--;" {
                    out.push(St::Guard);
                } else if l == "uint xcl_br = 0u;" {
                } else if l.starts_with("bool xcl_skip_") {
                } else if let Some(rest) = l.strip_prefix("xcl_skip_") {
                    let (name, v) = rest.split_once(" = ").unwrap_or_else(|| panic!("{l}"));
                    out.push(St::SetFlag(name.into(), v == "true;"));
                } else if let Some(rest) = l.strip_prefix("if (xcl_skip_") {
                    let name = rest.strip_suffix(") break;").unwrap_or_else(|| panic!("{l}"));
                    out.push(St::If(Cond::AnyFlag(vec![name.into()]), vec![St::Break], vec![]));
                } else if let Some(rest) = l.strip_prefix("if (!(xcl_skip_") {
                    let names = rest.strip_suffix(")) {").unwrap_or_else(|| panic!("{l}"));
                    let names = names.split(" || ").map(|n| n.trim_start_matches("xcl_skip_").to_string()).collect();
                    out.push(St::If(Cond::NoFlag(names), self.block(&["}"]).0, vec![]));
                } else if l == "break;" || l == "continue;" {
                    out.push(go(l.trim_end_matches(';')));
                } else if let Some(rest) = l.strip_prefix("xcl_br = ") {
                    let n = rest.strip_suffix("u; break;").unwrap_or_else(|| panic!("{l}"));
                    out.push(St::SetBr(n.parse().unwrap()));
                    out.push(St::Break);
                } else if l == "if (xcl_br != 0u) break;" {
                    out.push(St::If(Cond::BrSet, vec![St::Break], vec![]));
                } else if let Some(rest) = l.strip_prefix("if (xcl_br != 0u) { xcl_br = 0u; ") {
                    let j = rest.strip_suffix("; }").unwrap_or_else(|| panic!("{l}"));
                    out.push(St::If(Cond::BrSet, vec![St::SetBr(0), go(j)], vec![]));
                } else if let Some(rest) = l.strip_prefix("if (xcl_br != 0u) { if (xcl_br == ") {
                    let (n, rest) = rest.split_once("u) { xcl_br = 0u; ").unwrap_or_else(|| panic!("{l}"));
                    let j = rest.strip_suffix("; } break; }").unwrap_or_else(|| panic!("{l}"));
                    let own = St::If(Cond::BrIs(n.parse().unwrap()), vec![St::SetBr(0), go(j)], vec![]);
                    out.push(St::If(Cond::BrSet, vec![own, St::Break], vec![]));
                } else if l.starts_with("if (") && l.ends_with("; }") && !l.contains("break;") && !l.contains("continue;") {
                    // an instruction's own statement (a store under the lane's EXEC bit)
                } else if let Some(rest) = l.strip_prefix("if (") {
                    if let Some(c) = rest.strip_suffix(") {") {
                        let cond = self.condition(c);
                        let (then, closer) = self.block(&["}", "} else {"]);
                        let other = if closer == "} else {" { self.block(&["}"]).0 } else { vec![] };
                        out.push(St::If(cond, then, other));
                    } else if let Some(c) = rest.strip_suffix(") break;") {
                        out.push(St::If(self.condition(c), vec![St::Break], vec![]));
                    } else if let Some(c) = rest.strip_suffix(") continue;") {
                        out.push(St::If(self.condition(c), vec![St::Continue], vec![]));
                    } else if let Some((c, flag)) = rest.split_once(") xcl_skip_") {
                        let name = flag.strip_suffix(" = true;").unwrap_or_else(|| panic!("{l}"));
                        out.push(St::If(self.condition(c), vec![St::SetFlag(name.into(), true)], vec![]));
                    } else {
                        panic!("an `if` this reader does not know: {l}");
                    }
                } else {
                    // an instruction's statements: no control flow of their own
                    let control = ["}", "do ", "for ", "while ", "[loop]", "else", "break", "continue", "xcl_br", "return", "switch"];
                    assert!(!control.iter().any(|c| l.starts_with(c)), "a control statement this reader does not know: {l}");
                }
            }
        }
    }

    pub fn parse(hlsl: &str, wave64: bool) -> Vec<St> {
        let mut p = Parser { lines: body(hlsl), at: 0, wave64, branch: None };
        p.block(&[]).0
    }

    /// `parse` for another mode: the lines after the first one that contains `after`, up to
    /// the last line that is `end`.
    pub fn parse_between(hlsl: &str, after: &str, end: &str, wave64: bool) -> Vec<St> {
        let lines: Vec<&str> = hlsl.lines().collect();
        let start = lines.iter().position(|l| l.contains(after)).expect("start of the body") + 1;
        let end = lines.iter().rposition(|l| *l == end).expect("end of the body");
        let mut p = Parser { lines: lines[start..end].iter().map(|l| l.trim()).collect(), at: 0, wave64, branch: None };
        p.block(&[]).0
    }

    pub enum Out {
        Fall,
        Break,
        Continue,
    }

    pub struct Run<'a> {
        pub trace: Vec<u64>,
        pub decide: &'a mut Decider,
        pub cap: usize,
        /// the loop budget is spent: every `LOOP_GUARD` leaves its loop
        pub spent: bool,
        last: u64,
        br: u32,
        flags: std::collections::BTreeSet<String>,
    }

    impl<'a> Run<'a> {
        pub fn new(decide: &'a mut Decider, cap: usize) -> Run<'a> {
            Run { trace: Vec::new(), decide, cap, spent: false, last: 0, br: 0, flags: Default::default() }
        }

        /// Err: `cap` instructions ran.
        pub fn run(&mut self, sts: &[St]) -> Result<Out, ()> {
            for s in sts {
                match s {
                    St::Mark(pc) => {
                        if self.trace.len() >= self.cap + 4096 {
                            return Err(());
                        }
                        self.trace.push(*pc);
                        self.last = *pc;
                    }
                    St::If(c, then, other) => {
                        let v = match c {
                            Cond::Branch { negated } => self.decide.at(self.last) != *negated,
                            Cond::Const(v) => *v,
                            Cond::BrSet => self.br != 0,
                            Cond::BrIs(n) => self.br == *n,
                            Cond::AnyFlag(f) => f.iter().any(|x| self.flags.contains(x)),
                            Cond::NoFlag(f) => !f.iter().any(|x| self.flags.contains(x)),
                        };
                        match self.run(if v { then } else { other })? {
                            Out::Fall => {}
                            o => return Ok(o),
                        }
                    }
                    St::For(body) => loop {
                        let before = self.trace.len();
                        if let Out::Break = self.run(body)? {
                            break;
                        }
                        assert!(self.trace.len() > before, "a loop iteration without an instruction");
                    },
                    St::Do(body) => {
                        if let Out::Continue = self.run(body)? {
                            panic!("`continue` leaves a `do {{ }} while (false)`: it would end the block, not continue a loop");
                        }
                    }
                    St::Guard if self.spent => return Ok(Out::Break),
                    St::Guard => {}
                    St::Break => return Ok(Out::Break),
                    St::Continue => return Ok(Out::Continue),
                    St::SetBr(n) => self.br = *n,
                    St::SetFlag(f, true) => {
                        self.flags.insert(f.clone());
                    }
                    St::SetFlag(f, false) => {
                        self.flags.remove(f);
                    }
                }
            }
            Ok(Out::Fall)
        }
    }
}

/// How a program was laid out.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Form {
    /// structured: `structure.rs`, or the first structured form
    Structured,
    /// the `pc` dispatcher (with stretches in the first structured form)
    PcChain,
    /// no branch at all
    Straight,
}

fn form_of(hlsl: &str) -> Form {
    if hlsl.contains("    uint pc = 0;") {
        Form::PcChain
    } else if hlsl.contains("s_cbranch") || hlsl.contains("s_branch") || hlsl.contains("s_setpc") {
        Form::Structured
    } else {
        Form::Straight
    }
}

/// The three ways `translate` can lay a program out.
const FORMS: [&str; 3] = ["pc chain", "default", "general"];

/// `code` laid out as `form`: "pc chain" (`structured: false`: the dispatcher, which still
/// holds stretches in the first structured form), "default" (the first structured form where
/// it applies, else `structure.rs`, else the dispatcher) or "general" (`structure.rs` for
/// every reducible program).
fn translate_form(code: &[u8], opt: &Options, form: &str) -> Result<super::super::Translation, String> {
    match form {
        "pc chain" => translate(code, &Options { structured: false, ..opt.clone() }),
        "default" => translate(code, opt),
        _ => translate_general(code, opt),
    }
}

/// `form` ("default" or "general") of a wave32 kernel program, walked against its graph: Err
/// says where the HLSL runs another instruction than the program. Ok: the layout it got (a
/// `pc` chain is not walked). The general form is held to the exact instruction trace; the
/// default path (the first structured form) to the trace without the `s_endpgm` comments,
/// which it also reaches where it should not – they have no code.
fn walks_like_its_graph(code: &[u8], form: &str) -> Result<Form, String> {
    super::super::structure::CHECK.store(true, std::sync::atomic::Ordering::Relaxed);
    let g = Graph::of(code);
    let t = translate_form(code, &Options { wave64: false, ..Default::default() }, form)?;
    let layout = form_of(&t.hlsl);
    if layout != Form::Structured {
        return Ok(layout);
    }
    let sts = skeleton::parse(&t.hlsl, false);
    const CAP: usize = 3000;
    for run in 0..24u64 {
        let seed = 0xA076_1D64_78BD_642Fu64.wrapping_mul(run + 1) ^ code.len() as u64;
        let bias = [128u64, 40, 216, 12, 244, 96, 160, 128][run as usize % 8];
        let mut want = g.walk(&mut Decider::new(seed, bias), CAP);
        let mut d = Decider::new(seed, bias);
        let mut r = skeleton::Run::new(&mut d, CAP);
        let ended = match r.run(&sts) {
            Ok(skeleton::Out::Fall) => true,
            Ok(_) => return Err(format!("run {run}: a break / continue outside every loop")),
            Err(()) => false,
        };
        if form != "general" {
            want.retain(|pc| !g.ends.contains(pc));
            r.trace.retain(|pc| !g.ends.contains(pc));
        }
        let n = want.len().min(r.trace.len());
        // a walk cut at CAP: the HLSL must have come as far
        let same_length = if want.len() >= CAP { r.trace.len() >= want.len() } else { ended && want.len() == r.trace.len() };
        if want[..n] != r.trace[..n] || !same_length {
            let k = (0..n).find(|&k| want[k] != r.trace[k]).unwrap_or(n);
            let show = |v: &[u64]| v.get(k).map_or("the end".to_string(), |pc| format!("{pc:#06x}"));
            return Err(format!("run {run}, step {k}: the HLSL runs {}, the program {}", show(&r.trace), show(&want)));
        }
    }
    Ok(layout)
}

// ---------------------------------------------------------------- the generator

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        ((self.next() >> 33) % n as u64) as usize
    }

    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }
}

/// What decides a conditional block (true: its first successor).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Test {
    /// bit (step & 31) of s10 + k
    Input(u64),
    /// step < k
    Steps(u64),
    /// bit k of the trace hash
    Hash(u64),
    /// VCC != 0, read as a scalar
    AnyVcc,
    /// `s_and_saveexec`: EXEC & VCC != 0 (EXEC stays narrowed, the old one in s6)
    Narrowed,
    /// SCC as the block left it
    Scc,
    // read by the branch itself; not forced by the step counter
    VccZ,
    VccNz,
    ExecZ,
    ExecNz,
}

impl Test {
    fn raw(self) -> bool {
        matches!(self, Test::VccZ | Test::VccNz | Test::ExecZ | Test::ExecNz)
    }
}

#[derive(Clone, Copy, Debug)]
enum End {
    Exit,
    Jump(usize),
    Cond(usize, usize, Test),
}

/// Per-lane and cross-lane work of a block: what makes the active lanes observable.
#[derive(Clone, Copy, Debug)]
enum Work {
    /// v2 += constant (active lanes)
    Count,
    /// v3 += v0
    LaneSum,
    /// VCC = k > v1
    Mask(u64),
    /// VCC = s15 > v2
    MaskMoving,
    /// s4 = EXEC, EXEC &= VCC
    Save,
    /// EXEC = s<slot> (4 or 6)
    Restore(u64),
    /// EXEC |= s<slot>
    Widen(u64),
    /// EXEC = all lanes
    AllLanes,
    /// EXEC &= ~VCC
    Narrow,
    /// hash += v2 of the first active lane
    FirstLane,
    /// hash += v3 of lane k
    Lane(u64),
    /// v4 = DPP16 v2
    Dpp(u64),
    Barrier,
    /// v5 = VCC ? v2 : constant
    Select,
    /// hash += popcount(VCC)
    CountMask,
    /// memory[s16 + v6] = v2 (active lanes)
    Store,
}

#[derive(Clone, Debug)]
struct Block {
    work: Vec<Work>,
    end: End,
}

#[derive(Clone, Debug)]
struct Program {
    blocks: Vec<Block>,
    /// the order of the blocks in memory (block 0, the entry, first)
    layout: Vec<usize>,
    wave64: bool,
    /// how the terminators are spelled (fall through / `s_branch`, `s_endpgm` / jump to the end)
    spelling: u64,
}

/// Steps after which every scalar branch goes towards the exit.
const STEPS: u64 = 40;

fn ins(name: &str, enc: &str, fields: &[(&str, u64)]) -> Vec<u8> {
    encode(name, enc, fields, None).unwrap_or_else(|e| panic!("{name}: {e}"))
}

impl Program {
    fn successors(&self, b: usize) -> Vec<usize> {
        match self.blocks[b].end {
            End::Exit => vec![],
            End::Jump(t) => vec![t],
            End::Cond(t, f, _) => vec![t, f],
        }
    }

    /// Blocks to the exit (None: it cannot be reached).
    fn distance(&self) -> Vec<Option<usize>> {
        let n = self.blocks.len();
        let mut d: Vec<Option<usize>> = (0..n).map(|b| matches!(self.blocks[b].end, End::Exit).then_some(0)).collect();
        loop {
            let mut changed = false;
            for b in 0..n {
                let best = self.successors(b).into_iter().filter_map(|s| d[s]).min().map(|x| x + 1);
                if let Some(x) = best {
                    if d[b].is_none_or(|old| x < old) {
                        d[b] = Some(x);
                        changed = true;
                    }
                }
            }
            if !changed {
                return d;
            }
        }
    }

    fn reachable(&self) -> Vec<bool> {
        let mut seen = vec![false; self.blocks.len()];
        let mut work = vec![0usize];
        while let Some(b) = work.pop() {
            if !std::mem::replace(&mut seen[b], true) {
                work.extend(self.successors(b));
            }
        }
        seen
    }

    /// Every run ends: each reachable block reaches the exit, and once the step counter is
    /// spent (scalar tests forced towards the exit) no cycle is left. Branches on VCC / EXEC
    /// themselves are not forced: they become scalar tests where they could close a cycle.
    fn make_it_end(&mut self) -> bool {
        let d = self.distance();
        let reach = self.reachable();
        if (0..self.blocks.len()).any(|b| reach[b] && d[b].is_none()) {
            return false;
        }
        let forced = |p: &Program, b: usize| -> Vec<usize> {
            match p.blocks[b].end {
                End::Cond(t, f, test) if !test.raw() => vec![if d[t] <= d[f] && d[t].is_some() || d[f].is_none() { t } else { f }],
                _ => p.successors(b),
            }
        };
        let cyclic = |p: &Program| {
            // colours: 0 new, 1 on the path, 2 done
            fn visit(b: usize, colour: &mut [u8], next: &dyn Fn(usize) -> Vec<usize>) -> bool {
                colour[b] = 1;
                for s in next(b) {
                    if colour[s] == 1 || (colour[s] == 0 && visit(s, colour, next)) {
                        return true;
                    }
                }
                colour[b] = 2;
                false
            }
            let mut colour = vec![0u8; p.blocks.len()];
            (0..p.blocks.len()).any(|b| reach[b] && colour[b] == 0 && visit(b, &mut colour, &|x| forced(p, x)))
        };
        if cyclic(self) {
            for b in &mut self.blocks {
                if let End::Cond(_, _, test) = &mut b.end {
                    if test.raw() {
                        *test = Test::Input(0);
                    }
                }
            }
            assert!(!cyclic(self));
        }
        true
    }

    fn work(&self, w: Work, id: u64, out: &mut Vec<u8>) {
        let b = |n64: &'static str, n32: &'static str| if self.wave64 { n64 } else { n32 };
        let add_to_hash = |r: u64| ins("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", r)]);
        match w {
            Work::Count => out.extend(ins("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 129 + id % 60), ("VSRC1", 2)])),
            Work::LaneSum => out.extend(ins("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V), ("VSRC1", 3)])),
            Work::Mask(k) => out.extend(ins("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 128 + k), ("VSRC1", 1)])),
            Work::MaskMoving => out.extend(ins("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 15), ("VSRC1", 2)])),
            Work::Save => out.extend(ins(b("S_AND_SAVEEXEC_B64", "S_AND_SAVEEXEC_B32"), "ENC_SOP1", &[("SDST", 4), ("SSRC0", 106)])),
            Work::Restore(s) => out.extend(ins(b("S_MOV_B64", "S_MOV_B32"), "ENC_SOP1", &[("SDST", 126), ("SSRC0", s)])),
            Work::Widen(s) => out.extend(ins(b("S_OR_B64", "S_OR_B32"), "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", s)])),
            Work::AllLanes => out.extend(ins(b("S_MOV_B64", "S_MOV_B32"), "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)])),
            Work::Narrow => out.extend(ins(b("S_ANDN2_B64", "S_ANDN2_B32"), "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 106)])),
            Work::FirstLane => {
                out.extend(ins("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", V + 2)]));
                out.extend(add_to_hash(8));
            }
            Work::Lane(k) => {
                let k = if self.wave64 { k % 64 } else { k % 32 };
                out.extend(ins("V_READLANE_B32", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 3), ("SRC1", 128 + k)]));
                out.extend(add_to_hash(8));
            }
            Work::Dpp(ctrl) => out.extend(ins(
                "V_MOV_B32",
                "VOP1_VOP_DPP16",
                &[("VDST", 4), ("SRC0", 0xFA), ("VSRC0", 2), ("DPP_CTRL", ctrl), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF), ("BOUND_CTRL", 1)],
            )),
            Work::Barrier => out.extend(ins("S_BARRIER", "ENC_SOPP", &[])),
            Work::Select => out.extend(ins("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 129 + id % 60), ("VSRC1", 2)])),
            Work::CountMask => {
                out.extend(ins(b("S_BCNT1_I32_B64", "S_BCNT1_I32_B32"), "ENC_SOP1", &[("SDST", 3), ("SSRC0", 106)]));
                out.extend(add_to_hash(3));
            }
            Work::Store => out.extend(ins("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 6), ("DATA", 2), ("SADDR", 16), ("OFFSET", 0)])),
        }
    }

    /// The program's bytes. Registers: s0 trace hash, s1 steps, s2 / s3 / s8 scratch, s4:5 and
    /// s6:7 saved EXEC, s10..s14 inputs, s15 a threshold, s16:17 the memory the stores go to;
    /// v0 lane, v1 input, v2..v5 results, v6 the lane's offset in that memory.
    fn assemble(&self) -> Vec<u8> {
        enum Item {
            Code(Vec<u8>),
            /// branch (mnemonic) to a block (usize::MAX: the final s_endpgm)
            Branch(&'static str, usize),
        }
        const END: usize = usize::MAX;
        let d = self.distance();
        let b = |n64: &'static str, n32: &'static str| if self.wave64 { n64 } else { n32 };
        let mut items: Vec<(Option<usize>, Item)> = Vec::new();
        let mut spelling = self.spelling;
        let mut coin = move || {
            spelling = spelling.rotate_right(1);
            spelling & 1 == 1
        };
        for (p, &x) in self.layout.iter().enumerate() {
            let blk = &self.blocks[x];
            let mut c = Vec::new();
            c.extend(ins("S_ADD_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 129)]));
            c.extend(ins("S_MUL_I32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 128 + 31)]));
            c.extend(ins("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 129 + x as u64)]));
            for &w in &blk.work {
                self.work(w, x as u64, &mut c);
            }
            let next = self.layout.get(p + 1).copied();
            let mut tail = Vec::new();
            match blk.end {
                End::Exit => {
                    if next.is_none() && coin() {
                        // falls into the final s_endpgm
                    } else if coin() {
                        c.extend(ins("S_ENDPGM", "ENC_SOPP", &[]));
                    } else {
                        tail.push(Item::Branch("S_BRANCH", END));
                    }
                }
                End::Jump(t) => {
                    if next != Some(t) || coin() {
                        tail.push(Item::Branch("S_BRANCH", t));
                    }
                }
                End::Cond(t, f, test) => {
                    match test {
                        Test::Input(k) => c.extend(ins("S_BITCMP1_B32", "ENC_SOPC", &[("SSRC0", 10 + k % 5), ("SSRC1", 1)])),
                        Test::Steps(k) => c.extend(ins("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 1), ("SSRC1", 128 + k % 60)])),
                        Test::Hash(k) => c.extend(ins("S_BITCMP1_B32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128 + k % 32)])),
                        Test::AnyVcc => c.extend(ins(b("S_BCNT1_I32_B64", "S_BCNT1_I32_B32"), "ENC_SOP1", &[("SDST", 3), ("SSRC0", 106)])),
                        Test::Narrowed => c.extend(ins(b("S_AND_SAVEEXEC_B64", "S_AND_SAVEEXEC_B32"), "ENC_SOP1", &[("SDST", 6), ("SSRC0", 106)])),
                        Test::Scc | Test::VccZ | Test::VccNz | Test::ExecZ | Test::ExecNz => {}
                    }
                    if !test.raw() {
                        // s2 = spent ? towards the exit : the test
                        let to_t = d[t] <= d[f] && d[t].is_some() || d[f].is_none();
                        c.extend(ins("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 129), ("SSRC1", 128)]));
                        c.extend(ins("S_CMP_GE_U32", "ENC_SOPC", &[("SSRC0", 1), ("SSRC1", 128 + STEPS)]));
                        c.extend(ins("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 128 + to_t as u64), ("SSRC1", 2)]));
                        c.extend(ins("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 2), ("SSRC1", 128)]));
                    }
                    let (yes, no) = match test {
                        Test::VccZ => ("S_CBRANCH_VCCZ", "S_CBRANCH_VCCNZ"),
                        Test::VccNz => ("S_CBRANCH_VCCNZ", "S_CBRANCH_VCCZ"),
                        Test::ExecZ => ("S_CBRANCH_EXECZ", "S_CBRANCH_EXECNZ"),
                        Test::ExecNz => ("S_CBRANCH_EXECNZ", "S_CBRANCH_EXECZ"),
                        _ => ("S_CBRANCH_SCC1", "S_CBRANCH_SCC0"),
                    };
                    if next == Some(f) && coin() {
                        tail.push(Item::Branch(yes, t));
                    } else if next == Some(t) && coin() {
                        tail.push(Item::Branch(no, f));
                    } else {
                        tail.push(Item::Branch(yes, t));
                        tail.push(Item::Branch("S_BRANCH", f));
                    }
                }
            }
            items.push((Some(x), Item::Code(c)));
            items.extend(tail.into_iter().map(|t| (None, t)));
        }
        let mut at = HashMap::new();
        let mut pc = 0usize;
        for (label, item) in &items {
            if let Some(x) = label {
                at.insert(*x, pc);
            }
            pc += match item {
                Item::Code(c) => c.len(),
                Item::Branch(..) => 4,
            };
        }
        at.insert(END, pc);
        let mut code = Vec::new();
        for (_, item) in &items {
            match item {
                Item::Code(c) => code.extend(c),
                Item::Branch(name, to) => {
                    let off = (at[to] as i64 - (code.len() as i64 + 4)) / 4;
                    code.extend(ins(name, "ENC_SOPP", &[("SIMM16", off as i16 as u16 as u64)]));
                }
            }
        }
        code.extend(ins("S_ENDPGM", "ENC_SOPP", &[]));
        code
    }
}

fn random_work(rng: &mut Rng) -> Work {
    match rng.below(19) {
        18 => Work::Store,
        0 | 1 | 2 => Work::Count,
        3 => Work::LaneSum,
        4 | 5 => Work::Mask(1 + rng.below(62) as u64),
        6 => Work::MaskMoving,
        7 => Work::Save,
        8 => Work::Restore(4 + 2 * rng.below(2) as u64),
        9 => Work::Widen(4 + 2 * rng.below(2) as u64),
        10 => Work::AllLanes,
        11 => Work::Narrow,
        12 => Work::FirstLane,
        13 => Work::Lane(rng.below(64) as u64),
        14 => Work::Dpp([0x111, 0xB1, 0x140, 0x1B][rng.below(4)]),
        15 => Work::Barrier,
        16 => Work::Select,
        _ => Work::CountMask,
    }
}

fn random_test(rng: &mut Rng) -> Test {
    match rng.below(20) {
        0..=6 => Test::Input(rng.below(5) as u64),
        7 | 8 => Test::Steps(2 + rng.below(30) as u64),
        9 | 10 => Test::Hash(rng.below(32) as u64),
        11 | 12 => Test::AnyVcc,
        13 | 14 => Test::Narrowed,
        15 => Test::Scc,
        16 => Test::VccZ,
        17 => Test::VccNz,
        18 => Test::ExecZ,
        _ => Test::ExecNz,
    }
}

/// A random program. Three kinds of graph: a forward graph plus jumps back to dominators
/// (reducible: loops with several exits, heads and latches anywhere), a loop nest whose inner
/// blocks jump to every level's head and exit, and a forward graph plus jumps back to any
/// earlier block (often into the middle of a loop: irreducible).
/// Then the blocks are shuffled in memory – the layout the first structured form rejects.
fn generate(seed: u64, wave64: bool) -> Program {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    for _ in 0..8 {
        rng.next();
    }
    loop {
        let kind = rng.below(10);
        let mut ends: Vec<End> = Vec::new();
        if kind < 5 || kind >= 8 {
            let n = 3 + rng.below(8);
            for i in 0..n {
                let later = |rng: &mut Rng| i + 1 + if rng.chance(50) { 0 } else { rng.below(n - i - 1) };
                ends.push(if i + 1 == n || rng.chance(8) {
                    End::Exit
                } else if n - i - 1 < 2 || rng.chance(30) {
                    End::Jump(later(&mut rng))
                } else {
                    let (a, mut c) = (later(&mut rng), later(&mut rng));
                    if a == c {
                        c = if a + 1 < n { a + 1 } else { a - 1 };
                    }
                    End::Cond(a, c, random_test(&mut rng))
                });
            }
            // dominators of the forward graph, then jumps back to them
            let mut dom = vec![(1u64 << n) - 1; n];
            dom[0] = 1;
            for b in 1..n {
                let mut m = (1u64 << n) - 1;
                let mut any = false;
                for p in 0..b {
                    let to = match ends[p] {
                        End::Exit => vec![],
                        End::Jump(t) => vec![t],
                        End::Cond(t, f, _) => vec![t, f],
                    };
                    if to.contains(&b) && (p == 0 || dom[p] != (1u64 << n) - 1) {
                        m &= dom[p];
                        any = true;
                    }
                }
                dom[b] = if any { m | 1 << b } else { (1u64 << n) - 1 };
            }
            for _ in 0..rng.below(5) + if kind >= 8 { 3 } else { 0 } {
                let u = rng.below(n);
                // kind 8, 9: back to any earlier block, also into the middle of a loop (irreducible)
                let heads: Vec<usize> = (0..n).filter(|&v| dom[u] != (1u64 << n) - 1 && (dom[u] >> v & 1 == 1 || (kind >= 8 && v <= u))).collect();
                if heads.is_empty() {
                    continue;
                }
                let v = heads[rng.below(heads.len())];
                ends[u] = match ends[u] {
                    End::Exit => End::Jump(v),
                    End::Jump(t) if rng.chance(50) => End::Cond(v, t, random_test(&mut rng)),
                    End::Jump(t) => End::Cond(t, v, random_test(&mut rng)),
                    End::Cond(t, _, test) if rng.chance(50) => End::Cond(t, v, test),
                    End::Cond(_, f, test) => End::Cond(v, f, test),
                };
            }
        } else if kind < 8 {
            // heads 0..depth, inner blocks, then the exit of each level (inner first), which
            // is also the latch of the level around it
            let depth = 2 + rng.below(3);
            let inner = 1 + rng.below(4);
            let exit = |level: usize| depth + inner + (depth - 1 - level);
            let n = 2 * depth + inner;
            for h in 0..depth {
                let into = h + 1;
                ends.push(if rng.chance(60) { End::Jump(into) } else { End::Cond(into, exit(rng.below(h + 1)), random_test(&mut rng)) });
            }
            for k in 0..inner {
                let on = if k + 1 < inner { depth + k + 1 } else { exit(depth - 1) };
                let any = |rng: &mut Rng| match rng.below(5) {
                    0 | 1 => rng.below(depth),
                    2 | 3 => exit(rng.below(depth)),
                    _ => depth + rng.below(inner),
                };
                ends.push(match rng.below(6) {
                    0 => End::Jump(on),
                    1 | 2 => End::Cond(any(&mut rng), on, random_test(&mut rng)),
                    3 => End::Cond(on, any(&mut rng), random_test(&mut rng)),
                    _ => End::Cond(any(&mut rng), any(&mut rng), random_test(&mut rng)),
                });
            }
            for level in (0..depth).rev() {
                ends.push(if level == 0 {
                    End::Exit
                } else if rng.chance(70) {
                    End::Cond(level - 1, exit(level - 1), random_test(&mut rng))
                } else {
                    End::Cond(exit(rng.below(level)), level - 1, random_test(&mut rng))
                });
            }
            assert_eq!(ends.len(), n);
            for e in &mut ends {
                if let End::Cond(t, f, _) = *e {
                    if t == f {
                        *e = End::Jump(t);
                    }
                }
            }
        }
        let n = ends.len();
        let blocks: Vec<Block> = ends.into_iter().map(|end| Block { work: (0..rng.below(4)).map(|_| random_work(&mut rng)).collect(), end }).collect();
        let mut layout: Vec<usize> = (0..n).collect();
        if rng.chance(70) {
            for i in (2..n).rev() {
                layout.swap(i, 1 + rng.below(i));
            }
        }
        let mut p = Program { blocks, layout, wave64, spelling: rng.next() };
        if p.make_it_end() {
            return p;
        }
    }
}

// ---------------------------------------------------------------- on the GPU

/// An input: the registers the generated programs read.
fn input(case: u64, wave64: bool) -> KernelState {
    let mut rng = Rng(0x1234_5678_9ABC_DEF1u64.wrapping_mul(case + 1) | 1);
    let mut st = KernelState::default();
    for r in 10..15 {
        st.sgpr[r] = rng.next() as u32;
    }
    st.sgpr[15] = rng.below(48) as u32;
    st.sgpr[16] = STORES;
    for r in 4..8 {
        st.sgpr[r] = u32::MAX;
    }
    // every other input starts with some lanes off
    if case % 2 == 1 {
        st.sgpr[126] = rng.next() as u32 | 1 << rng.below(32);
        st.sgpr[127] = rng.next() as u32;
    }
    if !wave64 {
        st.sgpr[127] = 0;
        st.sgpr[5] = 0;
        st.sgpr[7] = 0;
    }
    for l in 0..64 {
        st.vgpr[l][0] = l as u32;
        st.vgpr[l][1] = rng.below(64) as u32;
        st.vgpr[l][2] = (l as u32 * 3 + case as u32) % 50;
        st.vgpr[l][6] = 4 * l as u32;
    }
    st
}

/// Where the generated programs store (a dword per lane), and the kernel's memory in dwords.
const STORES: u32 = 0x40;
const MEMORY: usize = 128;

/// The first register in which a GPU result differs from the reference run (the SGPRs and
/// VGPRs the program wrote, SCC), or the first dword of memory.
fn difference(want: &cpu::Machine, got: &runner::KernelResult, lanes: usize) -> Option<String> {
    if let Some(&r) = want.written_s.iter().find(|&&r| want.s[r] != got.sgpr[r]) {
        return Some(format!("s{r} = {:#x}, expected {:#x}", got.sgpr[r], want.s[r]));
    }
    if want.scc != got.scc {
        return Some(format!("scc = {}, expected {}", got.scc, want.scc));
    }
    for l in 0..lanes {
        if let Some(&r) = want.written_v.iter().find(|&&r| want.v[l][r] != got.vgpr[l][r]) {
            return Some(format!("lane {l}: v{r} = {:#x}, expected {:#x}", got.vgpr[l][r], want.v[l][r]));
        }
    }
    let at = (0..want.mem.len()).find(|&i| want.mem[i] != got.mem[i])?;
    Some(format!("memory +{:#x} = {:#x}, expected {:#x}", 4 * at, got.mem[at], want.mem[at]))
}

/// `code` in one of `FORMS`, compiled.
fn compile_form(code: &[u8], wave64: bool, form: &str) -> Result<Vec<u8>, String> {
    let t = translate_form(code, &on_gpu(Options { wave64, ..Default::default() }), form).map_err(|e| format!("{form}: {e}"))?;
    if !t.complete() {
        return Err(format!("{form}: {:?}", t.unsupported_why));
    }
    super::super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|e| format!("{form}: {e}"))
}

/// Run `code` on the GPU in each of `forms` with `cases` inputs and compare with the reference
/// run on the CPU: one line per form that differs (its first difference). Err: a form that
/// does not compile or run, or a program the reference cannot run.
fn against_the_reference(code: &[u8], wave64: bool, cases: &[KernelState], forms: &[&'static str]) -> Result<Vec<(&'static str, String)>, String> {
    let w = if wave64 { 64 } else { 32 };
    let mem = vec![0u32; MEMORY];
    let want: Vec<cpu::Machine> = cases.iter().map(|st| cpu::run(code, st, &mem, wave64, 100_000)).collect::<Result<_, _>>()?;
    let mut wrong = Vec::new();
    for &form in forms {
        let dxil = compile_form(code, wave64, form)?;
        for (k, st) in cases.iter().enumerate() {
            let got = runner::run(&dxil, st, &mem).map_err(|e| format!("{form}, input {k}: {e}"))?;
            if let Some(d) = difference(&want[k], &got, w) {
                wrong.push((form, format!("{form}, wave{w}, input {k}: {d}")));
                break;
            }
        }
    }
    Ok(wrong)
}

fn hex(code: &[u8]) -> String {
    code.chunks(4).map(|c| format!("{:08x}", u32::from_le_bytes(c.try_into().unwrap()))).collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------- random programs

/// `XCL_STRUCTURE_FUZZ=<n>`: how many random programs the two tests below take (default: a
/// short run). `XCL_STRUCTURE_FUZZ_SEED=<n>`: the first seed.
fn fuzz_range(default: u64) -> std::ops::Range<u64> {
    let n = std::env::var("XCL_STRUCTURE_FUZZ").ok().and_then(|v| v.parse().ok()).unwrap_or(default);
    let first = std::env::var("XCL_STRUCTURE_FUZZ_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    first..first + n
}

/// Random programs: a reducible one gets the general form and an irreducible one the `pc`
/// chain (never the reverse), and the general form's HLSL runs the program's instructions in
/// the program's order for the same branch decisions. The first structured form is walked
/// too where the default path uses it, and only counted: see
/// `first_form_runs_code_a_taken_jump_skips`.
#[test]
fn random_programs_structure_like_their_graph() {
    let (mut general, mut chain, mut first, mut first_wrong) = (0, 0, 0, Vec::new());
    for seed in fuzz_range(1000) {
        let p = generate(seed, false);
        let code = p.assemble();
        let g = Graph::of(&code);
        let form = walks_like_its_graph(&code, "general").unwrap_or_else(|e| panic!("seed {seed}: {e}\n{p:?}\n{}", hex(&code)));
        match form {
            Form::Structured => {
                assert!(g.reducible(), "seed {seed}: an irreducible program in the general form\n{p:?}");
                general += 1;
            }
            Form::PcChain => {
                assert!(!g.reducible(), "seed {seed}: a reducible program left on the pc chain\n{p:?}\n{}", hex(&code));
                chain += 1;
            }
            Form::Straight => {}
        }
        // the default path: the first structured form where `structurable` accepts the program
        let opt = Options { wave64: false, ..Default::default() };
        if form == Form::Structured && translate(&code, &opt).unwrap().hlsl != translate_general(&code, &opt).unwrap().hlsl {
            first += 1;
            if let Err(e) = walks_like_its_graph(&code, "default") {
                first_wrong.push(format!("seed {seed}: {e}"));
            }
        }
    }
    println!("general form: {general} programs walk like their graph; pc chain (irreducible): {chain}");
    println!("first structured form (default path): {} of {first} programs do not, e.g. {:?}", first_wrong.len(), first_wrong.first());
    assert!(general > chain);
}

/// Random programs on the GPU, as wave32 programs and as wave64 ones (split where the GPU has
/// 32 lanes): the general form computes what the reference run does. The other two forms are
/// only counted: where they differ, the first structured form is at fault (see
/// `first_form_runs_code_a_taken_jump_skips`), not `structure.rs`.
#[test]
fn gpu_random_programs_match_the_reference() {
    if !gpu_ready() {
        return;
    }
    let (mut ran, mut blocks) = (0, 0u64);
    let mut others: Vec<String> = Vec::new();
    for seed in fuzz_range(24) {
        for wave64 in [false, true] {
            let p = generate(seed, wave64);
            let code = p.assemble();
            let cases: Vec<KernelState> = (0..3).map(|k| input(seed * 3 + k, wave64)).collect();
            for st in &cases {
                // the step counter (s1): every run ends as the generator promises
                let steps = cpu::run(&code, st, &[0; MEMORY], wave64, 100_000).unwrap_or_else(|e| panic!("seed {seed}: {e}\n{p:?}")).s[1];
                assert!(steps <= STEPS as u32 + 2 * p.blocks.len() as u32, "seed {seed}: {steps} steps");
                blocks += steps as u64;
            }
            let wrong = against_the_reference(&code, wave64, &cases, &FORMS).unwrap_or_else(|e| panic!("seed {seed}: {e}\n{p:?}\n{}", hex(&code)));
            // an irreducible program: its "general" translation is the pc chain again
            let structured = Graph::of(&code).reducible();
            for (form, what) in wrong {
                assert!(form != "general" || !structured, "seed {seed}: {what}\n{p:?}\n{}", hex(&code));
                others.push(format!("seed {seed}: {what}"));
            }
            ran += 1;
        }
    }
    println!("{ran} programs ({blocks} blocks run): the general form matched the reference in all");
    println!("{} differences in the other forms (the first structured form):", others.len());
    for o in others.iter().take(12) {
        println!("  {o}");
    }
}

// ---------------------------------------------------------------- hand-picked programs

fn cmp_lt(reg: u64, k: u64) -> Asm<'static> {
    Asm::I("S_CMP_LT_U32", "ENC_SOPC", vec![("SSRC0", reg), ("SSRC1", 128 + k)])
}

fn end() -> Asm<'static> {
    Asm::I("S_ENDPGM", "ENC_SOPP", vec![])
}

/// `s_endpgm` inside two loops and a jump out of both to the program's last instruction.
fn ends_inside_loops_program() -> Vec<u8> {
    use Asm::*;
    assemble(&[
        L("outer"),
        s_add(0, 0, 129),
        L("inner"),
        s_add(1, 1, 129),
        s_bit(10, 1),
        Br("S_CBRANCH_SCC1", "on"),
        s_add(2, 2, 129),
        end(),
        L("on"),
        s_bit(11, 1),
        Br("S_CBRANCH_SCC1", "end"),
        cmp_lt(1, 6),
        Br("S_CBRANCH_SCC1", "inner"),
        s_add(3, 3, 129),
        cmp_lt(0, 3),
        Br("S_CBRANCH_SCC1", "outer"),
        s_add(4, 4, 129),
        L("end"),
    ])
}

/// Blocks no path reaches: a loop, a loop entered at two points (irreducible on its own), and
/// one that jumps into the live code.
fn unreachable_blocks_program() -> Vec<u8> {
    use Asm::*;
    assemble(&[
        s_add(0, 0, 129),
        Br("S_BRANCH", "live"),
        L("dead"),
        s_add(5, 5, 129),
        cmp_lt(5, 3),
        Br("S_CBRANCH_SCC1", "dead"),
        s_bit(10, 0),
        Br("S_CBRANCH_SCC1", "dx"),
        L("dy"),
        s_add(6, 6, 129),
        L("dx"),
        s_add(7, 7, 129),
        s_bit(11, 7),
        Br("S_CBRANCH_SCC1", "dy"),
        Br("S_BRANCH", "live"),
        L("live"),
        s_add(1, 1, 129),
        cmp_lt(1, 4),
        Br("S_CBRANCH_SCC1", "live"),
    ])
}

/// The entry block is a loop head with two latches; behind the loop a block that loops to
/// itself and is reached from both latches' ways out.
fn loop_heads_program() -> Vec<u8> {
    use Asm::*;
    assemble(&[
        L("top"),
        s_add(0, 0, 129),
        s_bit(10, 0),
        Br("S_CBRANCH_SCC1", "second"),
        cmp_lt(0, 5),
        Br("S_CBRANCH_SCC1", "top"),
        Br("S_BRANCH", "self"),
        L("second"),
        s_add(1, 1, 129),
        cmp_lt(0, 7),
        Br("S_CBRANCH_SCC1", "top"),
        L("self"),
        s_add(2, 2, 129),
        cmp_lt(2, 4),
        Br("S_CBRANCH_SCC1", "self"),
    ])
}

/// A conditional branch to the next instruction, and one whose two ways meet at once.
fn trivial_branches_program() -> Vec<u8> {
    use Asm::*;
    assemble(&[
        s_add(0, 0, 129),
        s_bit(10, 0),
        Br("S_CBRANCH_SCC1", "next"),
        L("next"),
        s_add(1, 1, 129),
        s_bit(11, 0),
        Br("S_CBRANCH_SCC1", "join"),
        Br("S_BRANCH", "join"),
        L("join"),
        s_add(2, 2, 129),
    ])
}

/// A conditional branch the emitter does not translate (never taken, in every form) out of
/// a loop.
fn untranslated_branch_program() -> Vec<u8> {
    use Asm::*;
    assemble(&[
        L("loop"),
        s_add(0, 0, 129),
        Br("S_CBRANCH_CDBGSYS", "out"),
        s_add(1, 1, 129),
        s_bit(10, 0),
        Br("S_CBRANCH_SCC1", "out"),
        cmp_lt(0, 4),
        Br("S_CBRANCH_SCC1", "loop"),
        s_add(3, 3, 129),
        L("out"),
        s_add(2, 2, 129),
    ])
}

/// An indirect jump (`s_setpc_b64` that is no far jump) in a loop with two exits.
fn indirect_jump_program() -> Vec<u8> {
    use Asm::*;
    assemble(&[
        L("loop"),
        s_add(0, 0, 129),
        s_bit(10, 0),
        Br("S_CBRANCH_SCC1", "out"),
        I("S_SETPC_B64", "ENC_SOP1", vec![("SSRC0", 20)]),
        s_add(1, 1, 129),
        s_bit(11, 0),
        Br("S_CBRANCH_SCC1", "out"),
        s_add(3, 3, 129),
        s_bit(12, 0),
        Br("S_CBRANCH_SCC1", "latch"),
        s_add(4, 4, 129),
        L("latch"),
        cmp_lt(0, 4),
        Br("S_CBRANCH_SCC1", "loop"),
        L("out"),
        s_add(2, 2, 129),
    ])
}

/// A loop closed by a far jump (`s_getpc_b64` … `s_setpc_b64`), left from two places.
fn far_jump_program() -> Vec<u8> {
    let one = |n: &str, e: &str, f: &[(&str, u64)], lit: Option<u32>| encode(n, e, f, lit).unwrap_or_else(|x| panic!("{n}: {x}"));
    let mut c = Vec::new();
    c.extend(one("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 129)], None)); // 0x00
    c.extend(one("S_BITCMP1_B32", "ENC_SOPC", &[("SSRC0", 10), ("SSRC1", 0)], None)); // 0x04
    c.extend(one("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 8)], None)); // 0x08 → 0x2c
    c.extend(one("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128 + 5)], None)); // 0x0c
    c.extend(one("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 5)], None)); // 0x10 → 0x28
    c.extend(one("S_GETPC_B64", "ENC_SOP1", &[("SDST", 20)], None)); // 0x14, the far jump's base: 0x18
    c.extend(one("S_SUB_U32", "ENC_SOP2", &[("SDST", 20), ("SSRC0", 20), ("SSRC1", 255)], Some(0x18))); // 0x18
    c.extend(one("S_SUBB_U32", "ENC_SOP2", &[("SDST", 21), ("SSRC0", 21), ("SSRC1", 128)], None)); // 0x20
    c.extend(one("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 20)], None)); // 0x24 → 0
    c.extend(one("S_ADD_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 129)], None)); // 0x28
    c.extend(one("S_ADD_U32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 2), ("SSRC1", 129)], None)); // 0x2c
    c.extend(one("S_ENDPGM", "ENC_SOPP", &[], None));
    c
}

/// `depth` forward branches, each over the next one's whole region (as many nested blocks).
fn nested_skips_program(depth: usize) -> Vec<u8> {
    let names: Vec<String> = (0..depth).map(|k| format!("j{k}")).collect();
    let mut items = Vec::new();
    for name in &names {
        items.push(s_add(0, 0, 129));
        items.push(s_bit(10, 0));
        items.push(Asm::Br("S_CBRANCH_SCC1", name));
    }
    for name in names.iter().rev() {
        items.push(Asm::L(name));
        items.push(s_add(1, 1, 0));
    }
    assemble(&items)
}

/// A loop with two exits whose body splits the wave by EXEC, as the console compiler does
/// (`s_and_saveexec`, `s_cbranch_execz` over each side, EXEC restored at the join), with a
/// lane read, a barrier and DPP inside the sides.
fn masked_sides_program(wave64: bool) -> Vec<u8> {
    use Asm::*;
    let b = |n64: &'static str, n32: &'static str| if wave64 { n64 } else { n32 };
    assemble(&[
        L("loop"),
        s_add(0, 0, 129),
        I("V_CMP_GT_U32", "ENC_VOPC", vec![("SRC0", 0), ("VSRC1", 1)]),
        I(b("S_AND_SAVEEXEC_B64", "S_AND_SAVEEXEC_B32"), "ENC_SOP1", vec![("SDST", 4), ("SSRC0", 106)]),
        Br("S_CBRANCH_EXECZ", "other"),
        I("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 2), ("SRC0", 129), ("VSRC1", 2)]),
        I("V_READFIRSTLANE_B32", "ENC_VOP1", vec![("VDST", 8), ("SRC0", V + 2)]),
        s_add(9, 9, 8),
        I("S_BARRIER", "ENC_SOPP", vec![]),
        I("V_MOV_B32", "VOP1_VOP_DPP16", vec![("VDST", 4), ("SRC0", 0xFA), ("VSRC0", 2), ("DPP_CTRL", 0x111), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF), ("BOUND_CTRL", 1)]),
        L("other"),
        I(b("S_ANDN2_B64", "S_ANDN2_B32"), "ENC_SOP2", vec![("SDST", 126), ("SSRC0", 4), ("SSRC1", 126)]),
        Br("S_CBRANCH_EXECZ", "join"),
        I("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 3), ("SRC0", 130), ("VSRC1", 3)]),
        I("V_READLANE_B32", "ENC_VOP3", vec![("VDST", 8), ("SRC0", V + 3), ("SRC1", 128 + 5)]),
        s_add(9, 9, 8),
        I("V_CMP_GT_U32", "ENC_VOPC", vec![("SRC0", 128 + 20), ("VSRC1", 3)]),
        Br("S_CBRANCH_VCCZ", "out"),
        L("join"),
        I(b("S_MOV_B64", "S_MOV_B32"), "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 4)]),
        s_bit(10, 0),
        Br("S_CBRANCH_SCC1", "out"),
        cmp_lt(0, 40),
        Br("S_CBRANCH_SCC1", "loop"),
        L("out"),
        I(b("S_MOV_B64", "S_MOV_B32"), "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 4)]),
        s_add(1, 1, 129),
    ])
}

/// The hand-picked programs with the form each must get.
fn hand_picked(wave64: bool) -> Vec<(&'static str, Vec<u8>, Form)> {
    vec![
        ("bottom entries", bottom_entries_program(), Form::Structured),
        ("nested exits", nested_exits_program(), Form::Structured),
        ("loop with if", loop_with_if_program(), Form::Structured),
        ("two entries", two_entry_loop_program(), Form::PcChain),
        ("ends inside loops", ends_inside_loops_program(), Form::Structured),
        ("unreachable blocks", unreachable_blocks_program(), Form::Structured),
        ("loop heads", loop_heads_program(), Form::Structured),
        ("trivial branches", trivial_branches_program(), Form::Structured),
        ("two breaks", two_breaks_program(), Form::Structured),
        ("masked sides", masked_sides_program(wave64), Form::Structured),
        ("nested skips, 40", nested_skips_program(40), Form::Structured),
    ]
}

/// Hand-picked shapes: each gets the form it must (the general one unless irreducible), and
/// the general form's HLSL walks like the program.
#[test]
fn hand_picked_programs_structure_like_their_graph() {
    let mut programs = hand_picked(false);
    programs.push(("untranslated branch", untranslated_branch_program(), Form::Structured));
    programs.push(("indirect jump", indirect_jump_program(), Form::Structured));
    programs.push(("far jump", far_jump_program(), Form::Structured));
    programs.push(("nested skips, 150", nested_skips_program(150), Form::Structured));
    for (name, code, want) in programs {
        let form = walks_like_its_graph(&code, "general").unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(form, want, "{name}");
        assert_eq!(Graph::of(&code).reducible(), want == Form::Structured, "{name}");
    }
}

/// A branch to the end of the code (no instruction there) is refused, whatever the form.
#[test]
fn a_branch_past_the_last_instruction_is_refused_in_every_form() {
    let mut code = ins("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 129)]);
    code.extend(ins("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)]));
    code.extend(ins("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0)]));
    for form in FORMS {
        let e = translate_form(&code, &Options::default(), form).err().unwrap_or_else(|| panic!("{form}: translated"));
        assert!(e.contains("not the start of an instruction"), "{form}: {e}");
    }
}

/// An instruction no form translates is counted alike by all three: the dry run of
/// `structure.rs` and its pass over the unreachable blocks do not count twice.
#[test]
fn untranslated_instructions_count_alike_in_every_form() {
    for (name, code) in [("untranslated branch", untranslated_branch_program()), ("indirect jump", indirect_jump_program()), ("unreachable blocks", unreachable_blocks_program())] {
        let opt = Options { wave64: false, ..Default::default() };
        let [pc, default, general] = FORMS.map(|f| translate_form(&code, &opt, f).unwrap());
        assert_eq!(form_of(&general.hlsl), Form::Structured, "{name}");
        for t in [&default, &general] {
            assert_eq!((&t.unsupported, &t.ignored, t.instructions), (&pc.unsupported, &pc.ignored, pc.instructions), "{name}");
        }
    }
}

/// The nesting the general form accepts (`structure::MAX_DEPTH`) compiles: 150 blocks inside
/// each other; beyond it the program stays on the `pc` chain.
#[test]
fn deep_nesting_compiles_or_falls_back() {
    let opt = Options { wave64: false, ..Default::default() };
    let deep = translate_general(&nested_skips_program(150), &opt).unwrap();
    assert_eq!(form_of(&deep.hlsl), Form::Structured);
    if super::super::dxc::location().is_ok() {
        super::super::dxc::compile(&deep.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("150 nested blocks: {e}"));
    }
    let deeper = translate_general(&nested_skips_program(200), &opt).unwrap();
    assert_eq!(form_of(&deeper.hlsl), Form::PcChain);
}

/// The hand-picked programs on the GPU: the general form computes what the reference run
/// does, as wave32 and wave64 programs.
#[test]
fn gpu_hand_picked_programs_match_the_reference() {
    if !gpu_ready() {
        return;
    }
    for wave64 in [false, true] {
        for (name, code, _) in hand_picked(wave64) {
            let mut cases: Vec<KernelState> = (0..4).map(|k| input(1000 + k, wave64)).collect();
            for st in &mut cases {
                // these programs count in s4..s7
                st.sgpr[4..8].fill(0);
            }
            let wrong = against_the_reference(&code, wave64, &cases, &["general"]).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(wrong.is_empty(), "{name}: {}", wrong[0].1);
        }
    }
}

// ---------------------------------------------------------------- counter-examples

/// A loop left by two conditional jumps to the same exit, with more code (an `if`) between
/// the second one and the latch.
fn two_breaks_program() -> Vec<u8> {
    use Asm::*;
    assemble(&[
        L("loop"),
        s_add(0, 0, 129),
        s_bit(10, 0),
        Br("S_CBRANCH_SCC1", "out"),
        s_add(1, 1, 129),
        s_bit(11, 0),
        Br("S_CBRANCH_SCC1", "out"),
        s_add(2, 2, 129),
        s_bit(12, 0),
        Br("S_CBRANCH_SCC1", "latch"),
        s_add(4, 4, 129),
        L("latch"),
        cmp_lt(0, 3),
        Br("S_CBRANCH_SCC1", "loop"),
        L("out"),
        s_add(3, 3, 129),
    ])
}

/// An if / else whose if branch jumps to the start of the else branch.
fn jump_into_else_program() -> Vec<u8> {
    use Asm::*;
    assemble(&[
        s_add(0, 0, 129),
        s_bit(10, 0),
        Br("S_CBRANCH_SCC1", "else"),
        s_add(1, 1, 129),
        s_bit(11, 0),
        Br("S_CBRANCH_SCC1", "else"),
        s_add(2, 2, 129),
        Br("S_BRANCH", "join"),
        L("else"),
        s_add(3, 3, 129),
        L("join"),
        s_add(4, 4, 129),
    ])
}

/// NOT #172: the first structured form (`emit_structured`, on `main` too). A jump to a target
/// that already has its skip flag pending at that level (here: the loop's second `break`) does
/// not close the open `if (!(xcl_skip_…))`, so the items up to the end of the level still run
/// after the jump was taken. Expected to fail until that form is fixed: drop `should_panic`
/// with the fix.
#[test]
#[should_panic(expected = "the HLSL runs 0x0018, the program 0x0030")]
fn first_form_runs_code_a_taken_jump_skips() {
    walks_like_its_graph(&two_breaks_program(), "default").unwrap();
}

/// NOT #172: the first structured form again. A conditional jump from the if branch to the
/// start of the else branch is taken for a skip to the end of the if branch: the else branch
/// does not run. Expected to fail until that form is fixed: drop `should_panic` with the fix.
#[test]
#[should_panic(expected = "the HLSL runs 0x0024, the program 0x0020")]
fn first_form_skips_the_else_branch_it_jumps_into() {
    walks_like_its_graph(&jump_into_else_program(), "default").unwrap();
}

/// The two programs above on the GPU: the general form computes what the program does; the
/// `pc` chain (`structured: false`, which keeps the first structured form inside its `pc`
/// guards) and the default path do not. So the `pc` chain is no reference for a differential
/// test of `structure.rs`: `gpu_general_structure_matches_the_pc_chain` compares with it.
#[test]
fn gpu_first_form_counter_examples_are_right_in_the_general_form() {
    if !gpu_ready() {
        return;
    }
    for (name, code) in [("two breaks", two_breaks_program()), ("jump into else", jump_into_else_program())] {
        // the first exit / the if not taken, the second taken
        let mut st = KernelState::default();
        st.sgpr[11] = 2;
        let wrong = against_the_reference(&code, false, std::slice::from_ref(&st), &FORMS).unwrap_or_else(|e| panic!("{name}: {e}"));
        for (form, what) in &wrong {
            assert!(*form != "general", "{name}: {what}");
            println!("{name}: {what}");
        }
    }
}

/// A wave64 loop whose exit block (before it in memory) reads the whole VCC and branches on
/// it into a block with a mask of its own; inside the loop, the mask the exit branches on and
/// two masks that only select lane by lane.
fn selector_masks_loop_program() -> Vec<u8> {
    use Asm::*;
    assemble(&[
        Br("S_BRANCH", "head"),
        L("exit"),
        I("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 2), ("SSRC0", 106)]),
        Br("S_CBRANCH_EXECZ", "done"),
        I("V_CMP_GT_U32", "ENC_VOPC", vec![("SRC0", 128 + 9), ("VSRC1", 0)]),
        I("S_MOV_B64", "ENC_SOP1", vec![("SDST", 6), ("SSRC0", 106)]),
        L("done"),
        end(),
        L("head"),
        I("V_CMP_GT_U32", "ENC_VOPC", vec![("SRC0", 128 + 4), ("VSRC1", 0)]),
        I("V_CNDMASK_B32", "ENC_VOP2", vec![("VDST", 1), ("SRC0", 128), ("VSRC1", 0)]),
        s_add(4, 4, 129),
        I("V_CMP_LT_U32", "ENC_VOPC", vec![("SRC0", 4), ("VSRC1", 0)]),
        Br("S_CBRANCH_VCCZ", "exit"),
        I("V_CMP_GT_U32", "ENC_VOPC", vec![("SRC0", 128 + 8), ("VSRC1", 0)]),
        I("V_CNDMASK_B32", "ENC_VOP2", vec![("VDST", 2), ("SRC0", 128), ("VSRC1", 0)]),
        Br("S_BRANCH", "head"),
    ])
}

/// #172, split wave64 (compute, 32-lane GPU): `x64_masks` leaves a mask in its half where no
/// path reads it whole, and takes a loop's `LOOP_GUARD` exit to go to the code after the loop
/// in memory, or to the program's end as on the `pc` chain. In the general form the guard's
/// `break` lands on the block placed after the `for` – here the exit block, which sits before
/// the loop in memory and reads the whole VCC while it holds the half-local mask of the
/// loop's last instructions. The halves then get a different EXEC, take the `s_cbranch_execz`
/// differently and no longer reach the same exchanges (group barriers). Only after a guard
/// trip, and not run on the GPU on purpose: a pair out of step is what the guard is there to
/// prevent. Expected to fail until a guard trip ends the program in this form too (or the
/// liveness follows it): drop `should_panic` with the fix.
#[test]
#[should_panic(expected = "half-local")]
fn split_wave64_halves_stay_in_step_after_a_loop_guard_trip() {
    let code = selector_masks_loop_program();
    let cs = super::super::CsInterface { threads: [64, 1, 1], user_sgprs: 6, ..Default::default() };
    let t = translate(&code, &Options { mode: Mode::Compute(cs), wave64: true, split_wave64: true, ..Default::default() }).unwrap();
    assert_eq!(form_of(&t.hlsl), Form::Structured);
    let local = t.hlsl.matches(" = xcl_x_ballot_local(").count();
    // the HLSL with the budget spent from the start: what runs after the loop's guard leaves it
    let mut d = Decider::new(1, 128);
    let mut r = skeleton::Run::new(&mut d, 100);
    r.spent = true;
    let _ = r.run(&skeleton::parse_between(&t.hlsl, "{ uint2 e0 = xcl_x_ballot(true);", "}", true));
    // 0x0004: s_and_saveexec_b64 s[2:3], vcc
    assert!(local == 0 || !r.trace.contains(&4), "after a guard trip the HLSL runs {:#x?}: the whole VCC is read while {local} masks of the loop are half-local", r.trace);
}

/// A guard trip itself (the same layout, counting up to s15): with 100 000 iterations to go,
/// every form stops at the budget; the `pc` chain ends the program there, the general form
/// goes on with the block after the loop (s3), as the first structured form does. Printed, not
/// asserted: which of the two a trip should do is the open point of the test above.
#[test]
fn gpu_loop_guard_trip_in_each_form() {
    if !gpu_ready() {
        return;
    }
    use Asm::*;
    let code = assemble(&[
        Br("S_BRANCH", "head"),
        L("exit"),
        s_add(3, 3, 129),
        end(),
        L("head"),
        s_add(0, 0, 129),
        I("S_CMP_GE_U32", "ENC_SOPC", vec![("SSRC0", 0), ("SSRC1", 15)]),
        Br("S_CBRANCH_SCC1", "exit"),
        Br("S_BRANCH", "head"),
    ]);
    for (bound, iterations) in [(100u32, 100), (100_000, 65_536)] {
        let mut st = KernelState::default();
        st.sgpr[15] = bound;
        for form in FORMS {
            let r = runner::run(&compile_form(&code, false, form).unwrap(), &st, &[]).unwrap();
            println!("{bound} iterations, {form}: s0 = {}, s3 = {}", r.sgpr[0], r.sgpr[3]);
            assert_eq!(r.sgpr[0], iterations, "{form}, {bound} iterations");
        }
    }
}
