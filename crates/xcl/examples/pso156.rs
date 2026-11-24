//! Issue #156 bench: how long the driver takes to build compute pipelines from synthetic
//! translated console programs, alone and with several creator threads, and what the creator
//! threads are doing meanwhile. Synthetic programs only (no game files).
//!
//! ```text
//! cargo run --release --example pso156 -- gen --out DIR --name NAME [shape options] --seeds A..B
//! cargo run --release --example pso156 -- run --threads N [--sample-ms 3000] [--cap-gb 20] FILE.dxil...
//! ```
//! `gen` assembles RDNA2 programs, translates them with XCL's recompiler (compute mode, 128
//! threads a group, `wave64` + `split_wave64` = the NVIDIA form) and compiles them with DXC
//! (`dxc::compile`, so the same -O1 / -O0 thresholds as the runtime unless `--dxc-arg -O..`).
//! Every seed gives different literals, so the driver has never seen the program.
//! `run` creates one compute pipeline per file with the XCL ABI root signature from N threads,
//! and reports wall / CPU time, peak private bytes and (with `--sample-ms`) what the threads
//! that are inside nvgpucomp64 are doing, from non-invasive cdb stack samples.

use std::ffi::c_void;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use windows::core::{Interface, PCWSTR};
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::{dxc, translate, CsInterface, Mode, Options};

const V: u64 = 256;

fn e(name: &str, enc: &str, f: &[(&str, u64)], lit: Option<u32>) -> Vec<u8> {
    encode(name, enc, f, lit).unwrap_or_else(|x| panic!("{name}: {x}"))
}

#[derive(Clone, Debug)]
struct Shape {
    rounds: u32,
    windows: u32,
    wave64: bool,
    split: bool,
    /// a load + store every `mem_every` rounds (0: none)
    mem_every: u32,
    /// a divergent if-block (with a load in it when `mem_every` > 0) every `branch_every` rounds
    branch_every: u32,
    /// extra VALU triples per round
    valu: u32,
}

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0 | 1
    }
}

/// The program: v[6:7] = s0 + tid * 4 : s1 (the high word, and so the memory window, is not
/// known when the program is compiled), then `rounds` rounds of an LCG step with fresh literals,
/// memory traffic and divergent blocks.
fn program(s: &Shape, seed: u32, pixel: bool) -> Vec<u8> {
    let mut v = Vec::new();
    let (save, or) = if s.wave64 { ("S_AND_SAVEEXEC_B64", "S_OR_B64") } else { ("S_AND_SAVEEXEC_B32", "S_OR_B32") };
    let mut rng = Lcg(seed.wrapping_mul(2_654_435_761) ^ 0x9E37_79B9);
    v.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 0)], None));
    v.extend(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 0), ("VSRC1", 6)], None));
    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 1)], None));
    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)], None));
    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 255)], Some(seed)));
    let glbl = |op: &str, f: &[(&str, u64)]| e(op, "ENC_FLAT_GLBL", f, None);
    for r in 0..s.rounds {
        v.extend(e("V_MUL_LO_U32", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 2), ("SRC1", 255)], Some(rng.next())));
        v.extend(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 255), ("VSRC1", 2)], Some(rng.next())));
        for _ in 0..s.valu {
            v.extend(e("V_XOR_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 2), ("VSRC1", 3)], None));
            v.extend(e("V_LSHRREV_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 141), ("VSRC1", 3)], None));
            v.extend(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 5), ("VSRC1", 3)], None));
        }
        let off = |k: u32| ((r * 4 + k * 4) % 2048) as u64;
        if s.mem_every > 0 && r % s.mem_every == 0 {
            v.extend(glbl("GLOBAL_LOAD_DWORD", &[("VDST", 4), ("ADDR", 6), ("SADDR", 0x7D), ("OFFSET", off(0))]));
            v.extend(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 4), ("VSRC1", 3)], None));
            v.extend(glbl("GLOBAL_STORE_DWORD", &[("ADDR", 6), ("DATA", 3), ("SADDR", 0x7D), ("OFFSET", off(1))]));
        }
        if s.branch_every > 0 && r % s.branch_every == s.branch_every - 1 {
            let mut body = e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 255), ("VSRC1", 3)], Some(rng.next()));
            if s.mem_every > 0 {
                body.extend(glbl("GLOBAL_LOAD_DWORD", &[("VDST", 8), ("ADDR", 6), ("SADDR", 0x7D), ("OFFSET", off(2))]));
                body.extend(e("V_XOR_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 8), ("VSRC1", 3)], None));
            }
            v.extend(e("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 2)], Some(rng.next())));
            v.extend(e(save, "ENC_SOP1", &[("SDST", 20), ("SSRC0", 106)], None));
            v.extend(e("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", (body.len() / 4) as u64)], None));
            v.extend(body);
            v.extend(e(or, "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 20)], None));
        }
    }
    v.extend(glbl("GLOBAL_STORE_DWORD", &[("ADDR", 6), ("DATA", 3), ("SADDR", 0x7D), ("OFFSET", 0)]));
    if pixel {
        v.extend(e("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 3), ("VSRC1", 2), ("VSRC2", 3), ("VSRC3", 2), ("DONE", 1), ("VM", 1)], None));
    }
    v.extend(e("S_ENDPGM", "ENC_SOPP", &[], None));
    v
}

fn el(name: &str, index: u32, sv: u32, ct: u32, reg: u32, mask: u8) -> xcl::gpu::dxbc::SigElement {
    xcl::gpu::dxbc::SigElement { name: name.into(), index, system_value: sv, component_type: ct, register: reg, mask, rw_mask: mask, stream: 0 }
}

/// The interfaces of the recompiler's own VS + PS draw test: POSITION / COLOR in, pos0 +
/// param0 out; the PS reads COLOR and writes SV_Target0.
fn psi() -> xcl::gpu::recomp::PsInterface {
    xcl::gpu::recomp::PsInterface {
        inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x302,
        attrs: None,
    }
}

fn vsi() -> xcl::gpu::recomp::VsInterface {
    xcl::gpu::recomp::VsInterface {
        inputs: vec![el("POSITION", 0, 0, 3, 0, 3), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF)],
        attr_base: 4,
        params: None,
    }
}

/// A pass-through console VS (position and COLOR), as DXIL.
fn pass_vs(split: bool) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None));
    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None));
    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None));
    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None));
    v.extend(e("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1)], None));
    v.extend(e("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None));
    v.extend(e("S_ENDPGM", "ENC_SOPP", &[], None));
    let t = translate(&v, &Options { mode: Mode::Vertex(vsi()), wave64: true, split_wave64: split, ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    dxc::compile(&t.hlsl, "main", "vs_6_6", &[]).unwrap()
}

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}
fn num(args: &[String], name: &str, d: u64) -> u64 {
    arg(args, name).map_or(d, |v| v.parse().unwrap_or_else(|_| panic!("{name} {v}")))
}
fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}
fn all<'a>(args: &'a [String], name: &str) -> Vec<&'a str> {
    args.windows(2).filter(|w| w[0] == name).map(|w| w[1].as_str()).collect()
}

fn gen(args: &[String]) {
    let out = PathBuf::from(arg(args, "--out").expect("--out DIR"));
    std::fs::create_dir_all(&out).unwrap();
    let name = arg(args, "--name").expect("--name");
    let s = Shape {
        rounds: num(args, "--rounds", 1000) as u32,
        windows: num(args, "--windows", 4) as u32,
        wave64: num(args, "--wave64", 1) == 1,
        split: num(args, "--split", 1) == 1,
        mem_every: num(args, "--mem-every", 1) as u32,
        branch_every: num(args, "--branch-every", 4) as u32,
        valu: num(args, "--valu", 0) as u32,
    };
    let seeds = arg(args, "--seeds").unwrap_or("1..2");
    let (a, b) = seeds.split_once("..").map(|(a, b)| (a.parse::<u32>().unwrap(), b.parse::<u32>().unwrap())).unwrap();
    let extra: Vec<String> = all(args, "--dxc-arg").iter().map(|s| s.to_string()).collect();
    let subs: Vec<(String, String)> = all(args, "--hlsl-sub").iter().map(|s| { let (f, t) = s.split_once("=>").expect("from=>to"); (f.to_string(), t.to_string()) }).collect();
    let save_hlsl = flag(args, "--save-hlsl");
    let pixel = arg(args, "--stage") == Some("ps");
    if pixel {
        std::fs::write(out.join("pass_vs.dxil.vs"), pass_vs(s.split)).unwrap();
    }
    let jobs = num(args, "--jobs", 4) as usize;
    let next = AtomicUsize::new(a as usize);
    let meta = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(out.join("meta.tsv")).unwrap());
    std::thread::scope(|sc| {
        for _ in 0..jobs {
            sc.spawn(|| loop {
                let seed = next.fetch_add(1, Ordering::Relaxed) as u32;
                if seed >= b {
                    return;
                }
                let code = program(&s, seed, pixel);
                let cs = CsInterface { threads: [128, 1, 1], user_sgprs: 2, tgid: [true, false, false], tg_size: false, tid_components: 1 };
                let mode = if pixel { Mode::Pixel(psi()) } else { Mode::Compute(cs) };
                let opt = Options { mode, wave64: s.wave64, split_wave64: s.split && s.wave64, mem_windows: s.windows, ..Default::default() };
                let t = translate(&code, &opt).unwrap();
                assert!(t.complete(), "{:?}", t.unsupported_why);
                let mut hlsl = t.hlsl.clone();
                for (f, to) in &subs {
                    let n = hlsl.matches(f.as_str()).count();
                    assert!(n > 0, "--hlsl-sub {f:?}: not in the HLSL");
                    hlsl = hlsl.replace(f.as_str(), to);
                }
                let file = out.join(format!("{name}_s{seed}.dxil"));
                if save_hlsl {
                    std::fs::write(file.with_extension("hlsl"), &hlsl).unwrap();
                }
                let ex: Vec<&str> = extra.iter().map(|s| s.as_str()).collect();
                let t0 = Instant::now();
                let dxil = match dxc::compile(&hlsl, "main", if pixel { "ps_6_6" } else { "cs_6_6" }, &ex) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("{name} seed {seed}: {}", &e[..e.len().min(2000)]);
                        continue;
                    }
                };
                let dt = t0.elapsed().as_secs_f64();
                std::fs::write(&file, &dxil).unwrap();
                let line = format!(
                    "{name}\t{seed}\t{}\t{}\t{}\t{:.2}\t{:?}\t{}\n",
                    code.len() / 4,
                    hlsl.len(),
                    dxil.len(),
                    dt,
                    s,
                    extra.join(" ")
                );
                print!("{line}");
                meta.lock().unwrap().write_all(line.as_bytes()).unwrap();
            });
        }
    });
}

// ---- run ----

struct Shared {
    dev: ID3D12Device,
    rs: ID3D12RootSignature,
    grs: ID3D12RootSignature,
    vs: Option<Vec<u8>>,
}

/// A graphics pipeline: the pass-through VS and `ps` (R32G32B32A32_FLOAT target, triangles).
unsafe fn graphics(sh: &Shared, ps: &[u8]) -> windows::core::Result<ID3D12PipelineState> {
    use windows::Win32::Graphics::Dxgi::Common::*;
    let vs = sh.vs.as_ref().unwrap();
    let names = [c"POSITION", c"COLOR"];
    let layout = [
        D3D12_INPUT_ELEMENT_DESC { SemanticName: windows::core::PCSTR(names[0].as_ptr() as _), SemanticIndex: 0, Format: DXGI_FORMAT_R32G32_FLOAT, InputSlot: 0, AlignedByteOffset: 0, InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, InstanceDataStepRate: 0 },
        D3D12_INPUT_ELEMENT_DESC { SemanticName: windows::core::PCSTR(names[1].as_ptr() as _), SemanticIndex: 0, Format: DXGI_FORMAT_R32G32B32A32_FLOAT, InputSlot: 0, AlignedByteOffset: 8, InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, InstanceDataStepRate: 0 },
    ];
    let mut blend = D3D12_BLEND_DESC::default();
    blend.RenderTarget[0].RenderTargetWriteMask = 0xF;
    let mut rtv = [DXGI_FORMAT_UNKNOWN; 8];
    rtv[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::ManuallyDrop::new(Some(sh.grs.clone())),
        VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr() as *const c_void, BytecodeLength: vs.len() },
        PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr() as *const c_void, BytecodeLength: ps.len() },
        BlendState: blend,
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() },
        InputLayout: D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: layout.as_ptr(), NumElements: 2 },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        RTVFormats: rtv,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    let r = sh.dev.CreateGraphicsPipelineState(&desc);
    drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
    r
}
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

fn ft(f: windows_sys::Win32::Foundation::FILETIME) -> f64 {
    ((f.dwHighDateTime as u64) << 32 | f.dwLowDateTime as u64) as f64 * 1e-7
}

/// (user + kernel) CPU seconds of the process
fn process_cpu() -> f64 {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    unsafe {
        let mut a = std::mem::zeroed();
        let mut b = std::mem::zeroed();
        let mut k = std::mem::zeroed();
        let mut u = std::mem::zeroed();
        GetProcessTimes(GetCurrentProcess(), &mut a, &mut b, &mut k, &mut u);
        ft(k) + ft(u)
    }
}

fn thread_cpu() -> f64 {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
    unsafe {
        let mut a = std::mem::zeroed();
        let mut b = std::mem::zeroed();
        let mut k = std::mem::zeroed();
        let mut u = std::mem::zeroed();
        GetThreadTimes(GetCurrentThread(), &mut a, &mut b, &mut k, &mut u);
        ft(k) + ft(u)
    }
}

/// (private bytes, peak commit of the process)
fn memory() -> (u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    unsafe {
        let mut c: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
        c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
        K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c as *mut _ as *mut PROCESS_MEMORY_COUNTERS, c.cb);
        (c.PrivateUsage as u64, c.PeakPagefileUsage as u64)
    }
}

fn avail_commit() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    unsafe {
        let mut m: MEMORYSTATUSEX = std::mem::zeroed();
        m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        GlobalMemoryStatusEx(&mut m);
        m.ullAvailPageFile
    }
}

/// Puts the process in a job whose commit limit is `gb`: an allocation past it fails (in the
/// driver: E_OUTOFMEMORY) instead of taking the machine's commit.
fn cap_commit(gb: u64) {
    use windows_sys::Win32::System::JobObjects::*;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        info.ProcessMemoryLimit = (gb << 30) as usize;
        let ok = SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info as *const _ as *const c_void, std::mem::size_of_val(&info) as u32);
        let ok2 = AssignProcessToJobObject(job, GetCurrentProcess());
        if ok == 0 || ok2 == 0 {
            eprintln!("commit cap: could not set ({ok}, {ok2})");
        }
    }
}

/// The process heap's kind from its signature at +0x10 (NT heap 0xFFEEFFEE, segment heap
/// 0xDDEEDDEE).
fn heap_kind() -> &'static str {
    unsafe {
        let h = windows_sys::Win32::System::Memory::GetProcessHeap() as *const u32;
        match *h.add(4) {
            0xFFEE_FFEE => "NT heap",
            0xDDEE_DDEE => "segment heap",
            _ => "unknown heap",
        }
    }
}


/// (logical processor, efficiency class) of every CPU set (P-cores have the higher class).
fn cpu_classes() -> Vec<(u8, u8)> {
    use windows_sys::Win32::System::SystemInformation::{GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION};
    unsafe {
        let mut len = 0u32;
        GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut len, std::ptr::null_mut(), 0);
        let mut buf = vec![0u8; len as usize];
        if GetSystemCpuSetInformation(buf.as_mut_ptr() as *mut SYSTEM_CPU_SET_INFORMATION, len, &mut len, std::ptr::null_mut(), 0) == 0 {
            return Vec::new();
        }
        let mut v = Vec::new();
        let mut o = 0usize;
        while o < len as usize {
            let e = &*(buf.as_ptr().add(o) as *const SYSTEM_CPU_SET_INFORMATION);
            let c = e.Anonymous.CpuSet;
            v.push((c.LogicalProcessorIndex, c.EfficiencyClass));
            o += e.Size as usize;
        }
        v
    }
}

/// Opt the process out of power throttling (EcoQoS): Windows 11 runs a process without a
/// foreground window on the efficient cores of a hybrid CPU.
fn no_throttle() {
    use windows_sys::Win32::System::Threading::*;
    unsafe {
        let st = PROCESS_POWER_THROTTLING_STATE { Version: 1, ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED, StateMask: 0 };
        let ok = SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &st as *const _ as *const c_void, std::mem::size_of_val(&st) as u32);
        println!("power throttling off: {}", ok != 0);
    }
}

/// Only the performance cores (the highest efficiency class).
fn pcores_only() {
    use windows_sys::Win32::System::Threading::*;
    let c = cpu_classes();
    let top = c.iter().map(|x| x.1).max().unwrap_or(0);
    let mask: usize = c.iter().filter(|x| x.1 == top).fold(0, |m, x| m | 1usize << x.0);
    let ok = unsafe { SetProcessAffinityMask(GetCurrentProcess(), mask) };
    println!("affinity: performance cores only ({mask:#x}): {}", ok != 0);
}

fn cpu_now_is_efficient(classes: &[(u8, u8)]) -> bool {
    let n = unsafe { windows_sys::Win32::System::Threading::GetCurrentProcessorNumber() } as u8;
    let top = classes.iter().map(|x| x.1).max().unwrap_or(0);
    classes.iter().any(|x| x.0 == n && x.1 < top)
}

/// The process's threads by CPU time (name or start module), the busiest first.
fn thread_report(min_s: f64) {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Diagnostics::ToolHelp::*;
    use windows_sys::Win32::System::Threading::*;
    #[link(name = "ntdll")]
    extern "system" {
        fn NtQueryInformationThread(h: *mut c_void, class: u32, info: *mut c_void, len: u32, ret: *mut u32) -> i32;
    }
    let mut out = Vec::new();
    unsafe {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        let mut te: THREADENTRY32 = std::mem::zeroed();
        te.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
        let pid = std::process::id();
        let mut ok = Thread32First(snap, &mut te);
        while ok != 0 {
            if te.th32OwnerProcessID == pid {
                let h = OpenThread(THREAD_QUERY_INFORMATION, 0, te.th32ThreadID);
                if !h.is_null() {
                    let (mut a, mut b, mut k, mut u) = std::mem::zeroed();
                    GetThreadTimes(h, &mut a, &mut b, &mut k, &mut u);
                    let cpu = ft(k) + ft(u);
                    let mut name: *mut u16 = std::ptr::null_mut();
                    let mut n = String::new();
                    if GetThreadDescription(h, &mut name) >= 0 && !name.is_null() {
                        let len = (0..).take_while(|&i| *name.add(i) != 0).count();
                        n = String::from_utf16_lossy(std::slice::from_raw_parts(name, len));
                        windows_sys::Win32::Foundation::LocalFree(name as _);
                    }
                    let mut start: usize = 0;
                    NtQueryInformationThread(h, 9, &mut start as *mut _ as *mut c_void, 8, std::ptr::null_mut());
                    let mut module = std::ptr::null_mut();
                    let mut file = [0u16; 260];
                    use windows_sys::Win32::System::LibraryLoader::*;
                    let m = if GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, start as *const u16, &mut module) != 0 {
                        let l = GetModuleFileNameW(module, file.as_mut_ptr(), 260) as usize;
                        String::from_utf16_lossy(&file[..l]).rsplit('\\').next().unwrap_or("").to_string()
                    } else {
                        "?".into()
                    };
                    if cpu >= min_s {
                        out.push((cpu, te.th32ThreadID, n, m));
                    }
                    CloseHandle(h);
                }
            }
            ok = Thread32Next(snap, &mut te);
        }
        CloseHandle(snap);
    }
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (cpu, id, n, m) in out {
        println!("  thread {id} {n:?} (starts in {m}): CPU {cpu:.1} s");
    }
}

fn device() -> (ID3D12Device, String) {
    unsafe {
        let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let sel = xcl::gpu::adapter::Selector::from_env_or("auto");
        let (a, desc) = xcl::gpu::adapter::candidates(&factory, &sel).into_iter().next().expect("no adapter");
        let dev = xcl::gpu::agility::create_device(&a, D3D_FEATURE_LEVEL_12_0).unwrap();
        (dev, xcl::gpu::adapter::name(&desc))
    }
}

#[derive(Default, Clone, Copy)]
struct Counts {
    samples: u64,
    in_driver: u64,
    heap_wait: u64,
    heap_run: u64,
    driver_run: u64,
    other_wait: u64,
    other: u64,
    /// thread-samples of threads not ours ("D3D Background Thread N", the driver's own) in nvgpucomp64
    background: u64,
}

/// Classify one thread's `kc` stack (frames top first): only threads with nvgpucomp64 on it.
fn classify(name: &str, frames: &[&str], c: &mut Counts) {
    if !frames.iter().any(|f| f.to_ascii_lowercase().contains("nvgpucomp64")) {
        return;
    }
    if !name.contains("pso-creator") {
        c.background += 1;
        return;
    }
    c.in_driver += 1;
    let is_heap = |f: &str| f.contains("Heap") || f.contains("RtlpHp") || f.contains("RtlpLowFrag");
    let is_wait = |f: &str| f.contains("WaitForAlertByThreadId") || f.contains("WaitOnCriticalSection") || f.contains("EnterCriticalSectionContended") || f.contains("WaitOnAddress") || f.contains("AcquireSRWLock") && f.contains("Contended") || f.contains("NtWaitFor") || f.contains("ZwWaitFor") || f.contains("NtDelayExecution");
    let top = frames.first().copied().unwrap_or("");
    // frames above the first nvgpucomp64 frame: what the driver called
    let above: Vec<&str> = frames.iter().copied().take_while(|f| !f.to_ascii_lowercase().contains("nvgpucomp64")).collect();
    let waiting = above.iter().any(|f| is_wait(f));
    let heap = above.iter().any(|f| is_heap(f));
    if waiting && heap {
        c.heap_wait += 1;
    } else if heap {
        c.heap_run += 1;
    } else if top.to_ascii_lowercase().contains("nvgpucomp64") {
        c.driver_run += 1;
    } else if waiting {
        c.other_wait += 1;
    } else {
        c.other += 1;
    }
}

fn parse_sample(text: &str, c: &mut Counts) {
    c.samples += 1;
    let mut frames: Vec<&str> = Vec::new();
    let mut in_stack = false;
    let mut name = "";
    for line in text.lines() {
        let l = line.trim();
        if l.contains(" Id: ") && l.contains("Teb:") {
            if in_stack {
                classify(name, &frames, c);
            }
            name = l.split_once('"').map_or("", |(_, r)| r.trim_end_matches('"'));
            frames.clear();
            in_stack = true;
            continue;
        }
        if l.starts_with("Call Site") || l.starts_with("#") || l.is_empty() {
            continue;
        }
        if in_stack && (l.contains('!') || l.contains("+0x") || l.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_')) {
            // `kn`-style lines: "00 ntdll!NtWaitForAlertByThreadId+0x14": drop the frame number
            let f = l.split_once(' ').filter(|(n, _)| n.chars().all(|ch| ch.is_ascii_hexdigit())).map_or(l, |(_, r)| r.trim());
            frames.push(f);
        }
    }
    if in_stack {
        classify(name, &frames, c);
    }
}

fn run(args: &[String]) {
    let cap = num(args, "--cap-gb", 20);
    if cap > 0 {
        cap_commit(cap);
    }
    if flag(args, "--no-throttle") {
        no_throttle();
    }
    if flag(args, "--pcores") {
        pcores_only();
    }
    // `--ws-max-mb M`: a hard working-set limit, so this process pages (memory pressure on the
    // process alone, as in a title that has most of the RAM committed)
    if let Some(mb) = arg(args, "--ws-max-mb") {
        use windows_sys::Win32::System::Memory::SetProcessWorkingSetSizeEx;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let max: usize = mb.parse::<usize>().unwrap() << 20;
        let ok = unsafe { SetProcessWorkingSetSizeEx(GetCurrentProcess(), 64 << 20, max, 0x4 /* QUOTA_LIMITS_HARDWS_MAX_ENABLE */) };
        println!("hard working-set limit {mb} MB: {}", ok != 0);
    }
    let classes = cpu_classes();
    let ecore_hits = AtomicUsize::new(0);
    let threads = num(args, "--threads", 1) as usize;
    let sample_ms = num(args, "--sample-ms", 0);
    let tag = arg(args, "--tag").unwrap_or("run").to_string();
    let logdir = PathBuf::from(arg(args, "--logdir").unwrap_or("."));
    std::fs::create_dir_all(&logdir).unwrap();
    let files: Vec<PathBuf> = args
        .iter()
        .filter(|a| a.ends_with(".dxil"))
        .map(PathBuf::from)
        .collect();
    let blobs: Vec<(String, Vec<u8>)> = files.iter().map(|f| (f.file_name().unwrap().to_string_lossy().into_owned(), std::fs::read(f).unwrap())).collect();
    let (dev, name) = device();
    // `--bg-mode N`: ID3D12Device6::SetBackgroundProcessingMode (2 = DISABLE_BACKGROUND_WORK)
    if let Some(m) = arg(args, "--bg-mode") {
        let d6: ID3D12Device6 = dev.cast().unwrap();
        let r = unsafe { d6.SetBackgroundProcessingMode(D3D12_BACKGROUND_PROCESSING_MODE(m.parse().unwrap()), D3D12_MEASUREMENTS_ACTION_KEEP_ALL, None, None) };
        println!("SetBackgroundProcessingMode({m}): {r:?}");
    }
    let rs = xcl::gpu::recomp::pso::abi_root_signature(&dev, false).unwrap();
    let vs: Option<Vec<u8>> = arg(args, "--vs").map(|p| std::fs::read(p).unwrap());
    let grs = xcl::gpu::recomp::pso::abi_root_signature(&dev, true).unwrap();
    let lib_in = arg(args, "--library-in").map(PathBuf::from);
    let lib_out = arg(args, "--library-out").map(PathBuf::from);
    let lib_blob = lib_in.as_ref().map(|p| std::fs::read(p).unwrap());
    let library: Option<ID3D12PipelineLibrary> = (lib_in.is_some() || lib_out.is_some()).then(|| unsafe {
        let d1: ID3D12Device1 = dev.cast().unwrap();
        match &lib_blob {
            Some(b) => d1.CreatePipelineLibrary(b).unwrap(),
            None => d1.CreatePipelineLibrary(&[]).unwrap(),
        }
    });
    struct Lib(Option<ID3D12PipelineLibrary>);
    unsafe impl Send for Lib {}
    unsafe impl Sync for Lib {}
    let library = Lib(library);
    let shared = Shared { dev, rs, grs, vs };
    let kb: f64 = blobs.iter().map(|b| b.1.len()).sum::<usize>() as f64 / 1024.0;
    println!(
        "{tag}: {name}, {}, pid {}, {} programs ({:.0} KB DXIL), {threads} creator thread(s), commit cap {cap} GB",
        heap_kind(),
        std::process::id(),
        blobs.len(),
        kb
    );
    let next = AtomicUsize::new(0);
    let done = AtomicBool::new(false);
    let peak_private = AtomicU64::new(0);
    let min_avail = AtomicU64::new(u64::MAX);
    let recs: Mutex<Vec<(usize, f64, f64, f64, i32)>> = Mutex::new(Vec::new());
    let psos: Mutex<Vec<ID3D12PipelineState>> = Mutex::new(Vec::new());
    let counts = Mutex::new(Counts::default());
    let wall_end = Mutex::new((0.0f64, 0.0f64));
    let dxc_done = AtomicUsize::new(0);
    let dxc_load = num(args, "--dxc-load", 0) as usize;
    let dxc_hlsl = arg(args, "--dxc-hlsl").map(|p| std::fs::read_to_string(p).unwrap());
    let cpu0 = process_cpu();
    let t0 = Instant::now();
    std::thread::scope(|sc| {
        sc.spawn(|| {
            while !done.load(Ordering::Acquire) {
                let (p, _) = memory();
                peak_private.fetch_max(p, Ordering::Relaxed);
                min_avail.fetch_min(avail_commit(), Ordering::Relaxed);
                if avail_commit() < 3 << 30 {
                    eprintln!("{tag}: system commit below 3 GB free – aborting to protect the machine");
                    std::process::exit(3);
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        });
        if sample_ms > 0 {
            let cdb = arg(args, "--cdb").unwrap_or(r"C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\cdb.exe").to_string();
            let sym = arg(args, "--sym").unwrap_or(r"srv*D:\ClaudeResourcesProjects\symcache*https://msdl.microsoft.com/download/symbols").to_string();
            let (counts, done, tag, logdir) = (&counts, &done, &tag, &logdir);
            sc.spawn(move || {
                let mut k = 0;
                let start = Instant::now();
                while !done.load(Ordering::Acquire) {
                    let slept = Instant::now();
                    while slept.elapsed() < Duration::from_millis(sample_ms) && !done.load(Ordering::Acquire) {
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    if done.load(Ordering::Acquire) {
                        break;
                    }
                    let out = std::process::Command::new(&cdb)
                        .args(["-pv", "-p", &std::process::id().to_string(), "-y", &sym, "-c", "~*kc 48;qd"])
                        .output();
                    let Ok(out) = out else { eprintln!("cdb failed to start"); return };
                    let text = String::from_utf8_lossy(&out.stdout).into_owned();
                    let _ = std::fs::write(logdir.join(format!("{tag}_stack{k:03}_{:.0}s.txt", start.elapsed().as_secs_f64())), &text);
                    let mut c = counts.lock().unwrap();
                    parse_sample(&text, &mut c);
                    k += 1;
                }
            });
        }
        // DXC compiles in the same process meanwhile (XCL's xcl-dxc workers share the process heap
        // with the driver's compiler): `--dxc-load K --dxc-hlsl FILE`
        for _ in 0..dxc_load {
            let (done, dxc_hlsl, dxc_done) = (&done, &dxc_hlsl, &dxc_done);
            sc.spawn(move || {
                while !done.load(Ordering::Acquire) {
                    dxc::compile(dxc_hlsl.as_ref().expect("--dxc-hlsl"), "main", "cs_6_6", &[]).unwrap();
                    dxc_done.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
        let mut workers = Vec::new();
        for t in 0..threads {
            let (shared, next, recs, psos, blobs, library, lib_blob, classes, ecore_hits) = (&shared, &next, &recs, &psos, &blobs, &library, &lib_blob, &classes, &ecore_hits);
            workers.push(std::thread::Builder::new().name(format!("pso-creator-{t}")).spawn_scoped(sc, move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= blobs.len() {
                    return;
                }
                let (fname, dxil) = &blobs[i];
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(shared.rs.clone())),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const c_void, BytecodeLength: dxil.len() },
                    ..Default::default()
                };
                let wname: Vec<u16> = fname.encode_utf16().chain([0]).collect();
                let c0 = thread_cpu();
                let s = t0.elapsed().as_secs_f64();
                let r: windows::core::Result<ID3D12PipelineState> = unsafe {
                    match (&library.0, lib_blob.is_some()) {
                        (Some(l), true) => l.LoadComputePipeline(PCWSTR(wname.as_ptr()), &desc),
                        _ if shared.vs.is_some() => graphics(shared, dxil),
                        _ => shared.dev.CreateComputePipelineState(&desc),
                    }
                };
                let end = t0.elapsed().as_secs_f64();
                let cpu = thread_cpu() - c0;
                let on_e = cpu_now_is_efficient(classes);
                if on_e {
                    ecore_hits.fetch_add(1, Ordering::Relaxed);
                }
                drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
                let hr = r.as_ref().err().map_or(0, |e| e.code().0);
                println!("  {fname}: {:.0} KB, {:.2} s (thread CPU {:.2} s, ended on {}-core){}", dxil.len() as f64 / 1024.0, end - s, cpu, if on_e { "E" } else { "P" }, if hr != 0 { format!(" FAILED {hr:#010x}") } else { String::new() });
                recs.lock().unwrap().push((dxil.len(), s, end, cpu, hr));
                if let Ok(p) = r {
                    if let (Some(l), false) = (&library.0, lib_blob.is_some()) {
                        unsafe { l.StorePipeline(PCWSTR(wname.as_ptr()), &p).unwrap() };
                    }
                    psos.lock().unwrap().push(p);
                }
            }).unwrap());
        }
        for w in workers {
            w.join().unwrap();
        }
        *wall_end.lock().unwrap() = (t0.elapsed().as_secs_f64(), process_cpu());
        done.store(true, Ordering::Release);
    });
    let (wall, cpu1) = *wall_end.lock().unwrap();
    let cpu = cpu1 - cpu0;
    let recs = recs.into_inner().unwrap();
    let sum: f64 = recs.iter().map(|r| r.2 - r.1).sum();
    let failed = recs.iter().filter(|r| r.4 != 0).count();
    let mut each: Vec<f64> = recs.iter().map(|r| r.2 - r.1).collect();
    each.sort_by(f64::total_cmp);
    let (_, peak_commit) = memory();
    println!(
        "{tag}: SUMMARY threads={threads} programs={} kb={:.0} wall={:.2}s sum={:.2}s concurrency={:.2} each_med={:.2}s each_max={:.2}s cpu={:.2}s cpu/wall={:.2} peak_private={:.2}GiB peak_commit={:.2}GiB min_sys_avail={:.1}GiB failed={failed} kb_per_s={:.1} heap={}",
        recs.len(),
        kb,
        wall,
        sum,
        sum / wall,
        each.get(each.len() / 2).copied().unwrap_or(0.0),
        each.last().copied().unwrap_or(0.0),
        cpu,
        cpu / wall,
        peak_private.load(Ordering::Relaxed) as f64 / (1u64 << 30) as f64,
        peak_commit as f64 / (1u64 << 30) as f64,
        min_avail.load(Ordering::Relaxed) as f64 / (1u64 << 30) as f64,
        kb / wall,
        heap_kind()
    );
    println!("{tag}: pipelines that ended on an efficiency core: {} of {}", ecore_hits.load(Ordering::Relaxed), recs.len());
    thread_report(1.0);
    if num(args, "--dxc-load", 0) > 0 {
        println!("{tag}: DXC compiles finished alongside: {}", dxc_done.load(Ordering::Relaxed));
    }
    let c = counts.into_inner().unwrap();
    if c.samples > 0 {
        let pct = |x: u64| if c.in_driver == 0 { 0.0 } else { 100.0 * x as f64 / c.in_driver as f64 };
        println!(
            "{tag}: STACKS samples={} creator-thread-samples-in-nvgpucomp64={} background-thread-samples-in-nvgpucomp64={} heap-lock-wait={:.1}% heap-running={:.1}% driver-code={:.1}% other-wait={:.1}% other={:.1}%",
            c.samples,
            c.in_driver,
            c.background,
            pct(c.heap_wait),
            pct(c.heap_run),
            pct(c.driver_run),
            pct(c.other_wait),
            pct(c.other)
        );
    }
    if let (Some(l), Some(out)) = (&library.0, &lib_out) {
        unsafe {
            let n = l.GetSerializedSize();
            let mut buf = vec![0u8; n];
            l.Serialize(&mut buf).unwrap();
            std::fs::write(out, &buf).unwrap();
            println!("{tag}: pipeline library {} bytes -> {}", n, out.display());
        }
    }
    drop(psos);
}

/// Re-parse saved stack samples (`STACKS` line for a set of files).
fn stacks(args: &[String]) {
    let mut c = Counts::default();
    for f in args.iter().filter(|a| Path::new(a).is_file()) {
        parse_sample(&std::fs::read_to_string(f).unwrap(), &mut c);
    }
    let pct = |x: u64| if c.in_driver == 0 { 0.0 } else { 100.0 * x as f64 / c.in_driver as f64 };
    println!(
        "STACKS samples={} creator-in-nvgpucomp64={} background-in-nvgpucomp64={} heap-lock-wait={:.1}% heap-running={:.1}% driver-code={:.1}% other-wait={:.1}% other={:.1}%",
        c.samples,
        c.in_driver,
        c.background,
        pct(c.heap_wait),
        pct(c.heap_run),
        pct(c.driver_run),
        pct(c.other_wait),
        pct(c.other)
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("gen") => gen(&args[1..]),
        Some("run") => run(&args[1..]),
        Some("stacks") => stacks(&args[1..]),
        _ => eprintln!("pso156 gen|run|stacks ... (see the file's docs)"),
    }
}
