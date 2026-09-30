//! Throughput and round-trip latency.
//!
//! `cargo run --release --example bench [-- <producer_cpu> <consumer_cpu>]`
//!
//! Round-trip latency is measured on a single core's clock (ping on ring A, echo on ring B), so
//! it is immune to cross-core counter skew. One-way latency is roughly half.

use std::hint::black_box;
use std::time::Instant;
use vortex::{hw_tick_hz, hw_ticks, ticks_to_nanos, SlotStamp, SpinThenYield, VortexRing};

/// Spin, then yield. With one core per thread the yield never triggers; with fewer cores than
/// threads (small VMs, CPU-quota'd containers) it lets the peer run instead of burning a whole
/// timeslice per handoff.
type Poll = SpinThenYield<4096>;

const MSG: usize = 64;
const THROUGHPUT_N: u64 = 50_000_000;
const LATENCY_N: usize = 2_000_000;
const LATENCY_WARMUP: usize = 100_000;

fn pin(cpu: Option<usize>) {
    #[cfg(target_os = "linux")]
    if let Some(cpu) = cpu {
        // SAFETY: zeroed cpu_set_t is a valid empty set; the pointer is valid for the call.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_SET(cpu, &mut set);
            if libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set) != 0 {
                eprintln!("sched_setaffinity({cpu}): {}", std::io::Error::last_os_error());
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpu;
}

fn throughput_single(cpus: (Option<usize>, Option<usize>)) {
    let mut ring = VortexRing::<4096, 4096>::new().unwrap();
    let (mut tx, mut rx) = ring.split();
    let msg = [0xA5u8; MSG];
    let elapsed = std::thread::scope(|s| {
        s.spawn(move || {
            pin(cpus.0);
            for _ in 0..THROUGHPUT_N {
                let mut slot = tx.reserve_ingress_wait(&mut Poll::default()).unwrap();
                slot.payload_mut()[..MSG].copy_from_slice(&msg);
                slot.commit_ingress(MSG, 0).unwrap();
            }
        });
        s.spawn(move || {
            pin(cpus.1);
            let mut sum = 0u64;
            let t0 = Instant::now();
            while let Some(slot) = rx.peek_egress_wait(&mut Poll::default()) {
                sum = sum.wrapping_add(slot.payload()[MSG - 1] as u64);
                slot.release_egress();
            }
            black_box(sum);
            t0.elapsed()
        })
        .join()
        .unwrap()
    });
    report_throughput("single-slot", elapsed);
}

fn throughput_burst(cpus: (Option<usize>, Option<usize>)) {
    const BURST: usize = 32;
    let mut ring = VortexRing::<4096, 4096>::new().unwrap();
    let (mut tx, mut rx) = ring.split();
    let msg = [0xA5u8; MSG];
    let elapsed = std::thread::scope(|s| {
        s.spawn(move || {
            pin(cpus.0);
            let mut sent = 0u64;
            while sent < THROUGHPUT_N {
                let mut b = tx.reserve_ingress_burst_wait(BURST, &mut Poll::default()).unwrap();
                let n = b.len().min((THROUGHPUT_N - sent) as usize);
                for i in 0..n {
                    b.payload_mut(i)[..MSG].copy_from_slice(&msg);
                }
                b.commit_ingress(n, |_| SlotStamp { len: MSG, hw_timestamp: 0, flags: 0 }).unwrap();
                sent += n as u64;
            }
        });
        s.spawn(move || {
            pin(cpus.1);
            let mut sum = 0u64;
            let t0 = Instant::now();
            while let Some(b) = rx.peek_egress_burst_wait(BURST, &mut Poll::default()) {
                for i in 0..b.len() {
                    sum = sum.wrapping_add(b.payload(i)[MSG - 1] as u64);
                }
                b.release_egress();
            }
            black_box(sum);
            t0.elapsed()
        })
        .join()
        .unwrap()
    });
    report_throughput("burst-32", elapsed);
}

fn report_throughput(name: &str, elapsed: std::time::Duration) {
    let secs = elapsed.as_secs_f64();
    let mpps = THROUGHPUT_N as f64 / secs / 1e6;
    let gbps = THROUGHPUT_N as f64 * MSG as f64 * 8.0 / secs / 1e9;
    println!("{name:<12} {mpps:>8.1} Mmsg/s   {gbps:>6.1} Gbit/s payload ({MSG} B)   {secs:.2} s");
}

fn latency(cpus: (Option<usize>, Option<usize>)) {
    let mut ping = VortexRing::<64, 4096>::new().unwrap();
    let mut pong = VortexRing::<64, 4096>::new().unwrap();
    let (mut ping_tx, mut ping_rx) = ping.split();
    let (mut pong_tx, mut pong_rx) = pong.split();
    let mut samples = Vec::with_capacity(LATENCY_N);

    std::thread::scope(|s| {
        s.spawn(move || {
            pin(cpus.1);
            while let Some(req) = ping_rx.peek_egress_wait(&mut Poll::default()) {
                let mut rsp = pong_tx.reserve_ingress_wait(&mut Poll::default()).unwrap();
                let len = req.payload().len();
                rsp.payload_mut()[..len].copy_from_slice(req.payload());
                req.release_egress();
                rsp.commit_ingress(len, 0).unwrap();
            }
        });
        pin(cpus.0);
        let msg = [0x5Au8; MSG];
        for i in 0..LATENCY_WARMUP + LATENCY_N {
            let t0 = hw_ticks();
            let mut req = ping_tx.reserve_ingress_wait(&mut Poll::default()).unwrap();
            req.payload_mut()[..MSG].copy_from_slice(&msg);
            req.commit_ingress(MSG, t0).unwrap();
            let rsp = pong_rx.peek_egress_wait(&mut Poll::default()).unwrap();
            black_box(rsp.payload());
            rsp.release_egress();
            let rtt = hw_ticks() - t0;
            if i >= LATENCY_WARMUP {
                samples.push(rtt);
            }
        }
        drop(ping_tx);
    });

    samples.sort_unstable();
    let pct = |p: f64| ticks_to_nanos(samples[((samples.len() as f64 * p) as usize).min(samples.len() - 1)]);
    let res = clock_resolution_ns();
    println!(
        "round-trip   p50 {} ns   p90 {} ns   p99 {} ns   p99.9 {} ns   p99.99 {} ns   max {} ns   (clock resolution {res:.1} ns, {LATENCY_N} samples)",
        pct(0.50),
        pct(0.90),
        pct(0.99),
        pct(0.999),
        pct(0.9999),
        ticks_to_nanos(*samples.last().unwrap()),
    );
}

/// Smallest observable counter increment. Differs from 1 / `hw_tick_hz` where the counter is
/// scaled (e.g. Apple Silicon reports a 1 GHz CNTFRQ but steps at the 24 MHz crystal rate).
fn clock_resolution_ns() -> f64 {
    let hz = hw_tick_hz() as f64;
    let mut min_step = u64::MAX;
    for _ in 0..1000 {
        let a = hw_ticks();
        let mut b = hw_ticks();
        while b == a {
            b = hw_ticks();
        }
        min_step = min_step.min(b - a);
    }
    min_step as f64 * 1e9 / hz
}

fn main() {
    let mut args = std::env::args().skip(1).map(|a| a.parse::<usize>().expect("cpu index"));
    let cpus = (args.next(), args.next());
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    if cores < 2 {
        println!("note: {cores} CPU available; producer and consumer share it, so figures reflect scheduler handoffs");
    }
    let probe = VortexRing::<4096, 4096>::new().unwrap();
    println!("arena: {:?}", probe.arena().report());
    drop(probe);
    throughput_single(cpus);
    throughput_burst(cpus);
    latency(cpus);
}
