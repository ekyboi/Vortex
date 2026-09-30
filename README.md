# Vortex

[![ci](https://github.com/ekyboi/Vortex/actions/workflows/ci.yml/badge.svg)](https://github.com/ekyboi/Vortex/actions/workflows/ci.yml)

**A zero-copy, single-producer / single-consumer ring for high-rate packet and frame ingestion in Rust.**

Vortex moves data from a producer thread (typically a network or storage driver loop) to a consumer thread (your processing code) without copying it and without touching the Rust heap. Frames live in memory mapped directly from the operating system. That memory is locked into RAM (or pre-faulted), backed by huge pages, optionally pinned to a NUMA node, and aligned so a NIC, io_uring, an RDMA adapter or a GPU can read and write it directly.

---

## Contents

- [What it does](#what-it-does)
- [When to use it](#when-to-use-it)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Guide](#guide)
  - [Creating a ring](#creating-a-ring)
  - [Producing](#producing)
  - [Consuming](#consuming)
  - [Bursts](#bursts)
  - [Waiting](#waiting)
  - [Shutdown and end of stream](#shutdown-and-end-of-stream)
  - [Long-lived threads](#long-lived-threads)
  - [Timestamps](#timestamps)
  - [Error handling](#error-handling)
- [Configuration](#configuration)
- [Working with devices (DMA, io_uring, RDMA, GPU)](#working-with-devices-dma-io_uring-rdma-gpu)
- [How it works](#how-it-works)
  - [Memory layout](#memory-layout)
  - [Building the arena](#building-the-arena)
  - [The ring protocol](#the-ring-protocol)
  - [Memory ordering](#memory-ordering)
  - [Cache-line isolation](#cache-line-isolation)
  - [Ownership and lifetimes](#ownership-and-lifetimes)
- [API reference](#api-reference)
- [Platform notes](#platform-notes)
- [Performance tuning](#performance-tuning)
- [Benchmarks](#benchmarks)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Limitations](#limitations)
- [Development](#development)
- [License](#license)

---

## What it does

- **Zero copy.** The producer writes straight into a slot of the ring, and the consumer reads that same memory. Nothing is copied between them, and the payload never passes through the Rust heap.
- **Off-heap, OS-mapped memory.** One `mmap` region holds everything. It has guard pages on both sides, and its control data is sealed read-only.
- **No page faults on the hot path.** The whole region is locked with `mlock`. If the lock limit forbids that, every page is touched before traffic starts instead.
- **Huge pages.** Transparent huge pages (THP) by default, or an explicit hugetlbfs pool of any size the kernel supports: 2 MiB, 32 MiB, 512 MiB, 1 GiB, and so on.
- **NUMA placement.** The arena can be bound to the NUMA node closest to your NIC or GPU.
- **DMA-ready frames.** Every slot is 4 KiB-aligned and contiguous. The whole pool, or a table of per-slot `iovec`s, can be registered with CUDA, RDMA or io_uring.
- **No false sharing.** Producer state, consumer state and every slot header live on separate 128-byte cache-line pairs. The build fails if a layout ever breaks that.
- **Compiler-enforced safety.** A payload slice cannot outlive its slot's reservation. That's checked at compile time, not runtime.
- **Static dispatch.** Ring size, slot size and wait strategy are all generic parameters. There are no trait objects or indirect calls on the hot path.
- **Minimal dependencies.** Only `std` and `libc`.

## When to use it

Vortex is a good fit when one thread produces large volumes of frames and one thread consumes them, and the cost of copying or allocating per frame matters. Examples:

- a DPDK, AF_XDP, io_uring or RDMA receive loop handing packets to a parser;
- a capture pipeline feeding frames to a GPU;
- a storage reader feeding a decoder.

It is **not** a general-purpose channel. It has exactly one producer and one consumer, fixed-size slots of at least 4 KiB, and a fixed capacity. For general message passing, use `std::sync::mpsc` or `crossbeam-channel`.

## Requirements

| | Supported |
|---|---|
| **Operating system** | Linux on arm64 or x86_64, any distribution (Arch, Arch Linux ARM, Debian, Ubuntu, Fedora, Alpine, …); macOS on arm64 (Apple Silicon) |
| **C library** | glibc or musl |
| **Kernel page size** | 4 KiB, 16 KiB or 64 KiB (all arm64 Linux configurations) |
| **Rust** | 1.80 or newer |
| **Crate dependencies** | `libc` |
| **Optional** | Reserved hugetlb pages, a raised memory-lock limit, `CAP_SYS_NICE` for NUMA binding in containers |

Other Unix systems build, but without hugetlb, NUMA binding or `madvise` hints. Windows is not supported.

## Installation

### 1. Install Rust

**Any Linux or macOS**, using [rustup](https://rustup.rs):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Arch Linux / Arch Linux ARM:**

```sh
sudo pacman -S --needed rustup base-devel
rustup default stable
```

(`sudo pacman -S rust base-devel` also works. Arch's packaged Rust is always recent enough.)

### 2. Install a linker

Skip this on Arch if you installed `base-devel` above.

| System | Command |
|---|---|
| macOS | `xcode-select --install` |
| Debian / Ubuntu | `sudo apt install build-essential` |
| Fedora | `sudo dnf install gcc` |
| Alpine | `apk add build-base` |

### 3. Add Vortex to your project

```sh
cargo add vortex --git https://github.com/ekyboi/Vortex
```

or add it to your project's `Cargo.toml` by hand:

```toml
[dependencies]
vortex = { git = "https://github.com/ekyboi/Vortex" }
```

To pin a specific version, add `tag = "v0.1.0"` or `rev = "<commit-hash>"`.

### Build from source

```sh
git clone https://github.com/ekyboi/Vortex
cd Vortex
cargo test                                   # run the full test suite
cargo run --release --example bench          # measure throughput and latency on this machine
```

## Quick start

```rust
use vortex::{hw_ticks, SpinThenYield, VortexRing};

// 256 slots, 4 KiB each. Both sizes are compile-time constants.
let mut ring = VortexRing::<256, 4096>::new()?;
let (mut tx, mut rx) = ring.split();

std::thread::scope(|s| {
    // Producer
    s.spawn(move || {
        let mut wait = SpinThenYield::<1024>::default();
        for seq in 0..10_000u64 {
            let mut slot = tx.reserve_ingress_wait(&mut wait).unwrap(); // claim a free slot
            slot.payload_mut()[..8].copy_from_slice(&seq.to_le_bytes()); // write in place
            slot.commit_ingress(8, hw_ticks()).unwrap();                  // publish 8 bytes
        }
    }); // `tx` is dropped here, which signals end of stream

    // Consumer
    s.spawn(move || {
        let mut wait = SpinThenYield::<1024>::default();
        let mut expected = 0u64;
        while let Some(slot) = rx.peek_egress_wait(&mut wait) {
            assert_eq!(slot.meta().sequence, expected);
            assert_eq!(slot.payload(), &expected.to_le_bytes()); // read in place
            slot.release_egress();                               // hand the slot back
            expected += 1;
        }
        assert_eq!(expected, 10_000);
    });
});
# Ok::<(), vortex::VortexError>(())
```

---

## Guide

### Creating a ring

```rust
use vortex::VortexRing;

let ring = VortexRing::<1024, 4096>::new()?;
# Ok::<(), vortex::VortexError>(())
```

- **`SLOTS`** is the capacity. It must be a non-zero power of two.
- **`SLOT_BYTES`** is the size of each frame. It must be a non-zero multiple of 4096 and at most `u32::MAX`.

Invalid parameters are rejected **at compile time**:

```text
error[E0080]: evaluation panicked: SLOTS must be a non-zero power of two
```

The payload pool takes `SLOTS × SLOT_BYTES` bytes. Metadata adds roughly `SLOTS × 144` bytes, and each region is rounded up to the page size. `VortexRing::new()` uses the default configuration. Use `VortexRing::with_config` to choose huge pages, NUMA placement and residency (see [Configuration](#configuration)).

`ring.split()` returns the ring's single `Producer` and `Consumer`. It borrows the ring mutably, so a second pair can't exist while the first is alive. After both handles are dropped you can call `split()` again, and cursor positions and any unconsumed data are preserved.

### Producing

```rust
# use vortex::*;
# let mut ring = VortexRing::<4, 4096>::new()?;
# let (mut tx, _rx) = ring.split();
if let Some(mut slot) = tx.reserve_ingress() {    // None if the ring is full
    let buf: &mut [u8] = slot.payload_mut();        // the whole SLOT_BYTES frame
    buf[..5].copy_from_slice(b"hello");
    let seq = slot.commit_ingress(5, hw_ticks())?;  // returns this frame's sequence number
    assert_eq!(seq, 0);
}
# Ok::<(), vortex::VortexError>(())
```

- `commit_ingress(len, hw_timestamp)` records the length and timestamp, then publishes the slot. `commit_ingress_flagged(len, ts, flags)` also stores 32 bits of your own flags, such as a queue ID or checksum status. `commit_ingress_now(len)` stamps the frame with `hw_ticks()`.
- If `len > SLOT_BYTES`, the commit returns `VortexError::PayloadOverflow` and publishes nothing.
- Dropping an `IngressSlot` without committing **abandons** it. The next `reserve_ingress` returns the same slot.

### Consuming

```rust
# use vortex::*;
# let mut ring = VortexRing::<4, 4096>::new()?;
# let (mut tx, mut rx) = ring.split();
# tx.reserve_ingress().unwrap().commit_ingress(5, 0)?;
if let Some(slot) = rx.peek_egress() {  // None if the ring is empty
    let meta = slot.meta();              // sequence, hw_timestamp, len, flags
    let data: &[u8] = slot.payload();    // exactly `meta.len` bytes
    assert_eq!(data.len(), meta.len as usize);
    slot.release_egress();               // hand the slot back to the producer
}
# Ok::<(), vortex::VortexError>(())
```

- `payload_mut()` gives mutable access for in-place work such as decryption or header rewriting.
- Dropping an `EgressSlot` without releasing it leaves the slot at the head. The next `peek_egress` returns it again.

### Bursts

Bursts reserve, publish or release many slots at once, with a **single** atomic store. That's the natural fit for drivers that fill or drain descriptor rings in batches.

```rust
use vortex::*;

let mut ring = VortexRing::<64, 4096>::new()?;
let (mut tx, mut rx) = ring.split();

// Producer: reserve up to 32 consecutive slots
let mut burst = tx.reserve_ingress_burst(32).unwrap();
let n = burst.len();                                    // may be fewer than 32
burst.for_each_payload_mut(|i, frame| frame[0] = i as u8);
let first_seq = burst.commit_ingress(n, |i| SlotStamp { len: 1, hw_timestamp: 0, flags: i as u32 })?;
assert_eq!(first_seq, 0);

// Consumer: take up to 32 committed slots
let batch = rx.peek_egress_burst(32).unwrap();
for i in 0..batch.len() {
    assert_eq!(batch.payload(i), &[i as u8]);
}
batch.release_egress();                                 // or release_egress_prefix(k)
# Ok::<(), vortex::VortexError>(())
```

- `IngressBurst::commit_ingress(n, stamp)` publishes the first `n` slots, calling `stamp(i)` for each one's length, timestamp and flags. Slots `n..len()` are abandoned. If `n > len()` it returns `BurstOverrun`.
- `EgressBurst::release_egress_prefix(k)` releases only the first `k` slots. The rest stay at the head.
- Indexing a burst out of range panics, just like slice indexing.

### Waiting

`reserve_ingress` and `peek_egress` never block. The `*_wait` variants poll using a **wait strategy**:

```rust
# use vortex::*;
# let mut ring = VortexRing::<4, 4096>::new()?;
# let (mut tx, mut rx) = ring.split();
use std::time::{Duration, Instant};

// Built-in strategies
assert!(rx.peek_egress_wait(&mut Bounded::<100>::default()).is_none()); // empty: gives up after 100 polls

drop(tx); // producer gone and ring empty: even an unbounded wait returns None
assert!(rx.peek_egress_wait(&mut SpinThenYield::<1024>::default()).is_none());

// Any FnMut() -> bool is a strategy: return false to give up
let deadline = Instant::now() + Duration::from_millis(1);
let mut until_deadline = || { std::hint::spin_loop(); Instant::now() < deadline };
assert!(rx.peek_egress_wait(&mut until_deadline).is_none());
# Ok::<(), vortex::VortexError>(())
```

| Strategy | Behaviour | Use when |
|---|---|---|
| `BusySpin` | Spins with the CPU's pause/yield hint forever | Each thread has its own dedicated, isolated core; lowest latency |
| `SpinThenYield<N>` | Spins `N` times, then calls `sched_yield` | **Default choice.** Near-spin latency on dedicated cores; still makes progress on 1-CPU VMs and CPU-limited containers |
| `Bounded<N>` | Spins up to `N` times, then gives up | You have other work to do between polls |
| `FnMut() -> bool` | Your code decides | Deadlines, shutdown flags, metrics |

Budgets reset at the start of every wait, so a single strategy value can be reused across calls. Strategies are generic parameters, so each one compiles into its call site with no dynamic dispatch.

> With `BusySpin` on a machine where producer and consumer share a CPU, every handoff waits for the scheduler to preempt the spinning thread. That can be thousands of times slower. Use `SpinThenYield` unless you have pinned, dedicated cores.

### Shutdown and end of stream

Dropping a `Producer` or `Consumer` marks it **closed**. The other side sees this in its `*_wait` calls:

- **Consumer:** `peek_egress_wait` keeps returning committed slots until the ring is drained. After that it returns `None`, which marks end of stream.
- **Producer:** `reserve_ingress_wait` returns `None` if the ring is full and the consumer is gone, since nothing can ever free a slot.
- You can check directly with `is_producer_closed()` / `is_consumer_closed()`.
- `split()` clears both flags.

### Long-lived threads

`split()` borrows the ring, so the handles can't outlive it. With `std::thread::scope` (as in the quick start) this works out naturally. For threads that run for the whole program, give the ring a `'static` lifetime once at startup:

```rust
use vortex::*;

let ring: &'static mut VortexRing<1024, 4096> = Box::leak(Box::new(VortexRing::new()?));
let (mut tx, mut rx) = ring.split();

let producer = std::thread::spawn(move || {
    let mut wait = SpinThenYield::<1024>::default();
    for _ in 0..100 {
        tx.reserve_ingress_wait(&mut wait).unwrap().commit_ingress_now(0).unwrap();
    }
});
let consumer = std::thread::spawn(move || {
    let mut wait = SpinThenYield::<1024>::default();
    let mut n = 0;
    while let Some(slot) = rx.peek_egress_wait(&mut wait) {
        slot.release_egress();
        n += 1;
    }
    n
});
producer.join().unwrap();
assert_eq!(consumer.join().unwrap(), 100);
# Ok::<(), vortex::VortexError>(())
```

The single `Box` holds only the small ring handle. The arena itself is always `mmap`-backed.

### Timestamps

Every committed slot carries a 64-bit `hw_timestamp`. Pass in the NIC's hardware timestamp if you have one; otherwise use the built-in counter:

| Function | Returns |
|---|---|
| `hw_ticks()` | Raw hardware counter: `CNTVCT_EL0` on arm64, `RDTSC` on x86_64, `CLOCK_MONOTONIC` in nanoseconds elsewhere |
| `hw_tick_hz()` | Counter frequency. Measured once (about 20 ms, sleeping), then cached. On arm64, the architectural `CNTFRQ_EL0` value is used if it agrees with that measurement. |
| `ticks_to_nanos(t)` | Converts a tick delta to nanoseconds |

Call `hw_tick_hz()` once at startup so the one-time 20 ms measurement doesn't land on your hot path.

### Error handling

Every OS failure is reported as a `VortexError` carrying the `errno`. `VortexError` implements `std::error::Error` and converts to `std::io::Error`.

```rust
use vortex::*;

let strict = ArenaConfig { residency: ResidencyPolicy::RequireLock, ..ArenaConfig::default() };
match VortexRing::<64, 4096>::with_config(strict) {
    Ok(ring) => println!("locked: {:?}", ring.arena().report().residency),
    Err(VortexError::Lock { errno, bytes }) => {
        eprintln!("cannot mlock {bytes} bytes (errno {errno}); raise `ulimit -l`");
    }
    Err(other) => return Err(other.into()),
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

| Variant | Meaning |
|---|---|
| `PageSize { errno }` | `sysconf(_SC_PAGESIZE)` failed |
| `GeometryOverflow` | `SLOTS × SLOT_BYTES` (plus metadata) doesn't fit in the address space |
| `Map { errno, bytes }` | `mmap` failed (usually out of address space or memory) |
| `Trim { errno }` | `munmap` of alignment slack failed |
| `Protect { errno }` | `mprotect` failed |
| `Lock { errno, bytes }` | `mlock` failed under `RequireLock`, or failed for a reason other than resource limits |
| `HugeTlb { errno }` | Hugetlb mapping failed with `fallback: false` (`ENOMEM`: no pages reserved; `EINVAL`: unsupported size; `ENOTSUP`: not Linux) |
| `Bind { errno, node }` | `mbind` failed (`EPERM`: container seccomp; `EINVAL`: no such node; `ENOTSUP`: not Linux) |
| `PayloadOverflow { len, capacity }` | Committed length exceeds `SLOT_BYTES` |
| `BurstOverrun { requested, reserved }` | Burst commit named more slots than were reserved |

---

## Configuration

```rust
use vortex::*;

let config = ArenaConfig {
    residency: ResidencyPolicy::LockOrPrefault,
    backing: PageBacking::HugeTlb {
        size: HugePageSize::system_default().unwrap_or(HugePageSize::SIZE_2M),
        fallback: true,
    },
    dont_fork: true,
    numa_node: None,
};
let ring = VortexRing::<1024, 4096>::with_config(config)?;
println!("{:#?}", ring.arena().report());
# Ok::<(), vortex::VortexError>(())
```

| Field | Default | Description |
|---|---|---|
| `residency` | `LockOrPrefault` | How pages are made resident before use (see below) |
| `backing` | `Transparent` | Page backing for the payload pool (see below) |
| `dont_fork` | `true` | `MADV_DONTFORK`: stops a `fork()` elsewhere in the process from copy-on-write-splitting pages a device has pinned (Linux) |
| `numa_node` | `None` | Bind the arena to this NUMA node before first touch (Linux) |

**`ResidencyPolicy`**

| Value | Behaviour |
|---|---|
| `LockOrPrefault` | `mlock` everything. If the lock limit forbids it (`EPERM`, `ENOMEM`, `EAGAIN`), write to every page instead to fault it in. |
| `RequireLock` | `mlock`, or fail with `VortexError::Lock` |
| `PrefaultOnly` | Write to every page; never `mlock` |

Locked pages are never swapped or reclaimed. Prefaulted pages don't fault on the hot path either, but the kernel can reclaim them under memory pressure.

**`PageBacking`**

| Value | Behaviour |
|---|---|
| `Base` | Ordinary pages, no huge-page hints |
| `Transparent` | `madvise(MADV_HUGEPAGE)`. The arena is aligned to the huge-page size if it's large enough to use one. Best effort. |
| `HugeTlb { size, fallback }` | Pool mapped from the reserved hugetlbfs pool (`MAP_HUGETLB`). Guaranteed huge TLB entries and physically contiguous pages. With `fallback: true`, falls back to `Transparent` if the pool is empty. |

**`HugePageSize`**: `SIZE_64K`, `SIZE_2M`, `SIZE_32M`, `SIZE_512M`, `SIZE_1G`, `from_bytes(n)`, or `system_default()` (reads `Hugepagesize` from `/proc/meminfo`). Which sizes exist depends on the kernel's base page size:

| Base page | Available hugetlb sizes |
|---|---|
| 4 KiB (x86_64, most arm64) | 64 KiB, 2 MiB, 32 MiB, 1 GiB |
| 16 KiB (some arm64) | 2 MiB, 32 MiB, 1 GiB |
| 64 KiB (some arm64 servers) | 2 MiB, 512 MiB, 16 GiB |

**`ArenaReport`**: returned by `ring.arena().report()`. Records what the kernel actually granted:

| Field | Meaning |
|---|---|
| `page_size` | OS base page size |
| `body_alignment` | Alignment of the arena body (the huge-page size when applicable) |
| `mapped_bytes` / `body_bytes` | Total mapped (including guard pages) / usable |
| `residency` | `Locked`, or `Prefaulted { mlock_errno }` |
| `backing` | `Base`, `Transparent { advice, hugetlb_errno }`, or `HugeTlb { size }` |
| `dont_fork` | `Applied`, `Rejected { errno }`, `Unsupported`, or `Disabled` |
| `numa_node` | The node the arena is bound to, if any |

---

## Working with devices (DMA, io_uring, RDMA, GPU)

All payload frames are 4 KiB-aligned, contiguous, and resident before any traffic flows.

| Need | API |
|---|---|
| Register the whole pool once | `ring.arena().dma_region()` returns a `DmaRegion` with `as_ptr()`, `as_mut_ptr()`, `addr()` and `len()` |
| Per-slot buffer table | `ring.arena().iovecs()` returns `&[libc::iovec]`, indexed by slot |
| Same table, typed | `ring.arena().descriptors()` returns `&[SlotDescriptor]`, with `base()`, `addr()` and `capacity()` |
| Address of a reserved slot (device writes into it) | `IngressSlot::dma_ptr()`, `IngressBurst::dma_ptr(i)`, plus `index()` / `index(i)` |
| Address of a committed slot (device reads from it) | `EgressSlot::dma_ptr()`, `dma_addr()`, `dma_region()`; `EgressBurst::dma_ptr(i)` |

**The receive pattern:** reserve slots, hand their addresses to the device, wait for completion, then commit.

```rust
use vortex::*;

let mut ring = VortexRing::<64, 4096>::new()?;
let (mut tx, mut rx) = ring.split();

let mut burst = tx.reserve_ingress_burst(8).unwrap();
let mut lens = [0usize; 8];
for (i, len) in lens.iter_mut().enumerate().take(burst.len()) {
    // A real driver posts (burst.index(i), burst.dma_ptr(i)) to the NIC here,
    // and later learns each frame's length from the completion queue.
    let frame = b"packet";
    unsafe { core::ptr::copy_nonoverlapping(frame.as_ptr(), burst.dma_ptr(i), frame.len()) };
    *len = frame.len();
}
let n = burst.len();
burst.commit_ingress(n, |i| SlotStamp { len: lens[i], hw_timestamp: hw_ticks(), flags: 0 })?;

assert_eq!(rx.peek_egress().unwrap().payload(), b"packet");
# Ok::<(), vortex::VortexError>(())
```

> **Important:** commit only **after** the device reports completion (an io_uring CQE, `ibv_poll_cq`, a descriptor "done" bit). The commit's `Release` store orders CPU writes. It cannot order a DMA transfer that is still in flight.

**io_uring fixed buffers** (sketch; uses your io_uring bindings):

```rust,ignore
let iov = ring.arena().iovecs();
io_uring_register(ring_fd, IORING_REGISTER_BUFFERS, iov.as_ptr().cast(), iov.len() as u32);
// Then for each reserved slot: IORING_OP_READ_FIXED with
//   addr = burst.dma_ptr(i), len = 4096, buf_index = burst.index(i)
```

**CUDA / GPUDirect** (sketch):

```rust,ignore
let pool = ring.arena().dma_region();
cuMemHostRegister(pool.as_mut_ptr().cast(), pool.len(), CU_MEMHOSTREGISTER_DEVICEMAP);
// Consumer: pass slot.dma_ptr() to cuMemcpyHtoDAsync; release_egress() after the copy completes.
```

**RDMA** (sketch): call `ibv_reg_mr(pd, pool.as_mut_ptr(), pool.len(), access)` once, then post receive WRs whose SGEs point at `burst.dma_ptr(i)`.

A `DmaRegion` or raw pointer is just an address, not a Rust borrow. Using it is only valid while the ring is alive, and only for slots your side currently owns.

---

## How it works

### Memory layout

Each ring owns one arena: a single contiguous virtual-memory reservation.

```text
 PROT_NONE   ┌───────────────── body (huge-page aligned, NUMA-bound, locked / prefaulted) ─────────────────┐  PROT_NONE
┌─────────┐  ┌──────────────────────────────┬─────────────────────────────────┬──────────────────────────┐  ┌─────────┐
│ guard   │  │ payload pool                 │ turnstile  │ slot headers       │ geometry │ descriptors   │  │ guard   │
│ page    │  │ SLOTS × SLOT_BYTES           │ tail│head  │ SLOTS × 128 B      │ 128 B    │ SLOTS × iovec │  │ page    │
└─────────┘  │ 4 KiB-aligned frames         │ 2 × 128 B  │                    │          │ (16 B each)   │  └─────────┘
             └────── read-write ────────────┴──────── read-write ─────────────┴─────── read-only ────────┘
```

| Region | Contents |
|---|---|
| **Payload pool** | The frames. It comes first, so it starts on the body's huge-page-aligned base, and is rounded up to its backing page size. |
| **Turnstile** | The producer cursor (`tail`) and the consumer cursor (`head`), each on its own 128-byte line together with that side's "closed" flag |
| **Slot headers** | One 128-byte line per slot, holding a `SlotMeta` (sequence, timestamp, length, flags) |
| **Geometry** | A 128-byte, C-ABI control block (magic `VORTEX01`, slot count and size, page size, region addresses), for validation by C or CUDA code |
| **Descriptors** | One `iovec` per slot. Sealed `PROT_READ` after setup, so a stray write faults instead of corrupting the table. |
| **Guard pages** | Never made accessible; an out-of-bounds access faults immediately |

### Building the arena

`VortexArena::map` performs these steps, in this order:

1. **Reserve.** `mmap` a `PROT_NONE` region big enough for the body, both guard pages and alignment slack. The body start is aligned to the page size, 4 KiB, and the huge-page size where relevant, and the slack is returned with `munmap`. Nothing is committed yet.
2. **Back the pool.** For `HugeTlb`, the pool range is replaced in place with a `MAP_FIXED | MAP_HUGETLB` mapping. If that fails, the range is re-reserved first (so the address space can't be claimed in between), then the arena either falls back to THP or returns the error.
3. **Commit.** `mprotect` the rest of the body read-write. Anything never committed stays `PROT_NONE` and acts as a guard.
4. **Place.** Before any page is touched: `MADV_HUGEPAGE` for THP, `mbind(MPOL_BIND)` for NUMA, and `MADV_DONTFORK`. Doing this before first touch means the first faults already get huge, node-local pages.
5. **Make resident.** `mlock`, or write to one byte per page. A read would only map the shared zero page, so the fault would still happen later on first write. Hugetlb pages are always touched explicitly, because Linux `mlock` skips hugetlb mappings. Otherwise the first packet into each huge page would pay to allocate and zero up to 1 GiB.
6. **Initialise.** Write the cursors, headers, geometry and descriptors.
7. **Seal.** `mprotect` the geometry and descriptors `PROT_READ`.

Dropping the ring `munmap`s the whole reservation, which also releases any memory locks.

### The ring protocol

- **Positions.** The cursors are 64-bit counters that only ever increase. Wrap-around is not a concern: at 10 billion slots per second, 2⁶⁴ takes 58 years. A position's slot is `pos & (SLOTS − 1)`, which is one AND instruction.
- **Invariant.** `head ≤ tail ≤ head + SLOTS`. Slots in `[head, tail)` belong to the consumer, and all others to the producer.
- **Private copies.** Each side keeps its own cursor in a private field. It never re-reads its own shared atomic, only stores to it.
- **Cached peer cursor.** Each side keeps a cached copy of the other side's cursor, and re-reads the real one only when the cache says the ring is full (producer) or empty (consumer). In steady state that's once per lap of the ring, not once per frame.
- **Sequence numbers.** A frame's sequence number is simply the position it was committed at, so the consumer can check it matches.

### Memory ordering

| Step | Who | Operation | Guarantees |
|---|---|---|---|
| 1 | Producer | Write payload and header (plain stores) | |
| 2 | Producer | `tail.store(pos + 1, Release)` | Step 1 is visible to anyone who observes the new `tail` |
| 3 | Consumer | `tail.load(Acquire)` | Sees everything the producer wrote before step 2 |
| 4 | Consumer | Read or modify payload (plain) | |
| 5 | Consumer | `head.store(pos + 1, Release)` | Step 4 completes before the producer may reuse the slot |
| 6 | Producer | `head.load(Acquire)` | Safe to overwrite the slot |

On x86_64 the `Release` store and `Acquire` load compile to plain `mov` instructions. On arm64 they're `stlr` and `ldar`. There are no locks, CAS loops or full fences. A burst publishes or releases many slots with one store.

Closing works the same way. `Drop` stores the closed flag with `Release`, after that handle's final cursor store. The peer loads it with `Acquire`, then re-reads the cursor once, so a commit racing with the drop is never lost.

### Cache-line isolation

Modern CPUs move memory between cores in 64-byte lines. Intel's spatial prefetcher pulls in pairs, and Apple and Neoverse cores use 128-byte lines. Vortex therefore treats **128 bytes** as the unit of isolation:

- `tail` and `head` are each on their own 128-byte-aligned line.
- Every slot header is exactly one 128-byte line. The producer stamping slot `n + 1` never invalidates the line the consumer is reading for slot `n`.
- The `Producer` and `Consumer` handles are themselves 128-byte aligned.

A `const` block in the source asserts all of this at **compile time**: sizes, alignments, offsets, and that no atomic straddles or shares a line. The build fails if a future change breaks it. `SlotDescriptor` is likewise checked to be ABI-identical to `libc::iovec`.

### Ownership and lifetimes

- `split(&mut self)` makes the producer/consumer pair unique.
- `IngressSlot`, `EgressSlot` and the burst types hold `&mut` borrows of their handle, so each side can have only one outstanding reservation at a time.
- Payload slices are borrowed from the guard. `commit_ingress` and `release_egress` take the guard **by value**, so the compiler rejects any use of a slice after hand-off:

```rust,compile_fail
let mut ring = vortex::VortexRing::<4, 4096>::new().unwrap();
let (mut tx, _rx) = ring.split();
let mut slot = tx.reserve_ingress().unwrap();
let buf: &mut [u8] = slot.payload_mut();
slot.commit_ingress(0, 0).unwrap();
buf[0] = 1; // error[E0505]: cannot move out of `slot` because it is borrowed
```

- Every `unsafe` block carries a `SAFETY` comment explaining why it's sound. The crate is compiled with `#![deny(unsafe_op_in_unsafe_fn)]`.

---

## API reference

Full rustdoc: `cargo doc --open`.

**Ring and handles**

| Item | Description |
|---|---|
| `VortexRing<SLOTS, SLOT_BYTES>` | `new()`, `with_config(cfg)`, `split()`, `arena()`, `occupancy()` |
| `Producer` | `reserve_ingress()`, `reserve_ingress_burst(max)`, `reserve_ingress_wait(w)`, `reserve_ingress_burst_wait(max, w)`, `free_slots()`, `next_sequence()`, `is_consumer_closed()`, `arena()` |
| `Consumer` | `peek_egress()`, `peek_egress_burst(max)`, `peek_egress_wait(w)`, `peek_egress_burst_wait(max, w)`, `pending()`, `next_sequence()`, `is_producer_closed()`, `arena()` |

**Slot guards**

| Item | Description |
|---|---|
| `IngressSlot` | `payload_mut()`, `payload()`, `sequence()`, `index()`, `capacity()`, `dma_ptr()`, `dma_region()`, `commit_ingress(len, ts)`, `commit_ingress_flagged(len, ts, flags)`, `commit_ingress_now(len)` |
| `IngressBurst` | `len()`, `first_sequence()`, `index(i)`, `payload_mut(i)`, `for_each_payload_mut(f)`, `dma_ptr(i)`, `commit_ingress(n, stamp)` |
| `EgressSlot` | `meta()`, `payload()`, `payload_mut()`, `index()`, `dma_ptr()`, `dma_addr()`, `dma_region()`, `release_egress()` |
| `EgressBurst` | `len()`, `first_sequence()`, `meta(i)`, `payload(i)`, `payload_mut(i)`, `dma_ptr(i)`, `release_egress()`, `release_egress_prefix(k)` |

**Metadata and device types**

| Item | Description |
|---|---|
| `SlotMeta` | `sequence: u64`, `hw_timestamp: u64`, `len: u32`, `flags: u32` |
| `SlotStamp` | `len: usize`, `hw_timestamp: u64`, `flags: u32`: per-slot values for burst commits |
| `VortexArena` | `report()`, `geometry()`, `descriptors()`, `iovecs()`, `dma_region()`, `SLOTS`, `SLOT_BYTES` |
| `Geometry` | `magic`, `slots`, `slot_bytes`, `page_size`, `pool_base`, `pool_bytes`, `headers_base`, `descriptors_base` (all `u64`, `#[repr(C)]`) |
| `SlotDescriptor` | `base()`, `addr()`, `capacity()`, `as_iovec()` |
| `DmaRegion` | `as_ptr()`, `as_mut_ptr()`, `addr()`, `len()`, `is_empty()` |

**Configuration, waiting and clock**

| Item | Description |
|---|---|
| `ArenaConfig`, `ResidencyPolicy`, `PageBacking`, `HugePageSize` | See [Configuration](#configuration) |
| `ArenaReport`, `Residency`, `Backing`, `Advice` | See [Configuration](#configuration) |
| `WaitStrategy` | Trait with `reset()` and `wait() -> bool`. Implemented by `BusySpin`, `SpinThenYield<N>`, `Bounded<N>` and every `FnMut() -> bool`. |
| `hw_ticks()`, `hw_tick_hz()`, `ticks_to_nanos(t)` | See [Timestamps](#timestamps) |
| `VortexError` | See [Error handling](#error-handling). Has `errno()`, `Display`, `std::error::Error`, and `From` for `std::io::Error`. |
| `COHERENCE_GRANULE` (128), `DMA_ALIGN` (4096), `GEOMETRY_MAGIC` | Constants |

---

## Platform notes

**Linux (arm64 and x86_64)**

- Every feature is available.
- Works with 4 KiB, 16 KiB and 64 KiB page kernels. The huge-page size is derived from the running kernel.
- Works on glibc and musl distributions, including Arch Linux and Arch Linux ARM.

**macOS (Apple Silicon)**

- `mlock` works. The page size is 16 KiB.
- There's no hugetlb, NUMA or `madvise` support:
  - `HugeTlb { fallback: true }` falls back and reports `hugetlb_errno: Some(ENOTSUP)`;
  - `HugeTlb { fallback: false }` and `numa_node: Some(_)` return errors with `ENOTSUP`.
- macOS has no CPU-affinity API, so benchmark threads can't be pinned and latency tails are noisier.
- The Apple Silicon counter reports 24 MHz or 1 GHz depending on the chip, but advances in coarse steps, so single-event timestamps resolve to tens of nanoseconds.

**Containers (Docker, Podman, Kubernetes)**

- The default memory-lock limit is small, so residency usually becomes `Prefaulted`. Raise it with `--ulimit memlock=-1`.
- Docker's default seccomp profile blocks `mbind`, so NUMA binding returns `EPERM`. Add `--cap-add SYS_NICE`, or leave `numa_node: None`.
- CPU quotas (`--cpus=1`) are fine with `SpinThenYield`.

---

## Performance tuning

None of this is required; each step removes a source of latency or jitter.

```sh
# 1. Allow the arena to be locked in RAM
ulimit -l unlimited                   # systemd: LimitMEMLOCK=infinity   docker: --ulimit memlock=-1

# 2. Reserve huge pages for PageBacking::HugeTlb (check the size with: grep Hugepagesize /proc/meminfo)
echo 512 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# 3. Find the NIC's NUMA node (interface names from `ip link`) and pass it as numa_node
cat /sys/class/net/<iface>/device/numa_node

# 4. Pin producer and consumer to separate physical cores on that node
cargo run --release --example bench -- 2 3
```

To make the settings persistent:

- **Memory-lock limit:** add a `memlock` entry in `/etc/security/limits.conf`, or `LimitMEMLOCK=` in your systemd unit.
- **Huge pages:** set `vm.nr_hugepages` in `/etc/sysctl.d/`.

For the lowest tail latency, also isolate the two cores from the scheduler (`isolcpus=` / `nohz_full=` kernel parameters) and use `BusySpin`.

In your code:

- use bursts when the device delivers frames in batches;
- call `hw_tick_hz()` once at startup;
- size `SLOTS` to absorb your longest consumer stall.

---

## Benchmarks

```sh
cargo run --release --example bench                  # unpinned
cargo run --release --example bench -- 2 3           # Linux: producer on CPU 2, consumer on CPU 3
```

The benchmark reports three things:

- single-slot throughput, over 50 million 64-byte messages;
- burst-of-32 throughput;
- round-trip latency percentiles over 2 million ping-pongs across two rings. This is measured on one core's clock, so it's unaffected by counter skew between cores.

Illustrative results from an Apple Silicon Mac with unpinned threads. They vary noticeably from run to run:

| Test | Result |
|---|---|
| Single-slot throughput | ~70–98 M msgs/s |
| Burst-32 throughput | ~110–137 M msgs/s |
| Round trip | p50 ~170–210 ns, p99 ~290–330 ns |

Tail percentiles beyond p99.9 are dominated by OS scheduling unless threads are pinned to isolated cores. Measure on your own hardware before relying on any figure.

---

## Troubleshooting

Start with `println!("{:#?}", ring.arena().report());`, which shows what the kernel actually granted.

| Symptom | Cause | Fix |
|---|---|---|
| `residency: Prefaulted { mlock_errno: Some(1 \| 11 \| 12) }` | Memory-lock limit too low | `ulimit -l unlimited`, or see [Performance tuning](#performance-tuning) |
| `VortexError::Lock` | `RequireLock` with too low a limit | Raise the limit, or use `LockOrPrefault` |
| `hugetlb_errno: Some(12)` / `VortexError::HugeTlb { errno: 12 }` | No huge pages reserved | Set `nr_hugepages` |
| `VortexError::HugeTlb { errno: 22 }` | That size isn't supported by this kernel | Use `HugePageSize::system_default()` |
| `VortexError::Bind { errno: 1, .. }` | Container seccomp blocks `mbind` | `--cap-add SYS_NICE`, or `numa_node: None` |
| `VortexError::Bind { errno: 22, .. }` | That NUMA node doesn't exist or has no memory | Check `/sys/devices/system/node/has_memory` |
| `advice: Rejected { errno: 22 }` | Kernel built without THP | Harmless; use `HugeTlb` if you need huge pages |
| Very slow handoffs, or a hang, on a small VM | `BusySpin` with producer and consumer sharing a CPU | Use `SpinThenYield` |
| Consumer waits forever | Producer still alive but idle | Drop the producer to signal end of stream, or use a `Bounded` / closure strategy |
| Compile error `SLOTS must be a non-zero power of two` | Invalid const parameter | Use 2, 4, 8, … slots and a multiple of 4096 bytes |

---

## FAQ

**Why must `SLOT_BYTES` be a multiple of 4096?**
Keeping every frame 4 KiB-aligned is what makes it registrable for DMA, io_uring fixed buffers and GPU access without copies. If most of your messages are small, pack several into one slot, or use a general-purpose channel instead.

**Can I have multiple producers or consumers?**
No. Use one ring per producer/consumer pair. Fan-in or fan-out topologies are built from several rings.

**Does it allocate on the hot path?**
No. Reserving, committing, peeking and releasing do no allocation and no system calls. The only allocation-like work happens at construction (`mmap` and friends).

**Is it `no_std`?**
Not currently. It uses `std` for errno, `sched_yield`, and clock calibration.

**What happens if a thread panics while holding a slot?**
The guard is dropped. An uncommitted reservation is abandoned, and an unreleased egress slot stays at the head. The handle is then dropped too, which marks that side closed for its peer.

---

## Limitations

- Single producer, single consumer.
- Fixed capacity, set at compile time.
- Minimum 4 KiB per slot.
- Commit only after a device's DMA has completed; Vortex cannot detect in-flight DMA.
- Prefaulted memory, used when `mlock` is not permitted, can still be reclaimed under memory pressure.
- Only arm64 Linux, x86_64 Linux and arm64 macOS are tested. Other Unix systems build, without the Linux-specific features.

---

## Development

```sh
cargo test                                            # unit tests, doctests (including this README), compile-fail tests
cargo test --release
cargo clippy --all-targets -- -D warnings
cargo doc --open

# Environment-gated tests, which assert that the fast paths actually engaged:
VORTEX_EXPECT_HUGETLB=1 cargo test hugetlb            # after reserving 2 MiB huge pages
bash -c 'ulimit -l 0 && VORTEX_EXPECT_PREFAULT=1 cargo test'   # mlock fallback path
```

**Layout**

| Path | Contents |
|---|---|
| `src/lib.rs` | The whole library |
| `examples/bench.rs` | Throughput and latency benchmark |
| `.github/workflows/ci.yml` | CI |

**CI** runs on every push and pull request:

| Job | What it checks |
|---|---|
| test | clippy, tests and the benchmark on arm64 Linux, arm64 macOS and x86_64 Linux |
| archlinux | Tests with Arch's packaged Rust |
| musl | Static musl build and tests |
| single-cpu | Everything pinned to one CPU with `taskset` |
| hugetlb | A reserved 2 MiB page pool (arm64 and x86_64) |
| memlock-fallback | Memory-lock limit of zero, forcing the prefault path |
| container-msrv | Rust 1.80, inside a 1-CPU Docker container with the default seccomp profile |

---

## License

Licensed under the [MIT License](LICENSE).
# Vortex
A zero-copy, single-producer / single-consumer ring for high-rate packet and frame ingestion in Rust
