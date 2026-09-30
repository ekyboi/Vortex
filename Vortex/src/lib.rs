//! Vortex: a hardware-aligned, zero-copy SPSC ingestion engine over off-heap, page-locked arenas.
//!
//! Memory topology of a single arena (one `PROT_NONE` reservation with committed sub-ranges, no
//! allocator involvement). The pool is base, THP, or hugetlbfs (2 MiB / 1 GiB) backed; the whole
//! body is optionally `MPOL_BIND`-ed to the NIC/GPU-local NUMA node before first touch.
//!
//! ```text
//!  PROT_NONE   ┌──────────────── body (huge-page-aligned, NUMA-bound, mlock'd / pre-faulted) ───────────────┐  PROT_NONE
//! ┌─────────┐  ┌──────────────────────────────┬─────────────────────────────────┬──────────────────────────┐  ┌─────────┐
//! │ guard   │  │ payload pool                 │ turnstile  │ slot headers       │ geometry │ descriptors   │  │ guard   │
//! │ (1 pg)  │  │ SLOTS × SLOT_BYTES, 4 KiB    │ tail│head  │ SLOTS × 128 B      │ 128 B    │ SLOTS × iovec │  │ (1 pg)  │
//! └─────────┘  │ aligned, DMA/GPU-registrable │ 2 × 128 B  │                    │          │               │  └─────────┘
//!              └──────────────────────────────┴───────────── RW ────────────────┴────────── RO ────────────┘
//! ```
//!
//! ```
//! use vortex::{hw_ticks, SpinThenYield, VortexRing};
//!
//! let mut ring = VortexRing::<256, 4096>::new().unwrap();
//! let (mut tx, mut rx) = ring.split();
//!
//! std::thread::scope(|s| {
//!     s.spawn(move || {
//!         for seq in 0..10_000u64 {
//!             let mut slot = tx.reserve_ingress_wait(&mut SpinThenYield::<1024>::default()).unwrap();
//!             slot.payload_mut()[..8].copy_from_slice(&seq.to_le_bytes());
//!             slot.commit_ingress(8, hw_ticks()).unwrap();
//!         }
//!         // Dropping `tx` signals end of stream.
//!     });
//!     s.spawn(move || {
//!         let mut expected = 0u64;
//!         while let Some(slot) = rx.peek_egress_wait(&mut SpinThenYield::<1024>::default()) {
//!             assert_eq!(slot.meta().sequence, expected);
//!             assert_eq!(slot.payload(), &expected.to_le_bytes());
//!             slot.release_egress();
//!             expected += 1;
//!         }
//!         assert_eq!(expected, 10_000);
//!     });
//! });
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, missing_debug_implementations)]

/// Compiles and runs the README's examples under `cargo test`, so they cannot drift.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

#[cfg(not(unix))]
compile_error!("vortex requires a POSIX virtual-memory interface (mmap / mprotect / mlock / madvise)");

use core::fmt;
use core::mem::{align_of, offset_of, size_of};
use core::ptr::NonNull;
use core::slice;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// Hardware constants & compile-time layout verification
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// Coherence isolation granule. 64 B lines are fetched in adjacent pairs by the L2 spatial
/// prefetcher on Intel (and Apple/Neoverse parts use 128 B lines outright), so every
/// independently-written object is isolated to its own 128 B-aligned granule.
pub const COHERENCE_GRANULE: usize = 128;

/// Alignment and stride quantum for payload frames: the NVMe/GDS/RDMA DMA page.
pub const DMA_ALIGN: usize = 4096;

/// `b"VORTEX01"` little-endian; first word of the read-only geometry block.
pub const GEOMETRY_MAGIC: u64 = u64::from_le_bytes(*b"VORTEX01");

/// `true` iff `[offset, offset + size)` lies entirely within a single coherence granule.
const fn confined_to_granule(offset: usize, size: usize) -> bool {
    size != 0
        && size <= COHERENCE_GRANULE
        && offset / COHERENCE_GRANULE == (offset + size - 1) / COHERENCE_GRANULE
}

/// `true` iff the two byte ranges share no coherence granule.
const fn granules_disjoint(a_off: usize, a_len: usize, b_off: usize, b_len: usize) -> bool {
    let a_first = a_off / COHERENCE_GRANULE;
    let a_last = (a_off + a_len - 1) / COHERENCE_GRANULE;
    let b_first = b_off / COHERENCE_GRANULE;
    let b_last = (b_off + b_len - 1) / COHERENCE_GRANULE;
    a_last < b_first || b_last < a_first
}

const fn align_up(value: usize, align: usize) -> Option<usize> {
    match value.checked_add(align - 1) {
        Some(v) => Some(v & !(align - 1)),
        None => None,
    }
}

const CURSOR_PAD: usize = COHERENCE_GRANULE - size_of::<AtomicU64>() - size_of::<AtomicBool>();

/// A ring cursor isolated to a private coherence granule, co-located with its owner's
/// disconnect flag (written once, at handle drop, so it adds no steady-state traffic).
#[repr(C, align(128))]
struct CursorLine {
    value: AtomicU64,
    closed: AtomicBool,
    _pad: [u8; CURSOR_PAD],
}

impl CursorLine {
    const fn new() -> Self {
        Self { value: AtomicU64::new(0), closed: AtomicBool::new(false), _pad: [0; CURSOR_PAD] }
    }
}

/// The turnstile: producer-owned `tail` and consumer-owned `head`, each on its own granule.
/// Each side writes exactly one granule and only reads the other's; the only coherence traffic
/// is the unavoidable publication transfer.
#[repr(C)]
struct Turnstile {
    tail: CursorLine,
    head: CursorLine,
}

impl Turnstile {
    const fn new() -> Self {
        Self { tail: CursorLine::new(), head: CursorLine::new() }
    }
}

/// Per-slot ingress metadata, stamped by the producer before publication.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SlotMeta {
    /// Monotonic ring position at commit; equals the consumer's expected position.
    pub sequence: u64,
    /// Hardware timestamp (NIC PTP clock, TSC, `cntvct_el0`, …) supplied by the driver loop.
    pub hw_timestamp: u64,
    /// Valid payload bytes; invariant `len <= SLOT_BYTES`.
    pub len: u32,
    /// Driver-defined flags (queue id, checksum status, truncation, …).
    pub flags: u32,
}

/// Header wrapper forcing each slot's metadata onto a private granule so the producer stamping
/// slot `n + 1` never invalidates the line the consumer is reading for slot `n`.
#[repr(C, align(128))]
struct SlotHeader {
    meta: SlotMeta,
}

impl SlotHeader {
    const VACANT: Self = Self { meta: SlotMeta { sequence: 0, hw_timestamp: 0, len: 0, flags: 0 } };
}

/// Immutable arena control block, sealed `PROT_READ` after initialisation. Laid out as a fixed
/// C ABI so out-of-process or C/CUDA peers can validate a mapping.
#[repr(C, align(128))]
#[derive(Debug, PartialEq, Eq)]
pub struct Geometry {
    /// [`GEOMETRY_MAGIC`].
    pub magic: u64,
    /// Slot count (power of two).
    pub slots: u64,
    /// Bytes per payload frame (multiple of [`DMA_ALIGN`]).
    pub slot_bytes: u64,
    /// OS base page size at map time.
    pub page_size: u64,
    /// Virtual address of payload frame 0.
    pub pool_base: u64,
    /// Total payload pool bytes.
    pub pool_bytes: u64,
    /// Virtual address of the slot header array.
    pub headers_base: u64,
    /// Virtual address of the descriptor array.
    pub descriptors_base: u64,
}

/// Immutable slot descriptor. ABI-identical to `struct iovec`, so the descriptor table can be
/// handed verbatim to `IORING_REGISTER_BUFFERS` (buffer index == slot index), `ibv_reg_mr`
/// loops, or `readv`/`preadv2` scatter lists.
#[repr(transparent)]
pub struct SlotDescriptor(libc::iovec);

impl SlotDescriptor {
    /// Base pointer of the slot's payload frame.
    #[inline(always)]
    pub fn base(&self) -> *mut u8 {
        self.0.iov_base.cast()
    }

    /// Base address of the slot's payload frame.
    #[inline(always)]
    pub fn addr(&self) -> usize {
        self.0.iov_base as usize
    }

    /// Frame capacity in bytes.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.0.iov_len
    }

    /// The underlying `iovec`.
    #[inline(always)]
    pub fn as_iovec(&self) -> &libc::iovec {
        &self.0
    }
}

impl fmt::Debug for SlotDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlotDescriptor")
            .field("base", &format_args!("{:#x}", self.addr()))
            .field("capacity", &self.capacity())
            .finish()
    }
}

const _: () = {
    // Coherence granule and DMA quantum are compatible powers of two.
    assert!(COHERENCE_GRANULE.is_power_of_two());
    assert!(DMA_ALIGN.is_power_of_two());
    assert!(DMA_ALIGN % COHERENCE_GRANULE == 0);

    // Cursor atomics: naturally aligned (single-copy atomic), confined to one granule, and the
    // cursor line is exactly one granule wide.
    assert!(align_of::<AtomicU64>() == size_of::<AtomicU64>());
    assert!(size_of::<CursorLine>() == COHERENCE_GRANULE);
    assert!(align_of::<CursorLine>() == COHERENCE_GRANULE);
    assert!(offset_of!(CursorLine, value) % align_of::<AtomicU64>() == 0);
    assert!(confined_to_granule(offset_of!(CursorLine, value), size_of::<AtomicU64>()));
    assert!(confined_to_granule(offset_of!(CursorLine, closed), size_of::<AtomicBool>()));

    // tail and head: granule-aligned and never co-resident in a granule.
    let tail = offset_of!(Turnstile, tail) + offset_of!(CursorLine, value);
    let head = offset_of!(Turnstile, head) + offset_of!(CursorLine, value);
    assert!(offset_of!(Turnstile, tail) % COHERENCE_GRANULE == 0);
    assert!(offset_of!(Turnstile, head) % COHERENCE_GRANULE == 0);
    assert!(confined_to_granule(tail, size_of::<AtomicU64>()));
    assert!(confined_to_granule(head, size_of::<AtomicU64>()));
    assert!(granules_disjoint(tail, size_of::<AtomicU64>(), head, size_of::<AtomicU64>()));
    assert!(size_of::<Turnstile>() == 2 * COHERENCE_GRANULE);
    assert!(align_of::<Turnstile>() == COHERENCE_GRANULE);

    // Slot headers: exactly one granule each; metadata never straddles.
    assert!(size_of::<SlotHeader>() == COHERENCE_GRANULE);
    assert!(align_of::<SlotHeader>() == COHERENCE_GRANULE);
    assert!(confined_to_granule(offset_of!(SlotHeader, meta), size_of::<SlotMeta>()));
    assert!(size_of::<SlotMeta>() == 24);

    // Geometry: one granule; descriptors tile granules exactly and follow it without padding.
    assert!(size_of::<Geometry>() == COHERENCE_GRANULE);
    assert!(align_of::<Geometry>() == COHERENCE_GRANULE);
    assert!(COHERENCE_GRANULE % size_of::<SlotDescriptor>() == 0);
    assert!(size_of::<Geometry>() % align_of::<SlotDescriptor>() == 0);

    // Descriptor ⇔ iovec ABI identity.
    assert!(size_of::<SlotDescriptor>() == size_of::<libc::iovec>());
    assert!(align_of::<SlotDescriptor>() == align_of::<libc::iovec>());
    assert!(offset_of!(libc::iovec, iov_base) == 0);
    assert!(offset_of!(libc::iovec, iov_len) == size_of::<*mut libc::c_void>());

    // Handle structs sit on distinct threads' stacks, but are still granule-isolated so a
    // co-located embedding (e.g. both in one struct) cannot reintroduce false sharing.
    assert!(align_of::<Producer<'static, 1, DMA_ALIGN>>() == COHERENCE_GRANULE);
    assert!(align_of::<Consumer<'static, 1, DMA_ALIGN>>() == COHERENCE_GRANULE);
    assert!(size_of::<Producer<'static, 1, DMA_ALIGN>>() == COHERENCE_GRANULE);
    assert!(size_of::<Consumer<'static, 1, DMA_ALIGN>>() == COHERENCE_GRANULE);
};

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// Errors & reporting
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// Every fallible OS interaction and API contract violation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum VortexError {
    /// `sysconf(_SC_PAGESIZE)` failed or returned a non-power-of-two.
    PageSize {
        /// `errno` (0 if the call succeeded with a nonsensical value).
        errno: i32,
    },
    /// Arena geometry overflows the address space.
    GeometryOverflow,
    /// `mmap` failed.
    Map {
        /// `errno`.
        errno: i32,
        /// Requested reservation size.
        bytes: usize,
    },
    /// `munmap` of alignment slack failed.
    Trim {
        /// `errno`.
        errno: i32,
    },
    /// `mprotect` of a guard page or the read-only control region failed.
    Protect {
        /// `errno`.
        errno: i32,
    },
    /// `mlock` failed under [`ResidencyPolicy::RequireLock`], or failed for a reason other than
    /// the `RLIMIT_MEMLOCK` family under [`ResidencyPolicy::LockOrPrefault`].
    Lock {
        /// `errno`.
        errno: i32,
        /// Bytes that were to be locked.
        bytes: usize,
    },
    /// `MAP_HUGETLB` failed (empty `nr_hugepages` pool, unsupported size or platform) under
    /// [`PageBacking::HugeTlb`] with `fallback: false`.
    HugeTlb {
        /// `errno` (`ENOTSUP` off Linux).
        errno: i32,
    },
    /// `mbind(MPOL_BIND)` to the requested NUMA node failed.
    Bind {
        /// `errno` (`ENOTSUP` off Linux).
        errno: i32,
        /// Requested node.
        node: u32,
    },
    /// A commit declared more payload bytes than the frame holds.
    PayloadOverflow {
        /// Declared length.
        len: usize,
        /// Frame capacity.
        capacity: usize,
    },
    /// A burst commit named more slots than were reserved.
    BurstOverrun {
        /// Slots named in the commit.
        requested: usize,
        /// Slots held by the reservation.
        reserved: usize,
    },
}

impl VortexError {
    /// The OS `errno`, for OS-originated variants.
    pub fn errno(&self) -> Option<i32> {
        match *self {
            Self::PageSize { errno }
            | Self::Map { errno, .. }
            | Self::Trim { errno }
            | Self::Protect { errno }
            | Self::Lock { errno, .. }
            | Self::HugeTlb { errno }
            | Self::Bind { errno, .. } => Some(errno),
            Self::GeometryOverflow | Self::PayloadOverflow { .. } | Self::BurstOverrun { .. } => None,
        }
    }
}

impl fmt::Display for VortexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let os = |f: &mut fmt::Formatter<'_>, what: &str, errno: i32| {
            write!(f, "{what}: {} (errno {errno})", std::io::Error::from_raw_os_error(errno))
        };
        match *self {
            Self::PageSize { errno } => os(f, "sysconf(_SC_PAGESIZE)", errno),
            Self::GeometryOverflow => f.write_str("arena geometry overflows the address space"),
            Self::Map { errno, bytes } => os(f, &format!("mmap({bytes} bytes)"), errno),
            Self::Trim { errno } => os(f, "munmap(alignment slack)", errno),
            Self::Protect { errno } => os(f, "mprotect", errno),
            Self::Lock { errno, bytes } => os(f, &format!("mlock({bytes} bytes)"), errno),
            Self::HugeTlb { errno } => os(f, "mmap(MAP_HUGETLB)", errno),
            Self::Bind { errno, node } => os(f, &format!("mbind(MPOL_BIND, node {node})"), errno),
            Self::PayloadOverflow { len, capacity } => {
                write!(f, "payload length {len} exceeds frame capacity {capacity}")
            }
            Self::BurstOverrun { requested, reserved } => {
                write!(f, "burst commit of {requested} slots exceeds reservation of {reserved}")
            }
        }
    }
}

impl std::error::Error for VortexError {}

impl From<VortexError> for std::io::Error {
    fn from(e: VortexError) -> Self {
        match e.errno() {
            Some(errno) => std::io::Error::from_raw_os_error(errno),
            None => std::io::Error::new(std::io::ErrorKind::InvalidInput, e),
        }
    }
}

/// How the arena body is made resident before traffic starts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResidencyPolicy {
    /// `mlock`; on `EPERM`/`ENOMEM`/`EAGAIN` (RLIMIT_MEMLOCK, unprivileged containers) fall back
    /// to write-faulting every page.
    #[default]
    LockOrPrefault,
    /// `mlock` or fail with [`VortexError::Lock`].
    RequireLock,
    /// Skip `mlock`; write-fault every page.
    PrefaultOnly,
}

/// Residency actually achieved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    /// Pages are wired: no major faults, no swap-out, no reclaim.
    Locked,
    /// Pages were write-faulted into private frames. No first-touch faults on the hot path, but
    /// the kernel may still reclaim under pressure.
    Prefaulted {
        /// The `mlock` errno that forced the fallback; `None` under [`ResidencyPolicy::PrefaultOnly`].
        mlock_errno: Option<i32>,
    },
}

/// Outcome of a best-effort `madvise`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Advice {
    /// The kernel accepted the advice.
    Applied,
    /// The kernel rejected the advice (e.g. `EINVAL` with THP compiled out).
    Rejected {
        /// `errno`.
        errno: i32,
    },
    /// Not available on this platform.
    Unsupported,
    /// Disabled by [`ArenaConfig`].
    Disabled,
}

/// Explicit hugetlbfs page size. Which sizes exist depends on the kernel's base page size:
///
/// | base page | contiguous-PTE | PMD      | contiguous-PMD / PUD |
/// |-----------|----------------|----------|----------------------|
/// | 4 KiB     | 64 KiB         | 2 MiB    | 32 MiB / 1 GiB       |
/// | 16 KiB    | 2 MiB          | 32 MiB   | 1 GiB                |
/// | 64 KiB    | 2 MiB          | 512 MiB  | 16 GiB               |
///
/// Use [`HugePageSize::system_default`] to pick whatever the running kernel is configured for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HugePageSize {
    shift: u8,
}

impl HugePageSize {
    /// 64 KiB.
    pub const SIZE_64K: Self = Self { shift: 16 };
    /// 2 MiB.
    pub const SIZE_2M: Self = Self { shift: 21 };
    /// 32 MiB.
    pub const SIZE_32M: Self = Self { shift: 25 };
    /// 512 MiB.
    pub const SIZE_512M: Self = Self { shift: 29 };
    /// 1 GiB.
    pub const SIZE_1G: Self = Self { shift: 30 };

    /// A power-of-two size of at least 4 KiB, representable in `usize`.
    pub const fn from_bytes(bytes: usize) -> Option<Self> {
        if bytes < 4096 || !bytes.is_power_of_two() {
            return None;
        }
        Some(Self { shift: bytes.trailing_zeros() as u8 })
    }

    /// The kernel's default hugetlb size (`Hugepagesize:` in `/proc/meminfo`). `None` off Linux
    /// or on kernels without hugetlbfs.
    pub fn system_default() -> Option<Self> {
        #[cfg(target_os = "linux")]
        {
            let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
            let line = meminfo.lines().find(|l| l.starts_with("Hugepagesize:"))?;
            let kib: usize = line.split_whitespace().nth(1)?.parse().ok()?;
            Self::from_bytes(kib.checked_mul(1024)?)
        }
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }

    /// log2 of the page size, as encoded into `mmap` flags (`MAP_HUGE_SHIFT`).
    pub const fn shift(self) -> u32 {
        self.shift as u32
    }

    /// Page size in bytes.
    pub const fn bytes(self) -> usize {
        1 << self.shift
    }
}

/// Page backing requested for the payload pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PageBacking {
    /// Base pages, no huge-page advice.
    Base,
    /// Body aligned to the PMD size and `madvise(MADV_HUGEPAGE)` (Linux). Best effort: khugepaged
    /// and fragmentation decide whether the fault is actually served from a huge page.
    #[default]
    Transparent,
    /// Pool mapped `MAP_HUGETLB` from the preallocated hugetlbfs pool (Linux,
    /// `vm.nr_hugepages` / `hugepages-1048576kB/nr_hugepages`). Guaranteed huge TLB entries and
    /// physically contiguous per page, which also shrinks IOMMU/MTT footprints for RDMA and GPU
    /// registration.
    HugeTlb {
        /// Page size.
        size: HugePageSize,
        /// On failure, fall back to [`PageBacking::Transparent`] instead of failing the map.
        fallback: bool,
    },
}

/// Arena construction knobs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaConfig {
    /// Residency strategy.
    pub residency: ResidencyPolicy,
    /// Payload pool page backing.
    pub backing: PageBacking,
    /// `madvise(MADV_DONTFORK)` the body so a `fork()` elsewhere in the process cannot COW-split
    /// frames that are pinned by an RDMA MR or a GPU registration (Linux).
    pub dont_fork: bool,
    /// Bind the body to this NUMA node with `mbind(MPOL_BIND)` before first touch (Linux). Pick
    /// the node local to the NIC / GPU (`/sys/class/net/<if>/device/numa_node`) so DMA and the
    /// consuming core never cross the socket interconnect.
    pub numa_node: Option<u32>,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            residency: ResidencyPolicy::LockOrPrefault,
            backing: PageBacking::Transparent,
            dont_fork: true,
            numa_node: None,
        }
    }
}

/// Page backing actually achieved for the payload pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backing {
    /// Base pages.
    Base,
    /// THP-eligible base mapping.
    Transparent {
        /// `MADV_HUGEPAGE` outcome.
        advice: Advice,
        /// The `MAP_HUGETLB` errno if this is a fallback from [`PageBacking::HugeTlb`].
        hugetlb_errno: Option<i32>,
    },
    /// hugetlbfs pages.
    HugeTlb {
        /// Page size.
        size: HugePageSize,
    },
}

/// What the kernel actually granted at map time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaReport {
    /// OS base page size.
    pub page_size: usize,
    /// Alignment of the body start.
    pub body_alignment: usize,
    /// Bytes mapped including both guard pages.
    pub mapped_bytes: usize,
    /// Bytes of the body (pool + turnstile + headers + control block + descriptors).
    pub body_bytes: usize,
    /// Residency outcome.
    pub residency: Residency,
    /// Payload pool backing.
    pub backing: Backing,
    /// `MADV_DONTFORK` outcome.
    pub dont_fork: Advice,
    /// NUMA node the body is bound to.
    pub numa_node: Option<u32>,
}

/// A raw, contiguous, DMA-aligned span of the arena, for `cuMemHostRegister`,
/// `cudaHostRegister`, `ibv_reg_mr`, `cuFileBufRegister`, or io_uring fixed buffers.
///
/// This is an address, not a borrow: it confers no access rights. Dereferencing it (from Rust or
/// from a device) is only sound while the owning arena is alive and while the slot(s) it covers
/// are held by the party performing the access.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DmaRegion {
    base: NonNull<u8>,
    len: usize,
}

// SAFETY: a DmaRegion is an inert (address, length) pair; every dereference is `unsafe` and
// carries its own contract.
unsafe impl Send for DmaRegion {}
// SAFETY: as above.
unsafe impl Sync for DmaRegion {}

impl DmaRegion {
    /// Base pointer.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const u8 {
        self.base.as_ptr()
    }

    /// Base pointer, mutable, for device write registration.
    #[inline(always)]
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.base.as_ptr()
    }

    /// Base address as `uintptr_t`.
    #[inline(always)]
    pub fn addr(&self) -> usize {
        self.base.as_ptr() as usize
    }

    /// Length in bytes.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if the span is zero bytes long.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl fmt::Debug for DmaRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DmaRegion")
            .field("addr", &format_args!("{:#x}", self.addr()))
            .field("len", &self.len)
            .finish()
    }
}

/// Stamp applied to one slot of an [`IngressBurst`] at commit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SlotStamp {
    /// Valid payload bytes.
    pub len: usize,
    /// Hardware timestamp.
    pub hw_timestamp: u64,
    /// Driver-defined flags.
    pub flags: u32,
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// Hardware clock
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// Invariant hardware counter for stamping when the NIC supplies no timestamp:
/// `RDTSC` on x86_64, `ISB; MRS CNTVCT_EL0` on aarch64, `CLOCK_MONOTONIC` ns elsewhere.
/// Units are counter ticks, not nanoseconds, on the first two.
#[inline(always)]
pub fn hw_ticks() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        #[allow(unused_unsafe)]
        // SAFETY: RDTSC is unprivileged unless CR4.TSD is set, which no mainstream OS does.
        unsafe {
            core::arch::x86_64::_rdtsc()
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let ticks: u64;
        // SAFETY: CNTVCT_EL0 is EL0-readable on Linux and Darwin (CNTKCTL_EL1.EL0VCTEN = 1). The
        // ISB prevents the read from being speculated ahead of preceding instructions.
        unsafe {
            core::arch::asm!(
                "isb",
                "mrs {ticks}, cntvct_el0",
                ticks = out(reg) ticks,
                options(nomem, nostack, preserves_flags),
            );
        }
        ticks
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: `ts` is a valid, exclusively borrowed out-parameter.
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
        (ts.tv_sec as u64).wrapping_mul(1_000_000_000).wrapping_add(ts.tv_nsec as u64)
    }
}

/// Frequency of [`hw_ticks`] in Hz, resolved once (~20 ms, sleeping) and cached. Never zero.
///
/// On aarch64 `CNTFRQ_EL0` is used when it agrees with a `CLOCK_MONOTONIC` measurement to
/// within 2%; the register is firmware-programmed and is zero or wrong on some boards and
/// hypervisors (Linux then takes the rate from the device tree instead). On x86_64 the
/// invariant TSC is always measured.
pub fn hw_tick_hz() -> u64 {
    static HZ: AtomicU64 = AtomicU64::new(0);
    let cached = HZ.load(Ordering::Relaxed);
    if cached != 0 {
        return cached;
    }
    let hz = resolve_tick_hz().max(1);
    HZ.store(hz, Ordering::Relaxed);
    hz
}

fn resolve_tick_hz() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        measure_tick_hz()
    }
    #[cfg(target_arch = "aarch64")]
    {
        let reported: u64;
        // SAFETY: CNTFRQ_EL0 is EL0-readable wherever CNTVCT_EL0 is.
        unsafe {
            core::arch::asm!(
                "mrs {hz}, cntfrq_el0",
                hz = out(reg) reported,
                options(nomem, nostack, preserves_flags),
            );
        }
        let measured = measure_tick_hz();
        if reported != 0 && reported.abs_diff(measured) <= measured / 50 {
            reported
        } else {
            measured
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        1_000_000_000
    }
}

/// Ticks per second against `CLOCK_MONOTONIC`. Preemption during the sleep does not bias the
/// ratio; preemption *between* the two clock reads would (badly, on a contended single CPU),
/// so each endpoint is the tightest of several tick-bracketed samples.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn measure_tick_hz() -> u64 {
    fn paired_sample() -> (std::time::Instant, u64) {
        let mut best = (std::time::Instant::now(), 0u64, u64::MAX);
        for _ in 0..16 {
            let before = hw_ticks();
            let now = std::time::Instant::now();
            let after = hw_ticks();
            let gap = after.wrapping_sub(before);
            if gap < best.2 {
                best = (now, before + gap / 2, gap);
            }
        }
        (best.0, best.1)
    }
    let (t0, c0) = paired_sample();
    std::thread::sleep(std::time::Duration::from_millis(20));
    let (t1, c1) = paired_sample();
    let ns = t1.duration_since(t0).as_nanos().max(1);
    (c1.wrapping_sub(c0) as u128 * 1_000_000_000 / ns) as u64
}

/// Convert a [`hw_ticks`] delta to nanoseconds.
#[inline]
pub fn ticks_to_nanos(ticks: u64) -> u64 {
    (ticks as u128 * 1_000_000_000 / hw_tick_hz() as u128) as u64
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// Wait strategies (statically dispatched)
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// Back-off policy for the `*_wait` APIs. `reset` is called at the start of every wait and
/// `wait` after each unsuccessful poll; returning `false` abandons the wait. Monomorphised into
/// the caller: no vtable, no indirect branch.
pub trait WaitStrategy {
    /// Start a new wait. Budgets are per wait, so one strategy value can be reused across calls.
    #[inline(always)]
    fn reset(&mut self) {}

    /// Pause once. Return `false` to give up.
    fn wait(&mut self) -> bool;
}

/// Pure `PAUSE` / `YIELD` spin. Lowest latency; owns the core. Never gives up (the wait still
/// ends when the peer handle is dropped).
#[derive(Clone, Copy, Debug, Default)]
pub struct BusySpin;

impl WaitStrategy for BusySpin {
    #[inline(always)]
    fn wait(&mut self) -> bool {
        core::hint::spin_loop();
        true
    }
}

/// Spin `SPINS` times per wait, then `sched_yield` on every subsequent poll. Never gives up.
/// The portable default: near-`BusySpin` latency when each side owns a core, and forward
/// progress when producer and consumer share one (small VMs, CPU-quota'd containers), where a
/// pure spin burns its whole timeslice before the peer can run.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpinThenYield<const SPINS: u32> {
    spins: u32,
}

impl<const SPINS: u32> WaitStrategy for SpinThenYield<SPINS> {
    #[inline(always)]
    fn reset(&mut self) {
        self.spins = 0;
    }

    #[inline(always)]
    fn wait(&mut self) -> bool {
        if self.spins < SPINS {
            self.spins += 1;
            core::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
        true
    }
}

/// Spin at most `ATTEMPTS` times per wait, then give up.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bounded<const ATTEMPTS: u32> {
    attempts: u32,
}

impl<const ATTEMPTS: u32> WaitStrategy for Bounded<ATTEMPTS> {
    #[inline(always)]
    fn reset(&mut self) {
        self.attempts = 0;
    }

    #[inline(always)]
    fn wait(&mut self) -> bool {
        self.attempts += 1;
        core::hint::spin_loop();
        self.attempts < ATTEMPTS
    }
}

/// Any `FnMut() -> bool` (deadline checks, shutdown flags, …) is a wait strategy.
impl<F: FnMut() -> bool> WaitStrategy for F {
    #[inline(always)]
    fn wait(&mut self) -> bool {
        self()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// Layers 1 & 2: OS mapping primitives
// ═══════════════════════════════════════════════════════════════════════════════════════════════

#[inline]
fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

fn page_size() -> Result<usize, VortexError> {
    // SAFETY: sysconf has no memory-safety preconditions.
    let rc = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if rc <= 0 {
        return Err(VortexError::PageSize { errno: if rc < 0 { errno() } else { 0 } });
    }
    let page = rc as usize;
    if !page.is_power_of_two() {
        return Err(VortexError::PageSize { errno: 0 });
    }
    Ok(page)
}

/// An owned anonymous address-space reservation, released with `munmap` on drop.
struct RawMapping {
    base: NonNull<u8>,
    len: usize,
}

impl RawMapping {
    /// Reserve `len` bytes of `PROT_NONE` address space. Nothing is committed or charged until
    /// sub-ranges are opened with `mprotect` / `MAP_FIXED`; untouched ranges act as guards.
    fn reserve(len: usize) -> Result<Self, VortexError> {
        // SAFETY: fresh anonymous private mapping at a kernel-chosen address; aliases nothing.
        let p = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(VortexError::Map { errno: errno(), bytes: len });
        }
        match NonNull::new(p.cast::<u8>()) {
            Some(base) => Ok(Self { base, len }),
            None => {
                // A mapping at address 0 is legal but unusable as a NonNull base; hand it back.
                // SAFETY: we own exactly this range.
                unsafe { libc::munmap(p, len) };
                Err(VortexError::Map { errno: libc::EFAULT, bytes: len })
            }
        }
    }

    /// Shrink the mapping to `[base + head, base + head + keep)`, unmapping the slack on both
    /// sides. `self` always describes exactly what is still mapped, including on error.
    fn trim(&mut self, head: usize, keep: usize) -> Result<(), VortexError> {
        assert!(head.checked_add(keep).is_some_and(|end| end <= self.len));
        let tail = self.len - head - keep;
        if head != 0 {
            // SAFETY: `[base, base + head)` is a prefix of our mapping.
            if unsafe { libc::munmap(self.base.as_ptr().cast(), head) } != 0 {
                return Err(VortexError::Trim { errno: errno() });
            }
            // SAFETY: `head < len`, so the result is inside the (formerly) mapped range.
            self.base = unsafe { self.base.add(head) };
            self.len -= head;
        }
        if tail != 0 {
            // SAFETY: `[base + keep, base + keep + tail)` is a suffix of our mapping.
            if unsafe { libc::munmap(self.base.as_ptr().add(keep).cast(), tail) } != 0 {
                return Err(VortexError::Trim { errno: errno() });
            }
            self.len = keep;
        }
        Ok(())
    }
}

impl Drop for RawMapping {
    fn drop(&mut self) {
        // SAFETY: `[base, base + len)` is exactly the range we still own; nothing borrows it
        // (every borrow is tied to the owning arena's lifetime). munmap implicitly munlocks.
        let rc = unsafe { libc::munmap(self.base.as_ptr().cast(), self.len) };
        debug_assert_eq!(rc, 0, "munmap failed: errno {}", errno());
    }
}

/// # Safety
/// `[ptr, ptr + len)` must be page-aligned and lie inside a mapping owned by the caller, and no
/// live reference may be invalidated by the new protection.
unsafe fn protect(ptr: *mut u8, len: usize, prot: libc::c_int) -> Result<(), VortexError> {
    // SAFETY: forwarded to the caller.
    if unsafe { libc::mprotect(ptr.cast(), len, prot) } != 0 {
        return Err(VortexError::Protect { errno: errno() });
    }
    Ok(())
}

/// Replace `[ptr, ptr + len)` of our reservation with hugetlbfs pages. Returns the errno on
/// failure; the anonymous-hugetlb failure paths (pool reservation, size validation) run before
/// the kernel touches the existing range, and the caller re-reserves it regardless.
///
/// # Safety
/// `[ptr, ptr + len)` must be aligned to `size`, lie inside a reservation owned by the caller,
/// and be unreferenced.
unsafe fn map_hugetlb_fixed(ptr: *mut u8, len: usize, size: HugePageSize) -> Result<(), i32> {
    #[cfg(target_os = "linux")]
    {
        // Encoded locally: libc only exports MAP_HUGE_{2MB,1GB} for some Linux libcs.
        const MAP_HUGE_SHIFT: libc::c_int = 26;
        let flags = libc::MAP_PRIVATE
            | libc::MAP_ANONYMOUS
            | libc::MAP_FIXED
            | libc::MAP_HUGETLB
            | ((size.shift() as libc::c_int) << MAP_HUGE_SHIFT);
        // SAFETY: MAP_FIXED over a range we own and nothing references, per the contract.
        let p = unsafe {
            libc::mmap(ptr.cast(), len, libc::PROT_READ | libc::PROT_WRITE, flags, -1, 0)
        };
        if p == libc::MAP_FAILED {
            return Err(errno());
        }
        debug_assert_eq!(p.cast::<u8>(), ptr);
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptr, len, size);
        Err(libc::ENOTSUP)
    }
}

/// Return `[ptr, ptr + len)` to a plain `PROT_NONE` reservation.
///
/// # Safety
/// As [`map_hugetlb_fixed`] with page alignment.
unsafe fn rereserve_fixed(ptr: *mut u8, len: usize) -> Result<(), VortexError> {
    // SAFETY: MAP_FIXED over a range we own and nothing references, per the contract.
    let p = unsafe {
        libc::mmap(
            ptr.cast(),
            len,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED,
            -1,
            0,
        )
    };
    if p == libc::MAP_FAILED {
        return Err(VortexError::Map { errno: errno(), bytes: len });
    }
    Ok(())
}

/// Bind `[ptr, ptr + len)` to `node` before first touch, so every subsequent fault (base, THP or
/// hugetlb) allocates node-local frames. `MPOL_MF_STRICT | MPOL_MF_MOVE` also migrates anything
/// already resident and fails if it cannot.
///
/// # Safety
/// `[ptr, ptr + len)` must be page-aligned and inside a mapping owned by the caller.
unsafe fn bind_node(ptr: *mut u8, len: usize, node: u32) -> Result<(), VortexError> {
    #[cfg(target_os = "linux")]
    {
        const MPOL_MF_STRICT: libc::c_ulong = 1 << 0;
        const MPOL_MF_MOVE: libc::c_ulong = 1 << 1;
        const WORD_BITS: usize = libc::c_ulong::BITS as usize;
        const MASK_WORDS: usize = 1024 / WORD_BITS;
        let n = node as usize;
        if n >= MASK_WORDS * WORD_BITS {
            return Err(VortexError::Bind { errno: libc::EINVAL, node });
        }
        let mut mask = [0 as libc::c_ulong; MASK_WORDS];
        mask[n / WORD_BITS] |= 1 << (n % WORD_BITS);
        // SAFETY: range owned per the contract; `mask` outlives the call. The kernel reads
        // `maxnode - 1` bits, hence the +1.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_mbind,
                ptr.cast::<libc::c_void>(),
                len as libc::c_ulong,
                libc::MPOL_BIND as libc::c_ulong,
                mask.as_ptr(),
                (MASK_WORDS * WORD_BITS + 1) as libc::c_ulong,
                MPOL_MF_STRICT | MPOL_MF_MOVE,
            )
        };
        if rc != 0 {
            return Err(VortexError::Bind { errno: errno(), node });
        }
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptr, len);
        Err(VortexError::Bind { errno: libc::ENOTSUP, node })
    }
}

/// # Safety
/// `[ptr, ptr + len)` must be page-aligned and inside a mapping owned by the caller.
#[cfg(target_os = "linux")]
unsafe fn advise(ptr: *mut u8, len: usize, advice: libc::c_int) -> Advice {
    // SAFETY: forwarded to the caller.
    if unsafe { libc::madvise(ptr.cast(), len, advice) } == 0 {
        Advice::Applied
    } else {
        Advice::Rejected { errno: errno() }
    }
}

/// # Safety
/// As [`advise`].
unsafe fn advise_huge_pages(ptr: *mut u8, len: usize) -> Advice {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: forwarded to the caller.
        unsafe { advise(ptr, len, libc::MADV_HUGEPAGE) }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptr, len);
        Advice::Unsupported
    }
}

/// # Safety
/// As [`advise`].
unsafe fn advise_dont_fork(ptr: *mut u8, len: usize, enabled: bool) -> Advice {
    if !enabled {
        return Advice::Disabled;
    }
    #[cfg(target_os = "linux")]
    {
        // SAFETY: forwarded to the caller.
        unsafe { advise(ptr, len, libc::MADV_DONTFORK) }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptr, len);
        Advice::Unsupported
    }
}

/// Force a private physical frame behind every page. A read would only map the shared zero page
/// (and a later write would still take a COW fault), so each page takes one volatile store.
/// Anonymous memory is already zero, so storing 0 is content-neutral.
///
/// # Safety
/// `[ptr, ptr + len)` must be writable, owned by the caller, and not yet hold live data.
unsafe fn prefault(ptr: *mut u8, len: usize, page: usize) {
    let mut off = 0;
    while off < len {
        // SAFETY: `off < len`; the range is writable and owned per the contract.
        unsafe { ptr.add(off).write_volatile(0) };
        off += page;
    }
}

/// # Safety
/// As [`prefault`].
unsafe fn establish_residency(
    ptr: *mut u8,
    len: usize,
    page: usize,
    policy: ResidencyPolicy,
) -> Result<Residency, VortexError> {
    if policy == ResidencyPolicy::PrefaultOnly {
        // SAFETY: forwarded to the caller.
        unsafe { prefault(ptr, len, page) };
        return Ok(Residency::Prefaulted { mlock_errno: None });
    }
    // SAFETY: range owned per the contract; mlock faults in and wires every page.
    if unsafe { libc::mlock(ptr.cast(), len) } == 0 {
        return Ok(Residency::Locked);
    }
    let e = errno();
    let rlimit_class = e == libc::EPERM || e == libc::ENOMEM || e == libc::EAGAIN;
    if policy == ResidencyPolicy::LockOrPrefault && rlimit_class {
        // SAFETY: forwarded to the caller.
        unsafe { prefault(ptr, len, page) };
        return Ok(Residency::Prefaulted { mlock_errno: Some(e) });
    }
    Err(VortexError::Lock { errno: e, bytes: len })
}

/// Byte offsets of each region within the body. Page granularity is a runtime quantity
/// (4 KiB on x86_64, 16 KiB on Apple Silicon, up to 64 KiB on some aarch64 kernels).
#[derive(Clone, Copy, Debug)]
struct ArenaLayout {
    pool_len: usize,
    /// Pool rounded to its backing page size; also the start of the metadata region.
    turnstile_off: usize,
    headers_off: usize,
    geometry_off: usize,
    descriptors_off: usize,
    ro_len: usize,
    body_len: usize,
}

impl ArenaLayout {
    /// `pool_page` is the page size backing the pool (base page, or the hugetlb size); the pool
    /// is rounded to it so the metadata region never shares a hugetlb page.
    fn compute(
        slots: usize,
        slot_bytes: usize,
        page: usize,
        pool_page: usize,
    ) -> Result<Self, VortexError> {
        const OVF: VortexError = VortexError::GeometryOverflow;
        debug_assert!(pool_page >= page && pool_page % page == 0);
        let pool_len = slots.checked_mul(slot_bytes).ok_or(OVF)?;
        let turnstile_off = align_up(pool_len, pool_page).ok_or(OVF)?;
        let headers_off = turnstile_off.checked_add(size_of::<Turnstile>()).ok_or(OVF)?;
        let headers_len = slots.checked_mul(size_of::<SlotHeader>()).ok_or(OVF)?;
        let rw_end = align_up(headers_off.checked_add(headers_len).ok_or(OVF)?, page).ok_or(OVF)?;
        let geometry_off = rw_end;
        let descriptors_off = geometry_off.checked_add(size_of::<Geometry>()).ok_or(OVF)?;
        let descriptors_len = slots.checked_mul(size_of::<SlotDescriptor>()).ok_or(OVF)?;
        let body_len =
            align_up(descriptors_off.checked_add(descriptors_len).ok_or(OVF)?, page).ok_or(OVF)?;
        if body_len > isize::MAX as usize / 2 {
            return Err(OVF);
        }
        debug_assert!(turnstile_off % COHERENCE_GRANULE == 0);
        debug_assert!(headers_off % COHERENCE_GRANULE == 0);
        debug_assert!(geometry_off % page == 0);
        Ok(Self {
            pool_len,
            turnstile_off,
            headers_off,
            geometry_off,
            descriptors_off,
            ro_len: body_len - geometry_off,
            body_len,
        })
    }
}

/// PMD-level THP size for a given base page: one page-table page of 8-byte PTEs maps
/// `page / 8` pages, i.e. 2 MiB on 4 KiB kernels, 32 MiB on 16 KiB, 512 MiB on 64 KiB (arm64).
#[cfg(target_os = "linux")]
const fn thp_pmd_size(page: usize) -> usize {
    page * (page / 8)
}

fn body_alignment(page: usize, backing: PageBacking, body_len: usize) -> usize {
    let base = page.max(DMA_ALIGN);
    match backing {
        PageBacking::Base => base,
        // A body smaller than one PMD can never be THP-mapped; aligning it would only burn
        // address space (512 MiB of slack per arena on 64 KiB-page arm64).
        #[cfg(target_os = "linux")]
        PageBacking::Transparent if body_len >= thp_pmd_size(page) => base.max(thp_pmd_size(page)),
        PageBacking::Transparent => base,
        // hugetlb pages must be naturally aligned; a THP fallback inherits this alignment.
        #[cfg(target_os = "linux")]
        PageBacking::HugeTlb { size, .. } => base.max(size.bytes()),
        #[cfg(not(target_os = "linux"))]
        PageBacking::HugeTlb { .. } => {
            let _ = body_len;
            base
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// Layer 2: VortexArena
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// Off-heap arena: one anonymous private mapping carved into a DMA-aligned payload pool, the
/// turnstile, per-slot headers, and a `PROT_READ`-sealed control block + descriptor table,
/// bracketed by `PROT_NONE` guard pages.
///
/// The arena exposes only read-only and address-level views; slot contents are reachable solely
/// through [`Producer`] / [`Consumer`] handles obtained from [`VortexRing::split`].
pub struct VortexArena<const SLOTS: usize, const SLOT_BYTES: usize> {
    pool: NonNull<u8>,
    turnstile: NonNull<Turnstile>,
    headers: NonNull<SlotHeader>,
    geometry: NonNull<Geometry>,
    descriptors: NonNull<SlotDescriptor>,
    report: ArenaReport,
    /// Owns the address range; `RawMapping::drop` munmaps it after every other field is gone.
    _mapping: RawMapping,
}

// SAFETY: the arena owns its mapping outright (no thread affinity). Shared access is sound
// because (a) the turnstile is accessed only through atomics, (b) geometry and descriptors are
// immutable after construction and physically PROT_READ, and (c) payload frames and headers are
// only materialised as references through the SPSC protocol, under which slot `i` is owned by at
// most one of {producer, consumer} at any instant.
unsafe impl<const SLOTS: usize, const SLOT_BYTES: usize> Send for VortexArena<SLOTS, SLOT_BYTES> {}
// SAFETY: see above.
unsafe impl<const SLOTS: usize, const SLOT_BYTES: usize> Sync for VortexArena<SLOTS, SLOT_BYTES> {}

impl<const SLOTS: usize, const SLOT_BYTES: usize> VortexArena<SLOTS, SLOT_BYTES> {
    /// Monomorphisation-time geometry gate: an invalid `<SLOTS, SLOT_BYTES>` fails the build.
    const GEOMETRY_OK: () = {
        assert!(SLOTS != 0 && SLOTS.is_power_of_two(), "SLOTS must be a non-zero power of two");
        assert!(SLOT_BYTES != 0, "SLOT_BYTES must be non-zero");
        assert!(SLOT_BYTES % DMA_ALIGN == 0, "SLOT_BYTES must be a multiple of DMA_ALIGN (4096)");
        assert!(SLOT_BYTES <= u32::MAX as usize, "SLOT_BYTES must fit SlotMeta::len (u32)");
        assert!(SLOTS.checked_mul(SLOT_BYTES).is_some(), "SLOTS * SLOT_BYTES overflows usize");
        assert!(
            SLOTS.checked_mul(size_of::<SlotHeader>()).is_some(),
            "SLOTS * header stride overflows usize"
        );
        assert!((SLOTS as u128) <= (u64::MAX as u128), "SLOTS must be representable in u64");
    };

    /// Map, guard, advise, make resident, initialise, and seal an arena.
    pub fn map(config: ArenaConfig) -> Result<Self, VortexError> {
        #[allow(clippy::let_unit_value)]
        let () = Self::GEOMETRY_OK;

        const OVF: VortexError = VortexError::GeometryOverflow;
        let page = page_size()?;
        let base_layout = ArenaLayout::compute(SLOTS, SLOT_BYTES, page, page)?;
        let hugetlb = match config.backing {
            PageBacking::HugeTlb { size, fallback } => {
                let pool_page = size.bytes().max(page);
                Some((size, fallback, ArenaLayout::compute(SLOTS, SLOT_BYTES, page, pool_page)?))
            }
            PageBacking::Base | PageBacking::Transparent => None,
        };
        let reserve_layout = hugetlb.map_or(base_layout, |(_, _, l)| l);
        let align = body_alignment(page, config.backing, reserve_layout.body_len);

        // 1. Reserve PROT_NONE address space for [guard | body | guard], over-sized so the body
        //    can be aligned to `align`, then hand the slack back. Everything not explicitly
        //    committed below stays PROT_NONE and traps stray accesses.
        let span = reserve_layout.body_len.checked_add(2 * page).ok_or(OVF)?;
        let reserve = span.checked_add(align - page).ok_or(OVF)?;
        let mut mapping = RawMapping::reserve(reserve)?;
        let raw = mapping.base.as_ptr() as usize;
        let body_addr = align_up(raw + page, align).ok_or(OVF)?;
        mapping.trim(body_addr - page - raw, span)?;

        // SAFETY: after trim the mapping is exactly [guard | body | guard].
        let body = unsafe { mapping.base.add(page) };
        debug_assert_eq!(body.as_ptr() as usize % align, 0);
        let body_ptr = body.as_ptr();

        // 2. Pool backing.
        let (layout, mut backing) = match hugetlb {
            Some((size, fallback, huge_layout)) => {
                // SAFETY: body is `size`-aligned (body_alignment), the pool region is a multiple
                // of `size` (ArenaLayout), and nothing references the reservation.
                match unsafe { map_hugetlb_fixed(body_ptr, huge_layout.turnstile_off, size) } {
                    Ok(()) => (huge_layout, Backing::HugeTlb { size }),
                    Err(e) => {
                        // Restore the reservation before deciding, so a hole can never be
                        // claimed by a concurrent mmap and later munmapped by us.
                        // SAFETY: as above.
                        unsafe { rereserve_fixed(body_ptr, huge_layout.turnstile_off)? };
                        if !fallback {
                            return Err(VortexError::HugeTlb { errno: e });
                        }
                        let backing =
                            Backing::Transparent { advice: Advice::Disabled, hugetlb_errno: Some(e) };
                        (base_layout, backing)
                    }
                }
            }
            None if config.backing == PageBacking::Base => (base_layout, Backing::Base),
            None => {
                (base_layout, Backing::Transparent { advice: Advice::Disabled, hugetlb_errno: None })
            }
        };
        let hugetlb_pool = match backing {
            Backing::HugeTlb { size } => Some(size.bytes()),
            Backing::Base | Backing::Transparent { .. } => None,
        };

        // 3. Commit everything not already backed by hugetlbfs.
        let rw_off = if hugetlb_pool.is_some() { layout.turnstile_off } else { 0 };
        // SAFETY: page-aligned sub-range of our reservation; nothing references it.
        unsafe {
            protect(body_ptr.add(rw_off), layout.body_len - rw_off, libc::PROT_READ | libc::PROT_WRITE)?
        };

        // 4. Placement policy precedes first touch so the initial faults are served from huge,
        //    node-local frames.
        if let Backing::Transparent { advice, .. } = &mut backing {
            // SAFETY: committed, page-aligned, owned.
            *advice = unsafe { advise_huge_pages(body_ptr, layout.body_len) };
        }
        if let Some(node) = config.numa_node {
            // SAFETY: as above.
            unsafe { bind_node(body_ptr, layout.body_len, node)? };
        }
        // SAFETY: as above.
        let dont_fork = unsafe { advise_dont_fork(body_ptr, layout.body_len, config.dont_fork) };

        // 5. Residency precedes initialisation so prefault stores cannot clobber live data.
        //    Linux mlock silently skips hugetlb VMAs (and they are never swapped anyway), so a
        //    hugetlb pool is faulted explicitly — otherwise the first packet into each page
        //    would pay for allocating and zeroing up to 1 GiB. Only the base-page remainder is
        //    subject to the residency policy, which also keeps a large hugetlb pool from
        //    tripping RLIMIT_MEMLOCK.
        // SAFETY: committed, owned, and holding no data yet.
        let residency = unsafe {
            if let Some(huge) = hugetlb_pool {
                prefault(body_ptr, layout.turnstile_off, huge);
            }
            establish_residency(
                body_ptr.add(rw_off),
                layout.body_len - rw_off,
                page,
                config.residency,
            )?
        };

        // SAFETY: every offset below comes from `layout`, lies within the body, and is aligned
        // for its target type (checked by the const block and the debug assertions in
        // `ArenaLayout::compute`).
        let (turnstile, headers, geometry, descriptors) = unsafe {
            (
                body.add(layout.turnstile_off).cast::<Turnstile>(),
                body.add(layout.headers_off).cast::<SlotHeader>(),
                body.add(layout.geometry_off).cast::<Geometry>(),
                body.add(layout.descriptors_off).cast::<SlotDescriptor>(),
            )
        };

        // SAFETY: exclusive, RW, in-bounds, correctly aligned; no references exist yet.
        unsafe {
            turnstile.write(Turnstile::new());
            for i in 0..SLOTS {
                headers.add(i).write(SlotHeader::VACANT);
                descriptors.add(i).write(SlotDescriptor(libc::iovec {
                    iov_base: body_ptr.add(i * SLOT_BYTES).cast(),
                    iov_len: SLOT_BYTES,
                }));
            }
            geometry.write(Geometry {
                magic: GEOMETRY_MAGIC,
                slots: SLOTS as u64,
                slot_bytes: SLOT_BYTES as u64,
                page_size: page as u64,
                pool_base: body_ptr as usize as u64,
                pool_bytes: layout.pool_len as u64,
                headers_base: headers.as_ptr() as usize as u64,
                descriptors_base: descriptors.as_ptr() as usize as u64,
            });
            // Seal the control block and descriptor table.
            protect(body_ptr.add(layout.geometry_off), layout.ro_len, libc::PROT_READ)?;
        }

        let report = ArenaReport {
            page_size: page,
            body_alignment: align,
            mapped_bytes: mapping.len,
            body_bytes: layout.body_len,
            residency,
            backing,
            dont_fork,
            numa_node: config.numa_node,
        };

        Ok(Self { pool: body, turnstile, headers, geometry, descriptors, report, _mapping: mapping })
    }

    /// What the kernel granted.
    #[inline]
    pub fn report(&self) -> &ArenaReport {
        &self.report
    }

    /// The sealed control block.
    #[inline]
    pub fn geometry(&self) -> &Geometry {
        // SAFETY: initialised, immutable (PROT_READ), lives as long as `self`.
        unsafe { self.geometry.as_ref() }
    }

    /// The sealed slot descriptor table, indexed by slot.
    #[inline]
    pub fn descriptors(&self) -> &[SlotDescriptor] {
        // SAFETY: SLOTS initialised, immutable descriptors, lifetime bound to `self`.
        unsafe { slice::from_raw_parts(self.descriptors.as_ptr(), SLOTS) }
    }

    /// The descriptor table as an `iovec` array (e.g. for `IORING_REGISTER_BUFFERS`).
    #[inline]
    pub fn iovecs(&self) -> &[libc::iovec] {
        // SAFETY: SlotDescriptor is repr(transparent) over libc::iovec.
        unsafe { slice::from_raw_parts(self.descriptors.as_ptr().cast::<libc::iovec>(), SLOTS) }
    }

    /// The entire payload pool as one contiguous region, for single-shot registration
    /// (`cuMemHostRegister`, `ibv_reg_mr`) rather than per-slot.
    #[inline]
    pub fn dma_region(&self) -> DmaRegion {
        DmaRegion { base: self.pool, len: SLOTS * SLOT_BYTES }
    }

    /// Slot count.
    pub const SLOTS: usize = SLOTS;
    /// Payload frame capacity.
    pub const SLOT_BYTES: usize = SLOT_BYTES;

    #[inline(always)]
    const fn slot_of(pos: u64) -> usize {
        (pos as usize) & (SLOTS - 1)
    }

    #[inline(always)]
    fn turnstile(&self) -> &Turnstile {
        // SAFETY: initialised in `map`, atomics only, lives as long as `self`.
        unsafe { self.turnstile.as_ref() }
    }

    #[inline(always)]
    fn payload_ptr(&self, pos: u64) -> *mut u8 {
        // SAFETY: slot_of(pos) < SLOTS, so the frame lies within the pool.
        unsafe { self.pool.as_ptr().add(Self::slot_of(pos) * SLOT_BYTES) }
    }

    #[inline(always)]
    fn header_ptr(&self, pos: u64) -> *mut SlotHeader {
        // SAFETY: slot_of(pos) < SLOTS, so the header lies within the header array.
        unsafe { self.headers.as_ptr().add(Self::slot_of(pos)) }
    }

    /// # Safety
    /// The caller must own slot `pos` (producer side) and no reference to its header may exist.
    #[inline(always)]
    unsafe fn stamp(&self, pos: u64, len: usize, hw_timestamp: u64, flags: u32) {
        debug_assert!(len <= SLOT_BYTES);
        let meta = SlotMeta { sequence: pos, hw_timestamp, len: len as u32, flags };
        // SAFETY: header in bounds; exclusivity per the contract.
        unsafe { core::ptr::addr_of_mut!((*self.header_ptr(pos)).meta).write(meta) };
    }

    /// # Safety
    /// The caller must own slot `pos` (consumer side) for the returned lifetime.
    #[inline(always)]
    unsafe fn meta(&self, pos: u64) -> &SlotMeta {
        // SAFETY: header in bounds, initialised, and not written while the consumer owns it.
        unsafe { &(*self.header_ptr(pos)).meta }
    }

    /// Committed length of slot `pos`, clamped to the frame (defence in depth; commit already
    /// enforces `len <= SLOT_BYTES`).
    ///
    /// # Safety
    /// As [`Self::meta`].
    #[inline(always)]
    unsafe fn committed_len(&self, pos: u64) -> usize {
        // SAFETY: forwarded.
        (unsafe { self.meta(pos) }.len as usize).min(SLOT_BYTES)
    }
}

impl<const SLOTS: usize, const SLOT_BYTES: usize> fmt::Debug for VortexArena<SLOTS, SLOT_BYTES> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VortexArena")
            .field("geometry", self.geometry())
            .field("report", &self.report)
            .finish_non_exhaustive()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// Layer 3: VortexRing — SPSC turnstile
// ═══════════════════════════════════════════════════════════════════════════════════════════════
//
// Cursors are unbounded u64 positions (2^64 slots ≈ 58 years at 10 Gpps); slot = pos & (SLOTS-1).
// Invariant: head <= tail <= head + SLOTS.
//
// Publication protocol:
//   producer: write payload + header (plain stores) ─► tail.store(pos+1, Release)
//   consumer: tail.load(Acquire) ─► read header + payload
//   consumer: read/modify payload (plain) ─► head.store(pos+1, Release)
//   producer: head.load(Acquire) ─► overwrite slot
//
// Each side keeps a private copy of its own cursor (never re-loads it) and a cached copy of the
// peer cursor, refreshed with an Acquire load only when the cache says full/empty. In steady
// state each side touches the peer granule once per SLOTS operations, not once per operation.

/// The ring: owns the arena, hands out exactly one producer/consumer pair at a time.
///
/// ```compile_fail
/// // SLOTS must be a power of two: rejected at monomorphisation.
/// let _ = vortex::VortexRing::<3, 4096>::new();
/// ```
///
/// ```compile_fail
/// // SLOT_BYTES must be a multiple of 4096.
/// let _ = vortex::VortexRing::<4, 1500>::new();
/// ```
///
/// ```compile_fail
/// // A payload slice cannot outlive its reservation.
/// let mut ring = vortex::VortexRing::<4, 4096>::new().unwrap();
/// let (mut tx, _rx) = ring.split();
/// let mut slot = tx.reserve_ingress().unwrap();
/// let buf: &mut [u8] = slot.payload_mut();
/// slot.commit_ingress(0, 0).unwrap();
/// buf[0] = 1;
/// ```
///
/// ```compile_fail
/// // A consumer view cannot outlive release.
/// let mut ring = vortex::VortexRing::<4, 4096>::new().unwrap();
/// let (_tx, mut rx) = ring.split();
/// let slot = rx.peek_egress().unwrap();
/// let view: &[u8] = slot.payload();
/// slot.release_egress();
/// let _ = view.len();
/// ```
pub struct VortexRing<const SLOTS: usize, const SLOT_BYTES: usize> {
    arena: VortexArena<SLOTS, SLOT_BYTES>,
}

impl<const SLOTS: usize, const SLOT_BYTES: usize> VortexRing<SLOTS, SLOT_BYTES> {
    /// Ring over an arena built with [`ArenaConfig::default`].
    pub fn new() -> Result<Self, VortexError> {
        Self::with_config(ArenaConfig::default())
    }

    /// Ring over an arena built with `config`.
    pub fn with_config(config: ArenaConfig) -> Result<Self, VortexError> {
        Ok(Self { arena: VortexArena::map(config)? })
    }

    /// The backing arena.
    #[inline]
    pub fn arena(&self) -> &VortexArena<SLOTS, SLOT_BYTES> {
        &self.arena
    }

    /// Committed-but-unreleased slot count: exact when no handles are live, otherwise a snapshot.
    #[inline]
    pub fn occupancy(&self) -> usize {
        let t = self.arena.turnstile();
        let head = t.head.value.load(Ordering::Acquire);
        let tail = t.tail.value.load(Ordering::Acquire);
        tail.wrapping_sub(head) as usize
    }

    /// Split into the unique producer and consumer. `&mut self` guarantees at most one pair is
    /// ever live; cursor state persists across successive splits.
    #[inline]
    pub fn split(&mut self) -> (Producer<'_, SLOTS, SLOT_BYTES>, Consumer<'_, SLOTS, SLOT_BYTES>) {
        let arena = &self.arena;
        let t = arena.turnstile();
        // Relaxed suffices: `&mut self` means any previous handles have been dropped, and
        // whatever transferred them back (join, scope exit, channel) established happens-before.
        let tail = t.tail.value.load(Ordering::Relaxed);
        let head = t.head.value.load(Ordering::Relaxed);
        t.tail.closed.store(false, Ordering::Relaxed);
        t.head.closed.store(false, Ordering::Relaxed);
        (
            Producer { arena, tail, head_cache: head },
            Consumer { arena, head, tail_cache: tail },
        )
    }
}

impl<const SLOTS: usize, const SLOT_BYTES: usize> fmt::Debug for VortexRing<SLOTS, SLOT_BYTES> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VortexRing")
            .field("occupancy", &self.occupancy())
            .field("arena", &self.arena)
            .finish()
    }
}

/// Producer half. `Send`, not `Clone`: exactly one exists per ring at a time.
#[repr(C, align(128))]
pub struct Producer<'a, const SLOTS: usize, const SLOT_BYTES: usize> {
    arena: &'a VortexArena<SLOTS, SLOT_BYTES>,
    /// Private copy of the published tail; the shared atomic is store-only from this side.
    tail: u64,
    /// Last observed consumer head.
    head_cache: u64,
}

impl<'a, const SLOTS: usize, const SLOT_BYTES: usize> Producer<'a, SLOTS, SLOT_BYTES> {
    /// The backing arena.
    #[inline]
    pub fn arena(&self) -> &'a VortexArena<SLOTS, SLOT_BYTES> {
        self.arena
    }

    /// Sequence number the next commit will carry.
    #[inline]
    pub fn next_sequence(&self) -> u64 {
        self.tail
    }

    #[inline(always)]
    fn refresh_head(&mut self) {
        self.head_cache = self.arena.turnstile().head.value.load(Ordering::Acquire);
    }

    #[inline(always)]
    fn free_cached(&self) -> u64 {
        SLOTS as u64 - self.tail.wrapping_sub(self.head_cache)
    }

    /// Free slots after refreshing the consumer cursor.
    #[inline]
    pub fn free_slots(&mut self) -> usize {
        self.refresh_head();
        self.free_cached() as usize
    }

    /// Reserve the slot at `tail`. `None` when the ring is full. Dropping the guard without
    /// committing abandons the reservation; the next reserve returns the same slot.
    #[inline]
    pub fn reserve_ingress(&mut self) -> Option<IngressSlot<'_, SLOTS, SLOT_BYTES>> {
        if self.free_cached() == 0 {
            self.refresh_head();
            if self.free_cached() == 0 {
                return None;
            }
        }
        let pos = self.tail;
        Some(IngressSlot { arena: self.arena, cursor: &mut self.tail, pos })
    }

    /// Reserve up to `max` consecutive slots for a driver RX burst (e.g. refilling a NIC
    /// descriptor ring or submitting a batch of io_uring `READ_FIXED` SQEs). `None` when full
    /// or `max == 0`.
    #[inline]
    pub fn reserve_ingress_burst(&mut self, max: usize) -> Option<IngressBurst<'_, SLOTS, SLOT_BYTES>> {
        let want = max.min(SLOTS) as u64;
        if want == 0 {
            return None;
        }
        if self.free_cached() < want {
            self.refresh_head();
        }
        let count = self.free_cached().min(want) as usize;
        if count == 0 {
            return None;
        }
        let start = self.tail;
        Some(IngressBurst { arena: self.arena, cursor: &mut self.tail, start, count })
    }

    /// `true` once the consumer handle has been dropped.
    #[inline]
    pub fn is_consumer_closed(&self) -> bool {
        self.arena.turnstile().head.closed.load(Ordering::Acquire)
    }

    /// Poll until a slot is free. `false` if `wait` gives up, or the ring is full and the
    /// consumer is gone (nothing will ever free a slot).
    #[inline]
    fn await_free<W: WaitStrategy>(&mut self, wait: &mut W) -> bool {
        wait.reset();
        loop {
            if self.free_cached() != 0 {
                return true;
            }
            self.refresh_head();
            if self.free_cached() != 0 {
                return true;
            }
            if self.is_consumer_closed() {
                // The consumer's final head store precedes its closed store (Release/Acquire);
                // re-read so a release racing with the drop is not missed.
                self.refresh_head();
                return self.free_cached() != 0;
            }
            if !wait.wait() {
                return false;
            }
        }
    }

    /// [`Self::reserve_ingress`], polling under `wait` while the ring is full. `None` if the
    /// strategy gives up or the consumer is dropped while the ring is full.
    #[inline]
    pub fn reserve_ingress_wait<W: WaitStrategy>(
        &mut self,
        wait: &mut W,
    ) -> Option<IngressSlot<'_, SLOTS, SLOT_BYTES>> {
        if !self.await_free(wait) {
            return None;
        }
        self.reserve_ingress()
    }

    /// [`Self::reserve_ingress_burst`], polling under `wait` until at least one slot is free.
    #[inline]
    pub fn reserve_ingress_burst_wait<W: WaitStrategy>(
        &mut self,
        max: usize,
        wait: &mut W,
    ) -> Option<IngressBurst<'_, SLOTS, SLOT_BYTES>> {
        if max == 0 || !self.await_free(wait) {
            return None;
        }
        self.reserve_ingress_burst(max)
    }
}

impl<const SLOTS: usize, const SLOT_BYTES: usize> Drop for Producer<'_, SLOTS, SLOT_BYTES> {
    /// Publish disconnection. Release orders it after this handle's final `tail` store.
    fn drop(&mut self) {
        self.arena.turnstile().tail.closed.store(true, Ordering::Release);
    }
}

impl<const SLOTS: usize, const SLOT_BYTES: usize> fmt::Debug for Producer<'_, SLOTS, SLOT_BYTES> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Producer")
            .field("tail", &self.tail)
            .field("head_cache", &self.head_cache)
            .finish()
    }
}

/// Consumer half. `Send`, not `Clone`: exactly one exists per ring at a time.
#[repr(C, align(128))]
pub struct Consumer<'a, const SLOTS: usize, const SLOT_BYTES: usize> {
    arena: &'a VortexArena<SLOTS, SLOT_BYTES>,
    /// Private copy of the published head; the shared atomic is store-only from this side.
    head: u64,
    /// Last observed producer tail.
    tail_cache: u64,
}

impl<'a, const SLOTS: usize, const SLOT_BYTES: usize> Consumer<'a, SLOTS, SLOT_BYTES> {
    /// The backing arena.
    #[inline]
    pub fn arena(&self) -> &'a VortexArena<SLOTS, SLOT_BYTES> {
        self.arena
    }

    /// Sequence number the next peek will observe.
    #[inline]
    pub fn next_sequence(&self) -> u64 {
        self.head
    }

    #[inline(always)]
    fn refresh_tail(&mut self) {
        self.tail_cache = self.arena.turnstile().tail.value.load(Ordering::Acquire);
    }

    #[inline(always)]
    fn pending_cached(&self) -> u64 {
        self.tail_cache.wrapping_sub(self.head)
    }

    /// Committed slots awaiting release, after refreshing the producer cursor.
    #[inline]
    pub fn pending(&mut self) -> usize {
        self.refresh_tail();
        self.pending_cached() as usize
    }

    /// Observe the slot at `head`. `None` when empty. Dropping the guard without releasing
    /// leaves the slot at the head; the next peek returns it again.
    #[inline]
    pub fn peek_egress(&mut self) -> Option<EgressSlot<'_, SLOTS, SLOT_BYTES>> {
        if self.pending_cached() == 0 {
            self.refresh_tail();
            if self.pending_cached() == 0 {
                return None;
            }
        }
        let pos = self.head;
        // SAFETY: head < tail (Acquire-observed), so the consumer owns `pos`.
        debug_assert_eq!(unsafe { self.arena.meta(pos) }.sequence, pos);
        Some(EgressSlot { arena: self.arena, cursor: &mut self.head, pos })
    }

    /// Observe up to `max` consecutive committed slots. `None` when empty or `max == 0`.
    #[inline]
    pub fn peek_egress_burst(&mut self, max: usize) -> Option<EgressBurst<'_, SLOTS, SLOT_BYTES>> {
        let want = max.min(SLOTS) as u64;
        if want == 0 {
            return None;
        }
        if self.pending_cached() < want {
            self.refresh_tail();
        }
        let count = self.pending_cached().min(want) as usize;
        if count == 0 {
            return None;
        }
        let start = self.head;
        Some(EgressBurst { arena: self.arena, cursor: &mut self.head, start, count })
    }

    /// `true` once the producer handle has been dropped. Slots it committed remain readable.
    #[inline]
    pub fn is_producer_closed(&self) -> bool {
        self.arena.turnstile().tail.closed.load(Ordering::Acquire)
    }

    /// Poll until a slot is committed. `false` if `wait` gives up, or the ring is drained and
    /// the producer is gone (end of stream).
    #[inline]
    fn await_pending<W: WaitStrategy>(&mut self, wait: &mut W) -> bool {
        wait.reset();
        loop {
            if self.pending_cached() != 0 {
                return true;
            }
            self.refresh_tail();
            if self.pending_cached() != 0 {
                return true;
            }
            if self.is_producer_closed() {
                // The producer's final tail store precedes its closed store (Release/Acquire);
                // re-read so a commit racing with the drop is not lost.
                self.refresh_tail();
                return self.pending_cached() != 0;
            }
            if !wait.wait() {
                return false;
            }
        }
    }

    /// [`Self::peek_egress`], polling under `wait` while the ring is empty. `None` if the
    /// strategy gives up or the producer has been dropped and every committed slot consumed.
    #[inline]
    pub fn peek_egress_wait<W: WaitStrategy>(
        &mut self,
        wait: &mut W,
    ) -> Option<EgressSlot<'_, SLOTS, SLOT_BYTES>> {
        if !self.await_pending(wait) {
            return None;
        }
        self.peek_egress()
    }

    /// [`Self::peek_egress_burst`], polling under `wait` until at least one slot is committed.
    #[inline]
    pub fn peek_egress_burst_wait<W: WaitStrategy>(
        &mut self,
        max: usize,
        wait: &mut W,
    ) -> Option<EgressBurst<'_, SLOTS, SLOT_BYTES>> {
        if max == 0 || !self.await_pending(wait) {
            return None;
        }
        self.peek_egress_burst(max)
    }
}

impl<const SLOTS: usize, const SLOT_BYTES: usize> Drop for Consumer<'_, SLOTS, SLOT_BYTES> {
    /// Publish disconnection. Release orders it after this handle's final `head` store.
    fn drop(&mut self) {
        self.arena.turnstile().head.closed.store(true, Ordering::Release);
    }
}

impl<const SLOTS: usize, const SLOT_BYTES: usize> fmt::Debug for Consumer<'_, SLOTS, SLOT_BYTES> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Consumer")
            .field("head", &self.head)
            .field("tail_cache", &self.tail_cache)
            .finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// Layer 4: zero-copy reference exchange
// ═══════════════════════════════════════════════════════════════════════════════════════════════
//
// Every guard holds `&'h mut` to its handle's private cursor, so:
//   * at most one guard per side exists at a time (the handle is exclusively borrowed);
//   * every slice a guard yields is borrowed from the guard, and commit/release consume the
//     guard by value — the borrow checker rejects any slice that would survive the hand-off.
// Raw DMA addresses are not borrows; their contracts are documented on each accessor.

/// Exclusive producer-side ownership of one uncommitted slot.
#[must_use = "dropping an IngressSlot abandons the reservation without publishing"]
pub struct IngressSlot<'h, const SLOTS: usize, const SLOT_BYTES: usize> {
    arena: &'h VortexArena<SLOTS, SLOT_BYTES>,
    cursor: &'h mut u64,
    pos: u64,
}

impl<const SLOTS: usize, const SLOT_BYTES: usize> IngressSlot<'_, SLOTS, SLOT_BYTES> {
    /// Sequence number this slot will carry.
    #[inline(always)]
    pub fn sequence(&self) -> u64 {
        self.pos
    }

    /// Slot index (== io_uring fixed-buffer index / descriptor index).
    #[inline(always)]
    pub fn index(&self) -> usize {
        VortexArena::<SLOTS, SLOT_BYTES>::slot_of(self.pos)
    }

    /// Frame capacity.
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        SLOT_BYTES
    }

    /// The full frame, mutably, straight into the mapping.
    #[inline(always)]
    pub fn payload_mut(&mut self) -> &mut [u8] {
        // SAFETY: the producer owns `pos` (tail - head < SLOTS, and the consumer never touches
        // positions >= tail); the frame is initialised (zero-filled or prior data), in bounds,
        // and exclusively borrowed through `&mut self`.
        unsafe { slice::from_raw_parts_mut(self.arena.payload_ptr(self.pos), SLOT_BYTES) }
    }

    /// The full frame, read-only.
    #[inline(always)]
    pub fn payload(&self) -> &[u8] {
        // SAFETY: as `payload_mut`, shared.
        unsafe { slice::from_raw_parts(self.arena.payload_ptr(self.pos), SLOT_BYTES) }
    }

    /// Frame base for device-side writes (NIC DMA, `IORING_OP_READ_FIXED`, RDMA RECV).
    /// The device write must complete before `commit_*`, and no `payload`/`payload_mut` borrow
    /// may be live while it is in flight.
    #[inline(always)]
    pub fn dma_ptr(&self) -> *mut u8 {
        self.arena.payload_ptr(self.pos)
    }

    /// The frame as a [`DmaRegion`] (4096-aligned, `SLOT_BYTES` long).
    #[inline(always)]
    pub fn dma_region(&self) -> DmaRegion {
        // SAFETY: payload_ptr is derived from the non-null pool base.
        DmaRegion { base: unsafe { NonNull::new_unchecked(self.dma_ptr()) }, len: SLOT_BYTES }
    }

    /// Stamp `len` / `hw_timestamp`, publish with a Release store of `tail`, and return the
    /// sequence number. On error nothing is published and the slot is abandoned.
    #[inline(always)]
    pub fn commit_ingress(self, len: usize, hw_timestamp: u64) -> Result<u64, VortexError> {
        self.commit_ingress_flagged(len, hw_timestamp, 0)
    }

    /// [`Self::commit_ingress`] stamped with [`hw_ticks`].
    #[inline(always)]
    pub fn commit_ingress_now(self, len: usize) -> Result<u64, VortexError> {
        self.commit_ingress_flagged(len, hw_ticks(), 0)
    }

    /// [`Self::commit_ingress`] with driver flags.
    #[inline(always)]
    pub fn commit_ingress_flagged(
        self,
        len: usize,
        hw_timestamp: u64,
        flags: u32,
    ) -> Result<u64, VortexError> {
        if len > SLOT_BYTES {
            return Err(VortexError::PayloadOverflow { len, capacity: SLOT_BYTES });
        }
        // SAFETY: producer owns `pos`; `self` is consumed, so no payload borrow is live.
        unsafe { self.arena.stamp(self.pos, len, hw_timestamp, flags) };
        let next = self.pos + 1;
        // Release: payload + header stores happen-before any consumer Acquire that sees `next`.
        self.arena.turnstile().tail.value.store(next, Ordering::Release);
        *self.cursor = next;
        Ok(self.pos)
    }
}

impl<const SLOTS: usize, const SLOT_BYTES: usize> fmt::Debug for IngressSlot<'_, SLOTS, SLOT_BYTES> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IngressSlot")
            .field("sequence", &self.pos)
            .field("index", &self.index())
            .finish()
    }
}

/// Exclusive producer-side ownership of `len()` consecutive uncommitted slots.
#[must_use = "dropping an IngressBurst abandons the reservation without publishing"]
pub struct IngressBurst<'h, const SLOTS: usize, const SLOT_BYTES: usize> {
    arena: &'h VortexArena<SLOTS, SLOT_BYTES>,
    cursor: &'h mut u64,
    start: u64,
    count: usize,
}

impl<const SLOTS: usize, const SLOT_BYTES: usize> IngressBurst<'_, SLOTS, SLOT_BYTES> {
    /// Reserved slot count (always `>= 1`).
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Always `false`; empty bursts are never handed out.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Sequence number of the first reserved slot.
    #[inline(always)]
    pub fn first_sequence(&self) -> u64 {
        self.start
    }

    #[inline(always)]
    fn pos(&self, i: usize) -> u64 {
        assert!(i < self.count, "burst index {i} out of range 0..{}", self.count);
        self.start + i as u64
    }

    /// Slot index of burst entry `i`. Panics if `i >= len()`.
    #[inline(always)]
    pub fn index(&self, i: usize) -> usize {
        VortexArena::<SLOTS, SLOT_BYTES>::slot_of(self.pos(i))
    }

    /// Frame `i`, mutably. Panics if `i >= len()`.
    #[inline(always)]
    pub fn payload_mut(&mut self, i: usize) -> &mut [u8] {
        let pos = self.pos(i);
        // SAFETY: producer owns every position in [start, start + count); exclusive via &mut.
        unsafe { slice::from_raw_parts_mut(self.arena.payload_ptr(pos), SLOT_BYTES) }
    }

    /// Visit every reserved frame mutably.
    #[inline]
    pub fn for_each_payload_mut<F: FnMut(usize, &mut [u8])>(&mut self, mut f: F) {
        for i in 0..self.count {
            let pos = self.start + i as u64;
            // SAFETY: distinct `i` map to distinct frames (count <= SLOTS), each owned by the
            // producer; only one slice is live per iteration.
            let frame =
                unsafe { slice::from_raw_parts_mut(self.arena.payload_ptr(pos), SLOT_BYTES) };
            f(i, frame);
        }
    }

    /// Frame base of entry `i` for device writes; same contract as [`IngressSlot::dma_ptr`].
    /// Panics if `i >= len()`.
    #[inline(always)]
    pub fn dma_ptr(&self, i: usize) -> *mut u8 {
        self.arena.payload_ptr(self.pos(i))
    }

    /// Stamp the first `n` entries with `stamp(i)` and publish them with a single Release store.
    /// Returns the first published sequence number. Entries `n..len()` are abandoned. On error
    /// nothing is published.
    #[inline]
    pub fn commit_ingress<F>(self, n: usize, mut stamp: F) -> Result<u64, VortexError>
    where
        F: FnMut(usize) -> SlotStamp,
    {
        if n > self.count {
            return Err(VortexError::BurstOverrun { requested: n, reserved: self.count });
        }
        for i in 0..n {
            let s = stamp(i);
            if s.len > SLOT_BYTES {
                return Err(VortexError::PayloadOverflow { len: s.len, capacity: SLOT_BYTES });
            }
            // SAFETY: producer owns this position; `self` is consumed so no borrow is live.
            unsafe { self.arena.stamp(self.start + i as u64, s.len, s.hw_timestamp, s.flags) };
        }
        if n != 0 {
            let next = self.start + n as u64;
            self.arena.turnstile().tail.value.store(next, Ordering::Release);
            *self.cursor = next;
        }
        Ok(self.start)
    }
}

impl<const SLOTS: usize, const SLOT_BYTES: usize> fmt::Debug for IngressBurst<'_, SLOTS, SLOT_BYTES> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IngressBurst")
            .field("first_sequence", &self.start)
            .field("len", &self.count)
            .finish()
    }
}

/// Exclusive consumer-side ownership of one committed slot.
#[must_use = "dropping an EgressSlot leaves it at the head; call release_egress to retire it"]
pub struct EgressSlot<'h, const SLOTS: usize, const SLOT_BYTES: usize> {
    arena: &'h VortexArena<SLOTS, SLOT_BYTES>,
    cursor: &'h mut u64,
    pos: u64,
}

impl<const SLOTS: usize, const SLOT_BYTES: usize> EgressSlot<'_, SLOTS, SLOT_BYTES> {
    /// Committed metadata.
    #[inline(always)]
    pub fn meta(&self) -> &SlotMeta {
        // SAFETY: consumer owns `pos` until release, which consumes `self`.
        unsafe { self.arena.meta(self.pos) }
    }

    /// Slot index.
    #[inline(always)]
    pub fn index(&self) -> usize {
        VortexArena::<SLOTS, SLOT_BYTES>::slot_of(self.pos)
    }

    /// The committed payload, straight out of the mapping.
    #[inline(always)]
    pub fn payload(&self) -> &[u8] {
        // SAFETY: consumer owns `pos`; producer stores to it happened-before our Acquire of
        // tail; the length is clamped to the frame.
        unsafe {
            slice::from_raw_parts(self.arena.payload_ptr(self.pos), self.arena.committed_len(self.pos))
        }
    }

    /// The committed payload, mutably, for in-place transforms (decrypt, decap, rewrite).
    #[inline(always)]
    pub fn payload_mut(&mut self) -> &mut [u8] {
        // SAFETY: as `payload`; exclusivity via &mut self.
        unsafe {
            slice::from_raw_parts_mut(
                self.arena.payload_ptr(self.pos),
                self.arena.committed_len(self.pos),
            )
        }
    }

    /// Base of the committed frame for handing to a downstream device (GPU H2D copy engine,
    /// GDS write, RDMA SEND). The device read must complete before `release_egress`.
    #[inline(always)]
    pub fn dma_ptr(&self) -> *const u8 {
        self.arena.payload_ptr(self.pos)
    }

    /// Base address of the committed frame as `uintptr_t`.
    #[inline(always)]
    pub fn dma_addr(&self) -> usize {
        self.dma_ptr() as usize
    }

    /// The committed bytes as a [`DmaRegion`] (4096-aligned base, `meta().len` long).
    #[inline(always)]
    pub fn dma_region(&self) -> DmaRegion {
        DmaRegion {
            // SAFETY: payload_ptr is derived from the non-null pool base.
            base: unsafe { NonNull::new_unchecked(self.arena.payload_ptr(self.pos)) },
            // SAFETY: consumer owns `pos`.
            len: unsafe { self.arena.committed_len(self.pos) },
        }
    }

    /// Retire the slot with a Release store of `head`, returning it to the producer.
    #[inline(always)]
    pub fn release_egress(self) {
        let next = self.pos + 1;
        // Release: all consumer reads/writes of the slot happen-before the producer's Acquire
        // that permits it to overwrite.
        self.arena.turnstile().head.value.store(next, Ordering::Release);
        *self.cursor = next;
    }
}

impl<const SLOTS: usize, const SLOT_BYTES: usize> fmt::Debug for EgressSlot<'_, SLOTS, SLOT_BYTES> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EgressSlot").field("meta", self.meta()).field("index", &self.index()).finish()
    }
}

/// Exclusive consumer-side ownership of `len()` consecutive committed slots.
#[must_use = "dropping an EgressBurst leaves its slots at the head; call release_egress to retire them"]
pub struct EgressBurst<'h, const SLOTS: usize, const SLOT_BYTES: usize> {
    arena: &'h VortexArena<SLOTS, SLOT_BYTES>,
    cursor: &'h mut u64,
    start: u64,
    count: usize,
}

impl<const SLOTS: usize, const SLOT_BYTES: usize> EgressBurst<'_, SLOTS, SLOT_BYTES> {
    /// Observed slot count (always `>= 1`).
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Always `false`; empty bursts are never handed out.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Sequence number of the first slot.
    #[inline(always)]
    pub fn first_sequence(&self) -> u64 {
        self.start
    }

    #[inline(always)]
    fn pos(&self, i: usize) -> u64 {
        assert!(i < self.count, "burst index {i} out of range 0..{}", self.count);
        self.start + i as u64
    }

    /// Metadata of entry `i`. Panics if `i >= len()`.
    #[inline(always)]
    pub fn meta(&self, i: usize) -> &SlotMeta {
        let pos = self.pos(i);
        // SAFETY: consumer owns [start, start + count) until release consumes `self`.
        unsafe { self.arena.meta(pos) }
    }

    /// Committed payload of entry `i`. Panics if `i >= len()`.
    #[inline(always)]
    pub fn payload(&self, i: usize) -> &[u8] {
        let pos = self.pos(i);
        // SAFETY: as `meta`; length clamped to the frame.
        unsafe { slice::from_raw_parts(self.arena.payload_ptr(pos), self.arena.committed_len(pos)) }
    }

    /// Committed payload of entry `i`, mutably. Panics if `i >= len()`.
    #[inline(always)]
    pub fn payload_mut(&mut self, i: usize) -> &mut [u8] {
        let pos = self.pos(i);
        // SAFETY: as `meta`; exclusivity via &mut self.
        unsafe {
            slice::from_raw_parts_mut(self.arena.payload_ptr(pos), self.arena.committed_len(pos))
        }
    }

    /// Frame base of entry `i`; same contract as [`EgressSlot::dma_ptr`]. Panics if `i >= len()`.
    #[inline(always)]
    pub fn dma_ptr(&self, i: usize) -> *const u8 {
        self.arena.payload_ptr(self.pos(i))
    }

    /// Retire every slot in the burst with a single Release store.
    #[inline(always)]
    pub fn release_egress(self) {
        let count = self.count;
        self.release_egress_prefix(count);
    }

    /// Retire the first `n` slots; the rest stay at the head. Panics if `n > len()`.
    #[inline(always)]
    pub fn release_egress_prefix(self, n: usize) {
        assert!(n <= self.count, "release of {n} slots exceeds burst of {}", self.count);
        if n != 0 {
            let next = self.start + n as u64;
            self.arena.turnstile().head.value.store(next, Ordering::Release);
            *self.cursor = next;
        }
    }
}

impl<const SLOTS: usize, const SLOT_BYTES: usize> fmt::Debug for EgressBurst<'_, SLOTS, SLOT_BYTES> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EgressBurst")
            .field("first_sequence", &self.start)
            .field("len", &self.count)
            .finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// Verification
// ═══════════════════════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const PREFAULT: ArenaConfig = ArenaConfig {
        residency: ResidencyPolicy::PrefaultOnly,
        backing: PageBacking::Transparent,
        dont_fork: true,
        numa_node: None,
    };

    #[test]
    fn geometry_and_alignment() {
        let ring = VortexRing::<64, 8192>::new().unwrap();
        let arena = ring.arena();
        let g = arena.geometry();
        assert_eq!(g.magic, GEOMETRY_MAGIC);
        assert_eq!(g.slots, 64);
        assert_eq!(g.slot_bytes, 8192);
        assert_eq!(g.pool_bytes, 64 * 8192);
        assert_eq!(g.pool_base as usize, arena.dma_region().addr());
        assert_eq!(g.pool_base as usize % DMA_ALIGN, 0);
        assert_eq!(g.headers_base as usize % COHERENCE_GRANULE, 0);
        assert_eq!(arena.turnstile.as_ptr() as usize % COHERENCE_GRANULE, 0);
        assert_eq!(
            &arena.turnstile().head as *const _ as usize - &arena.turnstile().tail as *const _ as usize,
            COHERENCE_GRANULE
        );

        let descs = arena.descriptors();
        assert_eq!(descs.len(), 64);
        for (i, d) in descs.iter().enumerate() {
            assert_eq!(d.addr() % DMA_ALIGN, 0);
            assert_eq!(d.addr(), g.pool_base as usize + i * 8192);
            assert_eq!(d.capacity(), 8192);
        }
        let iov = arena.iovecs();
        assert_eq!(iov[7].iov_base as usize, descs[7].addr());
        assert_eq!(iov[7].iov_len, 8192);

        let r = arena.report();
        assert_eq!(r.body_bytes % r.page_size, 0);
        assert_eq!(r.mapped_bytes, r.body_bytes + 2 * r.page_size);
    }

    #[test]
    fn prefault_policy_reports_prefaulted() {
        let ring = VortexRing::<8, 4096>::with_config(PREFAULT).unwrap();
        assert_eq!(ring.arena().report().residency, Residency::Prefaulted { mlock_errno: None });
    }

    #[test]
    fn full_empty_and_wraparound() {
        let mut ring = VortexRing::<4, 4096>::with_config(PREFAULT).unwrap();
        let (mut tx, mut rx) = ring.split();
        assert!(rx.peek_egress().is_none());

        for lap in 0..5u64 {
            for i in 0..4u64 {
                let mut s = tx.reserve_ingress().unwrap();
                let seq = lap * 4 + i;
                assert_eq!(s.sequence(), seq);
                assert_eq!(s.index(), i as usize);
                s.payload_mut()[..8].copy_from_slice(&seq.to_le_bytes());
                assert_eq!(s.commit_ingress(8, seq * 10).unwrap(), seq);
            }
            assert!(tx.reserve_ingress().is_none());
            assert_eq!(rx.pending(), 4);
            for i in 0..4u64 {
                let s = rx.peek_egress().unwrap();
                let seq = lap * 4 + i;
                assert_eq!(*s.meta(), SlotMeta { sequence: seq, hw_timestamp: seq * 10, len: 8, flags: 0 });
                assert_eq!(s.payload(), &seq.to_le_bytes());
                s.release_egress();
            }
            assert!(rx.peek_egress().is_none());
            assert_eq!(tx.free_slots(), 4);
        }
    }

    #[test]
    fn abandon_semantics() {
        let mut ring = VortexRing::<4, 4096>::with_config(PREFAULT).unwrap();
        let (mut tx, mut rx) = ring.split();

        let s = tx.reserve_ingress().unwrap();
        assert_eq!(s.sequence(), 0);
        drop(s);
        assert!(rx.peek_egress().is_none());

        tx.reserve_ingress().unwrap().commit_ingress_flagged(3, 1, 0xAB).unwrap();
        let s = rx.peek_egress().unwrap();
        assert_eq!(s.meta().flags, 0xAB);
        drop(s);
        let s = rx.peek_egress().unwrap();
        assert_eq!(s.meta().sequence, 0);
        s.release_egress();
        assert!(rx.peek_egress().is_none());
    }

    #[test]
    fn overflow_rejected_without_publication() {
        let mut ring = VortexRing::<4, 4096>::with_config(PREFAULT).unwrap();
        let (mut tx, mut rx) = ring.split();
        let err = tx.reserve_ingress().unwrap().commit_ingress(4097, 0).unwrap_err();
        assert_eq!(err, VortexError::PayloadOverflow { len: 4097, capacity: 4096 });
        assert!(rx.peek_egress().is_none());
        assert_eq!(tx.next_sequence(), 0);

        let err = tx.reserve_ingress_burst(2).unwrap().commit_ingress(3, |_| SlotStamp::default());
        assert_eq!(err.unwrap_err(), VortexError::BurstOverrun { requested: 3, reserved: 2 });
        assert!(rx.peek_egress().is_none());
    }

    #[test]
    fn cursors_survive_resplit() {
        let mut ring = VortexRing::<8, 4096>::with_config(PREFAULT).unwrap();
        {
            let (mut tx, _rx) = ring.split();
            for _ in 0..3 {
                tx.reserve_ingress().unwrap().commit_ingress_now(1).unwrap();
            }
        }
        assert_eq!(ring.occupancy(), 3);
        let (mut tx, mut rx) = ring.split();
        assert_eq!(tx.next_sequence(), 3);
        assert_eq!(rx.peek_egress().unwrap().meta().sequence, 0);
        assert_eq!(tx.free_slots(), 5);
    }

    #[test]
    fn burst_partial_commit_and_release() {
        let mut ring = VortexRing::<8, 4096>::with_config(PREFAULT).unwrap();
        let (mut tx, mut rx) = ring.split();

        let mut b = tx.reserve_ingress_burst(16).unwrap();
        assert_eq!(b.len(), 8);
        b.for_each_payload_mut(|i, f| f[0] = i as u8);
        let first = b
            .commit_ingress(5, |i| SlotStamp { len: 1, hw_timestamp: i as u64, flags: 0 })
            .unwrap();
        assert_eq!(first, 0);
        assert_eq!(tx.next_sequence(), 5);

        let e = rx.peek_egress_burst(8).unwrap();
        assert_eq!(e.len(), 5);
        for i in 0..5 {
            assert_eq!(e.payload(i), &[i as u8]);
            assert_eq!(e.meta(i).sequence, i as u64);
        }
        e.release_egress_prefix(2);
        assert_eq!(rx.next_sequence(), 2);
        assert_eq!(rx.peek_egress_burst(8).unwrap().len(), 3);

        // Burst reservation respects the consumer cursor: 5 committed, 2 released ⇒ 5 free.
        assert_eq!(tx.reserve_ingress_burst(8).unwrap().len(), 5);
    }

    #[test]
    fn consumer_in_place_mutation_and_dma_views() {
        let mut ring = VortexRing::<4, 4096>::with_config(PREFAULT).unwrap();
        let pool = ring.arena().dma_region();
        let (mut tx, mut rx) = ring.split();

        let mut s = tx.reserve_ingress().unwrap();
        assert_eq!(s.dma_region().addr(), pool.addr());
        assert_eq!(s.dma_region().len(), 4096);
        s.payload_mut()[..4].copy_from_slice(b"abcd");
        s.commit_ingress(4, 0).unwrap();

        let mut e = rx.peek_egress().unwrap();
        e.payload_mut().make_ascii_uppercase();
        assert_eq!(e.payload(), b"ABCD");
        assert_eq!(e.dma_addr(), pool.addr());
        assert_eq!(e.dma_addr() % DMA_ALIGN, 0);
        assert_eq!(e.dma_region().len(), 4);
        e.release_egress();
    }

    fn fill(seq: u64, frame: &mut [u8]) -> usize {
        let len = 8 + (seq % 251) as usize;
        frame[..8].copy_from_slice(&seq.to_le_bytes());
        frame[8..len].fill(seq as u8);
        len
    }

    fn verify(seq: u64, meta: &SlotMeta, payload: &[u8]) {
        assert_eq!(meta.sequence, seq);
        assert_eq!(payload.len(), 8 + (seq % 251) as usize);
        assert_eq!(&payload[..8], &seq.to_le_bytes());
        assert!(payload[8..].iter().all(|&b| b == seq as u8));
    }

    /// Spin, then yield: keeps the threaded tests fast on 1–2 vCPU machines and CPU-quota'd
    /// containers, where a pure spin burns a full timeslice before the peer can run.
    type Poll = SpinThenYield<1024>;

    #[test]
    fn spsc_stress_single() {
        const N: u64 = 2_000_000;
        let mut ring = VortexRing::<1024, 4096>::new().unwrap();
        let (mut tx, mut rx) = ring.split();
        std::thread::scope(|s| {
            s.spawn(move || {
                let mut wait = Poll::default();
                for seq in 0..N {
                    let mut slot = tx.reserve_ingress_wait(&mut wait).unwrap();
                    let len = fill(seq, slot.payload_mut());
                    assert_eq!(slot.commit_ingress_now(len).unwrap(), seq);
                }
            });
            s.spawn(move || {
                let mut wait = Poll::default();
                let mut seq = 0;
                while seq < N {
                    let slot = rx.peek_egress_wait(&mut wait).unwrap();
                    verify(seq, slot.meta(), slot.payload());
                    slot.release_egress();
                    seq += 1;
                }
            });
        });
        assert_eq!(ring.occupancy(), 0);
    }

    #[test]
    fn spsc_stress_burst() {
        const N: u64 = 2_000_000;
        let mut ring = VortexRing::<256, 4096>::new().unwrap();
        let (mut tx, mut rx) = ring.split();
        std::thread::scope(|s| {
            s.spawn(move || {
                let mut wait = Poll::default();
                let mut seq = 0;
                while seq < N {
                    let mut b = tx.reserve_ingress_burst_wait(32, &mut wait).unwrap();
                    let n = b.len().min((N - seq) as usize);
                    let mut lens = [0usize; 32];
                    for (i, len) in lens.iter_mut().enumerate().take(n) {
                        *len = fill(seq + i as u64, b.payload_mut(i));
                    }
                    let first = b
                        .commit_ingress(n, |i| SlotStamp { len: lens[i], hw_timestamp: hw_ticks(), flags: 0 })
                        .unwrap();
                    assert_eq!(first, seq);
                    seq += n as u64;
                }
            });
            s.spawn(move || {
                let mut wait = Poll::default();
                let mut seq = 0;
                while seq < N {
                    let b = rx.peek_egress_burst_wait(32, &mut wait).unwrap();
                    for i in 0..b.len() {
                        verify(seq + i as u64, b.meta(i), b.payload(i));
                    }
                    seq += b.len() as u64;
                    b.release_egress();
                }
            });
        });
        assert_eq!(ring.occupancy(), 0);
    }

    #[test]
    fn spsc_wait_stream_ends_on_producer_drop() {
        const N: u64 = 500_000;
        let mut ring = VortexRing::<64, 4096>::new().unwrap();
        let (mut tx, mut rx) = ring.split();
        std::thread::scope(|s| {
            s.spawn(move || {
                let mut wait = Poll::default();
                for seq in 0..N {
                    let mut slot = tx.reserve_ingress_wait(&mut wait).unwrap();
                    let len = fill(seq, slot.payload_mut());
                    slot.commit_ingress_now(len).unwrap();
                }
                // `tx` drops here: end of stream.
            });
            s.spawn(move || {
                let mut wait = Poll::default();
                let mut seq = 0;
                while let Some(slot) = rx.peek_egress_wait(&mut wait) {
                    verify(seq, slot.meta(), slot.payload());
                    slot.release_egress();
                    seq += 1;
                }
                assert!(rx.is_producer_closed());
                assert_eq!(seq, N);
            });
        });
    }

    #[test]
    fn wait_strategies_give_up_and_detect_closure() {
        let mut ring = VortexRing::<2, 4096>::with_config(PREFAULT).unwrap();
        {
            let (mut tx, mut rx) = ring.split();
            assert!(rx.peek_egress_wait(&mut Bounded::<16>::default()).is_none());
            let mut polls = 0;
            assert!(rx.peek_egress_wait(&mut || { polls += 1; polls < 3 }).is_none());
            assert_eq!(polls, 3);
            assert!(rx.peek_egress_burst_wait(0, &mut BusySpin).is_none());

            tx.reserve_ingress().unwrap().commit_ingress(0, 0).unwrap();
            tx.reserve_ingress().unwrap().commit_ingress(0, 0).unwrap();
            assert!(tx.reserve_ingress_wait(&mut Bounded::<4>::default()).is_none());
            assert!(!tx.is_consumer_closed());
            drop(rx);
            // Full and nobody left to drain it: an unbounded wait must still terminate.
            assert!(tx.is_consumer_closed());
            assert!(tx.reserve_ingress_wait(&mut BusySpin).is_none());
            assert!(tx.reserve_ingress_burst_wait(4, &mut BusySpin).is_none());
        }
        // Re-split clears both closure flags; committed data survives.
        let (tx, mut rx) = ring.split();
        assert!(!tx.is_consumer_closed() && !rx.is_producer_closed());
        drop(tx);
        let b = rx.peek_egress_burst_wait(8, &mut BusySpin).unwrap();
        assert_eq!(b.len(), 2);
        b.release_egress();
        assert!(rx.peek_egress_wait(&mut BusySpin).is_none());
    }

    #[test]
    fn base_backing() {
        let cfg = ArenaConfig { backing: PageBacking::Base, ..PREFAULT };
        let ring = VortexRing::<8, 4096>::with_config(cfg).unwrap();
        assert_eq!(ring.arena().report().backing, Backing::Base);
    }

    #[test]
    fn hugetlb_fallback_and_strict() {
        let cfg = ArenaConfig {
            backing: PageBacking::HugeTlb { size: HugePageSize::SIZE_2M, fallback: true },
            ..PREFAULT
        };
        let ring = VortexRing::<8, 4096>::with_config(cfg).unwrap();
        match ring.arena().report().backing {
            Backing::HugeTlb { size } => {
                assert_eq!(size, HugePageSize::SIZE_2M);
                assert_eq!(ring.arena().dma_region().addr() % (2 << 20), 0);
            }
            Backing::Transparent { hugetlb_errno, .. } => assert!(hugetlb_errno.is_some()),
            Backing::Base => panic!("hugetlb must fall back to THP, not base pages"),
        }
        let strict = ArenaConfig {
            backing: PageBacking::HugeTlb { size: HugePageSize::SIZE_2M, fallback: false },
            ..PREFAULT
        };
        match VortexRing::<8, 4096>::with_config(strict) {
            Ok(r) => assert!(matches!(r.arena().report().backing, Backing::HugeTlb { .. })),
            Err(e) => assert!(matches!(e, VortexError::HugeTlb { .. })),
        }
        #[cfg(not(target_os = "linux"))]
        assert_eq!(
            VortexRing::<8, 4096>::with_config(strict).unwrap_err(),
            VortexError::HugeTlb { errno: libc::ENOTSUP }
        );
    }

    /// CI sets VORTEX_EXPECT_HUGETLB=1 after reserving `vm.nr_hugepages`.
    #[test]
    fn hugetlb_granted_when_expected() {
        if std::env::var_os("VORTEX_EXPECT_HUGETLB").is_none() {
            return;
        }
        let cfg = ArenaConfig {
            backing: PageBacking::HugeTlb { size: HugePageSize::SIZE_2M, fallback: false },
            ..ArenaConfig::default()
        };
        let mut ring = VortexRing::<1024, 4096>::with_config(cfg).unwrap();
        assert_eq!(ring.arena().report().backing, Backing::HugeTlb { size: HugePageSize::SIZE_2M });
        let (mut tx, mut rx) = ring.split();
        for seq in 0..4096u64 {
            let mut s = tx.reserve_ingress().unwrap();
            let len = fill(seq, s.payload_mut());
            s.commit_ingress(len, 0).unwrap();
            let e = rx.peek_egress().unwrap();
            verify(seq, e.meta(), e.payload());
            e.release_egress();
        }
    }

    /// CI sets VORTEX_EXPECT_PREFAULT=1 under `ulimit -l 0`.
    #[test]
    fn mlock_fallback_when_expected() {
        if std::env::var_os("VORTEX_EXPECT_PREFAULT").is_none() {
            return;
        }
        let ring = VortexRing::<64, 4096>::new().unwrap();
        assert!(matches!(
            ring.arena().report().residency,
            Residency::Prefaulted { mlock_errno: Some(_) }
        ));
        let strict = ArenaConfig { residency: ResidencyPolicy::RequireLock, ..ArenaConfig::default() };
        assert!(matches!(
            VortexRing::<64, 4096>::with_config(strict).unwrap_err(),
            VortexError::Lock { .. }
        ));
    }

    #[test]
    fn numa_binding() {
        #[cfg(target_os = "linux")]
        {
            // First node with memory (node 0 can be memoryless on some arm64 servers).
            // Absent on kernels built without CONFIG_NUMA.
            let node = std::fs::read_to_string("/sys/devices/system/node/has_memory")
                .ok()
                .and_then(|l| l.trim().split([',', '-']).next()?.parse::<u32>().ok());
            let cfg = ArenaConfig { numa_node: Some(node.unwrap_or(0)), ..PREFAULT };
            match VortexRing::<8, 4096>::with_config(cfg) {
                Ok(r) => assert_eq!(r.arena().report().numa_node, cfg.numa_node),
                // ENOSYS: no CONFIG_NUMA. EPERM: container seccomp profiles (Docker's default)
                // deny mbind without CAP_SYS_NICE. Anything else is a bug.
                Err(VortexError::Bind { errno, .. }) => assert!(
                    errno == libc::EPERM || (errno == libc::ENOSYS && node.is_none()),
                    "mbind to node {node:?} failed with errno {errno}"
                ),
                Err(e) => panic!("{e}"),
            }
        }
        #[cfg(not(target_os = "linux"))]
        assert_eq!(
            VortexRing::<8, 4096>::with_config(ArenaConfig { numa_node: Some(0), ..PREFAULT })
                .unwrap_err(),
            VortexError::Bind { errno: libc::ENOTSUP, node: 0 }
        );
        let absurd = ArenaConfig { numa_node: Some(1 << 20), ..PREFAULT };
        assert!(matches!(
            VortexRing::<8, 4096>::with_config(absurd).unwrap_err(),
            VortexError::Bind { .. }
        ));
    }

    #[test]
    fn clock_calibration() {
        let hz = hw_tick_hz();
        assert!(hz >= 1_000_000, "implausible tick rate {hz}");
        let t0 = hw_ticks();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let ns = ticks_to_nanos(hw_ticks() - t0);
        assert!((15_000_000..2_000_000_000).contains(&ns), "20 ms sleep measured as {ns} ns");
    }

    #[test]
    fn wait_budget_is_per_call() {
        let mut ring = VortexRing::<2, 4096>::with_config(PREFAULT).unwrap();
        let (mut tx, mut rx) = ring.split();
        let mut wait = Bounded::<8>::default();
        // A reused Bounded strategy must not arrive exhausted at the next wait.
        assert!(rx.peek_egress_wait(&mut wait).is_none());
        tx.reserve_ingress().unwrap().commit_ingress(0, 0).unwrap();
        rx.peek_egress_wait(&mut wait).unwrap().release_egress();
        assert!(rx.peek_egress_wait(&mut wait).is_none());
    }

    #[test]
    fn huge_page_geometry() {
        assert_eq!(HugePageSize::from_bytes(2 << 20), Some(HugePageSize::SIZE_2M));
        assert_eq!(HugePageSize::from_bytes(512 << 20), Some(HugePageSize::SIZE_512M));
        assert_eq!(HugePageSize::from_bytes(3 << 20), None);
        assert_eq!(HugePageSize::from_bytes(2048), None);
        assert_eq!(HugePageSize::SIZE_1G.bytes(), 1 << 30);
        #[cfg(target_os = "linux")]
        {
            assert_eq!(thp_pmd_size(4 << 10), 2 << 20);
            assert_eq!(thp_pmd_size(16 << 10), 32 << 20);
            assert_eq!(thp_pmd_size(64 << 10), 512 << 20);
            // Small arenas are not PMD-aligned (no 512 MiB slack on 64 KiB kernels).
            assert_eq!(body_alignment(64 << 10, PageBacking::Transparent, 1 << 20), 64 << 10);
            assert_eq!(body_alignment(4 << 10, PageBacking::Transparent, 4 << 20), 2 << 20);
            if let Some(size) = HugePageSize::system_default() {
                assert!(size.bytes() >= page_size().unwrap());
            }
        }
        #[cfg(not(target_os = "linux"))]
        assert_eq!(HugePageSize::system_default(), None);
    }

    /// Whatever the kernel's default hugetlb size is (2 MiB, 32 MiB, 512 MiB, …), a fallback
    /// request must produce a working ring.
    #[test]
    fn system_default_hugetlb_with_fallback() {
        let Some(size) = HugePageSize::system_default() else { return };
        let cfg = ArenaConfig { backing: PageBacking::HugeTlb { size, fallback: true }, ..PREFAULT };
        let mut ring = VortexRing::<8, 4096>::with_config(cfg).unwrap();
        let (mut tx, mut rx) = ring.split();
        tx.reserve_ingress().unwrap().commit_ingress(1, 0).unwrap();
        rx.peek_egress().unwrap().release_egress();
    }

    #[test]
    fn layout_predicates() {
        assert!(confined_to_granule(0, 8));
        assert!(confined_to_granule(120, 8));
        assert!(!confined_to_granule(124, 8));
        assert!(granules_disjoint(0, 8, 128, 8));
        assert!(!granules_disjoint(0, 8, 64, 8));
        assert_eq!(align_up(1, 4096), Some(4096));
        assert_eq!(align_up(usize::MAX, 4096), None);
    }

    #[test]
    fn error_display_and_io_mapping() {
        let e = VortexError::Lock { errno: libc::ENOMEM, bytes: 4096 };
        assert!(e.to_string().starts_with("mlock(4096 bytes)"));
        assert_eq!(std::io::Error::from(e).raw_os_error(), Some(libc::ENOMEM));
        let e = VortexError::PayloadOverflow { len: 2, capacity: 1 };
        assert_eq!(std::io::Error::from(e).kind(), std::io::ErrorKind::InvalidInput);
    }
}
