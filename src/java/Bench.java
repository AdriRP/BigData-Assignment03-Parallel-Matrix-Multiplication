import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.management.ManagementFactory;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Locale;
import java.util.SplittableRandom;

/**
 * Benchmark harness for Assignment 3. One process measures one configuration.
 *
 *   java Bench --selftest
 *   java Bench seq <n> <A.bin> <B.bin> <blockSize> <warmups> <reps> <batch> <out.bin|-> [--hold]
 *   java Bench par <threads> <n> <A.bin> <B.bin> <blockSize> <warmups> <reps> <batch> <out.bin|-> [--hold]
 *
 * seq runs the sequential blocked baseline (DenseMethods.blocked); par runs
 * ParallelBlocked with a pool of the given number of threads. Each timed sample
 * covers <batch> consecutive multiplications (batch > 1 only for the smallest
 * sizes, whose single multiplication is too short to time reliably).
 *
 * Output lines:
 *   T,kernel,<W|M>,<index>,<nanoseconds for the batch>,<checksum>   one per timed sample
 *   I,<key>,<value>                                               extra information
 *   DONE
 * With --hold the process waits for a line on stdin before exiting, so the
 * driver can read its peak working set.
 */
public class Bench {

    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            usage();
        }
        boolean hold = args[args.length - 1].equals("--hold");
        switch (args[0]) {
            case "--selftest" -> System.exit(selftest());
            case "seq" -> run(1, args, 0);
            case "par" -> run(Integer.parseInt(args[1]), args, 1);
            default -> usage();
        }
        System.out.println("DONE");
        System.out.flush();
        if (hold) {
            System.in.read();
        }
    }

    static void usage() {
        System.err.println("see the class comment of Bench.java for the usage");
        System.exit(1);
    }

    /** off = 0 for seq, 1 for par (the thread count shifts the remaining arguments). */
    static void run(int threads, String[] args, int off) throws IOException {
        boolean parallel = off == 1;
        int n = Integer.parseInt(args[1 + off]);
        int bs = Integer.parseInt(args[4 + off]);
        int warmups = Integer.parseInt(args[5 + off]);
        int reps = Integer.parseInt(args[6 + off]);
        int batch = Integer.parseInt(args[7 + off]);
        String out = args[8 + off];
        double[][] a = readMatrix(args[2 + off], n);
        double[][] b = readMatrix(args[3 + off], n);
        double[][] c = new double[n][n];
        info("available_processors", Runtime.getRuntime().availableProcessors());
        info("batch", batch);
        try (ParallelBlocked par = parallel ? new ParallelBlocked(threads) : null) {
            info("worker_threads", parallel ? par.threads() : 0);
            prewarm(par, bs);
            for (int r = 0; r < warmups + reps; r++) {
                long t0 = System.nanoTime();
                for (int k = 0; k < batch; k++) {
                    if (parallel) {
                        par.multiply(a, b, c, n, bs);
                    } else {
                        DenseMethods.blocked(a, b, c, n, bs);
                    }
                }
                long t1 = System.nanoTime();
                timing(r, warmups, t1 - t0, DenseMethods.checksum(c));
            }
            // live JVM threads while the pool still exists: workers + main + JIT/GC/service threads
            info("jvm_live_threads", ManagementFactory.getThreadMXBean().getThreadCount());
        }
        info("result_sha256", sha256(c));
        if (!out.equals("-")) {
            writeMatrix(out, c);
        }
    }

    // ---------- JIT pre-warm ----------

    static final int PREWARM_N = 128;
    static final int PREWARM_CALLS = 200;

    /**
     * Calls the kernel PREWARM_CALLS times on a small random matrix so that
     * HotSpot compiles it with its optimising compiler (C2) before the warm-up
     * and measured runs at the real size (same pre-warm as Assignment 2).
     *
     * For the parallel method the band kernel is first pre-warmed on the main
     * thread alone, and only then through the pool. In a first version the
     * kernel was pre-warmed only through the pool: in some processes with 30 to
     * 32 threads every task then ran at the speed measured with
     * -XX:-UseSuperWord, i.e. the compiled kernel was not auto-vectorised.
     * Pre-warming on one thread, as for the sequential baseline, gives the
     * compiler the same single-threaded profile in every process.
     */
    static void prewarm(ParallelBlocked par, int bs) {
        SplittableRandom rng = new SplittableRandom(7);
        double[][] a = new double[PREWARM_N][PREWARM_N], b = new double[PREWARM_N][PREWARM_N];
        double[][] c = new double[PREWARM_N][PREWARM_N];
        for (int i = 0; i < PREWARM_N; i++) {
            for (int j = 0; j < PREWARM_N; j++) {
                a[i][j] = rng.nextDouble();
                b[i][j] = rng.nextDouble();
            }
        }
        int smallBs = Math.min(bs, PREWARM_N / 4);
        for (int r = 0; r < PREWARM_CALLS; r++) {
            if (par != null) {
                DenseMethods.blockedRows(a, b, c, PREWARM_N, smallBs, 0, PREWARM_N);
            } else {
                DenseMethods.blocked(a, b, c, PREWARM_N, smallBs);
            }
        }
        if (par != null) {
            for (int r = 0; r < PREWARM_CALLS; r++) {
                par.multiply(a, b, c, PREWARM_N, smallBs);
            }
        }
        info("prewarm", PREWARM_CALLS + " calls at n=" + PREWARM_N + (par != null ? " on one thread, then "
                + PREWARM_CALLS + " through the pool" : ""));
    }

    // ---------- output ----------

    static void timing(int r, int warmups, long ns, double checksum) {
        boolean warm = r < warmups;
        System.out.println("T,kernel," + (warm ? "W" : "M") + "," + (warm ? r : r - warmups) + "," + ns + ","
                + String.format(Locale.ROOT, "%.17g", checksum));
        System.out.flush();
    }

    static void info(String key, Object value) {
        System.out.println("I," + key + "," + value);
        System.out.flush();
    }

    /** SHA-256 of the raw IEEE-754 bits of C, row by row: equal hashes mean bit-identical results. */
    static String sha256(double[][] c) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            ByteBuffer buf = ByteBuffer.allocate(8 * c.length).order(ByteOrder.LITTLE_ENDIAN);
            for (double[] row : c) {
                buf.clear();
                buf.asDoubleBuffer().put(row);
                md.update(buf.array());
            }
            return HexFormat.of().formatHex(md.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    // ---------- binary matrix files (same format as Assignments 1 and 2) ----------

    static double[][] readMatrix(String path, int n) throws IOException {
        double[][] m = new double[n][n];
        try (FileChannel ch = FileChannel.open(Path.of(path), StandardOpenOption.READ)) {
            ByteBuffer buf = ByteBuffer.allocate(8 * n).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++) {
                buf.clear();
                while (buf.hasRemaining()) {
                    if (ch.read(buf) < 0) {
                        throw new IOException("short read in " + path);
                    }
                }
                buf.flip();
                buf.asDoubleBuffer().get(m[i]);
            }
        }
        return m;
    }

    static void writeMatrix(String path, double[][] m) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(8 * m.length).order(ByteOrder.LITTLE_ENDIAN);
        try (OutputStream out = new FileOutputStream(path)) {
            for (double[] row : m) {
                buf.clear();
                buf.asDoubleBuffer().put(row);
                out.write(buf.array());
            }
        }
    }

    // ---------- self-test ----------

    static int fails = 0;

    static void expect(String name, boolean ok) {
        System.out.println((ok ? "PASS " : "FAIL ") + name);
        if (!ok) {
            fails++;
        }
    }

    static boolean same(double[][] x, double[][] y) {
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x.length; j++) {
                if (Double.doubleToRawLongBits(x[i][j]) != Double.doubleToRawLongBits(y[i][j])) {
                    return false;
                }
            }
        }
        return true;
    }

    /** threads = 0 means the sequential baseline. */
    static double[][] multiply(int threads, double[][] a, double[][] b, int bs) {
        int n = a.length;
        double[][] c = new double[n][n];
        for (double[] row : c) {
            java.util.Arrays.fill(row, Double.NaN); // stale values must be overwritten
        }
        if (threads == 0) {
            DenseMethods.blocked(a, b, c, n, bs);
        } else {
            try (ParallelBlocked par = new ParallelBlocked(threads)) {
                par.multiply(a, b, c, n, bs);
            }
        }
        return c;
    }

    static String label(int threads) {
        return threads == 0 ? "seq" : "par(p=" + threads + ")";
    }

    static int selftest() {
        int[] configs = {0, 1, 2, 3, 7, 64};
        double[][] a2 = {{1, 2}, {3, 4}}, b2 = {{5, 6}, {7, 8}}, w2 = {{19, 22}, {43, 50}};
        double[][] a3 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, b3 = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
        double[][] w3 = {{30, 24, 18}, {84, 69, 54}, {138, 114, 90}};
        int n = 5;
        double[][] a = new double[n][n], id = new double[n][n], zero = new double[n][n];
        for (int i = 0; i < n * n; i++) {
            a[i / n][i % n] = (double) (i * 7 % 11) - 5.0;
            id[i / n][i % n] = (i / n == i % n) ? 1.0 : 0.0;
        }
        for (int t : configs) {
            String m = label(t);
            expect(m + " 2x2_known", same(multiply(t, a2, b2, 1), w2));
            expect(m + " 3x3_known (block 2)", same(multiply(t, a3, b3, 2), w3));
            expect(m + " A_times_identity", same(multiply(t, a, id, 2), a));
            expect(m + " identity_times_A", same(multiply(t, id, a, 2), a));
            expect(m + " A_times_zero", same(multiply(t, a, zero, 2), zero));
        }
        // random 37 x 37 (not a multiple of the block sizes or of the thread counts): bit-identical to ijk
        SplittableRandom rng = new SplittableRandom(1);
        int r = 37;
        double[][] x = new double[r][r], y = new double[r][r];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < r; j++) {
                x[i][j] = 2 * rng.nextDouble() - 1;
                y[i][j] = 2 * rng.nextDouble() - 1;
            }
        }
        double[][] ref = new double[r][r];
        DenseMethods.ijk(x, y, ref, r);
        for (int t : configs) {
            for (int bs : new int[] {1, 5, 16, 64}) {
                expect(label(t) + " bs=" + bs + " bit-identical to ijk (37x37)", same(multiply(t, x, y, bs), ref));
            }
        }
        // the same pool reused for many products (as in the benchmark) gives the same result every time
        boolean stable = true;
        double[][] c = new double[r][r];
        try (ParallelBlocked par = new ParallelBlocked(8)) {
            for (int k = 0; k < 100; k++) {
                par.multiply(x, y, c, r, 5);
                stable &= same(c, ref);
            }
        }
        expect("par(p=8) reused pool, 100 products bit-identical to ijk", stable);
        System.out.println(fails == 0 ? "SELFTEST OK" : "SELFTEST FAILED");
        return fails == 0 ? 0 : 1;
    }
}
