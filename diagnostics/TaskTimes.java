import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Diagnostic: duration of each band task of the parallel blocked product at
 * n = 2,000 (4 products). If every task is slow, the kernel itself is slow; if
 * only a few are, the slow ones are stragglers.
 *
 * The recorded output (diagnostics/diagnostics_log.txt) was produced while
 * Bench.prewarm pre-warmed the kernel only through the pool (first version);
 * with the current Bench.prewarm the slow case no longer appears.
 *
 *   javac -cp build/java -d build/diag diagnostics/TaskTimes.java
 *   java -Xmx4g -cp "build/java;build/diag" TaskTimes 32
 */
public class TaskTimes {
    public static void main(String[] args) throws Exception {
        int p = Integer.parseInt(args[0]);
        int n = 2000, bs = 256;
        double[][] a = Bench.readMatrix("data/inputs/A_" + n + ".bin", n);
        double[][] b = Bench.readMatrix("data/inputs/B_" + n + ".bin", n);
        double[][] c = new double[n][n];
        try (ParallelBlocked par = new ParallelBlocked(p)) {
            Bench.prewarm(par, bs);
        }
        ExecutorService pool = Executors.newFixedThreadPool(p);
        for (int rep = 0; rep < 4; rep++) {
            long[] dur = new long[p];
            List<Callable<Void>> tasks = new ArrayList<>();
            for (int t = 0; t < p; t++) {
                int r0 = (int) ((long) n * t / p), r1 = (int) ((long) n * (t + 1) / p), id = t;
                tasks.add(() -> {
                    long s = System.nanoTime();
                    DenseMethods.blockedRows(a, b, c, n, bs, r0, r1);
                    dur[id] = System.nanoTime() - s;
                    return null;
                });
            }
            long t0 = System.nanoTime();
            for (var f : pool.invokeAll(tasks)) {
                f.get();
            }
            long total = System.nanoTime() - t0;
            long[] s = dur.clone();
            Arrays.sort(s);
            System.out.printf("p=%d total=%.3f s  task min=%.3f median=%.3f max=%.3f  (tasks > 1.5x median: %d)%n", p,
                    total / 1e9, s[0] / 1e9, s[p / 2] / 1e9, s[p - 1] / 1e9,
                    Arrays.stream(dur).filter(d -> d > 1.5 * s[p / 2]).count());
        }
        pool.shutdownNow();
    }
}
