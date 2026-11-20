import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Parallel blocked multiplication with a fixed pool of p worker threads.
 *
 * Partitioning: the n rows of C are split into p contiguous bands whose sizes
 * differ by at most one row (band t holds rows floor(n t / p) to
 * floor(n (t + 1) / p) - 1). Each band is one task that runs
 * DenseMethods.blockedRows. With p > n some bands are empty.
 *
 * Shared data: all threads read A and B, which are not modified, and each
 * thread writes only the rows of its own band of C, so no two threads write
 * the same location and no locks are needed. invokeAll() returns only after
 * every task has finished, and Future.get() makes the workers' writes visible
 * to the calling thread (Java memory model happens-before).
 *
 * The pool and its threads are created once, before any timed run, and reused
 * by every multiplication. The timed region of multiply() includes creating
 * the p tasks, handing them to the pool and waiting for all of them.
 */
final class ParallelBlocked implements AutoCloseable {

    private final int p;
    private final ThreadPoolExecutor pool;

    ParallelBlocked(int p) {
        if (p < 1) {
            throw new IllegalArgumentException("thread count must be >= 1");
        }
        this.p = p;
        AtomicInteger id = new AtomicInteger();
        this.pool = new ThreadPoolExecutor(p, p, 0L, TimeUnit.MILLISECONDS, new LinkedBlockingQueue<>(), r -> {
            Thread t = new Thread(r, "mm-worker-" + id.getAndIncrement());
            t.setDaemon(true);
            return t;
        });
        pool.prestartAllCoreThreads();
    }

    int threads() {
        return p;
    }

    void multiply(double[][] a, double[][] b, double[][] c, int n, int bs) {
        List<Callable<Void>> tasks = new ArrayList<>(p);
        for (int t = 0; t < p; t++) {
            int r0 = (int) ((long) n * t / p);
            int r1 = (int) ((long) n * (t + 1) / p);
            tasks.add(() -> {
                DenseMethods.blockedRows(a, b, c, n, bs, r0, r1);
                return null;
            });
        }
        try {
            for (Future<Void> f : pool.invokeAll(tasks)) {
                f.get(); // rethrows any exception raised inside a task
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        } catch (ExecutionException e) {
            throw new IllegalStateException(e.getCause());
        }
    }

    @Override
    public void close() {
        pool.shutdownNow();
    }
}
