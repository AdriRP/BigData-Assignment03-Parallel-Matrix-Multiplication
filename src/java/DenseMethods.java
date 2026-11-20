import java.util.Arrays;

/**
 * Sequential dense kernels, C = A * B, on double[n][n] matrices.
 *
 * ijk and blocked are copied unchanged from Assignment 2 (DenseMethods.java).
 * blockedRows is the same blocked loop restricted to a band of rows of C; it is
 * the work that one thread of the parallel method performs.
 *
 * Every kernel adds the products of each entry in the same order
 * (k = 0, 1, ..., n-1), so with IEEE-754 arithmetic they produce bit-identical
 * results.
 */
final class DenseMethods {

    private DenseMethods() {}

    /** Basic i-j-k triple loop (Assignment 1). Used only as a reference in the self-test. */
    static void ijk(double[][] a, double[][] b, double[][] c, int n) {
        for (int i = 0; i < n; i++) {
            double[] ai = a[i];
            double[] ci = c[i];
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += ai[k] * b[k][j];
                }
                ci[j] = sum;
            }
        }
    }

    /**
     * Sequential baseline (Assignment 2, unchanged): the i-k-j loops over bs x bs
     * tiles so that the tiles of B and C in use stay in cache. C is zeroed inside
     * the kernel because each entry is accumulated over k.
     */
    static void blocked(double[][] a, double[][] b, double[][] c, int n, int bs) {
        for (int i = 0; i < n; i++) {
            Arrays.fill(c[i], 0.0);
        }
        for (int ii = 0; ii < n; ii += bs) {
            int iMax = Math.min(ii + bs, n);
            for (int kk = 0; kk < n; kk += bs) {
                int kMax = Math.min(kk + bs, n);
                for (int jj = 0; jj < n; jj += bs) {
                    int jMax = Math.min(jj + bs, n);
                    for (int i = ii; i < iMax; i++) {
                        double[] ai = a[i];
                        double[] ci = c[i];
                        for (int k = kk; k < kMax; k++) {
                            double aik = ai[k];
                            double[] bk = b[k];
                            for (int j = jj; j < jMax; j++) {
                                ci[j] += aik * bk[j];
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * The blocked kernel for rows r0 (inclusive) to r1 (exclusive) of C only.
     * The only differences from blocked() are the range of the outer tile loop and
     * the zeroing of those rows alone; with r0 = 0 and r1 = n it performs exactly
     * the same operations. It reads A and B and writes only rows r0..r1-1 of C.
     */
    static void blockedRows(double[][] a, double[][] b, double[][] c, int n, int bs, int r0, int r1) {
        for (int i = r0; i < r1; i++) {
            Arrays.fill(c[i], 0.0);
        }
        for (int ii = r0; ii < r1; ii += bs) {
            int iMax = Math.min(ii + bs, r1);
            for (int kk = 0; kk < n; kk += bs) {
                int kMax = Math.min(kk + bs, n);
                for (int jj = 0; jj < n; jj += bs) {
                    int jMax = Math.min(jj + bs, n);
                    for (int i = ii; i < iMax; i++) {
                        double[] ai = a[i];
                        double[] ci = c[i];
                        for (int k = kk; k < kMax; k++) {
                            double aik = ai[k];
                            double[] bk = b[k];
                            for (int j = jj; j < jMax; j++) {
                                ci[j] += aik * bk[j];
                            }
                        }
                    }
                }
            }
        }
    }

    static double checksum(double[][] c) {
        double s = 0.0;
        for (double[] row : c) {
            for (double v : row) {
                s += v;
            }
        }
        return s;
    }
}
