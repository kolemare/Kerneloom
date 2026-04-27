namespace kl::detail
{

    void matmul_cpu(
        const float *a,
        const float *b,
        float *c,
        int m,
        int n,
        int k)
    {
        for (int row = 0; row < m; ++row)
        {
            for (int col = 0; col < n; ++col)
            {
                float sum = 0.0f;

                for (int inner = 0; inner < k; ++inner)
                {
                    sum += a[row * k + inner] * b[inner * n + col];
                }

                c[row * n + col] = sum;
            }
        }
    }

}