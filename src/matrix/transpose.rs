/// Transpose a matrix: dst = src^T
///
/// Converts from row-major (rows × cols) to row-major (cols × rows).
/// After transpose, what was column j of src becomes row j of dst.
///
/// # Arguments
///
/// * `src` - Source matrix (rows × cols), row-major
/// * `dst` - Destination matrix (cols × rows), row-major
/// * `rows` - Number of rows in src
/// * `cols` - Number of columns in src
///
/// # Example
///
/// ```
/// use matmul::matrix::transpose::transpose;
///
/// let src = vec![1.0, 2.0, 3.0,   // 2×3 matrix
///                4.0, 5.0, 6.0];
/// let mut dst = vec![0.0; 6];      // will be 3×2
///
/// transpose(&src, &mut dst, 2, 3);
///
/// assert_eq!(dst, vec![1.0, 4.0,   // 3×2 matrix
///                      2.0, 5.0,
///                      3.0, 6.0]);
/// ```
pub fn transpose(src: &[f64], dst: &mut [f64], rows: usize, cols: usize) {
    for i in 0..rows {
        for j in 0..cols {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}
