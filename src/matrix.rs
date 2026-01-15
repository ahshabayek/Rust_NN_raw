//! Matrix module for neural network operations.

use std::fmt;

/// A 2D matrix of f64 values stored in row-major order.
///
/// Data is stored as a flat vector for cache-efficient access.
/// Element at position (row, col) is stored at index: row * cols + col
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    /// Number of rows in the matrix.
    pub rows: usize,
    /// Number of columns in the matrix.
    pub cols: usize,
    /// Flat storage of matrix elements in row-major order.
    pub data: Vec<f64>,
}

/// Trait for types that can be used as operands in matrix element-wise operations.
///
/// This trait enables generic operations that work with both:
/// - `Matrix` - operates element-by-element
/// - `f64` - broadcasts the scalar to all elements
///
/// # Example
/// ```
/// use rust_nn_raw::matrix::Matrix;
/// let m = Matrix::zeros(2, 2);
/// let _ = m.add(5.0);  // f64 implements MatrixOperand
/// let other = Matrix::zeros(2, 2);
/// let _ = m.add(&other);  // &Matrix implements MatrixOperand
/// ```
pub trait MatrixOperand {
    fn get_value(&self, index: usize) -> f64;
    fn dimensions(&self) -> Option<(usize, usize)>;
}

impl MatrixOperand for Matrix {
    fn get_value(&self, index: usize) -> f64 {
        self.data[index]
    }
    fn dimensions(&self) -> Option<(usize, usize)> {
        Some((self.rows, self.cols))
    }
}

impl MatrixOperand for f64 {
    fn get_value(&self, _index: usize) -> f64 {
        *self // Always return the same scalar
    }
    fn dimensions(&self) -> Option<(usize, usize)> {
        None // Scalar - no dimensions
    }
}

impl MatrixOperand for &Matrix {
    fn get_value(&self, index: usize) -> f64 {
        self.data[index]
    }
    fn dimensions(&self) -> Option<(usize, usize)> {
        Some((self.rows, self.cols))
    }
}

/// Display the matrix in a readable format.
impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{:8.4} ", self.get(i, j))?;
            }
            writeln!(f)?; // New line after each row
        }
        Ok(())
    }
}

impl Matrix {
    // === Constructors ===
    /// Creates a new matrix filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    ///
    /// A new `Matrix` instance filled with zeros.
    ///
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Creates a new matrix filled with ones.
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    ///
    /// A new `Matrix` instance filled with ones.
    ///
    #[must_use]
    pub fn ones(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![1.0; rows * cols],
        }
    }

    /// Creates a new matrix filled with random values.
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    ///
    /// A new `Matrix` instance filled with random values.
    ///
    #[must_use]
    pub fn random(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: (0..rows * cols).map(|_| rand::random()).collect(),
        }
    }
    /// Creates a new matrix from a vector of values.
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    /// * `data` - The vector of values to fill the matrix with.
    ///
    /// # Returns
    ///
    /// A new `Matrix` instance filled with the provided values.
    ///
    #[must_use]
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        // Safety check: make sure data length matches dimensions
        assert_eq!(
            data.len(),
            rows * cols,
            "Data length {} doesn't match {}x{} matrix size {}",
            data.len(),
            rows,
            cols,
            rows * cols
        );
        Matrix { rows, cols, data }
    }

    // === Helper Methods ===
    fn check_dimensions<Operand: MatrixOperand>(&self, other: &Operand) {
        if let Some((rows, cols)) = other.dimensions() {
            if self.rows != rows || self.cols != cols {
                panic!(
                    "Matrix dimensions do not match: {}x{} vs {}x{}",
                    self.rows, self.cols, rows, cols
                );
            }
        }
    }

    // === Accessors ===
    /// Returns the element at the specified position.
    ///
    /// # Panics
    ///
    /// Panics if `row` or `col` is out of bounds.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row >= self.rows || col >= self.cols {
            panic!(
                "Index ({}, {}) out of bounds for {}x{} matrix",
                row, col, self.rows, self.cols
            );
        }
        self.data[row * self.cols + col]
    }

    /// Sets the element at the specified position.
    ///
    /// # Panics
    ///
    /// Panics if `row` or `col` is out of bounds.
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        if row >= self.rows || col >= self.cols {
            panic!(
                "Index ({}, {}) out of bounds for {}x{} matrix",
                row, col, self.rows, self.cols
            );
        }
        self.data[row * self.cols + col] = value;
    }

    /// Applies a function to every element of the matrix.
    ///
    /// # Arguments
    /// * `f` - A function that takes f64 and returns f64
    ///
    /// # Example
    /// ```
    /// use rust_nn_raw::matrix::Matrix;
    /// let m = Matrix::from_vec(2, 2, vec![1.0, 4.0, 9.0, 16.0]);
    /// let sqrt_m = m.map(|x: f64| x.sqrt());
    /// // sqrt_m contains [1.0, 2.0, 3.0, 4.0]
    /// ```
    #[must_use]
    pub fn map<F>(&self, f: F) -> Matrix
    where
        F: Fn(f64) -> f64,
    {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }

    // === Operators ===
    //
    /// Adds two matrices element-wise.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different dimensions.
    #[must_use]
    pub fn add<Operand: MatrixOperand>(&self, other: Operand) -> Matrix {
        self.check_dimensions(&other);
        let data: Vec<f64> = (0..self.data.len())
            .map(|i| self.data[i] + other.get_value(i))
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Subtracts two matrices element-wise.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different dimensions.
    #[must_use]
    pub fn subtract<Operand: MatrixOperand>(&self, other: Operand) -> Matrix {
        self.check_dimensions(&other);
        let data: Vec<f64> = (0..self.data.len())
            .map(|i| self.data[i] - other.get_value(i))
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Multiplies two matrices element-wise.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different dimensions.
    #[must_use]
    pub fn elem_multiply<Operand: MatrixOperand>(&self, other: Operand) -> Matrix {
        self.check_dimensions(&other);
        let data: Vec<f64> = (0..self.data.len())
            .map(|i| self.data[i] * other.get_value(i))
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Divides two matrices element-wise.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different dimensions.
    #[must_use]
    pub fn elem_divide<Operand: MatrixOperand>(&self, other: Operand) -> Matrix {
        self.check_dimensions(&other);
        let data: Vec<f64> = (0..self.data.len())
            .map(|i| {
                let divisor = other.get_value(i);
                if divisor == 0.0 {
                    panic!("Division by zero in matrix");
                }
                self.data[i] / divisor
            })
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Multiplies two matrices.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have incompatible dimensions.
    #[must_use]
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!(
                "Matrix dimensions incompatible for multiplication: {}x{} * {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }

        let mut result = Matrix::zeros(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }

        result
    }

    /// Transposes a matrix.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty.
    #[must_use]
    pub fn transpose(&self) -> Matrix {
        // 1. Create result with flipped dimensions (cols × rows)
        let mut result = Matrix::zeros(self.cols, self.rows);

        // 2. For each element, swap its position
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let m = Matrix::zeros(3, 4);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 4);
        assert_eq!(m.data.len(), 12);
        assert_eq!(m.data, vec![0.0; 12]);
    }

    #[test]
    fn test_ones() {
        let m = Matrix::ones(2, 3);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.data.len(), 6);
        // All values should be 1.0
        for val in &m.data {
            assert_eq!(*val, 1.0);
        }
    }

    #[test]
    fn test_random() {
        let m = Matrix::random(3, 4);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 4);
        assert_eq!(m.data.len(), 12);
        // All values should be in range [0.0, 1.0)
        for val in &m.data {
            assert!(*val >= 0.0 && *val < 1.0);
        }
    }

    #[test]
    fn test_get_set() {
        let mut m = Matrix::zeros(3, 4);
        m.set(1, 2, 5.0);
        assert_eq!(m.get(1, 2), 5.0);
    }

    #[test]
    fn test_get_different_positions() {
        let mut m = Matrix::zeros(2, 3);
        m.set(0, 0, 1.0);
        m.set(0, 2, 2.0);
        m.set(1, 1, 3.0);

        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 2), 2.0);
        assert_eq!(m.get(1, 1), 3.0);
        assert_eq!(m.get(1, 0), 0.0); // Unchanged, still zero
    }

    #[test]
    #[should_panic]
    fn test_set_out_of_bounds() {
        let mut m = Matrix::zeros(2, 2);
        m.set(5, 5, 1.0); // Should panic
    }
    #[test]
    #[should_panic]
    fn test_get_out_of_bounds() {
        let m = Matrix::zeros(2, 2);
        let _ = m.get(5, 5); // Should panic, no assert needed
    }
    #[test]
    fn test_add_scalar() {
        let mut m = Matrix::zeros(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);

        let result = m.add(10.0);

        assert_eq!(result.get(0, 0), 11.0);
        assert_eq!(result.get(0, 1), 12.0);
        assert_eq!(result.get(1, 0), 13.0);
        assert_eq!(result.get(1, 1), 14.0);
    }

    #[test]
    fn test_elem_multiply_scalar() {
        let mut m = Matrix::zeros(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);

        let result = m.elem_multiply(2.0);

        assert_eq!(result.get(0, 0), 2.0);
        assert_eq!(result.get(0, 1), 4.0);
        assert_eq!(result.get(1, 0), 6.0);
        assert_eq!(result.get(1, 1), 8.0);
    }

    #[test]
    fn test_transpose() {
        let mut m = Matrix::zeros(2, 3);
        // Row 0: [1, 2, 3]
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(0, 2, 3.0);
        // Row 1: [4, 5, 6]
        m.set(1, 0, 4.0);
        m.set(1, 1, 5.0);
        m.set(1, 2, 6.0);

        let t = m.transpose();

        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(0, 1), 4.0);
        assert_eq!(t.get(1, 0), 2.0);
        assert_eq!(t.get(2, 1), 6.0);
    }
    #[test]
    fn test_matmul() {
        // A (2x3) * B (3x2) = C (2x2)
        let mut a = Matrix::zeros(2, 3);
        a.set(0, 0, 1.0);
        a.set(0, 1, 2.0);
        a.set(0, 2, 3.0);
        a.set(1, 0, 4.0);
        a.set(1, 1, 5.0);
        a.set(1, 2, 6.0);

        let mut b = Matrix::zeros(3, 2);
        b.set(0, 0, 7.0);
        b.set(0, 1, 8.0);
        b.set(1, 0, 9.0);
        b.set(1, 1, 10.0);
        b.set(2, 0, 11.0);
        b.set(2, 1, 12.0);

        let c = a.matmul(&b);

        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);
        assert_eq!(c.get(0, 0), 58.0); // 1*7 + 2*9 + 3*11
        assert_eq!(c.get(0, 1), 64.0); // 1*8 + 2*10 + 3*12
        assert_eq!(c.get(1, 0), 139.0); // 4*7 + 5*9 + 6*11
        assert_eq!(c.get(1, 1), 154.0); // 4*8 + 5*10 + 6*12
    }

    #[test]
    #[should_panic]
    fn test_matmul_incompatible() {
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(2, 2); // Incompatible: a.cols (3) != b.rows (2)
        let _ = a.matmul(&b);
    }

    #[test]
    fn test_add_matrix() {
        let mut a = Matrix::zeros(2, 2);
        a.set(0, 0, 1.0);
        a.set(0, 1, 2.0);
        a.set(1, 0, 3.0);
        a.set(1, 1, 4.0);

        let mut b = Matrix::zeros(2, 2);
        b.set(0, 0, 10.0);
        b.set(0, 1, 20.0);
        b.set(1, 0, 30.0);
        b.set(1, 1, 40.0);

        let c = a.add(&b);

        assert_eq!(c.get(0, 0), 11.0);
        assert_eq!(c.get(0, 1), 22.0);
        assert_eq!(c.get(1, 0), 33.0);
        assert_eq!(c.get(1, 1), 44.0);
    }

    #[test]
    fn test_subtract_scalar() {
        let mut m = Matrix::zeros(2, 2);
        m.set(0, 0, 10.0);
        m.set(0, 1, 20.0);
        m.set(1, 0, 30.0);
        m.set(1, 1, 40.0);

        let result = m.subtract(5.0);

        assert_eq!(result.get(0, 0), 5.0);
        assert_eq!(result.get(0, 1), 15.0);
        assert_eq!(result.get(1, 0), 25.0);
        assert_eq!(result.get(1, 1), 35.0);
    }

    #[test]
    fn test_subtract_matrix() {
        let mut a = Matrix::zeros(2, 2);
        a.set(0, 0, 10.0);
        a.set(0, 1, 20.0);
        a.set(1, 0, 30.0);
        a.set(1, 1, 40.0);

        let mut b = Matrix::zeros(2, 2);
        b.set(0, 0, 1.0);
        b.set(0, 1, 2.0);
        b.set(1, 0, 3.0);
        b.set(1, 1, 4.0);

        let c = a.subtract(&b);

        assert_eq!(c.get(0, 0), 9.0);
        assert_eq!(c.get(0, 1), 18.0);
        assert_eq!(c.get(1, 0), 27.0);
        assert_eq!(c.get(1, 1), 36.0);
    }

    #[test]
    fn test_elem_divide_scalar() {
        let mut m = Matrix::zeros(2, 2);
        m.set(0, 0, 10.0);
        m.set(0, 1, 20.0);
        m.set(1, 0, 30.0);
        m.set(1, 1, 40.0);

        let result = m.elem_divide(2.0);

        assert_eq!(result.get(0, 0), 5.0);
        assert_eq!(result.get(0, 1), 10.0);
        assert_eq!(result.get(1, 0), 15.0);
        assert_eq!(result.get(1, 1), 20.0);
    }

    #[test]
    #[should_panic]
    fn test_elem_divide_by_zero_scalar() {
        let m = Matrix::zeros(2, 2);
        let _ = m.elem_divide(0.0);
    }

    #[test]
    #[should_panic]
    fn test_elem_divide_by_zero_matrix() {
        let mut a = Matrix::zeros(2, 2);
        a.set(0, 0, 1.0);
        let b = Matrix::zeros(2, 2); // All zeros
        let _ = a.elem_divide(&b);
    }

    #[test]
    fn test_from_vec() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 2), 3.0);
        assert_eq!(m.get(1, 0), 4.0);
        assert_eq!(m.get(1, 2), 6.0);
    }

    #[test]
    #[should_panic]
    fn test_from_vec_wrong_size() {
        // Data has 4 elements, but 2x3 = 6 expected
        let _ = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_map() {
        let m = Matrix::from_vec(2, 2, vec![1.0, 4.0, 9.0, 16.0]);
        let result = m.map(|x| x.sqrt());
        assert_eq!(result.get(0, 0), 1.0);
        assert_eq!(result.get(0, 1), 2.0);
        assert_eq!(result.get(1, 0), 3.0);
        assert_eq!(result.get(1, 1), 4.0);
    }
}
