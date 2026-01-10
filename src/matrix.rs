//! Matrix module for neural network operations.

/// A 2D matrix of f64 values stored in row-major order.
///
/// Data is stored as a flat vector for cache-efficient access.
/// Element at position (row, col) is stored at index: row * cols + col
#[derive(Debug, Clone)]
pub struct Matrix {
    /// Number of rows in the matrix.
    pub rows: usize,
    /// Number of columns in the matrix.
    pub cols: usize,
    /// Flat storage of matrix elements in row-major order.
    pub data: Vec<f64>,
}

impl Matrix {
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
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub trait MatrixOperand {
        fn get_value(&self, index: usize) -> f64;
    }

    impl MatrixOperand for Matrix {
        fn get_value(&self, index: usize) -> f64 {
            self.data[index]
        }
    }

    impl MatrixOperand for f64 {
        fn get_value(&self, _index: usize) -> f64 {
            *self  // Always return the same scalar
        }
    }
    /// Returns the element at the specified position.
    ///
    /// # Panics
    ///
    /// Panics if `row` or `col` is out of bounds.
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

    /// Adds two matrices element-wise.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different dimensions.
    pub fn add(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Matrix dimensions do not match: {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );

        }
        let data: Vec<f64> = self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
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
    pub fn subtract(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Matrix dimensions do not match: {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        let data: Vec<f64> = self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
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
    pub fn elem_multiply(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Matrix dimensions do not match: {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        let data: Vec<f64> = self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
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
    pub fn elem_divide(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Matrix dimensions do not match: {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        let data: Vec<f64> = self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| {
                if *b == 0.0 {
                    panic!("Division by zero in matrix");
                }
                a / b
            })
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
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
        m.get(5, 5); // Should panic, no assert needed
    }
}
