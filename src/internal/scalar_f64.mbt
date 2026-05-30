// Scalar f64 reference impls. Linear algebra (matmul / gemv / transpose) +
// element-wise + reductions over FixedArray[Double].

///|
pub fn scalar_mean_f64(arr : FixedArray[Double]) -> Double {
  let len = arr.length()
  if len == 0 {
    return 0.0
  }
  scalar_sum_f64(arr) / len.to_double()
}

///|
/// Population variance (1/n * sum (x - mu)^2).
pub fn scalar_var_f64(arr : FixedArray[Double]) -> Double {
  let len = arr.length()
  if len == 0 {
    return 0.0
  }
  let mu = scalar_mean_f64(arr)
  let mut acc = 0.0
  for i in 0..<len {
    let d = arr[i] - mu
    acc = acc + d * d
  }
  acc / len.to_double()
}

///|
/// matmul with B already in transposed form (row-major B^T).
/// A is M x K, B^T is N x K, C is M x N - all row-major.
pub fn scalar_matmul_f64(
  a : FixedArray[Double],
  b_t : FixedArray[Double],
  c : FixedArray[Double],
  m : Int,
  k : Int,
  n : Int,
) -> Unit {
  for i in 0..<m {
    for j in 0..<n {
      let mut acc = 0.0
      for kk in 0..<k {
        acc = acc + a[i * k + kk] * b_t[j * k + kk]
      }
      c[i * n + j] = acc
    }
  }
}

///|
/// gemv: y = A * x.  A is M x N row-major, x is N, y is M.
pub fn scalar_gemv_f64(
  a : FixedArray[Double],
  x : FixedArray[Double],
  y : FixedArray[Double],
  m : Int,
  n : Int,
) -> Unit {
  for i in 0..<m {
    let mut acc = 0.0
    for j in 0..<n {
      acc = acc + a[i * n + j] * x[j]
    }
    y[i] = acc
  }
}

///|
/// Transpose: out[j*m + i] = a[i*n + j].
pub fn scalar_transpose_f64(
  a : FixedArray[Double],
  out : FixedArray[Double],
  m : Int,
  n : Int,
) -> Unit {
  for i in 0..<m {
    for j in 0..<n {
      out[j * m + i] = a[i * n + j]
    }
  }
}

///|
pub fn scalar_add_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = a[i] + b[i]
  }
}

///|
pub fn scalar_sub_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = a[i] - b[i]
  }
}

///|
pub fn scalar_mul_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = a[i] * b[i]
  }
}

///|
pub fn scalar_div_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = a[i] / b[i]
  }
}

///|
pub fn scalar_sqrt_f64(
  a : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = a[i].sqrt()
  }
}

///|
pub fn scalar_sum_f64(arr : FixedArray[Double]) -> Double {
  let len = arr.length()
  let mut acc = 0.0
  for i in 0..<len {
    acc = acc + arr[i]
  }
  acc
}

///|
pub fn scalar_dot_f64(a : FixedArray[Double], b : FixedArray[Double]) -> Double {
  let len = a.length()
  let mut acc = 0.0
  for i in 0..<len {
    acc = acc + a[i] * b[i]
  }
  acc
}

///|
/// Horizontal max reduction. Caller guarantees `arr` is non-empty.
pub fn scalar_max_reduce_f64(arr : FixedArray[Double]) -> Double {
  let mut acc = arr[0]
  for i in 1..<arr.length() {
    if arr[i] > acc {
      acc = arr[i]
    }
  }
  acc
}

///|
/// Horizontal min reduction. Caller guarantees `arr` is non-empty.
pub fn scalar_min_reduce_f64(arr : FixedArray[Double]) -> Double {
  let mut acc = arr[0]
  for i in 1..<arr.length() {
    if arr[i] < acc {
      acc = arr[i]
    }
  }
  acc
}

///|
/// Element-wise min (out[i] = min(a[i], b[i])).
pub fn scalar_min_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = a[i].min(b[i])
  }
}

///|
pub fn scalar_max_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = a[i].max(b[i])
  }
}
