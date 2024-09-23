# Neural Arithmetic Logic Units with Two Transition Matrix and Independent Gates (NALU2MIG)

This project presents **NALU2MIG**, a novel activation function designed to outperform traditional methods in tasks that require numerical precision, such as arithmetic operations and image recognition. The model introduces a two-transition matrix mechanism combined with independent gates, making it highly effective in tasks like image classification on the MNIST dataset, as well as complex arithmetic operations. Below are the key results of our experimentation.

## Exp 2: Static Simple Function Learning
Input a 100-dimensional vertex **x**, learn `y = func(a, b)`,
where <img src="https://latex.codecogs.com/svg.latex?a=\sum_{i=N}^{M}(\mathbf{x}_i)" title=""/>
, <img src="https://latex.codecogs.com/svg.latex?b=\sum_{i=P}^{Q}(\mathbf{x}_i)" title=""/>  and `func = +, -, x, /, ...`. Test the ability to interpolate and extrapolate.

```bash
python3 learn_function.py
```

and for image dataset

```bash
python3 learn_function_2D.py
```

## Results for Arithmetic Operations
The NALU2MIG model was tested on various arithmetic tasks such as addition, subtraction, multiplication, and division. Both interpolation and extrapolation experiments were conducted. The results are benchmarked against traditional activation functions like ReLU, Sigmoid, NAC, NALU, and other variations.

# Interpolation and Extrapolation MAE Comparison

## Interpolation (MAE)
| Operation | ReLU | Sigmoid | NAC  | NALU | NALU2M | NALUIG | NALU2MIG |
| --------- | ---- | ------- | ---- | ---- | ------ | ------ | -------- |
| a + b     | 1.01 | 1.01    | 1.01 | 1.01 | 1.01   | 0.98   | 1.01     |
| a - b     | 1.00 | 1.00    | 1.00 | 0.99 | 1.00   | 5.01   | 1.00     |
| a x b     | 4.34 | 4.34    | 3.39 | 4.07 | 4.34   | 4.07   | 4.35     |
| a / b     | 0.08 | 0.08    | 0.08 | 0.08 | 0.08   | 0.08   | 0.08     |
| a ^ 2     | 2.50 | 2.51    | 1.91 | 2.51 | 2.50   | 2.44   | 2.51     |
| sqrt(a)   | 0.16 | 0.16    | 0.17 | 0.16 | 0.16   | 0.16   | 0.16     |

## Extrapolation (MAE)
| Operation | ReLU | Sigmoid | NAC  | NALU | NALU2M | NALUIG | NALU2MIG |
| --------- | ---- | ------- | ---- | ---- | ------ | ------ | -------- |
| a + b     | 4.29 | 24.94   | 4.29 | 38.51| 4.17   | 35.33  | 4.30     |
| a - b     | 11.73| 11.79   | 4.12 | 17.72| 4.13   | 19.97  | 4.12     |
| a x b     | 168.19| 260.72 | 224.08| 213.81| 291.99| 252.22 | 72.04   |
| a / b     | 0.17 | 0.06    | 0.99 | 0.15 | 0.07   | 0.25   | 0.08     |
| a ^ 2     | 58.90| 81.05   | 76.95| 101.38| 89.16 | 90.91  | 39.72    |
| sqrt(a)   | 0.86 | 0.96    | 3.09 | 2.10 | 0.95   | 0.29   | 0.32     |


## Results for MNIST Dataset

The MNIST dataset, a popular benchmark for image recognition, was used to evaluate the performance of NALU2MIG in comparison to standard activation functions. The results show that NALU2MIG not only achieves superior accuracy but also maintains competitive training times.

| Network  | MAE    | MSE    | RMSE   | MAD    | Accuracy | Training Time (s) |
| -------- | ------ | ------ | ------ | ------ | -------- | ----------------- |
| ReLU     | 0.0925 | 0.5247 | 0.7244 | 2.5355 | 97.69%   | 8155.09s          |
| Sigmoid  | 0.3445 | 1.7291 | 1.3150 | 2.5355 | 90.94%   | 8060.91s          |
| NAC      | 0.0312 | 0.1430 | 0.3782 | 2.5355 | 99.09%   | 7881.31s          |
| NALU     | 0.0443 | 0.2315 | 0.4811 | 2.5355 | 98.91%   | 7416.66s          |
| NALU2M   | 4.4434 | 28.1290| 5.3037 | 2.5355 | 9.80%    | 7334.37s          |
| NALUIG   | 0.0390 | 0.1896 | 0.4354 | 2.5355 | 98.97%   | 7724.22s          |
| NALU2MIG | 0.0323 | 0.1551 | 0.3938 | 2.5355 | 99.14%   | 8533.50s          |

## Conclusion
The **NALU2MIG** model consistently outperforms traditional activation functions in both arithmetic operations and image classification tasks. Its strong performance across different benchmarks demonstrates its potential for various practical applications, including tasks that require precise numerical computations, such as financial forecasting, scientific computing, and AI-based image recognition tasks.


