
# coding: utf-8

# # Bổ trợ bài giảng về Đại số tuyến tính - Phần 1
# ## MaSSP 2018, Computer Science
# Tài liệu ngắn này đưa ra định nghĩa một số khái niệm cơ bản trong đại số tuyến tính liên quan đến vector và ma trận.
# 
# # 1. Một số khái niệm
# ## 1.1. Vô hướng (Scalar)
# Một `scalar` là một số bất kì thuộc tập số nào đó.
# Khi định nghĩa một số ta phải chỉ rõ tập số mà nó thuộc vào (gọi là `domain`).
# Ví dụ, $ n $ là số tự nhiên sẽ được kí hiệu: $ n \in \mathbb{N} $ (Natural numbers),
# hoặc $ x $ là số thực sẽ được kí hiệu: $ x \in \mathbb{R} $ (Real numbers).
# Trong Python số tự nhiên có thể là kiểu `int`, số thực có thể là kiểu `float`.
# <!---
# Một số thường có thể định nghĩa được bằng một kiểu dữ liệu nguyên thủy của các ngôn ngữ lập trình.
# Như số tự nhiên có thể là kiểu `int`, số thực có thể là kiểu `float` trong Python.
# --->

# In[1]:


x = 1
print(type(x))


# In[2]:


y = 2.0
print(type(y))


# ## 1.2. Véc-tơ (Vector)
# `Vector` là 1 mảng của các vô hướng scalars tương tự như mảng 1 chiều trong các ngôn ngữ lập trình.
# Các phần tử trong  vector cũng được đánh địa chỉ và có thể truy cập nó qua các địa chỉ tương ứng của nó.
# Trong toán học, một  vector có thể là  vector cột (`column vector`) nếu các nó được biểu diễn dạng một cột nhiều hàng, hoặc có thể là  vector hàng (`row vector`) nếu nó được biểu diễn dưới dạng một hàng của nhiều cột.
# 
# Một  vector cột có dạng như sau:
# 
# $$
# x =
# \begin{bmatrix}
# x_1 \\
# x_2 \\
# \vdots \\
# x_n
# \end{bmatrix}
# $$
# 
# Một  vector hàng có dạng như sau:
# $$
# x =
# \begin{bmatrix}
# x_1, &
# x_2, &
# \cdots &
# x_n
# \end{bmatrix}
# $$
# 
# Trong đó, $ x_1 $, $ x_2 $, ..., $ x_n $ là các phần tử `thứ 1`, `thứ 2`, ... `thứ n` của  vector.
# Lưu ý trong lập trình Python ta đánh số từ `0`: $x[0] = x_1, x[1] = x_2,...$.

# ## 1.3. Ma trận (Matrix)
# Ma trận là một mảng 2 chiều của các vô hướng tương tự như mảng 2 chiều trong các ngôn ngữ lập trình. Ví dụ dưới đây là một ma trận có $ m $ hàng và $ n $ cột:
# $$
# A =
# \begin{bmatrix}
# A _{1, 1} & A _{1, 2} & \cdots & A _{1, n} \\
# A _{2, 1} & A _{2, 2} & \cdots & A _{2, n} \\
# \vdots    & \vdots    & \vdots & \vdots    \\
# A _{m, 1} & A _{m, 2} & \cdots & A _{m, n}
# \end{bmatrix}
# $$
# 
# Khi định nghĩa một ma trận ta cần chỉ rõ số hàng và số cột cùng trường số của các phần tử có nó.
# Lúc này, $ mn $ được gọi là cấp của ma trận.
# Ví dụ, ma trận số thực $ A $ có m hàng và n cột được kí hiệu là: $ A \in \mathbb{R}^{m \times n} $.
# 
# Các phần tử trong ma trận được định danh bằng 2 địa chỉ hàng $ i $ và cột $ j $ tương ứng.
# Ví dụ phần tử hàng thứ 3, cột thứ 2 sẽ được kí hiệu là: $ A_{3,2} $.
# Ta cũng có thể kí hiệu các phần tử của hàng $ i $ là $ A _{i,:} $ và của cột $ j $ là $ A _{:,j} $.
# Nếu bạn để ý thì sẽ thấy $ A _{i,:} $ chính là  vector hàng, còn $ A _{:,j} $ là  vector cột.
# Như vậy,  vector có thể coi là trường hợp đặt biệt của ma trận với số hàng hoặc số cột là 1.
# 
# Các ma trận sẽ được kí hiệu: $ [A _{ij}] _{mn} $, trong đó $ A $ là tên của ma trận;
# $ m, n $ là cấp của ma trận; còn $ A _{ij} $ là các phần tử của ma trận tại hàng $ i $ và cột $ j $.
# 
# <!---
# Các  vector ta cũng sẽ biểu diễn tương tự.
#  vector hàng: $ [x_i]_n $, trong đó $ x $ là tên của  vector;
# $ n $ là cấp của  vector; $ x_i $ là phần tử của  vector tại vị trí $ i $.
#  vector cột ta sẽ biểu diễn thông qua phép chuyển vị của  vector hàng: $ [x_i]_n ^\intercal  $.
# 
# Ngoài ra, nếu một ma trận được biểu diễn dưới dạng: $ [A _{1j}] _{1n} $ thì ta cũng sẽ hiểu ngầm luôn nó là  vector hàng.
# Tương tự, với $ [A _{i1}] _{m1} $ thì ta có thể hiểu ngầm với nhau rằng nó là  vector cột.
# --->
# 
# Một điểm cần lưu ý nữa là các giá trị $ m, n, i, j $ khi được biểu điễn tường minh dưới dạng số, ta cần phải chèn dấu phẩy `,` vào giữa chúng.
# Ví dụ: $ [A _{ij}] _{9,4} $ là ma trận có cấp là `9, 4`. $ A _{5,25} $ là phần tử tại hàng `5` và cột `25`.
# Việc này giúp ta phân biệt được giữa ma trận và vector, nếu không ta sẽ bị nhầm ma trận thành vector.

# ## 1.4. Ten-xơ (Tensor)
# Tensor là một mảng nhiều chiều, nó là trường hợp tổng quát của việc biểu diễn số chiều.
# Như vậy, ma trận có thể coi là một  tensor 2 chiều,  vector là  tensor một nhiều còn scalar là tensor zero chiều.
# 
# Các phần tử của một  tensor cần được định danh bằng số địa chỉ tương ứng với số chiều của tensor đó. Ví dụ mộ  tensor 3 chiều $A$ có phần tử tại hàng $ i $, cột $ j $, cao $ k $ được kí hiệu là $ A_{i,j,k} $.
# <img src="./images/tensor1.png" alt="Tensor" style="height: 50%; width: 50%;"/>
# 
# Ví dụ ảnh trắng đen hoặc xám (`grayscale`) được biểu diễn bằng ma trận 2 chiều. Giá trị của mỗi phần tử trong ma trận là một số thập phân nằm trong khoảng từ 0 đến 1, ứng với độ đen trắng của từng điểm ảnh (`pixel`) (0 thể hiện màu đen và giá trị càng gần tới 1 thì càng trắng). Do hình ảnh có chiều dài và chiều rộng, ma trận của các điểm ảnh là ma trận 2 chiều.
# <img src="./images/MNIST_2.png" alt="grayscale" style="height: 25%; width: 25%;"/>
# 
# Một ảnh màu được biểu diễn bằng một tensor 3 chiều, 2 chiều đầu cũng để đánh số địa chỉ mỗi điểm ảnh dọc theo chiều dài và chiều rộng của ảnh. Chiều cuối cùng để phân biệt 3 màu cơ bản đỏ, xanh lá, xanh dương ($k=1,2,3$). Như vậy mỗi điểm ảnh được xác định bởi vị trí của nó, và thành phần 3 màu cơ bản.
# <img src="./images/tensor2.png" alt="color" style="height: 50%; width: 50%;"/>
# 
# Vậy đố các bạn biết, một đoạn phim đen trắng sẽ được biểu diễn bằng tensor mấy chiều? Một đoạn phim màu thì sao?
# <img src="./images/tensor4.png" alt="video" style="height: 50%; width: 50%;"/>
# 

# # 2. Một số ma trận đặc biệt
# ## 2.1. Ma trận không (zero matrix)
# Ma trận `zero` là ma trận mà tất cả các phần tử của nó đều bằng 0: $ A_{i,j} = 0, \forall{i,j}  $. Ví dụ:
# 
# $$
# \varnothing =
# \begin{bmatrix}
# 0 & 0 & 0 & 0 \\
# 0 & 0 & 0 & 0 \\
# 0 & 0 & 0 & 0
# \end{bmatrix}
# $$
# Ta có thể viết $\bf 0_{m\times n}$ để chỉ ma trận zero có size $m\times n$.
# 
# ## 2.2. Ma trận vuông (square matrix)
# Ma trận vuông là ma trận có số hàng bằng với số cột: $ A \in R^{n \times n} $.
# Ví dụ một ma trận vuông cấp 3 (số hàng và số cột là 3) có dạng như sau:
# 
# $$
# A =
# \begin{bmatrix}
# 2 & 1 & 9 \\
# 4 & 5 & 9 \\
# 8 & 0 & 5
# \end{bmatrix}
# $$
# 
# Với ma trận vuông, đường chéo bắt đầu từ góc trái trên cùng tới góc phải dưới cùng được gọi là đường chéo chính: $ \{ A _{i,i} \} $. Ký hiệu $\{ \cdots \}$ dùng để chỉ một tập hợp (`set`). Trong ví dụ trên, đường chéo chính đi qua các phần tử `2, 5, 5`.

# ## 2.3. Ma trận chéo
# Ma trận chéo là ma trận vuông có các phần từ nằm ngoài đường chéo chính bằng 0: $ A_{i,j} = 0, \forall{i \not = j} $.
# Ví dụ ma trận chéo cấp 4 (có 4 hàng và 4 cột) có dạng như sau:
# 
# $$
# A =
# \begin{bmatrix}
# 1 & 0 & 0 & 0 \\
# 0 & 2 & 0 & 0 \\
# 0 & 0 & 3 & 0 \\
# 0 & 0 & 0 & 4
# \end{bmatrix}
# $$
# 
# > Lưu ý rằng ma trận vuông zero (ma trận vuông có các phần tử bằng 0) cũng là một ma trận chéo, ký hiệu $\bf 0_n$.

# ## 2.4. Ma trận đơn vị
# Là ma trận chéo có các phần tử trên đường chéo bằng 1:
# $$
# \begin{cases}
# A _{i,j} = 0, \forall{i \not = j} \\
# A _{i,j} = 1, \forall{i = j}
# \end{cases}
# $$
# 
# Ma trận đơn vị được kí hiệu là $ I_n $ với $ n $ là cấp của ma trận. Ví dụ ma trận đơn vị  cấp 3 được biểu diễn như sau:
# 
# $$
# I_{3} =
# \begin{bmatrix}
# 1 & 0 & 0 \\
# 0 & 1 & 0 \\
# 0 & 0 & 1
# \end{bmatrix}
# $$
# 
# <!--- đã nói ở phần định nghĩa ma trận
# ## 2.5. Ma trận cột
# Ma trận cột chính là  vector cột, tức là ma trận chỉ có 1 cột.
# 
# ## 2.6. Ma trận hàng
# Tương tự như ma trận cột, ma trận hàng chính là  vector hàng, tức là ma trận chỉ có 1 hàng.
# --->

# ## 2.5. Ma trận chuyển vị
# Ma trận chuyển vị là ma trận nhận được sau khi ta đổi hàng thành cột và cột thành hàng.
# 
# $$
# \begin{cases}
# A \in \mathbb{R}^{m\times n} \\
# B \in \mathbb{R}^{n\times m} \\
# A _{i,j} = B _{j,i}, \forall{i,j}
# \end{cases}
# $$
# 
# Ma trận chuyển vị của $ A $ được kí hiệu là $ A^\intercal $. Như vậy: $ (A^\intercal)_{i,j} = A _{j,i} $.
# 
# $$
#     \begin{bmatrix}
#         1 & 2 & 3 \\
#         10 & 15 & 20
#     \end{bmatrix}
# ^\intercal =
#     \begin{bmatrix}
#         1 & 10 \\
#         2 & 15 \\
#         3 & 20
#     \end{bmatrix}
# $$
# 
# Vector cũng là một ma trận nên mọi phép toán với ma trận đều có thể áp dụng được cho vector, bao gồm cả phép chuyển vị ma trận.
# Sử dụng phép chuyển vị ta có thể biến một  vector hàng thành  vector cột và ngược lại.
# 
# Mặc định (`by default, convention`) trong toán học khi cho một vector $x\in\mathbb{R}^n$ ta hiểu đây là một vector cột. Đôi lúc để viết cho ngắn gọi người ta thường sử dụng phép chuyển vị để định nghĩa vector cột, ví dụ $ x = [x_1, x_2, ..., x_n]^\intercal $.
# 
# <!---Do đó ở ví dụ về vector hàng, theo chuẩn ta nên viết $x^{\top} =
# \begin{bmatrix}
# x_1, &
# x_2, &
# \cdots &
# x_n
# \end{bmatrix}$. --->
# <!---
# # 3. Các kí hiệu
# Để thuận tiện, từ nay về sau tôi sẽ mặc định các vô hướng, phần tử của ma trận (bao gồm cả  vector) mà chúng ta làm việc là thuộc trường số thực $ \mathbb{R} $. Tôi cũng sẽ sử dụng một số kí hiệu bổ sung như dưới đây.
# 
# Các ma trận sẽ được kí hiệu: $ [A _{ij}] _{mn} $, trong đó $ A $ là tên của ma trận;
# $ m, n $ là cấp của ma trận; còn $ A _{ij} $ là các phần tử của ma trận tại hàng $ i $ và cột $ j $.
# 
# Các  vector ta cũng sẽ biểu diễn tương tự.
#  vector hàng: $ [x_i]_n $, trong đó $ x $ là tên của  vector;
# $ n $ là cấp của  vector; $ x_i $ là phần tử của  vector tại vị trí $ i $.
#  vector cột ta sẽ biểu diễn thông qua phép chuyển vị của  vector hàng: $ [x_i]_n ^\intercal  $.
# 
# Ngoài ra, nếu một ma trận được biểu diễn dưới dạng: $ [A _{1j}] _{1n} $ thì ta cũng sẽ hiểu ngầm luôn nó là  vector hàng.
# Tương tự, với $ [A _{i1}] _{m1} $ thì ta có thể hiểu ngầm với nhau rằng nó là  vector cột.
# 
# Một điểm cần lưu ý nữa là các giá trị $ m, n, i, j $ khi được biểu điễn tường minh dưới dạng số,
# ta cần phải chèn dấu phẩy `,` vào giữa chúng.
# Ví dụ: $ [A _{ij}] _{9,4} $ là ma trận có cấp là `9, 4`. $ A _{5,25} $ là phần tử tại hàng `5` và cột `25`.
# Việc này giúp ta phân biệt được giữa ma trận và  vector, nếu không ta sẽ bị nhầm ma trận thành  vector.
# 
# Trên đây là một số khái niệm cơ bản để làm việc với ma trận, trong phần sau tôi sẽ đề cập tới các phép toán của ma trận.
# Việc biến đổi ma trận và các phép toán trên ma trận là rất cần thiết để làm việc với các bài toán về học máy sau này.
# --->
