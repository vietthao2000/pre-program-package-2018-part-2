
# coding: utf-8

# # Bài giảng về Đại số tuyến tính - Phần 1
# ## MaSSP 2018, Computer Science

# Bài giảng sau đây giới thiệu một số kiến thức cơ bản về đại số tuyến tính cần thiết để hiểu và thực hiện chương trình <b>Deep Learning</b>.
# 
# Bài giảng này bỏ qua một số kiến thức quan trọng (ví dụ như vector riêng - eigen vector), và không đi sâu vào phân tích tính chất của các khái niệm.

# Đối tượng cơ bản trong đại số tuyến tính là __ma trận__ và __vector__. Ví dụ cơ bản sau đây giới thiệu cách thức tương tác với ma trận trong numpy và được thực hiện với __ma trận 2 chiều__, tuy nhiên các phép toán cơ bản là không thay đổi nếu với số chiều khác.

# Bài giảng này sử dụng thư viện <b>numpy</b> trong Python để thực hiện các ví dụ minh hoạ.

# # 1. Giới thiệu về Numpy
# [Numpy](https://github.com/numpy/numpy) là một trong những thư viện mã nguồn mở viết bằng Python phổ biến nhất hiện nay để làm việc với các phép tính toán khoa học, đặc biệt được sử dụng rộng rãi trong nhiều chương trình <b>Machine Learning</b> nói chung và <b>Deep Learning</b> nói riêng. Các hàm nền tảng để xây dựng các chương trình học máy hầu hết được viết bằng Python và có sử dụng Numpy. 
# 
# Mục tiêu chính của numpy là tối ưu và đơn giản hoá các phép toán liên quan đến đại số tuyến tính. Không chỉ vậy, Numpy còn có nhiều kiểu dữ liệu đa chiều giúp cho việc tính toán, lập trình, làm việc với các hệ cơ sở dữ liệu cực kì thuận tiện.
# 
# Như vậy, việc nắm được cách sử dụng Numpy là một lợi thế lớn để giúp bạn nhanh chóng tiếp cận được với học máy.
# 
# Gói phần mềm này gồm một số phần cơ bản sau:
# 
# * Tập các mảng đa chiều hữu ích
# * Tập các hàm tính toán tinh vi
# * Thuận tiện khi làm việc với đại số tuyến tính
# * Khả năng tạo các mảng số ngẫu nhiên tiện lợi
# * Có thể tích hợp với C/C++ và Fortran
# * Hỗ trợ biến đổi Fourier
# 
# Phần này sẽ không đi chi tiết vào Numpy mà chỉ đề cập tới một số giao diện lập trình (APIs) có thể làm việc được trong bài viết này.
# Để tìm hiểu các APIs khác, các bạn nên tìm hiểu thêm qua [trang chủ của Numpy](http://www.numpy.org/).
# 
# Trước khi bắt đầu, hãy cài đặt numpy, và bắt đầu chương trình Python (hoặc notebook) bằng những lệnh sau:

# In[56]:


import numpy as np


# Lưu ý: khi bạn tải lại notebook này, bạn phải chạy lại lệnh import nói trên.

# # 2. Giới thiệu về vector và ma trận
# Các bạn hãy đọc qua các định nghĩa cơ bản của ma trận và vector trong bài viết [Đỗ Minh Hải - Ma trận là gì?](https://dominhhai.github.io/vi/2017/09/what-is-matrix/) để có cái nhìn tổng quát. Sau đó, các bạn hãy tham khảo note [Intro_vector_and_matrix.ipynb](Intro_vector_and_matrix.ipynb) có kèm giải thích và hình ảnh minh hoạ.

# ## 2.1 Khởi tạo vector và ma trận đơn giản
# Để khởi tạo một vector bằng numpy rất đơn giản, ta chỉ cần đưa cho hàm `numpy.array` một list các phần tử thể hiện vector. Ví dụ để khởi tạo vector `[1, 2]` ta làm như sau:

# In[47]:


v = np.array([1, 2])
print(v)


# Với ma trận, chúng ta hãy bắt đầu với việc khởi tạo ma trận A và in ra các thông tin cơ bản của A.
# 
# Ma trận A có kích thước 2 x 5, các phần tử từ 0 đến 9, theo thứ tự tăng dần từ trái qua phải và từ trên xuống.
# 
# $$ A = 
#     \begin{bmatrix}
#         0 & 1 & 2 & 3 & 4 \\
#         5 & 6 & 7 & 8 & 9
#     \end{bmatrix}
# $$

# Để tạo một ma trận ta có thể sử dụng `ndarray` (viết gọn là `array`) của Numpy.
# 
# Lưu ý rằng mảng `array` của Numpy là khác với mảng thuần của Python.
# Mảng thuần của Python không có được nhiều tiện ích tính toán như của Numpy.
# Về cơ bản, `array` này là một đối tượng mảng đa chiều thuần nhất tức là mọi phần tử đều cùng 1 kiểu.
# 
# Để khởi tạo ma trận A, ta sử dụng $np.array$ và nhóm các phần tử trong cùng một hàng vào một list, và nhóm các hàng vào thành một list lớn hơn như sau:

# In[ ]:


A = np.array([[0, 1, 2, 3, 4],
              [5, 6, 7, 8, 9]])
print(A)


# Hãy thử khởi tạo ma trận sau đây:
# $$ A2 = 
#     \begin{bmatrix}
#         3 & 4 \\
#         5 & 6 \\
#         7 & 8
#     \end{bmatrix}
# $$

# In[ ]:


# code


# _Lưu ý:_ Luôn luôn phải có $[\ ]$ để xác định các phần tử trong ma trận.
# 
# Ví dụ sau đây là <span style="color: red">SAI</span>. Hãy chạy cell này và quan sát lỗi nhận được

# In[ ]:


B1 = np.array(1,2,3,4)


# Quay lại với ma trận $A$ có kích thước 2x5 ban đầu, ta sẽ in các thông tin cơ bản của $A$.
# 
# Trước hết, $A.shape$ sẽ cho thông tin về kích thước của ma trận.

# In[ ]:


""" A
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]]
"""
print("Shape: {}".format(A.shape))      # In thông tin về kích thước của A


# Kết quả thu được là một $tuple$. Để truy cập thông tin của A theo từng chiều, ta sử dụng chỉ số với $A.shape$.

# In[ ]:


""" A
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]]
"""
print(A.shape[0])  
print(A.shape[1])


# Ngoài ra, $A.ndim$ sẽ cho biết số chiều của ma trận, và $A.size$ cho biết số phần tử có trong A.

# In[ ]:


""" A
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]]
"""
print("Dimension: {}".format(A.ndim))   # In số chiều của A

print("Size: {}".format(A.size))        # In số phần tử có trong A


# $np.array$ tự động xác định kiểu dữ liệu của ma trận khi khởi tạo. Hãy in thông tin từ $A.dtype$ để biết kiểu dữ liệu hiện tại của các phần tử trong A.

# In[ ]:


# code


# Tuy vậy ta có thể sử dụng __dtype__ để ép kiểu dữ liệu trong ma trận khi khởi tạo. Ví dụ nếu muốn khởi tạo ma trận $A2$ sao cho các phần tử của $A2$ có kiểu số thực, ta làm như sau:

# In[ ]:


A2 = np.array([[3, 4], [5, 6]], dtype=float)  # Sử dụng kiểu số thực

print(A2)
""" Kết quả:
[[ 3.  4.]
 [ 5.  6.]]
"""

print(A2.dtype)  # Kết quả: float64


# __Checkpoint 1__: Cho vector $b$, bạn hãy in ra các thông tin của $b$ tương tự như đã làm với ma trận $A$.
# $$ b = (6.0, 7.0, 8.0, 9.0, 10.0) $$

# In[ ]:


# Khởi tạo b
b = np.array([6., 7., 8., 9., 10.])
# In thông tin về kích thước

# số chiều

# kiểu dữ liệu

# số phần tử


# ## 2.2. Khởi tạo các ma trận đặc biệt
# 
# * Nếu cần khởi tạo ma trận toàn 0, sử dụng __np.zeros__, kèm theo thông tin về kích thước ma trận

# In[ ]:


A3 = np.zeros((6, 5), dtype=int)
print(A3)

""" Kết quả:
[[0 0 0 0 0]
 [0 0 0 0 0]
 [0 0 0 0 0]
 [0 0 0 0 0]
 [0 0 0 0 0]
 [0 0 0 0 0]]
"""


# * Tương tự như thế, bạn có thể khởi tạo ma trận toàn 1 bằng __np.ones__:

# In[ ]:


A4 = np.ones((4, 2))
print(A4)
""" Kết quả:
[[ 1.  1.]
 [ 1.  1.]
 [ 1.  1.]
 [ 1.  1.]]
"""


# * Ma trận đơn vị được khởi tạo bằng __np.eye__.
# 
# $$ A5 = 
#     \begin{bmatrix}
#         1 & 0 & 0 \\
#         0 & 1 & 0 \\
#         0 & 0 & 1
#     \end{bmatrix}
# $$
# 
# _Lưu ý ma trận đơn vị chỉ có 2 chiều và luôn là ma trận vuông._

# In[ ]:


A5 = np.eye(3)
print(A5)


# * Cuối cùng, __np.arange__ tạo ra vector với các phần tử liên tiếp nhau.
# 
#     __reshape__ được sử dụng để thay đổi kích thước ma trận hoặc vector.
#     
#     Kết hợp 2 hàm này ta sẽ thu được một ma trận với các phần tử liên tiếp từ trái qua phải, trên xuống dưới.

# In[ ]:


A6 = np.arange(10, 20).reshape(2, 5)
print(A6)

""" Kết quả:
[[10 11 12 13 14]
 [15 16 17 18 19]]
"""


# # 3. Truy cập và thay đổi phần tử trong ma trận
# 
# Mặc dù hầu hết các phép toán trong Deep Learning không cần truy cập từng phần tử của ma trận, bạn cũng nên biết một vài cách thức cơ bản để thao tác với từng phần tử trong ma trận.

# Chúng ta sẽ sử dụng ma trận sau để minh hoạ
# 
# $$ A = 
#     \begin{bmatrix}
#         0 & 1 & 2 & 3 & 4 \\
#         5 & 6 & 7 & 8 & 9 \\
#         10 & 11 & 12 & 13 & 14 \\
#         15 & 16 & 17 & 18 & 19
#     \end{bmatrix}
# $$

# In[ ]:


A = np.arange(20).reshape(4, 5)


# Trong numpy (và Python), các phần tử được đánh số từ 0. Numpy cho phép đọc và thay đổi từng phần tử trong ma trận.
# 
# Bạn có thể truy cập phần tử bằng $A[i, j]$...

# In[ ]:


""" A
[[0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18  19]]
"""
print(A[0, 0])
print(A[3, 4])


# ... và thay đổi những phần tử này bằng cách gán chúng với một giá trị khác.

# In[ ]:


A[0, 0] = 19
A[3, 4] = 0
print(A)
""" Kết quả:
[[19  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18  0]]
"""

# khôi phục lại giá trị ban đầu của ma trận A
A[0, 0] = 0
A[3, 4] = 19


# Để truy cập cả một dòng hay một cột, sử dụng dấu 2 chấm __":"__ thay cho chỉ số không cần thiết
# 
# $$ A = 
#     \begin{bmatrix}
#         0 & 1 & 2 & 3 & 4 \\
#         5 & 6 & 7 & 8 & 9 \\
#         10 & 11 & 12 & 13 & 14 \\
#         15 & 16 & 17 & 18 & 19
#     \end{bmatrix}
# $$

# In[ ]:


print(A[0, :])  # In ra dòng đầu tiên. Kết quả: [0 1 2 3 4]
print(A[:, 0])  # In ra cột đầu tiên. Kết quả: [0 5 10 15]


# Để truy cập một ma trận con, sử dụng $A[i_1:i_2, j_1:j_2]$ để lấy ra phần chỉ số cần thiết.
# 
# _Lưu ý: dùng $i_1:i_2$ để kí hiệu các phần tử từ $i_1$ đến $i_2 - 1$._

# $$ A = 
#     \begin{bmatrix}
#         0 & 1 & 2 & 3 & 4 \\
#         5 & 6 & 7 & 8 & 9 \\
#         10 & 11 & 12 & 13 & 14 \\
#         15 & 16 & 17 & 18 & 19
#     \end{bmatrix}
# $$

# In[ ]:


print(A[2:4, 2:5])
""" Kết quả
[[12 13 14]
 [17 18 19]]
"""


# In[ ]:


print(A[0:3, :])
""" Kết quả
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
"""


# Ma trận hoặc vector con có thể được gán sang biến khác và truy cập / thay đổi tương ứng.
# 
# **CẢNH BÁO**: Python không sao chép ma trận trong phép gán =, cho nên LUÔN LUÔN sử dụng copy để tạo ra một ma trận mới và gán = với ma trận đó.

# In[ ]:


# Ví dụ SAI
B1 = A[0:1, 0:3]
B1[0, 0] = -1   # Phép toán này thay đổi giá trị tương ứng trong A
print(B1)
""" Kết quả
[[-1  1  2]]
"""

print(A)
""" Kết quả
[[-1  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
"""


# In[ ]:


# Khôi phục lại giá trị đúng của A
A[0, 0] = 0


# In[ ]:


# Ví dụ đúng
B2 = A[0:1, 0:3].copy()
B2[0, 0] = -1
print(B2)
""" Kết quả
[[-1  1  2]]
"""

print(A)
""" Kết quả
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
"""


# Bạn có thể thay đổi nhiều hơn 1 phần tử trong 1 lệnh như sau

# In[ ]:


A1 = A.copy()      # tạo A1 là copy của A để không thay đổi giá trị của A
A1[0:2, 0:3] = -1
print(A1)
""" Kết quả
[[-1 -1 -1  3  4]
 [-1 -1 -1  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
"""


# __Checkpoint 2__: Hãy thực hiện những thao tác trên với vector $b$
# $$ b = (6.0, 7.0, 8.0, 9.0, 10.0) $$

# In[ ]:


# Khởi tạo b

# Truy cập 1 phần tử tồn tại trong b

# Thử truy cập 1 phần tử không tồn tại trong b - quan sát lỗi

# Tạo một vector con của b

# Thay đổi một phần tử trong b và in giá trị mới


# # 4. Tính toán trên ma trận
# 
# Các phép toán trên ma trận là những tính toán cơ bản khi làm việc với học máy.
# Trong phần này, tôi không đề cập đầy đủ tất cả các phép toán của nó,
# mà chỉ đề cập tới các phép toán cơ bản để có thể dùng được với học máy cơ bản.
# 
# Chúng ta sẽ thực hiện các tính toán trong phần này trên ma trận sau:
# $$ A = 
#     \begin{bmatrix}
#         1 & 2 & 3 \\
#         10 & 15 & 20
#     \end{bmatrix}
# $$

# In[ ]:


# code khởi tạo A


# ## 4.1. Nhân ma trận với một số scalar
# Nhân ma trận với một số (vô hướng) $\alpha$ là phép nhân số đó với từng phần tử của ma trận.
# 
# $$ \alpha [A _{ij}] _{mn} = [\alpha . A _{ij}] _{mn} $$
# 
# Ví dụ:
# $$
# 5 \begin{bmatrix}
# 1 & 2 & 3  \\
# 10 & 15 & 20
# \end{bmatrix}
#  = \begin{bmatrix}
# 5 & 10 & 15  \\
# 50 & 75 & 100
# \end{bmatrix}
# $$
# 
# Các tính chất:
# 
# * Tính giao hoán: $ \alpha A = A \alpha $
# * Tính kết hợp: $ \alpha(\beta A) = (\alpha \beta) A $
# * Tính phân phối: $ (\alpha + \beta) A = \alpha A + \beta A $
# 
# Ngoài ra, nhân ma trận với 1 sẽ không làm thay đổi ma trận: $ 1A = A $,
# còn nhân với 0 sẽ biến ma trận thành ma trận không $ 0A = \bf 0_{m\times n} $.

# Trong numpy, bạn có thể thực hiện các phép toán tác động lên __toàn bộ__ các phần tử của ma trận y như bạn thao tác trên một phần tử riêng biệt.
# 
# Khi viết $A * c$ hoặc $A / c$, numpy sẽ nhân (chia) tất cả các phần tử của $A$ cho $c$.

# In[ ]:


A = np.array([[1, 2, 3], [10, 15, 20]], dtype=float)
print(A * 3)
print(A / 2)


# __Checkpoint 3__: Làm thế nào để thực hiện hàm $sin$, $exp$, và bình phương lên ma trận $A$?

# In[ ]:


# sin

# exp

# bình phương


# Những phép toán trên cũng có thể được thực hiện lên một ma trận con của A một cách tương tự.

# In[ ]:


A1 = A.copy()
A1[:, 0] *= 3      # cách viết này tương đương với A1[:, 0] = A1[:, 0] * 3
print(A1)


# ## 4.2 Cộng 2 ma trận
# Là phép cộng từng phần tử tương ứng của <span style="color:blue">2 ma trận cùng cấp</span> với nhau.
# 
# $$ [A _{ij}] _{mn} + [B _{ij}] _{mn} = [A _{ij} + B _{ij}] _{mn} $$
# 
# Ví dụ:
# $$
# \begin{bmatrix}
# 5 & 10 & 15  \\
# 20 & 25 & 30
# \end{bmatrix} +
# \begin{bmatrix}
# 1 & 2 & 3  \\
# 4 & 5 & 6
# \end{bmatrix} =
# \begin{bmatrix}
# 6 & 12 & 18  \\
# 24 & 29 & 36
# \end{bmatrix}
# $$
# 
# Các tính chất:
# 
# * Tính giao hoán: $ A + B = B + A $
# * Tính kết hợp: $ A + (B + C) = (A + B) + C $
# * Tính phân phối: $ \alpha (A + B) = \alpha A + \alpha B $
# 
# Ngoài ra, dễ dàng thấy rằng cộng một ma trận với ma trận không thì không làm thay đổi ma trận đó: $ A + \varnothing = A $.
# 
# Từ phép nhân ma trận với một số, ta có thể định nghĩa được phép trừ ma trận là phép trừ từng phần tử tương ứng trong ma trận: $ A - \lambda B = A + (-\lambda)B$ với $\lambda \in \mathbb{R} $.

# Hãy khởi tạo ma trận B như dưới đây, và thực hiện phép cộng $A+B$, và phép trừ $A-B$.
# 
# $A = \begin{bmatrix}
#         1 & 2 & 3 \\
#         10 & 15 & 20
#     \end{bmatrix}; 
#  \ \ \ B = \begin{bmatrix}
#         1 & 1 & 1 \\
#         2 & 3 & 4
#     \end{bmatrix}
# $

# In[ ]:


# code


# ## 4.3 Nhân 2 ma trận
# Nhân 2 ma trận là phép lấy tổng của tích từng phần tử của hàng tương ứng với cột tương ứng.
# Phép nhân này chỉ <span style="color:blue">khả thi khi số cột của ma trận bên trái bằng với số hàng của ma trận bên phải</span>.
# Cho 2 ma trận $ [A]_{mp} $ và $ [B]_{pn} $, tích của chúng theo thứ tự đó sẽ là một ma trận có số hàng bằng với số hàng của $ A $ và số cột bằng với số cột của $ B $: $ [AB]_{mn} $.
# 
# $$C_{ij} = (AB)_{ij} = \sum_{k=1}^p {A_{ik} B_{kj}}~~~, \forall{i = \overline{1,m}; j = \overline{1,n}}$$
# 
# Ví dụ:
# $$
# \begin{bmatrix}
# 1 & 2 & 3  \\
# 4 & 5 & 6
# \end{bmatrix}
# \begin{bmatrix}
# 1 & 2  \\
# 3 & 4  \\
# 5 & 6
# \end{bmatrix} =
# \begin{bmatrix}
# 22 & 28  \\
# 49 & 64
# \end{bmatrix}~~~~~~~~~~~ (1)
# $$
# 
# $$
# \begin{bmatrix}
# 1 & 2 & 3  \\
# 4 & 5 & 6
# \end{bmatrix}
# \begin{bmatrix}
# 1 & 0 & 0  \\
# 0 & 1 & 0  \\
# 0 & 0 & 1
# \end{bmatrix} =
# \begin{bmatrix}
# 1 & 2 & 3  \\
# 4 & 5 & 6
# \end{bmatrix}~~~ (2)
# $$

# Các tính chất:
# 
# * Tính kết hợp: $ A(BC) = (AB)C $
# * Tính phân phối: $ A(B+C) = AB + AC $, $ (A+B)C = AC + BC $.
# 
# >Lưu ý là phép nhân 2 ma trận không có tính chất giao hoán: $ AB \not = BA $.
# 
# Nếu bạn để ý ở công thức 2 phía trên thì sẽ thấy rằng việc nhân với ma trận đơn vị không làm thay đổi ma trận đó: $ AI = IA = A $. (Xem lại định nghĩa của [ma trận đơn vị](https://dominhhai.github.io/vi/2017/09/what-is-matrix/#2-4-ma-tr%E1%BA%ADn-%C4%91%C6%A1n-v%E1%BB%8B)) 
# 
# Cách biểu diễn với Numpy:

# In[3]:


# create matrix A
A = np.array([(1, 2, 3), (4, 5, 6)])
print(A)
# [[1 2 3]
#  [4 5 6]]

# create matrix B
B = np.array([(0, 5), (4, 9), (9, 0)])
print(B)
# [[0 5]
#  [4 9]
#  [9 0]]

# product of A and B
C = A.dot(B)
print(C)
# [[35 23]
#  [74 65]]

C = np.dot(A, B)
# [[35 23]
#  [74 65]]


# __Checkpoint 3__: Làm thế nào để thực hiện phép nhân 2 ma trận này?
# 
# $A = \begin{bmatrix}
#         1 & 2 & 3 \\
#         10 & 15 & 20
#     \end{bmatrix}; 
#  \ \ \ B = \begin{bmatrix}
#         1 & 1 & 1 \\
#         2 & 3 & 4
#     \end{bmatrix}
# $

# Nhân 2 ma trận $AB$ yêu cầu số cột của A bằng số dòng của B!
#     
# **CẢNH BÁO**: tuyệt đối không sử dụng $*$ để nhân ma trận theo đúng nghĩa nhân ma trận trong toán học. $A*B$ sẽ trả về ma trận mới với mỗi phần tử là tích của 2 phần tử ở vị trí tương ứng trong A và B, gọi là phép nhân từng phần tử Hamadard.

# In[4]:


A = np.array([[1, 2, 3], [10, 15, 20]])
B = np.array([[1, 1, 1], [2, 3, 4]])

A*B


# Tuy vậy, chúng ta có thể thực hiện phép nhân giữa ma trận $A$ và chuyển vị của ma trận $B$.

# ## 4.4 Chuyển vị ma trận (matrix transpose)
# Chuyển vị là phép biến cột thành hàng và hàng thành cột của một ma trận.
# Cho ma trận $ [A] _{mn} $ thì chuyển vị của nó là $ [B _{ij}] _{nm} = [A _{ji}] _{mn}^\intercal $ ($ \intercal $ là kí hiệu của phép chuyển vị)
# có $ B _{ij} = A _{ji} ~~~, \forall i,j $. Ví dụ:
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
# Các tính chất:
# 
# * $ (A^\intercal)^\intercal = A $
# * $ (A + B)^\intercal = A^\intercal + B^\intercal $
# * $ (AB)^\intercal = B^\intercal A^\intercal $
# * $ (\alpha A)^\intercal = \alpha A^\intercal $   (với scalar: $\alpha^\intercal = \alpha$)
# 
# Ngoài ra, ta có thể thực hiện phép nhân số học 2 vector để thu được 1 scalar bằng cách nhân với chuyển của 1 trong 2  vector: $ a\cdot b = a^\intercal b = \sum_{i=1}^n a_i b_i$.
# Phép nhân kiểu này còn được gọi là phép nhân vô hướng (`scalar product`, còn được gọi là `inner/dot product`), tức là tổng của tích mỗi phần tử tương ứng của 2 vectors.
# 
# Trong numpy, ta chuyển vị một ma trận bằng hàm $np.transpose$.

# In[5]:


print(np.transpose(A))


# In[ ]:


# code


# ## 4.5 Ma trận nghịch đảo (inverse matrix)
# Ma trận nghịch đảo của ma trận <span style="color:blue">vuông khả nghịch</span> $ A $ cấp n là ma trận $ B $ sao cho tích của chúng là ma trận đơn vị cùng cấp: $ AB = I_n $.
# Ma trận nghịch đảo được kí hiệu là $ A^{-1} $, tức là $ A A^{-1} = A^{-1}A = I_n $. Ma trận vuông $A$ có ma trận nghịch đảo gọi là khả nghịch (`invertible`); không phải tất cả các ma trận vuông đều khả nghịch.
# 
# Các tính chất:
# 
# * $ (A^{-1})^{-1} = A $
# * $ (kA)^{-1} = k^{-1} A^{-1} ~~~, \forall k \not = 0 $
# * $ (AB)^{-1} = B^{-1} A^{-1} $
# * $ (A^\intercal)^{-1} = (A^{-1})^\intercal $
# 
# Ngoài ra nếu để ý sẽ thấy ma trận đơn vị luôn có nghịch đảo là chính nó: $ I_n^{-1} = I_n $. Ma trận zero không tồn tại nghịch đảo, nghĩa là không khả nghịch.
# 
# Để xem một ma trận vuông có khả nghịch hay không và cách tìm ma trận nghịch đảo tương ứng của nó tôi sẽ trình bày trong bài viết tới. Tạm thời bạn cứ nắm được khái niệm của nó và cách lập trình đã nhé.

# In[6]:


# create matrix a
A = np.array([[1., 2.], [3., 4.]])

# inverse matrix of A
B = np.linalg.inv(A)
print(B)
# [[-2.   1. ]
#  [ 1.5 -0.5]]


# ## 4.6. Phép nhân từng phần tử Hadamard
# Là phép nhân từng phần tử tương ứng của <span style="color:blue">2 ma trận cùng cấp</span> với nhau.
# 
# $$ [A _{ij}] _{mn} \circ [B _{ij}] _{mn} = [A _{ij}  B _{ij}] _{mn} $$
# 
# Ví dụ:
# $$
# \begin{bmatrix}
# 5 & 10 & 15  \\
# 20 & 25 & 30
# \end{bmatrix} \circ
# \begin{bmatrix}
# 1 & 2 & 3  \\
# 4 & 5 & 6
# \end{bmatrix} =
# \begin{bmatrix}
# 5 & 20 & 45  \\
# 80 & 115 & 180
# \end{bmatrix}
# $$
# 
# Các tính chất:
# 
# * Tính giao hoán: $ A \circ B = B \circ A $
# * Tính kết hợp: $ A \circ (B \circ C) = (A \circ B) \circ C $
# * Tính phân phối: $ A \circ (B + C) = A \circ B + A \circ C $
# 

# In[ ]:


# create matrix a
a = np.array([(1, 2, 3), (4, 5, 6)])
print(a)
# [[1 2 3]
#  [4 5 6]]

# create matrix b
b = np.array([(0, 5, 25), (4, 9, 9)])
print(b)
# [[0 5 25]
#  [4 9  9]]

# Multiplying of a and b
c = a * b
print(c)
# [[ 0 10 75]
#  [16 45 54]]

c = np.multiply(a, b)
print(c)
# [[ 0 10 75]
#  [16 45 54]]


# Ngoài phép nhân Hadamard theo từng phần tử, ta cũng có các phép biến đổi khác tương tự như:
# 
# Phép chia Hadamard: $ [A _{ij}] _{mn} \oslash [B _{ij}] _{mn} = [A _{ij} / B _{ij}] _{mn} $.
# 
# Phép lũy thừa Hadamard: $ [A _{ij}] _{mn}^p = [A _{ij}^p] _{mn} ~~~, \forall p \in \mathbb{R} $
# 
# Từ phép lũy thừa với số mũ phân số, ta có thể viết lại dưới dạng phép khai căn Hadamard: $ \sqrt[p]{[A _{ij}] _{mn}} = [\sqrt[p]{A _{ij}}] _{mn} ~~~, \forall p \in \mathbb{N} $

# In[ ]:


# create matrix a
a = np.array([(1., 2., 3.), (4., 5., 6.)])
print(a)
# [[1. 2. 3.]
#  [4. 5. 6.]]

# create matrix b
b = np.array([(0., 5., 25.), (4., 9., 9.)])
print(b)
# [[0. 5. 25.]
#  [4. 9.  9.]]

# divide
c = b / a # c = np.divide(b, a)
print(c)
# [[ 0.          2.5         8.33333333]
#  [ 1.          1.8         1.5       ]]

# power of 2 (luỹ thừa)
c = a ** 2
print(c)
# [[  1.   4.   9.]
#  [ 16.  25.  36.]]


# ## Bài tập

# __Bài tập 1__: Hãy khởi tạo 1 ma trận $A$ dưới đây, sau đó dùng numpy để đảo ngược vị trí các phần tử trong từng hàng.
# 
# $$
# A =    \begin{bmatrix}
#         1 & 2 & 3 \\
#         10 & 15 & 20
#     \end{bmatrix}
#  =>
#     \begin{bmatrix}
#         3 & 2 & 1 \\
#         20 & 15 & 10
#     \end{bmatrix}
# $$

# In[ ]:


# code


# **Bài tập 2**: Sử dụng numpy để giải quyết bài toán sau đây.
# 
# Cho hệ phương trình sau:
# $$
# \begin{cases}
# 2x - 3y = 1 \\
# x + 2y = 3
# \end{cases}
# $$
# 
# * Viết lại hệ phương trình dưới dạng $Az = b$
# * Mô tả ma trận $A$, vector $b$ và $z$
# * Tìm $z$ 
# 
# Để xem các lệnh bạn có thể sử dụng được trong Numpy, hãy truy cập https://docs.scipy.org/doc/numpy/user/.

# In[ ]:


# code

