# COMP5421 Homework4 Report

-----









## Problem 2.1. Eight-Point Algorithm

----

The matrix F is:

```python
[[ 9.80213865e-10 -1.32271663e-07  1.12586847e-03]
 [-5.72416248e-08  2.97011941e-09 -1.17899320e-05]
 [-1.08270296e-03  3.05098538e-05 -4.46974798e-03]]
```



And the image view is:

![](D:\Year3S\COMP5421\assignment\HW4_code_data\yyubm.assets\eight_point.png)



## Problem 2.2 Seven-Point Algorithm.

----

We randomly picked the seven points from the database and compute the fundamental matrix.

We tried several times and find that once we could get result like this:

```python
 [[ 9.05782568e-08  2.10609816e-08  9.84762396e-04]
  [-3.91155235e-07  1.71458944e-07 -5.79509543e-04]
  [-9.75321578e-04  5.31396432e-04  9.02122037e-03]]
```

And the visualization is:

![](D:\Year3S\COMP5421\assignment\HW4_code_data\yyubm.assets\seven2.png)

**First, it is purely random and largely depends on our chosen data points. Sometimes we got some very bad result.**

**Second, this is still not as good as the previous 8-point algorithm.**

> Below is another try for the algorithm and we get a good result. But since it is purely random I could not recurrent it.

![](D:\Year3S\COMP5421\assignment\HW4_code_data\yyubm.assets\seven_point.png)



## Problem 3.1.

---

The essential matrix is 

```python
[[-1.92592123e-02  3.00526429e-01 -1.73693252e+00]
 [ 1.51113724e-01  1.32873151e-02 -3.08885271e-02]
 [ 1.73986815e+00  9.11774760e-02  3.90697726e-04]]
```



## Problem 3.2. Triangulate

---

```python
[[-0.9998386   0.01763739  0.00342032 -0.02599827]
 [ 0.01789571  0.99453463  0.10286209 -1.        ]
 [-0.00158741  0.1029067  -0.99468975  0.07961991]]
```

