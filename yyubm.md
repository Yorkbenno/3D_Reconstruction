# COMP5421 Homework4 Report

-----

## Problem 1. Theory

![](D:\Year3S\COMP5421\assignment\HW4_code_data\yyubm.assets\Assignment2problem1_1.jpg)

![](D:\Year3S\COMP5421\assignment\HW4_code_data\yyubm.assets\Assignment2problem1_2.jpg)





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
[[ 2.26587821e-03 -3.06867395e-01  1.66257398e+00]
 [-1.32799331e-01  6.91553934e-03 -4.32775554e-02]
 [-1.66717617e+00 -1.33444257e-02 -6.72047195e-04]]
```



## Problem 3.2. Triangulate

---

The matrix of M2 is attached below for reference:

```python
[[ 0.99942697  0.03331533  0.00598477 -0.02599827]
 [-0.03372859  0.96531605  0.25889634 -1.        ]
 [ 0.00284802 -0.25894984  0.96588657  0.07961991]]
```

![](D:\Year3S\COMP5421\assignment\HW4_code_data\yyubm.assets\Assignment2Q3_1.jpg)



## Problem 4.1.  Epipolar Correspondence

----

We finally choose window size 4 which means we will generate a 9*9 matrix each time.

We iterate from y - 25 to y + 25 and find the one with minimum error.

The results are shown below:

![](D:\Year3S\COMP5421\assignment\HW4_code_data\yyubm.assets\Q4.1results.png)

We could see that the results are quiet good.



## Problem 4.2. 3D Visualization

-----

![](D:\Year3S\COMP5421\assignment\HW4_code_data\yyubm.assets\1.png)

![](D:\Year3S\COMP5421\assignment\HW4_code_data\yyubm.assets\2.png)

![](D:\Year3S\COMP5421\assignment\HW4_code_data\yyubm.assets\3.png)

![](D:\Year3S\COMP5421\assignment\HW4_code_data\yyubm.assets\4.png)