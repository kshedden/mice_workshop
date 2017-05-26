Handling missing data in Python Statsmodels with MICE
-----------------------------------------------------

[Link](
https://github.com/statsmodels/statsmodels/blob/master/statsmodels/imputation/mice.py) to the Statsmodels MICE source code

In a regression analysis, individual data values may be missing for
either the dependent variable or for one one or more of the
independent variables.  Most standard regression analysis techniques
such as linear modeling using ordinary least squares (OLS) cannot
automatically handle such missing values.  Here we will discuss
several ways to incorporate data with missing data values into a
regression analysis, focusing in particular on an approach called
_Multiple Imputation with Chained Equations_ (MICE).

In a standard regression analysis with one dependent variable (DV) and
one or more independent variables (IV's), four basic patterns of
missingness can occur, as described below (comments below about
"information" are not precise and may depend on additional aspects of
the missingness mechanism):

* _All IV's and the DV are missing_: in general such an observation
  contains no useful information and can be excluded from the
  analysis.

* _All IV's are observed, but the DV is missing_: usually an
  observation with this missingness pattern can be excluded, as it
  contains no information about the regression parameters.  However
  in some settings (e.g. a survey analysis in which the goal is to
  estimate a population total), it should be retained.  We do not
  consider this setting further here.

* _The DV is observed, but all of the IV's are missing_: in most
  settings there will be very little information in these cases about
  the model parameters, however in some settings, e.g. logistic
  regression, these cases could contribute a small amount of
  information about the prevalence.

* _The DV is observed, and some, but not all of the IV's are
  observed_: cases with this missingness pattern can potentially
  contain a lot of information about the regression parameters,
  especially if the missing IV's can be predicted from the non-missing
  IV's.  This missingness pattern is the main focus of the methods
  discussed here, as we can aim to recover some of the information
  present in cases with this missingness pattern.

## Missingness mechanisms

This is a big topic, we only cover it briefly here.  The essential
issue is whether the status of different data values being missing are
independent random events.  To formalize this, let I(i, j) = 1 if
covariate j in observation i is missing, and I(i, j) = 0 otherwise.
Similarly, let J(i) = 1 if the outcome (DV) is missing for observation
i, and J(i) = 0 otherwise.

The most basic type of missingness mechanism is "missing completely at
random" (MCAR).  This means that whether a given data value is missing
is completely independent of whether any other data value is missing,
and further is unrelated to the values of any of the covariates, or to
the values of the outcome (DV).  Stated slightly more formally, this
means that the elements of I and J are mutually independent of each
other, and of X and Y.

A slightly more realistic missingness mechanism is "missing at random"
(MAR).  This means that whether a value is missing is independent of
any other value being missing, conditioned on the observed data.
Slightly more precisely, it means that the elements of I and J are
mutually independent, given Xobs and Yobs (the observed components of
X and Y).

Any other missing data pattern falls into the broad category of
"missing not at random" (MNAR) data.  This includes many settings
where J(i) depends on Y(i), such as in a survey where people with high
income are less likely to report their income.

MNAR data are much harder to handle rigorously than MCAR or MAR data.
Here we focus on methods that are known to work well for MCAR and MAR
data.

In general, the performance of procedures for handling missing data
can be evaluated in terms of their bias and variance for estimating
parameters of interest.  The goal of almost any statistical procedure
is to have small or negligible bias, while also having the smallest
achievable variance.  Unfortunately, the bias and variance of most
procedures for handling missing data will depend on the missingness
mechanism, the details of which are generally unknown.  There is a
good theoretical understanding of how various procedures for handling
missing data compare in relative terms in the MAR, MCAR, and in
(certain) MNAR settings.

## Complete case analysis

A very simple way to deal with missing data in a regression analysis
is to drop all cases with any missing values, either in the IV's or in
the DV.  This is called "complete case analysis", or sometimes,
"list-wise deletion".  Most statistical models in Statsmodels can be
fit using complete case analysis by specifying the `missing='drop'`
keyword argument.

Complete case analysis will not introduce bias in the MAR/MCAR
setting, but substantial bias can result by applying it in the MNAR
setting.  As for variance, complete case analysis has higher variance
than other options (discussed below) in the MAR and MCAR settings.
Its main advantage is its simplicity.

## Single imputation

Single imputation refers to any procedure that replaces each missing
values with a prediction of the unobserved data value.  This includes
imputation procedures based on the mean as well as various
regression-based procedures that we will not cover further here.  It
also includes variations on the "last observation carried forward"
approach for time series or other sequential data.

We will not discuss single imputation in detail here, but see the
documentation for the Pandas [fillna](
https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html)
method for procedures that facilitate single imputation with Pandas
data frames.

## Multiple imputation

Multiple Imputation (MI) is any setting in which the missing values
are imputed multiple times, resulting in several complete data sets.
The imputation should be random, so that the different imputed data
sets are different from each other.  In addition, the imputation
should be conducted such that the variation in the imputed values for
a given missing value reflects the uncertainty in our ability to
predict that value.

The usual approach for MI is to fit the model of interest to each
imputed data set (which is straightforward since the imputed data sets
are complete).  Say we do this m times.  We then obtain m estimated
values for the parameter of interest, b(1), ..., b(m), and m
corresponding standard errors, s(1), ..., s(m).  These values are then
combined as follows:

* The estimates are pooled taking their mean: b* = [b(1) + ... +
  b(m)]/m.

* The standard deviations are pooled by averaging their squares: v =
  [s(1)^2 + ... + s(m)^2]/m.  This is the average uncertainty in b*
  (on the variance scale) if we had observed complete data.

* The uncertainty due to missingness is the variance of b(1), ...,
  b(m).  Call this vm.

* The standard error of b* can be estimated as sqrt(v + vm).  That is,
  the uncertainty in estimating b with b* is given as the sum of the
  uncertainty we would have had with a complete data set, and the
  uncertainty due to the imputation.

This pooling procedure is sometimes referred to as "Rubin's rule".
There is a slightly more complex form that depends on m (the number of
imputed data sets).  If m is large enough (say m ~ 20) the simple
approach given above is accurate enough.

## Multiple Imputation with Chained Equations (MICE)

MICE is one way to produce multiply imputed data sets, that can then
be analyzed as described in the preceding section.  To illustrate,
suppose we have a regression with a DV Y and two IV's X1 and X2.  The
basic idea of MICE is that we first temporarily impute the missing
values using any convenient procedure, say mean imputation.  This
produces a "working data set".  We then fit three regression models to
the working data:

* Y ~ X1 + X2

* X1 ~ Y + X2

* X2 ~ Y + X1

These models could be fit using OLS, GLM, or any other modeling
procedure.  Next, a technique called _predictive mean matching_ (PMM)
is used to update the missing values.  To illustrate, suppose Y(1) is
missing.  We obtain the fitted value Y_hat(i) for i=1, ..., n, and
obtain the k closest Y_hat(j) values to Y_hat(i), where k is a tuning
parameter of the PMM procedure.  We then randomly choose on index
value from this set, say j*, and update the imputed Y(1) with Y(j*).
