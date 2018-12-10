# Data cleaning
## spot anomaly
1. data.describe()
2. see graph

## Fill Missing
total: 1460

*LotFrontage      1201 (NA) => mean
Alley            91  => X
MasVnrType       1452 (NA) => None
MasVnrArea       1452 (NA) => 0
*BsmtQual         1423 (NA) => guess by calc
*BsmtCond         1423 (NA) => guess by calc
BsmtExposure     1422 (NA)
BsmtFinType1     1423 (NA) => X
BsmtFinType2     1422 (NA) => X
*Electrical       1459 (NA) => Sbrkr
FireplaceQu      770
GarageType       1379
GarageYrBlt      1379
GarageFinish     1379
GarageQual       1379
GarageCond       1379
PoolQC           7 
Fence            281 
MiscFeature      54 
------------
BsmtFinSF1    (0) => keep?
BsmtFinSF2    (0)

## Outlier
SaleCondition: discard abnormal?


# Data transformation
source 80 -> 510 fields

## Discretization
OverallQual: 1-3: Bad, 4-6: average, 7-10: good
OverallCond: 1-3: Bad, 4-6: average, 7-10: good
YearBuilt
YearRemodAdd
LotShape: (Reg),(IR1,IR2,IR3)
BsmtFinSF1Level (0, 0.5, 1)
TotalBsmtSFLevel (0, 0.5, 1)
1stFlrSFLevel
2ndFlrSFLevel
GrLivAreaLevel
GarageAreaLevel
OpenPorchLevel

## label -> num
MSSubClass
MSZoning
Street: 0,1
LotShape
LandContour
Utilities
LotConfig
Alley
ExterQual
ExterCond
BsmtQual
BsmtCond
BsmtExposure
HeatingQC (0, 0.25, 0.5, 0.75, 1)

## OneHotEncoding
Street
Alley
LotShape
LandContour
Utilities (only allpub,nosewr afterall)
LotConfig
LandSlope
Neighborhood
Condition1
Condition2
BldgType
HouseStyle
RoofStyle
RoofMatl (compshg,tar,wdshake,wdshngl) [only 4/8]
Exterior1st
Exterior2nd
MasVnrType
ExterQual
ExterCond
Foundation
BsmtQual
BsmtCond (no Ex)
BsmtExposure
BsmtFinType1
BsmtFinType2
Heating
HeatingQC
CentralAir (really need?)
Electrical
KitchenQual
Functional
FireplaceQu
GarageType
GarageFinish
GarageQual
GarageCond
PavedDrive
PoolQC
Fence
MiscFeature
MoSold
SaleType
SaleCondition

## IsExist (more important, 1/0)
LotShape: Regular
LandContour: Lvl
LandSlope: Gentle
LotConfig: Insdie
Condition1: Norm
Condition2: Norm
BldgType: 1Fam
RoofStyle: Gable
RoofMatl: CompShg
Heating: GasA
PavedDrive: Y|(P,N)
HasEnclosedPorch: >0
Has3SsnPorch: >0
HasScreenPorch: >0
SaleType: WD
SaleCondition: Normal

## normalization
LotFrontage
LotArea
Overallqual
OverallCond
YearBuilt
YearRemodAdd
MasVnrArea
BsmtFinSF1
BsmtFinSF2
BsmtUnfSF
TotalBsmtSF
1stFlrSF
2ndFlrSF
LowQualFinSF (better treatment?)
GrLivArea
FullBath
HalfBath
TotRmsAbvGrd
Fireplaces
FireplaceQu
GarageYrBlt
GarageCars
GarageArea
OpenPorchSF
EnclosedPorch
3SsnPorch
ScreenPorch
PoolArea (mostly 0)
Fence
MiscVal
YrSold

## unchange
BsmtFullBath
BsmtHalfBath

## new field
- top 15 important features add ^2, ^3, sqrt new fields
LotAreaLevel [s2,s3,sq]
MSSubClass (mean)
LotFrontage [s2,s3,sq]
SimplOverallQual [s2,s3,sq]
SimplOverallCond
Now_YearBuilt
Now_YearRemodAdd
MasVnrAreaLevel
SimplExterQual
SimplExterCond
SimplBsmtQual
SimplBsmtCond
SimplBsmtFinType1
BsmtFinSF1 [s2,s3,sq]
SimplBsmtFinType2 
BsmtUnfSF [s2,s3,sq]
SimplHeatingQC
2ndFlrSF [s2,s3,sq]
GrLivArea [s2,s3,sq]
SimplKitchenQual
SimplFunctional
FireplaceScore [s2,s3,sq]
SimplFireplaceQu
Now_GarageYrBlt
GarageYrBlt [s2,s3,sq]
SimplGarageQual
SimplGarageCond
TotalPorchSF
PoolScore (combine PoolArea, PoolQC)
SimplPoolQC
YearSoldMeanPrice
MonthSaledMeanPrice
MonthSaledCount

## inconsistent test & train data
LotFrontage: train > 200 [2 record]
LotArea: train > 100,000 [4 record]







=======================================================

1. one-hot encoded categorical variable
2. split into features and data
3. convert to array
4. split data into training and testing

# SalePrice VS xxx
## Strong response
OverallQual
GrLivArea
GarageArea
Neighborhood (Some district obviously better)
YearBuilt (newer -> higher price)
ExterQual
BsmtQual
BsmtCond
TotalBsmtSF
1stFlrSF
2ndFlrSF (very strong correlation)
GarageArea

## Normal response
BsmtFinSF1
FullBath
TotRmsAbvGrd
Fireplaces


## Very weak response
LotArea
LotFrontage
ExterCond
BsmtFinSF2
BsmtFullBath
BsmtHalfBath
BedroomAbvGr
GarageCars
WoodDeckSF
SaleType
SaleCondition

## Nearly no relation
MasVnrArea
BsmtUnfSF
Heating
GarageYrBlt
OpenPorchSF
EnclosedPorch
3SsnPorch
ScreenPorch
PoolArea (too few data)
MiscVal

## Absolutely not related
MoSold
YrSold

Zonning? <- think no need to care

