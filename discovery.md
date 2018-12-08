# 
## feature transformation
create new features (maybe combine features)

# process
## spot anomaly
1. data.describe()
2. see graph

## Invalid/null fields
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

## group values
OverallQual: 1-3: Bad, 4-6: average, 7-10: good
OverallCond: 1-3: Bad, 4-6: average, 7-10: good

## discard value
SaleCondition: discard abnormal?

## inconsistent test & train data
LotFrontage: train > 200 [2 record]
LotArea: train > 100,000 [4 record]




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

