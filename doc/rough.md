
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

