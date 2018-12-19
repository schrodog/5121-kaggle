# Note
{ } = explanation

## Data Cleaning
remove Id, SalePrice
- remove outlier
lotFrontage, LotArea, MasVnrArea
- where data range exist only in training, not testing data
combine test & train dataset

### find missing data, fill missing
LotFrontage: plot median, mean; group by neighborhood
Alley: 'NA' ??
MasVnrType: most frequent
Bsmt*: string->'NA'
BsmtFinSF1/2: 0
BsmtFinType1, BsmtFinType2: 'Unf'
  {basement of type1 and type2 are different, one can already finished and another is not finished}

Electrical: missing 1, {almost normal data} => most freq
FireplaceQu: 'NA'
  { refer to no fireplace, Fireplaces=0}
PoolQC: 'NA'
  {becoz poolArea=0 for all NA}
MiscFeature,Fence: 'NA'
GarageFinish, GarageType, GarageQual, GarageType: 'NA'


## Feature engineering
### add new features
1. year 
RemodAge = YrSold - YearRemodAdd
HouseAge = YrSold - YearBuilt
Oldness = HouseAge*0.5 + RemodAge
GarageAge = -1 Or YrSold - max(GarageYrBlt, YearRemodAdd)

2. grade, value
OverallGrad = OverallQual + OverallCond
GarageGrade = GarageQual + GarageCond
ExterGrade = ExterQual + ExterCond
KitchenValue = KitchenAbvGr * KitchenQual
FireplaceValue = Fireplaces * FireplaceQu
GarageValue = GarageArea * GarageQual
PoolValue = PoolArea * PoolQC
  {PoolArea only 7 record, 400 < x < 750 }

BsmtValue = BsmtFinType1*BsmtFinSF1 + BsmtFinType2*BsmtFinSF2 + 0.2*BsmtUnfSF + BsmtCond*BsmtQual*TotalBsmtSF*0.3
{BsmtFinSF2: >0 count=167, BsmtFinSF1 >0 count=467
BsmtUnfSF >0 count=1342
}

BathValue = BsmtFullBath + 0.5*BsmtHalfBath + 2*FullBath + 1.5*HalfBath

3. total area
TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch
{count_0: ScreenPorch=1344, 3SsnPorch=1436}

TotalSF = 1stFlrSF + 2ndFlrSF + GrLivArea + 0.4*(LowQualFinSF + TotalBsmtSF) + 0.1*(WoodDeckSF + TotalPorchSF + PoolArea + LotArea + GarageArea)


### change features
1. find features with very low var
- change to binary variable
'RoofStyle', 'LotShape', 'MSZoning', 'Electrical', 'Condition2', 'RoofMatl', 'Alley', 'Heating', 'LandSlope', 'MiscFeature', 'CentralAir', 'PoolQC', 'Street', 'Utilities'

RoofStyle -> Gable, Hip, others
LotShape -> Reg, IR1, others
LandSlope -> Gtl, others
MSZoning -> RL, RM, FV, others
Heating -> GasA, others
Alley -> Yes,No
Electrical -> SBrkr, others

### Drop features
CentralAir
PoolQC {almost all None...} ?
Condition2
RoofMatl
Street
Utilities
MiscFeature {Only Shed is meaningful}

### Discretize (num -> class)
MiscVal: 0 -> 0, >0 -> 1
 {1408 records == 0}

GarageQual, GarageCond -> 0-5
GarageQual, GarageCond -> 0-5


### OneHot Encoding
Integer Encoding: for field with natural ordering, comparable
OneHot Encoding: no apparent ordering



- discover month more house sold, price inverse with sold count
- price,count ~ MSSubClass

- Encode comparable label with num
Street, Alley, ...

Neighborhood location -> numerical latitude, longitude -> NeighborDistance
NeighborDistance = sqrt( (longitude-min)^2 + (latitude-min)^2 )

bin Neighborhood [0-4]
NeighborPrice, NeighborBin

- separate string,number features
create binary fields on popular features
eg. LotShape, LandContour ...

- simplify existing feature, discretize


Neighborhood binning

- find important feature by XGBRegressor
15 most import feature
d^2, d^3, sqrt(d)

- for boolean features, do not scatter and skewed
use max scater, 95% quantile to scale data

- transform skewed numeric features by taking log(feature+1)
- make features more normal
skewed > 0.75
NeighborPrice, NeighborPrice-s2, NeighborPrice-s3 -> log(1+x)
MonthSaledMeanPrice, MSSubClassMeanPrice, NeighborPrice.. -> log(1+x)

- drop columns missing in test data
exterior1st_imstucc,stone, ...













