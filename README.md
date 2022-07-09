# probabilistic_matrix_factorization

## MovieLens Dataset
| Dataset       | num of users | num of items | num of ratings |
|---------------|--------------|--------------|----------------|
| MovieLens100K |     1,000    |     1,700    |      100K      |
| MovieLens1M   |     6,000    |     4,000    |       1M       |
| MovieLens10M  |    72,000    |    10,000    |       10M      |
| MovieLens20M  |    138,000   |    27,000    |       20M      |

## Training details:
### MovieLens100K（收敛）
latent feature dim = 10

batch size = 1024

number of epoch = 500

learning rate = 0.1

optimizer = SGD

momentum = 0.9

weight_decay = 1e-4

![ml100k](https://user-images.githubusercontent.com/13879402/177429223-d873f447-fab6-4d56-9035-f95f642d0d5e.png)
### MovieLens1M（收敛）
latent feature dim = 15

batch size = 1024

number of epoch = 200

learning rate = 0.1

optimizer = SGD

momentum = 0.9

weight_decay = 1e-4

![ml1m](https://user-images.githubusercontent.com/13879402/177429312-a794f91c-41d4-4a92-bd67-b69df0872207.png)
### MovieLens10M（有收敛趋势，但未接近1，后面尝试调参）
latent feature dim = 30

batch size = 8096

number of epoch = 800

learning rate = 0.1

optimizer = SGD

momentum = 0.9

weight_decay = 1e-4

![ml10m](https://user-images.githubusercontent.com/13879402/178045191-9aa9ab97-bf5b-4c68-931f-5352e9ff99f4.png)

### MovieLens20M（收敛性更差）
latent feature dim = 30

batch size = 20000

number of epoch = 500

learning rate = 0.1

optimizer = SGD

momentum = 0.9

weight_decay = 1e-4

![rmse_curve](https://user-images.githubusercontent.com/13879402/178113855-ae5e632b-add7-460d-a2bc-77ac10c1be21.png)
