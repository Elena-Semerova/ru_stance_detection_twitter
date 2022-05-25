# "Analysis of Recent Topics in Social Networks Using Natural Language Processing"

The purpose of our work was to collect Russian-language dataset for Stance Detection task based on posts by users of social network Twitter, also we made some experiments on our data. We called our dataset `TwiRuS` (Twitter Russian Stances).

Collected dataset: [TwiRuS](https://drive.google.com/drive/u/0/folders/16YCOfDelmdFguYzyPIZO2x7YnscJQ5Dh)

In final version of our dataset we have 5 targets (topics): `cancel culture`, `feminism`, `LGBTQ+`, `ageism` and `lookism`. Additionally we collected sentiment labels for each tweet.

### Total count of tweets for each topic

| Topic | Count of tweets |
|------------|:------------:|
| cancel culture | 1567 |
| feminism | 4375 |
| LGBTQ+ | 12332 |
| ageism | 1620 |
| lookism | 1290 |

### Total count of labels for stances and sentiments

| Stance Label | Count of tweets | Sentiment Label | Count of tweets |
|--------------|:---------------:|-----------------|:---------------:|
| Favor | 5916 | Positive | 2972 |
| Against | 6476 | Negative | 9514 |
| Neutral | 8792 | Neutral | 8698 |

### Distributions of stance and sentiment labels for each topic

#### cancel culture

<img src="https://sun1.userapi.com/sun1-91/s/v1/ig2/Z3uDWjhmAnTKga2G6hn1fsF0_TPzuetlJVqhMFQ8RmNqcdEvdioxRDmw1GAk61tpz33PYmfO7J-CHW0dl6ZU94dI.jpg?size=864x504&quality=95&type=album" width="400"/> <img src="https://sun1.userapi.com/sun1-56/s/v1/ig2/oRpVrKYOZHLSEZjbTgt4zbd3uKvIjqJtfXAqlRtkbfaRIGrH6ZU3yEz2IGKqVITA7Hi1zHYEooXmlZQbJVXgxVwJ.jpg?size=864x504&quality=95&type=album" width="400"/> 

#### feminism

<img src="https://sun1.userapi.com/sun1-89/s/v1/ig2/IcZv61CkcgDV0IEqwwIHgBRY3HxLkq7vwZHvyrDIXdNX1mTtywiFEeI-2lHvXqDRfqYbx6wa6ZHgii6v2fEde4do.jpg?size=864x504&quality=95&type=album" width="400"/> <img src="https://sun1.userapi.com/sun1-25/s/v1/ig2/DVy7NLvqj6jL649RKV_EyACU29MnBWNs0UrNRJsi8TrJMJdzy8lY-cjRt-bIo8f4wQGOfrk7W8d_qVdGto9v4gXj.jpg?size=864x504&quality=95&type=album" width="400"/> 

#### LGBTQ+

<img src="https://sun1.userapi.com/sun1-16/s/v1/ig2/nWL9dmiAVmMQohq4l9ka92zFym0QNEXu3wj209PDQmxbMBLSfw4EPatTLopJaM6MC1qZZblO_jsI9toOPz7LLoFZ.jpg?size=864x504&quality=95&type=album" width="400"/> <img src="https://sun9-east.userapi.com/sun9-28/s/v1/ig2/I3e7RbGO6tSFObn4YEv63L__9jGDLJjvZKXt7kwRq79SG62DeFENSpOyXzYEWDdUfX0cxbFKioc-meumqtJ2R3_P.jpg?size=864x504&quality=95&type=album" width="400"/> 

#### ageism

<img src="https://sun1.userapi.com/sun1-26/s/v1/ig2/1xp_gUaSdjM1mMAuCNKeLO1nIlQc5P3wMRNielrINunJmytrGbrY-AAY42bSwxsNnu5zn3ik67jU6m1UtaRTP7gV.jpg?size=864x504&quality=95&type=album" width="400"/> <img src="https://sun1.userapi.com/sun1-94/s/v1/ig2/Prw13Y64J768rz2YpQKnNG6yTqIT0BL2Et3GpDGvquxQ0p-m5SJJG_NnnH6vNBXD2U6u5h41mC3ZU-s4aCmg0ZGg.jpg?size=864x504&quality=95&type=album" width="400"/> 

#### lookism

<img src="https://sun9-west.userapi.com/sun9-16/s/v1/ig2/vCGEz8rEBRPdDX4HpnYySvgsHCGFXPJjlmE0_5_33ddmu06dthnSiuhkO751Cbd_k7UPfS426RQYE7szmBxnaW0N.jpg?size=864x504&quality=95&type=album" width="400"/> <img src="https://sun1.userapi.com/sun1-19/s/v1/ig2/xtiz_PFipiPlMB1CwW-4TlCh6eGY_JJOxIb6MzXwOnRCjwyoGfPQpnryQbuLCvhPHJsq1DjHNZcWqaXNuxhIrsgi.jpg?size=864x504&quality=95&type=album" width="400"/> 

